"""
知识库向量检索服务。

本模块把知识库文档写入本地 FAISS 向量索引，并在查询时执行向量召回。
设计目标是：
1. 沿用项目现有的 `Qwen Embedding + FAISS + 本地 JSON metadata` 技术路线。
2. 为知识库检索补齐真正的向量 RAG 基础，而不是只做关键词匹配。
3. 在 embedding 或 FAISS 不可用时优雅退化，不影响主问答链路和 CRUD。

当前策略采用“全量知识库单索引 + 查询时再按 tenant/category 过滤”的方式，
实现简单且足够支撑当前项目规模。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeDocument
from app.db.session import DATA_DIR

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = Any  # type: ignore[assignment]

try:
    import faiss
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
    np = None


VECTOR_DIR = DATA_DIR / "faiss"
INDEX_PATH = VECTOR_DIR / "knowledge_documents.index"
METADATA_PATH = VECTOR_DIR / "knowledge_documents.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_BATCH_SIZE = 10

logger = logging.getLogger("uvicorn.error")


def is_available() -> bool:
    """只有 FAISS 和 numpy 都可用时，才启用向量链路。"""

    return faiss is not None and np is not None


def rebuild_knowledge_index(db: Session) -> None:
    """根据当前启用的知识库文档重建整个向量索引。"""

    if not is_available():
        logger.info("FAISS unavailable, skip knowledge vector rebuild")
        return

    documents = list(
        db.scalars(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.enabled.is_(True))
            .order_by(KnowledgeDocument.id.asc())
        ).all()
    )
    VECTOR_DIR.mkdir(exist_ok=True)

    if not documents:
        _save_metadata([])
        _rebuild_index([], _embedding_dimensions())
        return

    metadata = [_serialize_document(item) for item in documents]
    try:
        embeddings = _embed_documents([item["document"] for item in metadata])
    except ValueError as exc:
        logger.warning("Knowledge vector rebuild skipped: %s", exc)
        return

    _save_metadata(metadata)
    _rebuild_index(embeddings, _embedding_dimensions())
    logger.info(
        "Knowledge vector index rebuilt, document_count=%s index=%s",
        len(metadata),
        INDEX_PATH,
    )


def search_knowledge_documents(
    db: Session,
    query: str,
    tenant_code: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """执行知识库向量检索，并返回文档 ID 与分数。"""

    if not is_available() or not query.strip():
        return []

    metadata = _load_metadata()
    if not metadata or not INDEX_PATH.exists():
        rebuild_knowledge_index(db)
        metadata = _load_metadata()
    if not metadata or not INDEX_PATH.exists():
        return []

    index = faiss.read_index(str(INDEX_PATH))
    dimensions = _embedding_dimensions()
    if index.d != dimensions:
        rebuild_knowledge_index(db)
        if not INDEX_PATH.exists():
            return []
        index = faiss.read_index(str(INDEX_PATH))

    try:
        query_vector = np.array([_embed_query(query)], dtype="float32")
    except ValueError as exc:
        logger.warning("Knowledge vector search skipped: %s", exc)
        return []

    raw_top_n = max(top_n * 5, 20)
    scores, positions = index.search(query_vector, raw_top_n)
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0 or position >= len(metadata):
            continue
        item = metadata[int(position)]
        if tenant_code and item.get("tenant_code") not in {tenant_code, None, ""}:
            continue
        if category and item.get("category") != category:
            continue
        results.append(
            {
                "id": int(item["id"]),
                "vector_score": float(score),
            }
        )
        if len(results) >= top_n:
            break
    return results


def _serialize_document(document: KnowledgeDocument) -> dict[str, Any]:
    """把 ORM 文档转成 metadata 记录。"""

    return {
        "id": document.id,
        "tenant_code": document.tenant_code,
        "title": document.title,
        "category": document.category,
        "source_url": document.source_url,
        "document": _build_document_text(document),
    }


def _build_document_text(document: KnowledgeDocument) -> str:
    """构造用于 embedding 的文档文本。"""

    return "\n".join(
        part
        for part in [
            document.title or "",
            document.category or "",
            document.keywords or "",
            document.content or "",
        ]
        if part
    )


def _load_metadata() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _save_metadata(metadata: list[dict[str, Any]]) -> None:
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def _rebuild_index(embeddings: list[list[float]], embedding_dimensions: int) -> None:
    index = faiss.IndexFlatIP(embedding_dimensions)
    if embeddings:
        matrix = np.array(embeddings, dtype="float32")
        index.add(matrix)
    faiss.write_index(index, str(INDEX_PATH))


def _embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _embedding_client()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[start : start + MAX_BATCH_SIZE]
        response = client.embeddings.create(
            model=_embedding_model(),
            input=batch,
            dimensions=_embedding_dimensions(),
        )
        embeddings.extend(_normalize_embedding(item.embedding) for item in response.data)
    return embeddings


def _embed_query(text: str) -> list[float]:
    return _embed_documents([text])[0]


def _normalize_embedding(embedding: list[float]) -> list[float]:
    vector = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def _embedding_client() -> OpenAI:
    try:
        from openai import OpenAI as OpenAIClient
    except ImportError as exc:
        raise ValueError(
            "openai package is required for knowledge embeddings. Run: pip install openai"
        ) from exc

    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("QWEN_API_KEY is not configured")
    return OpenAIClient(
        api_key=api_key,
        base_url=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
    )


def _embedding_model() -> str:
    return os.getenv("QWEN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def _embedding_dimensions() -> int:
    raw_value = os.getenv("QWEN_EMBEDDING_DIMENSIONS")
    if raw_value is None:
        return DEFAULT_EMBEDDING_DIMENSIONS
    if not re.fullmatch(r"\d+", raw_value):
        raise ValueError("QWEN_EMBEDDING_DIMENSIONS must be an integer")
    return int(raw_value)
