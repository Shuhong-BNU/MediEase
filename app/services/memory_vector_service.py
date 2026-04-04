"""
长期关键事件的向量化与本地 FAISS 检索。

本模块负责：
1. 把长期关键事件转换成 embedding。
2. 把向量写入本地 FAISS 索引和 JSON 元数据。
3. 在可用时执行向量检索，不可用时优雅退化。

OpenAI SDK 同样采用延迟导入，避免环境缺依赖时应用启动即失败。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from app.db.models import MemoryEvent
from app.db.session import DATA_DIR

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = Any  # type: ignore[assignment]

VECTOR_DIR = DATA_DIR / "faiss"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_BATCH_SIZE = 10

try:
    import faiss
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    faiss = None
    np = None

logger = logging.getLogger("uvicorn.error")


def is_available() -> bool:
    return faiss is not None and np is not None


def replace_memory_events(
    patient_id: int,
    events: list[MemoryEvent],
    source_types: Optional[list[str]] = None,
) -> None:
    if not is_available():
        logger.info("FAISS unavailable, skip vector write for patient_id=%s", patient_id)
        return

    VECTOR_DIR.mkdir(exist_ok=True)
    metadata = _load_metadata(patient_id)
    embedding_dimensions = _embedding_dimensions()
    logger.info(
        "Writing memory event vectors for patient_id=%s source_types=%s event_count=%s dir=%s model=%s dimensions=%s",
        patient_id,
        source_types or ["all"],
        len(events),
        VECTOR_DIR,
        _embedding_model(),
        embedding_dimensions,
    )
    if source_types:
        metadata = [item for item in metadata if item["source_type"] not in source_types]
    else:
        metadata = []

    for event in events:
        metadata.append(
            {
                "event_id": event.id,
                "patient_id": event.patient_id,
                "event_type": event.event_type,
                "event_time": event.event_time.isoformat(),
                "source_type": event.source_type,
                "source_id": event.source_id or "",
                "document": _build_event_document(event),
            }
        )

    _save_metadata(patient_id, metadata)
    _rebuild_index(
        patient_id=patient_id,
        metadata=metadata,
        embeddings=_embed_documents([item["document"] for item in metadata]),
        embedding_dimensions=embedding_dimensions,
    )
    logger.info(
        "Memory event vectors written for patient_id=%s index=%s metadata=%s total_vectors=%s",
        patient_id,
        _index_path(patient_id),
        _metadata_path(patient_id),
        len(metadata),
    )


def search_memory_events(
    patient_id: int,
    query: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if not is_available():
        logger.info("FAISS unavailable, skip vector search for patient_id=%s", patient_id)
        return []
    if not query.strip():
        return []

    metadata = _load_metadata(patient_id)
    if not metadata:
        logger.info("No vector metadata found for patient_id=%s", patient_id)
        return []

    try:
        embedding_dimensions = _embedding_dimensions()
        index = _load_or_rebuild_index(patient_id, metadata, embedding_dimensions)
        query_vector = np.array([_embed_query(query)], dtype="float32")
    except Exception as exc:
        logger.warning(
            "Vector search degraded for patient_id=%s query=%r because FAISS index is unavailable: %s",
            patient_id,
            query,
            exc,
        )
        return []

    scores, positions = index.search(query_vector, max(top_n, 1))

    rows: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0 or position >= len(metadata):
            continue
        item = metadata[int(position)]
        rows.append(
            {
                "event_id": int(item["event_id"]),
                "vector_score": float(score),
            }
        )
    logger.info(
        "Vector search complete for patient_id=%s query=%r top_n=%s hit_count=%s",
        patient_id,
        query,
        top_n,
        len(rows),
    )
    return rows


def _rebuild_index(
    patient_id: int,
    metadata: list[dict[str, Any]],
    embeddings: list[list[float]],
    embedding_dimensions: int,
) -> None:
    index = _build_index(metadata, embeddings, embedding_dimensions)
    faiss.write_index(index, str(_index_path(patient_id)))


def _load_or_rebuild_index(
    patient_id: int,
    metadata: list[dict[str, Any]],
    embedding_dimensions: int,
):
    index_path = _index_path(patient_id)
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            if index.d == embedding_dimensions:
                return index
            logger.info(
                "Existing FAISS index dimension mismatch for patient_id=%s old=%s new=%s, rebuilding",
                patient_id,
                index.d,
                embedding_dimensions,
            )
        except Exception as exc:
            logger.warning(
                "Failed to read FAISS index for patient_id=%s path=%s, rebuilding in memory: %s",
                patient_id,
                index_path,
                exc,
            )
    else:
        logger.info("Missing FAISS index for patient_id=%s at %s, rebuilding", patient_id, index_path)

    embeddings = _embed_documents([item["document"] for item in metadata])
    index = _build_index(metadata, embeddings, embedding_dimensions)
    try:
        faiss.write_index(index, str(index_path))
        logger.info("Persisted rebuilt FAISS index for patient_id=%s at %s", patient_id, index_path)
    except Exception as exc:
        # Windows + FAISS 在少数路径场景下会出现文件读写异常，运行时退回内存索引即可。
        logger.warning(
            "Failed to persist rebuilt FAISS index for patient_id=%s path=%s, continue with in-memory index: %s",
            patient_id,
            index_path,
            exc,
        )
    return index


def _build_index(
    metadata: list[dict[str, Any]],
    embeddings: list[list[float]],
    embedding_dimensions: int,
):
    index = faiss.IndexFlatIP(embedding_dimensions)
    if metadata:
        matrix = np.array(embeddings, dtype="float32")
        index.add(matrix)
    return index


def _metadata_path(patient_id: int) -> Path:
    return VECTOR_DIR / f"patient_{patient_id}.json"


def _index_path(patient_id: int) -> Path:
    return VECTOR_DIR / f"patient_{patient_id}.index"


def _load_metadata(patient_id: int) -> list[dict[str, Any]]:
    path = _metadata_path(patient_id)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_metadata(patient_id: int, metadata: list[dict[str, Any]]) -> None:
    _metadata_path(patient_id).write_text(
        json.dumps(metadata, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_event_document(event: MemoryEvent) -> str:
    parts = [
        event.event_type,
        event.title,
        event.summary or "",
        event.source_type,
    ]
    return "\n".join(part for part in parts if part)


def _embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _embedding_client()
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[start : start + MAX_BATCH_SIZE]
        logger.info(
            "Requesting Qwen embeddings batch_size=%s model=%s dimensions=%s",
            len(batch),
            _embedding_model(),
            _embedding_dimensions(),
        )
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
    """延迟创建 embedding client，缺依赖时给出明确错误。"""

    try:
        from openai import OpenAI as OpenAIClient
    except ImportError as exc:
        raise ValueError(
            "openai package is required for memory embeddings. Run: pip install openai"
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
