"""
知识库服务。

本模块现在同时承担三层职责：
1. 知识库文档的 CRUD。
2. 知识库的混合检索：关键词召回 + 向量召回。
3. LangChain 风格的 Retriever 封装，供 LangGraph / RAG 场景直接调用。

这意味着当前项目的知识库链路已经从“轻量 keyword search”升级成真正可讲清楚的
向量 RAG 基础设施：
- 文档会被写入 SQLite。
- 启用文档会同步进入本地 FAISS 索引。
- 查询时会合并 keyword / vector 命中，并给出 retrieval label。
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.models import KnowledgeDocument
from app.schemas.knowledge import (
    KnowledgeDocumentCreate,
    KnowledgeDocumentUpdate,
    KnowledgeSearchItem,
)
from app.services import knowledge_vector_service


class KnowledgeBaseRetriever(BaseRetriever):
    """LangChain / LangGraph 可直接复用的知识库 Retriever。"""

    db: Session
    tenant_code: Optional[str] = None
    category: Optional[str] = None
    top_n: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        results = search_knowledge_documents(
            self.db,
            query=query,
            tenant_code=self.tenant_code,
            category=self.category,
            top_n=self.top_n,
        )
        if not results:
            return []

        doc_map = {
            document.id: document
            for document in self.db.scalars(
                select(KnowledgeDocument).where(
                    KnowledgeDocument.id.in_([item.id for item in results])
                )
            ).all()
        }
        ordered_documents: list[Document] = []
        for item in results:
            source = doc_map.get(item.id)
            ordered_documents.append(
                Document(
                    page_content=source.content if source is not None else item.snippet,
                    metadata={
                        "id": item.id,
                        "title": item.title,
                        "category": item.category,
                        "source_url": item.source_url,
                        "snippet": item.snippet,
                        "score": item.score,
                        "retrieval_sources": item.retrieval_sources,
                        "retrieval_label": item.retrieval_label,
                        "keyword_score": item.keyword_score,
                        "vector_score": item.vector_score,
                    },
                )
            )
        return ordered_documents


def create_knowledge_document(
    db: Session,
    payload: KnowledgeDocumentCreate,
) -> KnowledgeDocument:
    """创建知识库文档，并尽量同步向量索引。"""

    document = KnowledgeDocument(**payload.model_dump())
    db.add(document)
    db.commit()
    db.refresh(document)
    knowledge_vector_service.rebuild_knowledge_index(db)
    return document


def list_knowledge_documents(
    db: Session,
    tenant_code: Optional[str] = None,
    enabled_only: bool = False,
) -> list[KnowledgeDocument]:
    """列出知识库文档。"""

    stmt = select(KnowledgeDocument).order_by(
        KnowledgeDocument.updated_at.desc(),
        KnowledgeDocument.id.desc(),
    )
    if tenant_code:
        stmt = stmt.where(KnowledgeDocument.tenant_code == tenant_code)
    if enabled_only:
        stmt = stmt.where(KnowledgeDocument.enabled.is_(True))
    return list(db.scalars(stmt).all())


def get_knowledge_document(db: Session, document_id: int) -> Optional[KnowledgeDocument]:
    """按主键读取文档。"""

    return db.get(KnowledgeDocument, document_id)


def update_knowledge_document(
    db: Session,
    document: KnowledgeDocument,
    payload: KnowledgeDocumentUpdate,
) -> KnowledgeDocument:
    """更新知识库文档，并同步向量索引。"""

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(document, field, value)
    db.add(document)
    db.commit()
    db.refresh(document)
    knowledge_vector_service.rebuild_knowledge_index(db)
    return document


def search_knowledge_documents(
    db: Session,
    query: str,
    tenant_code: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 5,
) -> list[KnowledgeSearchItem]:
    """执行知识库混合检索，并返回统一结果结构。"""

    keyword_results = _keyword_search_documents(
        db,
        query=query,
        tenant_code=tenant_code,
        category=category,
        top_n=max(top_n * 2, 10),
    )
    vector_results = knowledge_vector_service.search_knowledge_documents(
        db,
        query=query,
        tenant_code=tenant_code,
        category=category,
        top_n=max(top_n * 2, 10),
    )

    merged_scores: dict[int, dict[str, Any]] = {}
    for item in keyword_results:
        merged_scores[item.id] = {
            "keyword_score": item.score,
            "vector_score": 0.0,
            "snippet": item.snippet,
        }

    for item in vector_results:
        bucket = merged_scores.setdefault(
            item["id"],
            {
                "keyword_score": 0.0,
                "vector_score": 0.0,
                "snippet": "",
            },
        )
        bucket["vector_score"] = item["vector_score"]

    if not merged_scores:
        return []

    documents = {
        document.id: document
        for document in db.scalars(
            select(KnowledgeDocument).where(KnowledgeDocument.id.in_(list(merged_scores)))
        ).all()
    }
    results: list[KnowledgeSearchItem] = []
    for document_id, score_bundle in merged_scores.items():
        document = documents.get(document_id)
        if document is None:
            continue
        keyword_score = float(score_bundle["keyword_score"])
        vector_score = float(score_bundle["vector_score"])
        retrieval_sources = []
        if keyword_score > 0:
            retrieval_sources.append("keyword")
        if vector_score > 0:
            retrieval_sources.append("vector")
        results.append(
            KnowledgeSearchItem(
                id=document.id,
                title=document.title,
                category=document.category,
                snippet=score_bundle["snippet"]
                or _build_snippet(document.content, [query]),
                source_url=document.source_url,
                score=keyword_score + vector_score * 2.0,
                retrieval_sources=retrieval_sources,
                retrieval_label=_build_retrieval_label(retrieval_sources),
                keyword_score=keyword_score,
                vector_score=vector_score,
            )
        )

    results.sort(key=lambda item: (-item.score, item.id))
    return results[:top_n]


def build_knowledge_retriever(
    db: Session,
    tenant_code: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 5,
) -> KnowledgeBaseRetriever:
    """构造可被 LangChain / LangGraph 直接调用的 Retriever。"""

    return KnowledgeBaseRetriever(
        db=db,
        tenant_code=tenant_code,
        category=category,
        top_n=top_n,
    )


def serialize_knowledge_document(document: KnowledgeDocument) -> dict:
    """把知识库 ORM 对象转成稳定字典。"""

    return {
        "id": document.id,
        "tenant_code": document.tenant_code,
        "title": document.title,
        "category": document.category,
        "content": document.content,
        "keywords": document.keywords,
        "source_url": document.source_url,
        "enabled": document.enabled,
    }


def _keyword_search_documents(
    db: Session,
    query: str,
    tenant_code: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 5,
) -> list[KnowledgeSearchItem]:
    """基于标题、关键词和正文做轻量排序检索。"""

    stmt = select(KnowledgeDocument).where(KnowledgeDocument.enabled.is_(True))
    if tenant_code:
        stmt = stmt.where(
            or_(
                KnowledgeDocument.tenant_code == tenant_code,
                KnowledgeDocument.tenant_code.is_(None),
            )
        )
    if category:
        stmt = stmt.where(KnowledgeDocument.category == category)

    tokens = [token.strip() for token in query.replace("；", " ").split() if token.strip()]
    candidates = list(db.scalars(stmt).all())
    scored: list[KnowledgeSearchItem] = []
    for document in candidates:
        haystack = " ".join(
            [
                document.title or "",
                document.category or "",
                document.keywords or "",
                document.content or "",
            ]
        )
        score = 0.0
        for token in tokens or [query]:
            if token in (document.title or ""):
                score += 2.5
            if token in (document.keywords or ""):
                score += 2.0
            if token in (document.content or ""):
                score += 1.0
        if score <= 0 and query not in haystack:
            continue
        scored.append(
            KnowledgeSearchItem(
                id=document.id,
                title=document.title,
                category=document.category,
                snippet=_build_snippet(document.content, tokens or [query]),
                source_url=document.source_url,
                score=score or 0.5,
                retrieval_sources=["keyword"],
                retrieval_label="keyword",
                keyword_score=score or 0.5,
                vector_score=0.0,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.id))
    return scored[:top_n]


def _build_retrieval_label(retrieval_sources: list[str]) -> Optional[str]:
    if {"keyword", "vector"}.issubset(set(retrieval_sources)):
        return "hybrid"
    if "vector" in retrieval_sources:
        return "vector"
    if "keyword" in retrieval_sources:
        return "keyword"
    return None


def _build_snippet(content: str, tokens: list[str]) -> str:
    """截取包含命中词的片段，便于前端展示来源。"""

    clean_content = content.strip()
    if not clean_content:
        return ""
    for token in tokens:
        index = clean_content.find(token)
        if index != -1:
            start = max(index - 30, 0)
            end = min(index + 90, len(clean_content))
            return clean_content[start:end]
    return clean_content[:120]
