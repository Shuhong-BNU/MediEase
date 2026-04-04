"""
知识库 Schema。
职责概览：
1. 约束知识库文档的创建、读取与搜索结构。
2. 为 Agent 工具调用提供稳定的检索结果格式。
3. 为后续租户级知识空间扩展保留 tenant_code 字段。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeDocumentBase(BaseModel):
    """知识库文档公共字段。"""

    tenant_code: Optional[str] = None
    title: str
    category: Optional[str] = None
    content: str
    keywords: Optional[str] = None
    source_url: Optional[str] = None
    enabled: bool = True


class KnowledgeDocumentCreate(KnowledgeDocumentBase):
    """创建知识库文档。"""


class KnowledgeDocumentUpdate(BaseModel):
    """更新知识库文档。"""

    tenant_code: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None
    keywords: Optional[str] = None
    source_url: Optional[str] = None
    enabled: Optional[bool] = None


class KnowledgeDocumentRead(KnowledgeDocumentBase):
    """知识库文档读取结构。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class KnowledgeSearchRequest(BaseModel):
    """知识库搜索请求。"""

    query: str
    tenant_code: Optional[str] = None
    category: Optional[str] = None
    top_n: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchItem(BaseModel):
    """知识库搜索命中项。"""

    id: int
    title: str
    category: Optional[str] = None
    snippet: str
    source_url: Optional[str] = None
    score: float
    retrieval_sources: list[str] = Field(default_factory=list)
    retrieval_label: Optional[str] = None
    keyword_score: float = 0.0
    vector_score: float = 0.0


class KnowledgeSearchResponse(BaseModel):
    """知识库搜索结果集合。"""

    query: str
    top_n: int
    results: list[KnowledgeSearchItem]
