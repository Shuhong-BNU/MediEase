"""
记忆与检索相关的请求/响应模型。

覆盖：
1. 短期记忆读写。
2. 长期关键事件抽取与检索。
3. 长期用户画像读取。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BusinessMemoryExtractRequest(BaseModel):
    """从业务数据中提炼长期关键事件时的入参。"""

    patient_id: Optional[int] = None
    patient_code: Optional[str] = None


class ConversationMemoryExtractRequest(BaseModel):
    """从最近若干条短期对话中提炼长期记忆的入参。"""

    patient_id: int
    recent_limit: int = Field(default=10, ge=1, le=50)


class ConversationMemoryCreate(BaseModel):
    """手动写入一条短期记忆。"""

    patient_id: int
    session_id: str
    role: str
    content: str
    multimodal_payload: Optional[str] = None


class ConversationMemoryRead(BaseModel):
    """短期记忆读取结果。"""

    id: int
    patient_id: int
    session_id: str
    role: str
    content: str
    multimodal_payload: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MemoryEventRead(BaseModel):
    """长期关键事件读取结果。"""

    id: int
    patient_id: int
    event_type: str
    event_time: datetime
    title: str
    summary: Optional[str] = None
    source_type: str
    source_id: Optional[str] = None

    model_config = {"from_attributes": True}


class MemoryEventSearchRequest(BaseModel):
    """长期关键事件混合检索的入参。"""

    patient_id: Optional[int] = None
    patient_code: Optional[str] = None
    query: str
    top_n: int = Field(default=5, ge=1, le=20)


class MemoryEventSearchItem(MemoryEventRead):
    """单条检索结果，包含多种召回分数与来源。"""

    retrieval_score: float
    retrieval_sources: list[str]
    retrieval_label: str
    matched_by_keyword: bool
    matched_by_vector: bool
    keyword_score: float
    vector_score: float


class MemoryEventSearchResponse(BaseModel):
    """长期关键事件检索结果集合。"""

    patient_id: int
    query: str
    top_n: int
    results: list[MemoryEventSearchItem]


class UserProfileRead(BaseModel):
    """长期用户画像读取结果。"""

    id: int
    patient_id: int
    profile_summary: Optional[str] = None
    communication_style: Optional[str] = None
    preferred_topics: Optional[str] = None
    stable_preferences: Optional[str] = None
    source_summary: Optional[str] = None
    refreshed_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BusinessMemoryExtractResponse(BaseModel):
    """业务数据提炼完成后的事件集合。"""

    patient_id: int
    event_count: int
    memory_events: list[MemoryEventRead]


class ConversationMemoryExtractResponse(BaseModel):
    """对话记忆提炼完成后的事件与画像。"""

    patient_id: int
    event_count: int
    profile_updated: bool
    memory_events: list[MemoryEventRead]
    user_profile: Optional[UserProfileRead] = None
