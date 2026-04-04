"""
长期偏好 Schema。

职责概览：
- 描述用户希望长期保存的沟通偏好与关注主题。
- 支持按患者维度 upsert，供记忆层和 Agent 汇总使用。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MemoryPreferenceBase(BaseModel):
    """长期偏好的公共字段。"""

    preferred_name: Optional[str] = None
    response_style: Optional[str] = None
    response_length: Optional[str] = None
    preferred_language: Optional[str] = None
    focus_topics: Optional[str] = None
    additional_preferences: Optional[str] = None


class MemoryPreferenceUpsert(MemoryPreferenceBase):
    """新增或更新偏好时需要的载荷。"""

    patient_id: int


class MemoryPreferenceRead(MemoryPreferenceBase):
    """长期偏好读取结果。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    patient_id: int
    created_at: datetime
    updated_at: datetime
