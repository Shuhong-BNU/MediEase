"""
就诊记录 Schema。

职责概览：
- 约束就诊记录的创建、更新和读取结构。
- 为 Agent 工具返回和后台管理接口提供统一格式。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class VisitRecordBase(BaseModel):
    """就诊记录的公共字段。"""

    patient_id: int
    visit_code: str
    visit_type: str
    department: Optional[str] = None
    physician_name: Optional[str] = None
    visit_time: Optional[datetime] = None
    summary: Optional[str] = None
    notes: Optional[str] = None


class VisitRecordCreate(VisitRecordBase):
    """创建就诊记录时使用的结构。"""


class VisitRecordUpdate(BaseModel):
    """更新就诊记录时允许变更的字段。"""

    visit_type: Optional[str] = None
    department: Optional[str] = None
    physician_name: Optional[str] = None
    visit_time: Optional[datetime] = None
    summary: Optional[str] = None
    notes: Optional[str] = None


class VisitRecordRead(VisitRecordBase):
    """就诊记录读取结果。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    visit_time: datetime
    created_at: datetime
    updated_at: datetime
