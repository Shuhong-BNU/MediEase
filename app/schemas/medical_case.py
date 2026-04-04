"""
病例 Schema。

职责概览：
- 定义病例创建、更新、读取时的请求响应结构。
- 为 API 层和服务层之间提供统一的数据边界。
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class MedicalCaseBase(BaseModel):
    """病例的公共字段定义。"""

    patient_id: int
    case_code: str
    diagnosis: str
    chief_complaint: Optional[str] = None
    present_illness: Optional[str] = None
    past_history: Optional[str] = None
    treatment_plan: Optional[str] = None
    attending_physician: Optional[str] = None
    recorded_at: Optional[datetime] = None


class MedicalCaseCreate(MedicalCaseBase):
    """创建病例时要求的字段。"""


class MedicalCaseUpdate(BaseModel):
    """更新病例时允许变更的字段。"""

    diagnosis: Optional[str] = None
    chief_complaint: Optional[str] = None
    present_illness: Optional[str] = None
    past_history: Optional[str] = None
    treatment_plan: Optional[str] = None
    attending_physician: Optional[str] = None
    recorded_at: Optional[datetime] = None


class MedicalCaseRead(MedicalCaseBase):
    """病例读取结果。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    recorded_at: datetime
    created_at: datetime
    updated_at: datetime
