"""
患者 Schema。

职责概览：
- 约束患者基础信息的输入与输出格式。
- 与数据库模型解耦，避免 API 直接暴露 ORM 对象。
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PatientBase(BaseModel):
    """患者主档案的公共字段。"""

    patient_code: str
    full_name: str
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    phone: Optional[str] = None
    id_number: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None


class PatientCreate(PatientBase):
    """创建患者时使用的结构。"""


class PatientUpdate(BaseModel):
    """更新患者时允许变更的字段。"""

    full_name: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[date] = None
    phone: Optional[str] = None
    id_number: Optional[str] = None
    address: Optional[str] = None
    emergency_contact_name: Optional[str] = None
    emergency_contact_phone: Optional[str] = None


class PatientRead(PatientBase):
    """患者读取结果。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
