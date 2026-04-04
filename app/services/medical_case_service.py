"""
病例服务。

职责概览：
- 封装病例的增删改查逻辑。
- 供 API 层和记忆提炼逻辑复用，避免在多个地方直接拼装 SQLAlchemy 操作。
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import MedicalCase, Patient
from app.schemas.medical_case import MedicalCaseCreate, MedicalCaseUpdate


def create_medical_case(db: Session, payload: MedicalCaseCreate) -> MedicalCase:
    """创建病例并返回持久化结果。"""

    medical_case = MedicalCase(**payload.model_dump(exclude_none=True))
    db.add(medical_case)
    db.commit()
    db.refresh(medical_case)
    return medical_case


def get_medical_case_by_id(db: Session, case_id: int) -> Optional[MedicalCase]:
    """按主键读取病例。"""

    return db.get(MedicalCase, case_id)


def list_medical_cases(
    db: Session, patient_id: Optional[int] = None
) -> list[MedicalCase]:
    """按记录时间倒序列出病例，可选限制到某个患者。"""

    stmt = select(MedicalCase).order_by(MedicalCase.recorded_at.desc())
    if patient_id is not None:
        stmt = stmt.where(MedicalCase.patient_id == patient_id)
    return list(db.scalars(stmt).all())


def update_medical_case(
    db: Session, medical_case: MedicalCase, payload: MedicalCaseUpdate
) -> MedicalCase:
    """仅更新请求中显式提供的字段。"""

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(medical_case, field, value)
    db.add(medical_case)
    db.commit()
    db.refresh(medical_case)
    return medical_case


def patient_exists(db: Session, patient_id: int) -> bool:
    """供路由层在创建前做患者存在性校验。"""

    return db.get(Patient, patient_id) is not None
