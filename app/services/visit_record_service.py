"""
就诊记录服务。

职责概览：
- 管理就诊记录的增查改逻辑。
- 为 Agent 工具和长期记忆抽取提供统一访问入口。
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Patient, VisitRecord
from app.schemas.visit_record import VisitRecordCreate, VisitRecordUpdate


def create_visit_record(db: Session, payload: VisitRecordCreate) -> VisitRecord:
    """创建就诊记录。"""

    visit_record = VisitRecord(**payload.model_dump(exclude_none=True))
    db.add(visit_record)
    db.commit()
    db.refresh(visit_record)
    return visit_record


def get_visit_record_by_id(
    db: Session, visit_record_id: int
) -> Optional[VisitRecord]:
    """按主键读取单条就诊记录。"""

    return db.get(VisitRecord, visit_record_id)


def list_visit_records(
    db: Session,
    patient_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> list[VisitRecord]:
    """按就诊时间倒序列出记录，支持患者过滤和数量限制。"""

    stmt = select(VisitRecord).order_by(VisitRecord.visit_time.desc())
    if patient_id is not None:
        stmt = stmt.where(VisitRecord.patient_id == patient_id)
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())


def update_visit_record(
    db: Session, visit_record: VisitRecord, payload: VisitRecordUpdate
) -> VisitRecord:
    """仅更新显式传入的字段。"""

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(visit_record, field, value)
    db.add(visit_record)
    db.commit()
    db.refresh(visit_record)
    return visit_record


def patient_exists(db: Session, patient_id: int) -> bool:
    """供路由层在创建前做患者存在性校验。"""

    return db.get(Patient, patient_id) is not None
