"""
患者服务。

职责概览：
- 封装患者主档案的增查改逻辑。
- 为路由层、身份校验、记忆层提供统一的数据访问接口。
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Patient
from app.schemas.patient import PatientCreate, PatientUpdate
from app.services.crypto_service import encrypt_optional_text


def _apply_encrypted_fields(patient: Patient) -> None:
    """为高敏字段同步生成加密镜像，便于后续逐步切换到加密存储读取。"""

    patient.phone_encrypted = encrypt_optional_text(patient.phone)
    patient.id_number_encrypted = encrypt_optional_text(patient.id_number)
    patient.address_encrypted = encrypt_optional_text(patient.address)
    patient.emergency_contact_phone_encrypted = encrypt_optional_text(
        patient.emergency_contact_phone
    )


def create_patient(db: Session, payload: PatientCreate) -> Patient:
    """创建患者主档案。"""

    patient = Patient(**payload.model_dump())
    _apply_encrypted_fields(patient)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def list_patients(db: Session) -> list[Patient]:
    """按主键倒序返回患者列表。"""

    return list(db.scalars(select(Patient).order_by(Patient.id.desc())).all())


def get_patient_by_id(db: Session, patient_id: int) -> Optional[Patient]:
    """按主键读取患者。"""

    return db.get(Patient, patient_id)


def get_patient_by_code(db: Session, patient_code: str) -> Optional[Patient]:
    """按患者编码读取患者。"""

    stmt = select(Patient).where(Patient.patient_code == patient_code)
    return db.scalar(stmt)


def get_patient_by_phone(db: Session, phone: str) -> Optional[Patient]:
    """按手机号读取患者，供身份识别和工具路由使用。"""

    stmt = select(Patient).where(Patient.phone == phone)
    return db.scalar(stmt)


def update_patient(db: Session, patient: Patient, payload: PatientUpdate) -> Patient:
    """仅更新用户显式传入的患者字段。"""

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(patient, field, value)
    _apply_encrypted_fields(patient)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient
