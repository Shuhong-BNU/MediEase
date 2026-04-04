"""
患者身份验证与脱敏工具。

本模块负责：
1. 校验患者身份信息是否匹配。
2. 对手机号、身份证号、地址等字段做默认脱敏。
3. 为工具层和 API 层提供统一的身份展示结构。
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Patient
from app.services import patient_service


def mask_id_number(id_number: Optional[str]) -> Optional[str]:
    """对身份证号做中间位脱敏。"""

    if not id_number or len(id_number) < 8:
        return id_number
    return f"{id_number[:4]}********{id_number[-4:]}"


def mask_phone(phone: Optional[str]) -> Optional[str]:
    """对手机号做中间位脱敏。"""

    if not phone or len(phone) < 7:
        return phone
    return f"{phone[:3]}****{phone[-4:]}"


def mask_address(address: Optional[str]) -> Optional[str]:
    """对地址做轻量脱敏，仅保留前缀以便识别。"""

    if not address:
        return address
    if len(address) <= 6:
        return f"{address}***"
    return f"{address[:6]}***"


def verify_patient_identity(
    db: Session,
    patient_code: str,
    phone: Optional[str] = None,
    id_number: Optional[str] = None,
) -> dict:
    """通过患者编号搭配手机号或身份证号完成身份验证。"""

    patient = patient_service.get_patient_by_code(db, patient_code)
    if patient is None:
        return {
            "verified": False,
            "reason": "patient not found",
            "patient_code": patient_code,
        }

    if not phone and not id_number:
        return {
            "verified": False,
            "reason": "phone or id_number is required",
            "patient_code": patient_code,
        }

    phone_match = phone is not None and phone == patient.phone
    id_match = _match_id_number(id_number, patient.id_number)
    verified = phone_match or id_match

    return {
        "verified": verified,
        "reason": "ok" if verified else "credential mismatch",
        "patient": serialize_patient_identity(patient),
    }


def serialize_patient_identity(patient: Patient) -> dict:
    """返回默认可展示的患者身份信息。"""

    return {
        "id": patient.id,
        "patient_code": patient.patient_code,
        "full_name": patient.full_name,
        "gender": patient.gender,
        "phone_masked": mask_phone(patient.phone),
        "id_number_masked": mask_id_number(patient.id_number),
    }


def _match_id_number(provided: Optional[str], actual: Optional[str]) -> bool:
    """支持完整身份证号或后四位匹配。"""

    if not provided or not actual:
        return False
    if provided == actual:
        return True
    return len(provided) == 4 and actual.endswith(provided)

