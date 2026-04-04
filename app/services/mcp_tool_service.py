"""
患者领域工具服务。

本模块把患者资料、病例、就诊记录等业务能力整理成可供 Agent / MCP 工具层调用的函数。
这里聚焦“业务查询和统一序列化”，不直接负责权限决策；权限由 Agent 执行上下文统一收口。
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import MedicalCase, MedicalReport, Patient, VisitRecord
from app.services import (
    escalation_service,
    identity_service,
    knowledge_service,
    medical_case_service,
    patient_service,
    report_service,
    visit_record_service,
)


def serialize_patient_profile(
    patient: Patient,
    include_sensitive: bool = False,
) -> dict:
    """统一序列化患者资料，默认仅返回安全可展示字段。"""

    data = {
        "id": patient.id,
        "patient_code": patient.patient_code,
        "full_name": patient.full_name,
        "gender": patient.gender,
        "date_of_birth": patient.date_of_birth.isoformat()
        if patient.date_of_birth
        else None,
        "emergency_contact_name": patient.emergency_contact_name,
    }
    if include_sensitive:
        data.update(
            {
                "phone": patient.phone,
                "id_number": patient.id_number,
                "address": patient.address,
                "emergency_contact_phone": patient.emergency_contact_phone,
            }
        )
        return data

    data.update(
        {
            "phone_masked": identity_service.mask_phone(patient.phone),
            "id_number_masked": identity_service.mask_id_number(patient.id_number),
            "address_masked": identity_service.mask_address(patient.address),
            "emergency_contact_phone_masked": identity_service.mask_phone(
                patient.emergency_contact_phone
            ),
        }
    )
    return data


def serialize_medical_case(medical_case: MedicalCase) -> dict:
    """统一序列化病例结果。"""

    return {
        "id": medical_case.id,
        "patient_id": medical_case.patient_id,
        "case_code": medical_case.case_code,
        "diagnosis": medical_case.diagnosis,
        "chief_complaint": medical_case.chief_complaint,
        "present_illness": medical_case.present_illness,
        "past_history": medical_case.past_history,
        "treatment_plan": medical_case.treatment_plan,
        "attending_physician": medical_case.attending_physician,
        "recorded_at": medical_case.recorded_at.isoformat(),
        "created_at": medical_case.created_at.isoformat(),
        "updated_at": medical_case.updated_at.isoformat(),
    }


def serialize_visit_record(visit_record: VisitRecord) -> dict:
    """统一序列化就诊记录结果。"""

    return {
        "id": visit_record.id,
        "patient_id": visit_record.patient_id,
        "visit_code": visit_record.visit_code,
        "visit_type": visit_record.visit_type,
        "department": visit_record.department,
        "physician_name": visit_record.physician_name,
        "visit_time": visit_record.visit_time.isoformat(),
        "summary": visit_record.summary,
        "notes": visit_record.notes,
        "created_at": visit_record.created_at.isoformat(),
        "updated_at": visit_record.updated_at.isoformat(),
    }


def serialize_medical_report(medical_report: MedicalReport) -> dict:
    """统一序列化检验检查报告。"""

    return report_service.serialize_medical_report(medical_report)


def build_verification_required_response(
    tool_name: str,
    patient: Optional[Patient] = None,
) -> dict:
    """构造统一的“需要先完成身份验证”响应。"""

    response = {
        "found": patient is not None,
        "access_granted": False,
        "requires_identity_verification": True,
        "reason": "identity verification required",
        "tool_name": tool_name,
        "next_action": "请先提供患者编号并补充手机号或身份证号（支持身份证后四位）完成身份验证。",
    }
    if patient is not None:
        response["patient"] = identity_service.serialize_patient_identity(patient)
    return response


def get_patient_profile(
    db: Session,
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
    include_sensitive: bool = False,
) -> dict:
    """查询患者基础资料，默认返回安全可展示字段。"""

    patient = _resolve_patient(db, patient_id=patient_id, patient_code=patient_code)
    if patient is None:
        return {"found": False, "reason": "patient not found"}

    return {
        "found": True,
        "patient": serialize_patient_profile(patient, include_sensitive=include_sensitive),
    }


def get_patient_medical_cases(
    db: Session,
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
) -> dict:
    """查询患者病例信息。"""

    patient = _resolve_patient(db, patient_id=patient_id, patient_code=patient_code)
    if patient is None:
        return {"found": False, "reason": "patient not found", "medical_cases": []}

    medical_cases = medical_case_service.list_medical_cases(db, patient_id=patient.id)
    return {
        "found": True,
        "access_granted": True,
        "patient": identity_service.serialize_patient_identity(patient),
        "medical_cases": [serialize_medical_case(item) for item in medical_cases],
    }


def get_patient_visit_records(
    db: Session,
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """查询患者就诊记录。"""

    patient = _resolve_patient(db, patient_id=patient_id, patient_code=patient_code)
    if patient is None:
        return {"found": False, "reason": "patient not found", "visit_records": []}

    visit_records = visit_record_service.list_visit_records(
        db,
        patient_id=patient.id,
        limit=limit,
    )
    return {
        "found": True,
        "access_granted": True,
        "patient": identity_service.serialize_patient_identity(patient),
        "count": len(visit_records),
        "visit_records": [serialize_visit_record(item) for item in visit_records],
    }


def get_patient_medical_reports(
    db: Session,
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """查询患者检验检查报告。"""

    patient = _resolve_patient(db, patient_id=patient_id, patient_code=patient_code)
    if patient is None:
        return {"found": False, "reason": "patient not found", "medical_reports": []}

    reports = report_service.list_medical_reports(
        db,
        patient_id=patient.id,
        limit=limit,
    )
    return {
        "found": True,
        "access_granted": True,
        "patient": identity_service.serialize_patient_identity(patient),
        "count": len(reports),
        "medical_reports": [serialize_medical_report(item) for item in reports],
    }


def search_knowledge_base(
    db: Session,
    query: str,
    tenant_code: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 5,
) -> dict:
    """搜索知识库文档，内部走 LangChain 风格 Retriever。"""

    retriever = knowledge_service.build_knowledge_retriever(
        db,
        tenant_code=tenant_code,
        category=category,
        top_n=top_n,
    )
    documents = retriever.invoke(query)
    return {
        "found": bool(documents),
        "query": query,
        "count": len(documents),
        "results": [
            {
                "id": document.metadata.get("id"),
                "title": document.metadata.get("title"),
                "category": document.metadata.get("category"),
                "snippet": document.metadata.get("snippet") or document.page_content[:160],
                "source_url": document.metadata.get("source_url"),
                "score": document.metadata.get("score", 0.0),
                "retrieval_label": document.metadata.get("retrieval_label"),
                "retrieval_sources": document.metadata.get("retrieval_sources", []),
                "keyword_score": document.metadata.get("keyword_score", 0.0),
                "vector_score": document.metadata.get("vector_score", 0.0),
            }
            for document in documents
        ],
    }


def create_manual_escalation(
    db: Session,
    conversation_session_id: Optional[str] = None,
    patient_id: Optional[int] = None,
    risk_level: str = "medium",
    trigger_reason: str = "",
    recommended_action: str = "",
) -> dict:
    """创建人工升级事件，供高风险问答链路主动调用。"""

    event = escalation_service.create_manual_escalation_event(
        db,
        conversation_session_id=conversation_session_id,
        patient_id=patient_id,
        risk_level=risk_level,
        trigger_reason=trigger_reason or "需要人工进一步判断",
        recommended_action=recommended_action or "建议人工客服或医护人员跟进。",
    )
    return {
        "found": True,
        "access_granted": True,
        "manual_escalation": escalation_service.serialize_manual_escalation_event(event),
    }


def verify_patient(
    db: Session,
    patient_code: str,
    phone: Optional[str] = None,
    id_number: Optional[str] = None,
) -> dict:
    """代理到统一身份验证逻辑。"""

    return identity_service.verify_patient_identity(
        db,
        patient_code=patient_code,
        phone=phone,
        id_number=id_number,
    )


def resolve_patient_from_identity(
    db: Session,
    patient_code: Optional[str] = None,
    patient_id: Optional[int] = None,
) -> Optional[Patient]:
    """供 Agent 包装层复用的患者解析函数。"""

    return _resolve_patient(db, patient_id=patient_id, patient_code=patient_code)


def _resolve_patient(
    db: Session,
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
) -> Optional[Patient]:
    """根据 ID 或患者编号解析患者实体。"""

    if patient_id is not None:
        return patient_service.get_patient_by_id(db, patient_id)
    if patient_code is not None:
        return patient_service.get_patient_by_code(db, patient_code)
    return None
