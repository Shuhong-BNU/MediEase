"""
人工升级服务。
职责概览：
1. 在高风险对话场景下记录“建议转人工”的事件。
2. 为前端展示和审计留出稳定的事件结构。
3. 当前只做事件记录，不直接接入真实人工客服系统。
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ManualEscalationEvent


def create_manual_escalation_event(
    db: Session,
    conversation_session_id: Optional[str],
    patient_id: Optional[int],
    risk_level: str,
    trigger_reason: str,
    recommended_action: str,
) -> ManualEscalationEvent:
    """创建一条人工升级建议事件。"""

    event = ManualEscalationEvent(
        conversation_session_id=conversation_session_id,
        patient_id=patient_id,
        risk_level=risk_level,
        trigger_reason=trigger_reason,
        recommended_action=recommended_action,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def list_manual_escalation_events(
    db: Session,
    conversation_session_id: Optional[str] = None,
    patient_id: Optional[int] = None,
) -> list[ManualEscalationEvent]:
    """按会话或患者查询升级事件。"""

    stmt = select(ManualEscalationEvent).order_by(
        ManualEscalationEvent.created_at.desc(),
        ManualEscalationEvent.id.desc(),
    )
    if conversation_session_id:
        stmt = stmt.where(
            ManualEscalationEvent.conversation_session_id == conversation_session_id
        )
    if patient_id is not None:
        stmt = stmt.where(ManualEscalationEvent.patient_id == patient_id)
    return list(db.scalars(stmt).all())


def serialize_manual_escalation_event(event: ManualEscalationEvent) -> dict:
    """把升级事件转成稳定响应。"""

    return {
        "id": event.id,
        "conversation_session_id": event.conversation_session_id,
        "patient_id": event.patient_id,
        "risk_level": event.risk_level,
        "trigger_reason": event.trigger_reason,
        "recommended_action": event.recommended_action,
        "status": event.status,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }
