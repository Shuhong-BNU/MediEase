"""
稳定会话服务。

本模块为 `/api/agent/query` 提供服务端稳定会话语义：
1. 创建或复用 `conversation_session_id`。
2. 记录当前会话绑定到哪个患者。
3. 记录当前会话已验证通过的患者身份。
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentConversationSession


def get_or_create_session(
    db: Session,
    session_id: str | None = None,
) -> AgentConversationSession:
    """按外部传入 session_id 复用会话，不存在时自动创建。"""

    if session_id:
        session = get_session_by_id(db, session_id)
        if session is not None:
            touch_session(db, session)
            return session

    session = AgentConversationSession(session_id=f"agent-session-{uuid4().hex}")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_id(
    db: Session,
    session_id: str,
) -> AgentConversationSession | None:
    """按稳定会话 ID 查询会话实体。"""

    stmt = select(AgentConversationSession).where(
        AgentConversationSession.session_id == session_id
    )
    return db.scalar(stmt)


def update_session_patient_context(
    db: Session,
    session: AgentConversationSession,
    patient_id: int | None = None,
    verified_patient_id: int | None = None,
) -> AgentConversationSession:
    """更新会话绑定患者和已验证患者。"""

    if patient_id is not None:
        session.patient_id = patient_id
    if verified_patient_id is not None:
        session.verified_patient_id = verified_patient_id
        session.patient_id = verified_patient_id

    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def touch_session(
    db: Session,
    session: AgentConversationSession,
) -> AgentConversationSession:
    """刷新会话活跃时间。"""

    db.add(session)
    db.commit()
    db.refresh(session)
    return session

