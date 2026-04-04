"""
短期记忆服务。

本模块负责：
1. 写入用户和助手的短期对话记忆。
2. 按患者维度或按稳定会话维度读取最近对话。
3. 提供长期沉淀阈值所需的消息计数。
"""

from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import ConversationMemory
from app.schemas.memory import ConversationMemoryCreate


def create_conversation_memory(
    db: Session,
    payload: ConversationMemoryCreate,
) -> ConversationMemory:
    """写入一条短期记忆并立即返回落库结果。"""

    memory = ConversationMemory(**payload.model_dump())
    db.add(memory)
    db.commit()
    db.refresh(memory)
    return memory


def list_conversation_memories(
    db: Session,
    patient_id: int,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[ConversationMemory]:
    """查询某个患者的短期记忆，可选按 session 过滤。"""

    stmt = (
        select(ConversationMemory)
        .where(ConversationMemory.patient_id == patient_id)
        .order_by(ConversationMemory.created_at.desc(), ConversationMemory.id.desc())
    )
    if session_id is not None:
        stmt = stmt.where(ConversationMemory.session_id == session_id)
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())


def list_recent_conversation_memories(
    db: Session,
    patient_id: int,
    session_id: Optional[str] = None,
    limit: int = 6,
) -> list[ConversationMemory]:
    """优先按稳定会话读取最近 N 条短期记忆，必要时可回退到患者维度。"""

    stmt = (
        select(ConversationMemory)
        .where(ConversationMemory.patient_id == patient_id)
        .order_by(ConversationMemory.created_at.desc(), ConversationMemory.id.desc())
        .limit(limit)
    )
    if session_id is not None:
        stmt = stmt.where(ConversationMemory.session_id == session_id)

    memories = list(db.scalars(stmt).all())
    memories.reverse()
    return memories


def count_conversation_memories(
    db: Session,
    patient_id: int,
) -> int:
    """按患者统计短期记忆总量，用于 5 轮阈值触发长期沉淀。"""

    stmt = select(func.count(ConversationMemory.id)).where(
        ConversationMemory.patient_id == patient_id
    )
    return int(db.scalar(stmt) or 0)


def clear_conversation_memories(
    db: Session,
    patient_id: Optional[int] = None,
    session_id: Optional[str] = None,
) -> int:
    """按患者或会话清空短期记忆，返回删除条数。"""

    stmt = delete(ConversationMemory)
    if patient_id is not None:
        stmt = stmt.where(ConversationMemory.patient_id == patient_id)
    if session_id is not None:
        stmt = stmt.where(ConversationMemory.session_id == session_id)
    result = db.execute(stmt)
    db.commit()
    return int(result.rowcount or 0)
