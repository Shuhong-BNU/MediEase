"""
记忆服务。

职责概览：
- 管理长期事件、用户画像和对话抽取所需的聚合逻辑。
- 从业务数据与对话数据两条链路刷新长期记忆。
- 对外提供带 keyword/vector/recent 降级策略的混合检索接口。

与其他模块的关系：
- 依赖 `patient_service`、`medical_case_service`、`visit_record_service`
  获取结构化业务数据。
- 依赖 `memory_vector_service` 维护 FAISS 向量索引。
- 被 API 层用于注入记忆上下文，也被长期沉淀流程调用。
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import (
    ConversationMemory,
    MemoryEvent,
    MemoryPreference,
    Patient,
    UserProfile,
)
from app.services import (
    medical_case_service,
    memory_vector_service,
    patient_service,
    visit_record_service,
)

logger = logging.getLogger("uvicorn.error")


MEDICAL_KEYWORDS = [
    "心内科",
    "冠心病",
    "心绞痛",
    "高血压",
    "糖尿病",
    "复诊",
    "用药",
    "检查",
    "住院",
]

STYLE_KEYWORDS = {
    "简短": "简洁",
    "简单": "简洁",
    "直接": "直接给结论",
    "详细": "详细说明",
    "通俗": "通俗解释",
    "专业": "专业表达",
}

EVENT_MATCH_KEYWORDS = [
    "最近",
    "最新",
    "复诊",
    "住院",
    "用药",
    "检查",
    "心内科",
    "胸痛",
    "心绞痛",
    "冠心病",
    "医生",
]


def get_patient(
    db: Session,
    patient_id: Optional[int],
    patient_code: Optional[str],
) -> Optional[Patient]:
    """按主键或患者编码定位患者。"""

    if patient_id is not None:
        return patient_service.get_patient_by_id(db, patient_id)
    if patient_code is not None:
        return patient_service.get_patient_by_code(db, patient_code)
    return None


def list_memory_events(db: Session, patient_id: int) -> list[MemoryEvent]:
    """按时间倒序列出患者全部长期事件。"""

    stmt = (
        select(MemoryEvent)
        .where(MemoryEvent.patient_id == patient_id)
        .order_by(MemoryEvent.event_time.desc(), MemoryEvent.id.desc())
    )
    return list(db.scalars(stmt).all())


def get_user_profile(db: Session, patient_id: int) -> Optional[UserProfile]:
    """读取患者长期画像。"""

    stmt = select(UserProfile).where(UserProfile.patient_id == patient_id)
    return db.scalar(stmt)


def clear_long_term_memory(db: Session, patient_id: int) -> dict[str, int]:
    """清空患者的长期画像、事件和偏好，供治理接口按需调用。"""

    deleted_events = db.execute(
        delete(MemoryEvent).where(MemoryEvent.patient_id == patient_id)
    )
    deleted_profiles = db.execute(
        delete(UserProfile).where(UserProfile.patient_id == patient_id)
    )
    deleted_preferences = db.execute(
        delete(MemoryPreference).where(MemoryPreference.patient_id == patient_id)
    )
    db.commit()
    memory_vector_service.replace_memory_events(
        patient_id=patient_id,
        events=[],
        source_types=["medical_case", "visit_record", "conversation"],
    )
    return {
        "memory_events": int(deleted_events.rowcount or 0),
        "user_profiles": int(deleted_profiles.rowcount or 0),
        "memory_preferences": int(deleted_preferences.rowcount or 0),
    }


def get_relevant_memory_events(
    db: Session,
    patient_id: int,
    query: str,
    limit: int = 5,
) -> list[MemoryEvent]:
    """对外暴露简化版相关事件检索结果。"""

    results = search_memory_events(db, patient_id, query, top_n=limit)
    return [item["event"] for item in results]


def get_conversation_texts_for_extraction(
    db: Session,
    patient_id: int,
    session_id: Optional[str] = None,
    limit: int = 10,
) -> list[str]:
    """提取最近若干条对话文本，供长期记忆沉淀使用。"""

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
    return [memory.content for memory in memories if memory.content.strip()]


def refresh_business_memory(
    db: Session,
    patient: Patient,
) -> list[MemoryEvent]:
    """基于病例与就诊记录重建业务侧长期事件。"""

    events = _rebuild_business_memory_events(db, patient)
    db.commit()
    logger.info(
        "Business memory extracted for patient_id=%s patient_code=%s event_count=%s",
        patient.id,
        patient.patient_code,
        len(events),
    )
    memory_vector_service.replace_memory_events(
        patient_id=patient.id,
        events=events,
        source_types=["medical_case", "visit_record"],
    )
    return events


def refresh_conversation_memory(
    db: Session,
    patient: Patient,
    conversation_texts: list[str],
) -> tuple[list[MemoryEvent], UserProfile]:
    """基于对话内容重建会话侧事件与用户画像。"""

    events = _rebuild_conversation_memory_events(db, patient, conversation_texts)
    profile = _upsert_user_profile(db, patient, conversation_texts)
    db.commit()
    logger.info(
        "Conversation memory extracted for patient_id=%s patient_code=%s event_count=%s conversation_count=%s",
        patient.id,
        patient.patient_code,
        len(events),
        len(conversation_texts),
    )
    memory_vector_service.replace_memory_events(
        patient_id=patient.id,
        events=events,
        source_types=["conversation"],
    )
    db.refresh(profile)
    return events, profile


def search_memory_events(
    db: Session,
    patient_id: int,
    query: str,
    top_n: int = 5,
) -> list[dict]:
    """执行 keyword + vector 混合检索，必要时回退到 recent。"""

    events = list_memory_events(db, patient_id)
    if not events:
        logger.info("No memory events available for hybrid search patient_id=%s", patient_id)
        return []

    event_map = {event.id: event for event in events}
    ranked: dict[int, dict] = {}
    query_keywords = [keyword for keyword in EVENT_MATCH_KEYWORDS if keyword in query]

    # 先做轻量关键词召回，保证在向量能力不可用时仍能返回稳定结果。
    for event in events:
        keyword_hits = _count_keyword_hits(query_keywords, event)
        if keyword_hits <= 0:
            continue
        keyword_score = 0.6 + min(keyword_hits, 5) * 0.08
        ranked[event.id] = {
            "event": event,
            "retrieval_score": keyword_score,
            "retrieval_sources": ["keyword"],
            "keyword_score": keyword_score,
            "vector_score": 0.0,
        }

    # 再叠加向量召回，形成可解释的混合排序。
    vector_rows = memory_vector_service.search_memory_events(
        patient_id=patient_id,
        query=query,
        top_n=top_n * 2,
    )
    for row in vector_rows:
        event = event_map.get(row["event_id"])
        if event is None:
            continue
        vector_score = row["vector_score"] * 0.7
        existing = ranked.get(event.id)
        if existing is None:
            ranked[event.id] = {
                "event": event,
                "retrieval_score": vector_score,
                "retrieval_sources": ["vector"],
                "keyword_score": 0.0,
                "vector_score": vector_score,
            }
            continue
        existing["retrieval_score"] += vector_score
        existing["vector_score"] += vector_score
        if "vector" not in existing["retrieval_sources"]:
            existing["retrieval_sources"].append("vector")

    results = sorted(
        ranked.values(),
        key=lambda item: (
            item["retrieval_score"],
            item["event"].event_time,
            item["event"].id,
        ),
        reverse=True,
    )
    if results:
        logger.info(
            "Hybrid memory search patient_id=%s query=%r top_n=%s keyword_hits=%s vector_hits=%s merged_hits=%s",
            patient_id,
            query,
            top_n,
            sum(1 for item in ranked.values() if "keyword" in item["retrieval_sources"]),
            len(vector_rows),
            len(results[:top_n]),
        )
        return results[:top_n]

    logger.info(
        "Hybrid memory search fallback to recent events for patient_id=%s query=%r top_n=%s",
        patient_id,
        query,
        top_n,
    )
    return [
        {
            "event": event,
            "retrieval_score": 0.0,
            "retrieval_sources": ["recent"],
            "keyword_score": 0.0,
            "vector_score": 0.0,
        }
        for event in events[:top_n]
    ]


def _rebuild_business_memory_events(
    db: Session,
    patient: Patient,
) -> list[MemoryEvent]:
    """重建来源于病例和就诊记录的长期事件。"""

    db.execute(
        delete(MemoryEvent).where(
            MemoryEvent.patient_id == patient.id,
            MemoryEvent.source_type.in_(["medical_case", "visit_record"]),
        )
    )

    events: list[MemoryEvent] = []
    medical_cases = medical_case_service.list_medical_cases(db, patient_id=patient.id)
    visit_records = visit_record_service.list_visit_records(db, patient_id=patient.id)

    # 业务侧事件每次都全量重建，确保长期记忆与主数据保持一致。
    for medical_case in medical_cases:
        events.append(
            MemoryEvent(
                patient_id=patient.id,
                event_type="medical_case",
                event_time=medical_case.recorded_at,
                title=f"病例诊断：{medical_case.diagnosis}",
                summary=medical_case.chief_complaint or medical_case.treatment_plan,
                source_type="medical_case",
                source_id=str(medical_case.id),
            )
        )

    for visit_record in visit_records:
        events.append(
            MemoryEvent(
                patient_id=patient.id,
                event_type="visit_record",
                event_time=visit_record.visit_time,
                title=f"{visit_record.department or '门诊'}就诊",
                summary=visit_record.summary or visit_record.notes,
                source_type="visit_record",
                source_id=str(visit_record.id),
            )
        )

    for event in events:
        db.add(event)
    db.flush()
    return [
        event
        for event in list_memory_events(db, patient.id)
        if event.source_type in {"medical_case", "visit_record"}
    ]


def _rebuild_conversation_memory_events(
    db: Session,
    patient: Patient,
    conversation_texts: list[str],
) -> list[MemoryEvent]:
    """从近期对话中抽取医疗相关事件。"""

    db.execute(
        delete(MemoryEvent).where(
            MemoryEvent.patient_id == patient.id,
            MemoryEvent.source_type == "conversation",
        )
    )

    events: list[MemoryEvent] = []
    for index, text in enumerate(conversation_texts, start=1):
        if not text.strip():
            continue
        matched_keywords = [keyword for keyword in MEDICAL_KEYWORDS if keyword in text]
        if not matched_keywords:
            continue
        events.append(
            MemoryEvent(
                patient_id=patient.id,
                event_type="conversation_medical_hint",
                event_time=datetime.utcnow(),
                title=f"对话提及：{'、'.join(matched_keywords[:3])}",
                summary=text[:300],
                source_type="conversation",
                source_id=f"conversation-{index}",
            )
        )

    for event in events:
        db.add(event)
    db.flush()
    return [event for event in list_memory_events(db, patient.id) if event.source_type == "conversation"]


def _upsert_user_profile(
    db: Session,
    patient: Patient,
    conversation_texts: list[str],
) -> UserProfile:
    """根据长期偏好和近期对话重建用户画像。"""

    profile = get_user_profile(db, patient.id)
    if profile is None:
        profile = UserProfile(patient_id=patient.id)

    preference = _get_memory_preference(db, patient.id)
    conversation_blob = "\n".join(conversation_texts)
    topics = _extract_topics(patient, preference, conversation_blob)
    communication_style = _extract_communication_style(preference, conversation_blob)
    stable_preferences = _build_stable_preferences(preference, communication_style)
    profile.profile_summary = _build_profile_summary(patient, topics, stable_preferences)
    profile.communication_style = communication_style
    profile.preferred_topics = topics
    profile.stable_preferences = stable_preferences
    profile.source_summary = _build_source_summary(preference, conversation_texts)
    profile.refreshed_at = datetime.utcnow()

    db.add(profile)
    db.flush()
    return profile


def _get_memory_preference(db: Session, patient_id: int) -> Optional[MemoryPreference]:
    """读取患者长期偏好配置。"""

    stmt = select(MemoryPreference).where(MemoryPreference.patient_id == patient_id)
    return db.scalar(stmt)


def _extract_topics(
    patient: Patient,
    preference: Optional[MemoryPreference],
    conversation_blob: str,
) -> str:
    """从显式偏好和近期对话中提取高频关注主题。"""

    topics: list[str] = []
    if preference and preference.focus_topics:
        topics.extend(_split_items(preference.focus_topics))
    for keyword in MEDICAL_KEYWORDS:
        if keyword in conversation_blob and keyword not in topics:
            topics.append(keyword)
    return "、".join(topics[:6]) or "常规健康咨询"


def _extract_communication_style(
    preference: Optional[MemoryPreference],
    conversation_blob: str,
) -> str:
    """优先采用显式偏好，否则根据对话关键词做轻量推断。"""

    if preference and preference.response_style:
        return preference.response_style
    styles = [value for keyword, value in STYLE_KEYWORDS.items() if keyword in conversation_blob]
    return "、".join(dict.fromkeys(styles)) if styles else "稳健、清晰"


def _build_stable_preferences(
    preference: Optional[MemoryPreference],
    communication_style: str,
) -> str:
    """把长期偏好拼成可直接展示的摘要文本。"""

    parts: list[str] = []
    if preference and preference.preferred_name:
        parts.append(f"偏好称呼：{preference.preferred_name}")
    if preference and preference.response_length:
        parts.append(f"回答长度：{preference.response_length}")
    if preference and preference.preferred_language:
        parts.append(f"语言：{preference.preferred_language}")
    parts.append(f"表达风格：{communication_style}")
    if preference and preference.additional_preferences:
        parts.append(f"补充偏好：{preference.additional_preferences}")
    return "；".join(parts)


def _build_profile_summary(
    patient: Patient,
    topics: str,
    stable_preferences: str,
) -> str:
    """生成对外展示用的用户画像摘要。"""

    return (
        f"{patient.full_name}（{patient.patient_code}）长期关注 {topics} 相关内容。"
        f"稳定偏好：{stable_preferences}。"
    )


def _build_source_summary(
    preference: Optional[MemoryPreference],
    conversation_texts: list[str],
) -> str:
    """记录画像的来源，方便前端调试和后续审计。"""

    source_parts = ["业务数据：病例与诊疗记录"]
    if preference is not None:
        source_parts.append("用户配置：长期偏好设置")
    if conversation_texts:
        source_parts.append(f"短期对话：{len(conversation_texts)}条")
    return "；".join(source_parts)


def _split_items(text: str) -> list[str]:
    """兼容中文和英文分隔符。"""

    normalized = text.replace("；", ",").replace("、", ",").replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _count_keyword_hits(query_keywords: list[str], event: MemoryEvent) -> int:
    """统计查询关键词与记忆事件文本的交集命中数。"""

    if not query_keywords:
        return 0
    haystack = f"{event.title} {event.summary or ''}"
    return sum(1 for keyword in query_keywords if keyword in haystack)
