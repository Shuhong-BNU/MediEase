"""
项目全部 ORM 数据模型。

模型分为四类：
1. 患者业务数据：患者、病例、就诊记录。
2. 记忆数据：长期偏好、长期画像、长期关键事件、短期对话。
3. Agent 运行时数据：稳定会话上下文。
4. 审计数据：工具调用和敏感读取日志。

这些模型共同支撑 `/api/agent/query` 的主链路、记忆沉淀链路和后续调试/审计能力。
"""

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Patient(Base):
    """患者基础身份信息。"""

    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(128))
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    phone_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    id_number: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    id_number_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    address: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    address_encrypted: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emergency_contact_name: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )
    emergency_contact_phone: Mapped[Optional[str]] = mapped_column(
        String(32),
        nullable=True,
    )
    emergency_contact_phone_encrypted: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    medical_cases: Mapped[list["MedicalCase"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    visit_records: Mapped[list["VisitRecord"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    medical_reports: Mapped[list["MedicalReport"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    memory_preference: Mapped[Optional["MemoryPreference"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    user_profile: Mapped[Optional["UserProfile"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    memory_events: Mapped[list["MemoryEvent"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    conversation_memories: Mapped[list["ConversationMemory"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    agent_sessions: Mapped[list["AgentConversationSession"]] = relationship(
        back_populates="patient",
        foreign_keys="AgentConversationSession.patient_id",
    )
    verified_agent_sessions: Mapped[list["AgentConversationSession"]] = relationship(
        back_populates="verified_patient",
        foreign_keys="AgentConversationSession.verified_patient_id",
    )
    tool_audit_logs: Mapped[list["ToolAuditLog"]] = relationship(
        back_populates="patient",
    )
    manual_escalation_events: Mapped[list["ManualEscalationEvent"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class MedicalCase(Base):
    """患者病例记录，偏向诊断与治疗事实。"""

    __tablename__ = "medical_cases"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    case_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    diagnosis: Mapped[str] = mapped_column(String(255))
    chief_complaint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    present_illness: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    past_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    treatment_plan: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attending_physician: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="medical_cases")


class VisitRecord(Base):
    """患者单次就诊行为记录，偏向时间线和具体就诊事实。"""

    __tablename__ = "visit_records"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    visit_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    visit_type: Mapped[str] = mapped_column(String(32))
    department: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    physician_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    visit_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="visit_records")


class MedicalReport(Base):
    """检验检查报告，承载文本/图片报告的结构化摘要与原始内容。"""

    __tablename__ = "medical_reports"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    report_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    report_type: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(255))
    department: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    report_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    structured_data_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    abnormal_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), default="manual")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="medical_reports")


class MemoryPreference(Base):
    """用户主动配置的长期偏好。"""

    __tablename__ = "memory_preferences"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        unique=True,
        index=True,
    )
    preferred_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    response_style: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    response_length: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    preferred_language: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    focus_topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    additional_preferences: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="memory_preference")


class UserProfile(Base):
    """从业务数据和对话中沉淀出的长期用户画像。"""

    __tablename__ = "user_profiles"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id"),
        unique=True,
        index=True,
    )
    profile_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    communication_style: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    preferred_topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stable_preferences: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    correction_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    refreshed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="user_profile")


class MemoryEvent(Base):
    """长期关键事件，用于结构化查询和混合检索。"""

    __tablename__ = "memory_events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    event_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(64))
    source_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    correction_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="memory_events")


class ConversationMemory(Base):
    """短期对话记忆，按患者与会话双维度保存。"""

    __tablename__ = "conversation_memories"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    multimodal_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="conversation_memories")


class AgentConversationSession(Base):
    """Agent 侧稳定会话上下文，承载患者归属与已验证身份。"""

    __tablename__ = "agent_conversation_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    patient_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("patients.id"),
        nullable=True,
        index=True,
    )
    verified_patient_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("patients.id"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    last_active_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped[Optional["Patient"]] = relationship(
        back_populates="agent_sessions",
        foreign_keys=[patient_id],
    )
    verified_patient: Mapped[Optional["Patient"]] = relationship(
        back_populates="verified_agent_sessions",
        foreign_keys=[verified_patient_id],
    )
    tool_audit_logs: Mapped[list["ToolAuditLog"]] = relationship(
        back_populates="conversation_session",
        cascade="all, delete-orphan",
    )


class ToolAuditLog(Base):
    """工具调用与敏感读取审计日志。"""

    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    conversation_session_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversation_sessions.session_id"),
        index=True,
    )
    patient_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("patients.id"),
        nullable=True,
        index=True,
    )
    tool_name: Mapped[str] = mapped_column(String(128), index=True)
    arguments_json: Mapped[str] = mapped_column(Text)
    result_summary_json: Mapped[str] = mapped_column(Text)
    access_granted: Mapped[bool] = mapped_column(Boolean, default=True)
    denial_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )

    conversation_session: Mapped["AgentConversationSession"] = relationship(
        back_populates="tool_audit_logs",
    )
    patient: Mapped[Optional["Patient"]] = relationship(back_populates="tool_audit_logs")


class ManualEscalationEvent(Base):
    """记录高风险对话生成的人工升级建议，便于后续人工接管或回访。"""

    __tablename__ = "manual_escalation_events"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    conversation_session_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("agent_conversation_sessions.session_id"),
        nullable=True,
        index=True,
    )
    patient_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("patients.id"),
        nullable=True,
        index=True,
    )
    risk_level: Mapped[str] = mapped_column(String(32), index=True)
    trigger_reason: Mapped[str] = mapped_column(Text)
    recommended_action: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    conversation_session: Mapped[Optional["AgentConversationSession"]] = relationship()
    patient: Mapped[Optional["Patient"]] = relationship(
        back_populates="manual_escalation_events"
    )


class KnowledgeDocument(Base):
    """知识库文档，作为业务数据之外的第二证据源供 Agent 检索。"""

    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tenant_code: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    category: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TenantConfig(Base):
    """多租户配置骨架，当前用于预留 prompt 与工具开关的扩展位。"""

    __tablename__ = "tenant_configs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    tenant_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    prompt_preamble: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enabled_tools_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    knowledge_namespace: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    response_style_overrides: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
