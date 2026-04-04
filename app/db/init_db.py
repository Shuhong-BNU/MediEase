"""
数据库初始化与轻量迁移。

项目启动时会调用 `init_db()`：
1. 对当前 ORM 模型执行建表。
2. 对历史 SQLite 数据做必要的轻量补列迁移。

本文件刻意保持“轻量、幂等、可重复执行”，便于本地原型项目直接升级。
"""

from sqlalchemy import text

from app.db.base import Base
from app.db.models import (
    AgentConversationSession,
    ConversationMemory,
    KnowledgeDocument,
    MedicalCase,
    MedicalReport,
    ManualEscalationEvent,
    MemoryEvent,
    MemoryPreference,
    Patient,
    TenantConfig,
    ToolAuditLog,
    UserProfile,
    VisitRecord,
)
from app.db.session import engine


def init_db() -> None:
    """创建全部表，并执行兼容老库的轻量迁移。"""

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()


def _ensure_sqlite_columns() -> None:
    """为历史 SQLite 库补齐新增字段，避免要求手工重建数据库。"""

    with engine.begin() as connection:
        _ensure_column(
            connection,
            "conversation_memories",
            "multimodal_payload",
            "TEXT",
        )
        _ensure_column(connection, "patients", "phone_encrypted", "TEXT")
        _ensure_column(connection, "patients", "id_number_encrypted", "TEXT")
        _ensure_column(connection, "patients", "address_encrypted", "TEXT")
        _ensure_column(
            connection,
            "patients",
            "emergency_contact_phone_encrypted",
            "TEXT",
        )
        _ensure_column(connection, "user_profiles", "correction_note", "TEXT")
        _ensure_column(connection, "user_profiles", "expires_at", "DATETIME")
        _ensure_column(connection, "memory_events", "correction_note", "TEXT")
        _ensure_column(connection, "memory_events", "expires_at", "DATETIME")


def _ensure_column(connection, table_name: str, column_name: str, column_sql: str) -> None:
    """涓哄崟涓〃鎵ц骞呯瓑鐨勮ˉ鍒楄縼绉汇€?"""

    columns = connection.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    column_names = {column[1] for column in columns}
    if column_name not in column_names:
        connection.execute(
            text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")
        )
