"""
记忆与检索相关测试。

覆盖点：
- 短期记忆优先按会话读取，确保多轮对话上下文边界清晰。
- 当向量检索不可用或无结果时，混合检索会退化到 recent 模式。
"""

from __future__ import annotations

from datetime import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.base import Base
from app.db.models import ConversationMemory, MemoryEvent, Patient
from app.services import conversation_memory_service, memory_service


class MemoryServiceTests(unittest.TestCase):
    """验证记忆读取与检索退化逻辑。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "memory_test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)
        self.db = self.SessionLocal()

        patient = Patient(patient_code="P0002", full_name="李阿梅")
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        self.patient = patient

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_recent_conversation_memories_prefer_session_scope(self) -> None:
        """同一患者有多段会话时，应优先返回指定会话的上下文。"""

        self.db.add_all(
            [
                ConversationMemory(
                    patient_id=self.patient.id,
                    session_id="session-a",
                    role="user",
                    content="A1",
                ),
                ConversationMemory(
                    patient_id=self.patient.id,
                    session_id="session-a",
                    role="assistant",
                    content="A2",
                ),
                ConversationMemory(
                    patient_id=self.patient.id,
                    session_id="session-b",
                    role="user",
                    content="B1",
                ),
            ]
        )
        self.db.commit()

        records = conversation_memory_service.list_recent_conversation_memories(
            self.db,
            patient_id=self.patient.id,
            session_id="session-a",
            limit=10,
        )

        self.assertEqual([item.content for item in records], ["A1", "A2"])

    def test_memory_search_falls_back_to_recent_when_vector_unavailable(self) -> None:
        """当 keyword 和 vector 都没有命中时，应回退到最近事件列表。"""

        self.db.add_all(
            [
                MemoryEvent(
                    patient_id=self.patient.id,
                    event_type="visit_record",
                    event_time=datetime(2026, 4, 1, 10, 0, 0),
                    title="心内科复诊",
                    summary="复诊稳定",
                    source_type="visit_record",
                    source_id="1",
                ),
                MemoryEvent(
                    patient_id=self.patient.id,
                    event_type="medical_case",
                    event_time=datetime(2026, 4, 2, 10, 0, 0),
                    title="病例诊断：高血压",
                    summary="持续观察血压",
                    source_type="medical_case",
                    source_id="2",
                ),
            ]
        )
        self.db.commit()

        with patch(
            "app.services.memory_vector_service.search_memory_events",
            return_value=[],
        ):
            results = memory_service.search_memory_events(
                self.db,
                patient_id=self.patient.id,
                query="这次完全不含关键词",
                top_n=2,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["retrieval_sources"], ["recent"])


if __name__ == "__main__":
    unittest.main()
