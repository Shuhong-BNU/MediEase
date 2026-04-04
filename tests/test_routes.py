"""
API 路由测试。
覆盖点：
- 稳定会话复用与短期记忆回写。
- 五轮阈值触发长期沉淀。
- 语音播报成功/失败降级。
- 风险提示、引用来源、人工升级和记忆清理接口。
- 报告解读接口最小链路。
"""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.api.routes import get_db
from app.db.base import Base
from app.db.models import ConversationMemory, KnowledgeDocument, Patient
from app.main import app


class StubAgent:
    """用于路由测试的极简 Agent。"""

    def __init__(self, db, llm_client, execution_context):
        self.execution_context = execution_context

    def run(self, user_query, images=None, debug_planner=False, memory_context=None):
        return {
            "answer": f"回答：{user_query}",
            "conversation_session_id": self.execution_context.conversation_session_id,
            "tool_outputs": [],
            "planner_debug": None,
            "resolved_patient_id": 1,
            "verified_patient_id": 1,
        }


class StubAgentWithEvidence:
    """返回知识库工具结果，便于验证 citations 与风险升级。"""

    def __init__(self, db, llm_client, execution_context):
        self.execution_context = execution_context

    def run(self, user_query, images=None, debug_planner=False, memory_context=None):
        return {
            "answer": "你可以立即停药，基本可以确诊。",
            "conversation_session_id": self.execution_context.conversation_session_id,
            "tool_outputs": [
                {
                    "tool_name": "search_knowledge_base",
                    "arguments": {"query": "停药 风险"},
                    "result": {
                        "found": True,
                        "results": [
                            {
                                "id": 1,
                                "title": "停药风险说明",
                                "snippet": "擅自停药可能导致病情波动。",
                                "source_url": "https://example.com/kb/1",
                            }
                        ],
                    },
                    "access_granted": True,
                    "denial_reason": None,
                }
            ],
            "planner_debug": None,
            "resolved_patient_id": 1,
            "verified_patient_id": 1,
        }


class StubSpeechClient:
    def synthesize(self, text, voice="longanyang", audio_format="mp3"):
        return {
            "audio_base64": base64.b64encode(b"fake-audio").decode("utf-8"),
            "mime_type": "audio/mp3",
            "model": "fake-tts",
            "voice": voice,
        }


class FailingSpeechClient:
    def synthesize(self, text, voice="longanyang", audio_format="mp3"):
        raise ValueError("tts failed")


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "route_test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)

        with self.SessionLocal() as db:
            patient = Patient(
                id=1,
                patient_code="P0001",
                full_name="王建国",
                phone="13800000001",
                id_number="310101195911051234",
            )
            db.add(patient)
            db.add(
                KnowledgeDocument(
                    id=1,
                    title="停药风险说明",
                    category="用药",
                    content="擅自停药可能导致病情波动。",
                    enabled=True,
                )
            )
            db.commit()

        def override_get_db():
            db: Session = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_query_returns_and_reuses_conversation_session_id(self) -> None:
        with patch("app.api.routes.QwenClient", return_value=object()), patch(
            "app.api.routes.QwenMCPAgent", StubAgent
        ):
            first = self.client.post("/api/agent/query", json={"query": "你好"})
            self.assertEqual(first.status_code, 200)
            first_session_id = first.json()["conversation_session_id"]

            second = self.client.post(
                "/api/agent/query",
                json={
                    "query": "继续追问",
                    "conversation_session_id": first_session_id,
                },
            )
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["conversation_session_id"], first_session_id)

        with self.SessionLocal() as db:
            memories = db.query(ConversationMemory).order_by(ConversationMemory.id.asc()).all()
            self.assertEqual(len(memories), 4)
            self.assertTrue(all(item.session_id == first_session_id for item in memories))

    def test_query_triggers_long_term_refresh_after_five_rounds(self) -> None:
        with patch("app.api.routes.QwenClient", return_value=object()), patch(
            "app.api.routes.QwenMCPAgent", StubAgent
        ), patch(
            "app.api.routes.memory_service.refresh_conversation_memory",
            return_value=([], None),
        ) as refresh_mock:
            for index in range(5):
                response = self.client.post(
                    "/api/agent/query",
                    json={"query": f"第{index + 1}轮"},
                )
                self.assertEqual(response.status_code, 200)

        self.assertEqual(refresh_mock.call_count, 1)

    def test_query_returns_speech_download_url_when_tts_succeeds(self) -> None:
        with patch("app.api.routes.QwenClient", return_value=object()), patch(
            "app.api.routes.QwenMCPAgent", StubAgent
        ), patch("app.api.routes.QwenSpeechClient", StubSpeechClient), patch(
            "app.api.routes._save_speech_audio",
            return_value=("D:/tmp/fake.mp3", "/media/generated_audio/fake.mp3"),
        ):
            response = self.client.post(
                "/api/agent/query",
                json={
                    "query": "请播报",
                    "enable_speech": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.json()["speech_download_url"].endswith("/media/generated_audio/fake.mp3")
        )

    def test_query_preserves_text_when_tts_fails(self) -> None:
        with patch("app.api.routes.QwenClient", return_value=object()), patch(
            "app.api.routes.QwenMCPAgent", StubAgent
        ), patch("app.api.routes.QwenSpeechClient", FailingSpeechClient):
            response = self.client.post(
                "/api/agent/query",
                json={
                    "query": "请播报但允许失败",
                    "enable_speech": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "回答：请播报但允许失败")
        self.assertIsNone(response.json().get("speech_download_url"))

    def test_query_returns_citations_and_manual_escalation_when_high_risk(self) -> None:
        with patch("app.api.routes.QwenClient", return_value=object()), patch(
            "app.api.routes.QwenMCPAgent", StubAgentWithEvidence
        ):
            response = self.client.post(
                "/api/agent/query",
                json={"query": "我胸痛得厉害，现在可以停药吗？"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["citations"])
        self.assertTrue(payload["risk_alerts"])
        self.assertIsNotNone(payload["manual_escalation"])
        self.assertTrue(any("线下就医" in item for item in payload["recommended_actions"]))

    def test_memory_clear_endpoint_deletes_short_term_records(self) -> None:
        with self.SessionLocal() as db:
            db.add_all(
                [
                    ConversationMemory(
                        patient_id=1,
                        session_id="session-1",
                        role="user",
                        content="hi",
                    ),
                    ConversationMemory(
                        patient_id=1,
                        session_id="session-1",
                        role="assistant",
                        content="hello",
                    ),
                ]
            )
            db.commit()

        response = self.client.post("/api/memory/clear?patient_id=1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["deleted_short_term"], 2)

    def test_report_interpret_endpoint_works_with_plain_text(self) -> None:
        response = self.client.post(
            "/api/reports/interpret",
            json={
                "title": "血常规复查",
                "report_type": "检验",
                "report_text": "白细胞 12.6 偏高\nC 反应蛋白 36 异常",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["risk_level"], "medium")
        self.assertGreaterEqual(len(payload["abnormal_items"]), 2)


if __name__ == "__main__":
    unittest.main()
