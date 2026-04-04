"""
Agent 执行器测试。

覆盖 Planner 记忆注入、强制验权、验权后访问同一患者数据等核心行为。
这些测试直接调用 `QwenMCPAgent`，避免把问题掩盖在 API 层之下。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import Patient, ToolAuditLog, VisitRecord
from app.llm.qwen_mcp_agent import AgentExecutionContext, QwenMCPAgent


class SequenceLLM:
    """用固定序列模拟 Planner、工具调用和 Finalizer。"""

    def __init__(self, tool_batches=None, final_answer="最终回答") -> None:
        self.tool_batches = list(tool_batches or [])
        self.final_answer = final_answer
        self.planner_messages = []

    def complete(self, messages, temperature=0):
        if messages and messages[0]["role"] == "system" and "内部 Planner" in messages[0]["content"]:
            self.planner_messages.append(messages)
            return {
                "content": json.dumps(
                    {
                        "objective": "完成患者问题回答",
                        "need_identity_verification": True,
                        "image_reasoning": False,
                        "tool_sequence": ["verify_patient_identity", "get_patient_visit_records"],
                        "steps": ["先验权", "再查询记录"],
                        "final_answer_focus": ["结论", "依据"],
                    },
                    ensure_ascii=False,
                )
            }
        return {"content": self.final_answer}

    def complete_with_tools(self, messages, tools, tool_choice="auto", temperature=0):
        if self.tool_batches:
            batch = self.tool_batches.pop(0)
            tool_calls = []
            for index, call in enumerate(batch, start=1):
                tool_calls.append(
                    {
                        "id": f"call-{index}",
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                )
            return {
                "content": "",
                "assistant_message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item["id"],
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": json.dumps(item["arguments"], ensure_ascii=False),
                            },
                        }
                        for item in tool_calls
                    ],
                },
                "tool_calls": tool_calls,
            }
        return {
            "content": "草稿回答",
            "assistant_message": {
                "role": "assistant",
                "content": "草稿回答",
                "tool_calls": [],
            },
            "tool_calls": [],
        }


class AgentExecutionTests(unittest.TestCase):
    """直接验证 Agent 控制流与权限行为。"""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "agent_test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)
        self.db: Session = self.SessionLocal()

        patient = Patient(
            patient_code="P0001",
            full_name="王建国",
            phone="13800000001",
            id_number="310101195911051234",
        )
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)
        self.patient = patient

        visit_record = VisitRecord(
            patient_id=patient.id,
            visit_code="V0001",
            visit_type="outpatient",
            department="心内科",
            physician_name="李医生",
            summary="复诊稳定",
        )
        self.db.add(visit_record)
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_planner_receives_memory_summary(self) -> None:
        llm = SequenceLLM()
        agent = QwenMCPAgent(
            db=self.db,
            llm_client=llm,
            execution_context=AgentExecutionContext(conversation_session_id="session-1"),
        )

        agent.run(
            "请总结最近一次复诊重点",
            memory_context={
                "short_term_memories": [
                    {"role": "user", "content": "我最近胸闷", "multimodal_payload": None},
                    {"role": "assistant", "content": "建议复诊", "multimodal_payload": None},
                ],
                "user_profile": {
                    "profile_summary": "长期关注心内科复诊与用药提醒",
                    "stable_preferences": "偏好简洁表达",
                    "preferred_topics": "心内科、复诊",
                },
                "relevant_events": [
                    {
                        "event_time": "2026-04-01T10:00:00",
                        "title": "心内科复诊",
                        "summary": "复诊稳定",
                    }
                ],
            },
        )

        planner_prompt = llm.planner_messages[0][1]["content"]
        self.assertIn("相关记忆摘要：", planner_prompt)
        self.assertIn("长期画像=", planner_prompt)
        self.assertIn("相关事件=", planner_prompt)

    def test_debug_output_contains_langgraph_trace(self) -> None:
        llm = SequenceLLM()
        agent = QwenMCPAgent(
            db=self.db,
            llm_client=llm,
            execution_context=AgentExecutionContext(conversation_session_id="session-debug"),
        )

        result = agent.run("debug trace", debug_planner=True)

        self.assertEqual(result["planner_debug"]["workflow_framework"], "langgraph")
        self.assertEqual(
            result["planner_debug"]["graph_trace"],
            ["planner", "tool_routing", "finalizer", "risk_check"],
        )

    def test_sensitive_tool_denied_without_identity_verification(self) -> None:
        llm = SequenceLLM(
            tool_batches=[
                [
                    {
                        "name": "get_patient_visit_records",
                        "arguments": {"patient_code": "P0001", "limit": 1},
                    }
                ]
            ]
        )
        agent = QwenMCPAgent(
            db=self.db,
            llm_client=llm,
            execution_context=AgentExecutionContext(
                conversation_session_id="session-2",
                resolved_patient_id=self.patient.id,
            ),
        )

        result = agent.run("帮我看最近一次就诊记录")

        self.assertEqual(len(result["tool_outputs"]), 1)
        tool_output = result["tool_outputs"][0]
        self.assertFalse(tool_output["access_granted"])
        self.assertEqual(tool_output["denial_reason"], "identity verification required")

        audit_logs = self.db.query(ToolAuditLog).all()
        self.assertEqual(len(audit_logs), 1)
        self.assertFalse(audit_logs[0].access_granted)

    def test_sensitive_tool_allowed_after_successful_verification(self) -> None:
        llm = SequenceLLM(
            tool_batches=[
                [
                    {
                        "name": "verify_patient_identity",
                        "arguments": {
                            "patient_code": "P0001",
                            "phone": "13800000001",
                        },
                    }
                ],
                [
                    {
                        "name": "get_patient_visit_records",
                        "arguments": {"patient_code": "P0001", "limit": 1},
                    }
                ],
            ]
        )
        agent = QwenMCPAgent(
            db=self.db,
            llm_client=llm,
            execution_context=AgentExecutionContext(conversation_session_id="session-3"),
        )

        result = agent.run("我是 P0001，请帮我看最近一次就诊记录")

        self.assertEqual(len(result["tool_outputs"]), 2)
        self.assertTrue(result["tool_outputs"][0]["access_granted"])
        self.assertTrue(result["tool_outputs"][1]["access_granted"])
        self.assertEqual(result["verified_patient_id"], self.patient.id)
        self.assertEqual(
            result["tool_outputs"][1]["result"]["visit_records"][0]["visit_code"],
            "V0001",
        )


if __name__ == "__main__":
    unittest.main()
