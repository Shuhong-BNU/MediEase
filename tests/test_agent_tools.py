"""
Agent 新工具测试。
覆盖点：
- 知识库检索工具可被 Agent 执行并返回结构化结果。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.db.models import AgentConversationSession, KnowledgeDocument
from app.llm.qwen_mcp_agent import AgentExecutionContext, QwenMCPAgent
from app.services import knowledge_service


class KnowledgeToolLLM:
    def complete(self, messages, temperature=0):
        if messages and messages[0]["role"] == "system" and "Planner" in messages[0]["content"]:
            return {
                "content": json.dumps(
                    {
                        "objective": "搜索知识库并回答",
                        "need_identity_verification": False,
                        "image_reasoning": False,
                        "tool_sequence": ["search_knowledge_base"],
                        "steps": ["检索知识库", "整理回答"],
                        "final_answer_focus": ["结论", "依据"],
                    },
                    ensure_ascii=False,
                )
            }
        return {"content": "整理后的回答"}

    def complete_with_tools(self, messages, tools, tool_choice="auto", temperature=0):
        has_tool_response = any(message.get("role") == "tool" for message in messages)
        if not has_tool_response:
            return {
                "content": "",
                "assistant_message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "search_knowledge_base",
                                "arguments": json.dumps({"query": "停药 风险"}, ensure_ascii=False),
                            },
                        }
                    ],
                },
                "tool_calls": [
                    {
                        "id": "call-1",
                        "name": "search_knowledge_base",
                        "arguments": {"query": "停药 风险"},
                    }
                ],
            }
        return {
            "content": "不要擅自停药。",
            "assistant_message": {"role": "assistant", "content": "不要擅自停药。", "tool_calls": []},
            "tool_calls": [],
        }


class AgentToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "agent_tool_test.db"
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(self.engine)
        self.db: Session = self.SessionLocal()

        self.db.add(AgentConversationSession(session_id="session-tool"))
        self.db.add(
            KnowledgeDocument(
                title="停药风险说明",
                category="用药",
                content="擅自停药可能导致病情波动。",
                keywords="停药 风险 PILL_RISK",
                enabled=True,
            )
        )
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_agent_can_execute_knowledge_base_tool(self) -> None:
        agent = QwenMCPAgent(
            db=self.db,
            llm_client=KnowledgeToolLLM(),
            execution_context=AgentExecutionContext(conversation_session_id="session-tool"),
        )

        result = agent.run("停药有哪些风险？")

        self.assertEqual(len(result["tool_outputs"]), 1)
        self.assertEqual(result["tool_outputs"][0]["tool_name"], "search_knowledge_base")
        self.assertTrue(result["tool_outputs"][0]["result"]["found"])
        self.assertEqual(
            result["tool_outputs"][0]["result"]["results"][0]["title"],
            "停药风险说明",
        )


    def test_langchain_retriever_returns_documents(self) -> None:
        retriever = knowledge_service.build_knowledge_retriever(self.db, top_n=3)

        documents = retriever.invoke("PILL_RISK")

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["title"], "停药风险说明")
        self.assertTrue(documents[0].page_content)


if __name__ == "__main__":
    unittest.main()
