"""
MediEase 的 LangGraph 编排层。

本模块保留原有 `QwenMCPAgent` 对外接口不变，但把内部执行流迁移为
LangGraph 状态图，核心目标是：
1. 把 Planner、工具路由、工具执行、Finalizer、风险检查、人工升级做成显式节点。
2. 继续复用现有 service 层、权限策略、审计日志和数据库模型。
3. 保持 `/api/agent/query`、单元测试和前端调用契约基本不变。
"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from app.llm.qwen_client import QwenClient
from app.services import (
    escalation_service,
    mcp_tool_service,
    safety_service,
    tool_audit_service,
    tool_registry_service,
)


ToolHandler = Callable[..., Dict[str, Any]]


SYSTEM_PROMPT = """
你是医院患者智能辅助 Agent。你可以调用内部工具完成身份验证、病例查询、就诊记录调取、报告查询、知识库检索和人工升级。
涉及患者隐私数据时，必须先完成身份验证，未验证前不要读取病例、就诊记录、报告和完整患者资料。
回答必须基于工具结果，不要编造病例或就诊信息。当用户要求“最近一次”“最新一次”就诊记录时，使用 get_patient_visit_records，并传入 limit=1。
不要臆造不存在的工具名称。采用“先规划、再行动、最后校验”的工作方式：
1. 先根据问题形成简短内部计划。
2. 每一步只执行当前最必要的工具。
3. 工具结果不足时继续补查，足够时停止。
4. 最终答案只保留结论和证据，不暴露冗长内部思维。
如果提供了“短期记忆、用户画像、关键事件、风险提示”上下文，回答前优先参考这些信息，并在与用户当前问题相关时加以利用。
""".strip()

PLANNER_PROMPT = """
你是 Agent 的内部 Planner，需要为当前问题生成简洁计划。要求结合以下策略：
1. CoT：先拆分目标、约束、所需证据。
2. ReAct：明确下一步最合适的动作和工具顺序。
3. Self-Consistency：你会被多次采样，输出要稳定、可执行、短。
仅输出 JSON，不要输出 markdown，不要解释：
{
  "objective": "一句话目标",
  "need_identity_verification": true,
  "image_reasoning": false,
  "tool_sequence": ["verify_patient_identity", "get_patient_visit_records"],
  "steps": ["步骤1", "步骤2"],
  "final_answer_focus": ["回答应覆盖的重点1", "重点2"]
}
""".strip()

FINALIZER_PROMPT = """
你是最终答案整理器。请基于用户问题、执行计划和工具结果给出最终回答。要求：
1. 只能使用已有工具结果和已知图片内容。
2. 如果证据不足，明确指出不足。
3. 直接给结论、依据和必要提醒，不暴露内部思维链。
""".strip()

MAX_TOOL_STEPS = 6
PLAN_TEMPERATURES = [0.1, 0.4, 0.7]


@dataclass
class AgentExecutionContext:
    """一次问答执行过程中共享的可变上下文。"""

    conversation_session_id: str
    resolved_patient_id: Optional[int] = None
    verified_patient_id: Optional[int] = None
    memory_context_summary: str = ""
    tool_audit_context: dict[str, Any] = field(default_factory=dict)


class AgentGraphState(TypedDict, total=False):
    """LangGraph 里流转的状态。"""

    user_query: str
    images: list[dict[str, Any]]
    has_images: bool
    memory_context: dict[str, Any]
    memory_context_summary: str
    execution_plan: dict[str, Any]
    planner_debug: dict[str, Any]
    messages: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    tool_steps: int
    pending_tool_calls: list[dict[str, Any]]
    latest_response: dict[str, Any]
    draft_answer: str
    answer: str
    post_safety_assessment: dict[str, Any]
    manual_escalation: dict[str, Any] | None
    execution_trace: list[str]
    max_steps_reached: bool


class QwenMCPAgent:
    """基于 LangGraph 的 Qwen Agent 执行器。"""

    def __init__(
        self,
        db: Session,
        llm_client: QwenClient,
        execution_context: AgentExecutionContext,
    ) -> None:
        self.db = db
        self.llm_client = llm_client
        self.execution_context = execution_context
        self.tools = self._build_tool_registry()
        self.graph = self._build_graph()

    def run(
        self,
        user_query: str,
        images: Optional[List[Dict[str, Any]]] = None,
        debug_planner: bool = False,
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行一次完整问答。"""

        initial_state: AgentGraphState = {
            "user_query": user_query,
            "images": list(images or []),
            "has_images": bool(images),
            "memory_context": dict(memory_context or {}),
            "tool_outputs": [],
            "tool_steps": 0,
            "execution_trace": [],
            "max_steps_reached": False,
            "manual_escalation": None,
        }
        final_state = self.graph.invoke(initial_state)
        return self._build_result(
            answer=final_state.get("answer", ""),
            tool_outputs=final_state.get("tool_outputs", []),
            planner_debug=final_state.get("planner_debug", {}),
            execution_plan=final_state.get("execution_plan", {}),
            debug_planner=debug_planner,
            post_safety_assessment=final_state.get("post_safety_assessment"),
            manual_escalation=final_state.get("manual_escalation"),
            execution_trace=final_state.get("execution_trace", []),
        )

    def _build_graph(self):
        """构建一次问答使用的 LangGraph 状态图。"""

        workflow = StateGraph(AgentGraphState)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("tool_routing", self._tool_routing_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("finalizer", self._finalizer_node)
        workflow.add_node("risk_check", self._risk_check_node)
        workflow.add_node("escalation", self._escalation_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "tool_routing")
        workflow.add_conditional_edges(
            "tool_routing",
            self._route_after_tool_routing,
            {"tool_execution": "tool_execution", "finalizer": "finalizer"},
        )
        workflow.add_edge("tool_execution", "tool_routing")
        workflow.add_edge("finalizer", "risk_check")
        workflow.add_conditional_edges(
            "risk_check",
            self._route_after_risk_check,
            {"escalation": "escalation", "finish": END},
        )
        workflow.add_edge("escalation", END)
        return workflow.compile()

    def _planner_node(self, state: AgentGraphState) -> dict[str, Any]:
        memory_context = state.get("memory_context") or {}
        memory_context_summary = self._build_memory_summary(memory_context)
        self.execution_context.memory_context_summary = memory_context_summary
        execution_plan, planner_debug = self._build_execution_plan(
            user_query=state["user_query"],
            has_images=state.get("has_images", False),
            memory_context_summary=memory_context_summary,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._build_memory_messages(memory_context),
            {"role": "system", "content": self._format_execution_plan(execution_plan)},
            {
                "role": "user",
                "content": self._build_user_content(state["user_query"], state.get("images", [])),
            },
        ]
        return {
            "memory_context_summary": memory_context_summary,
            "execution_plan": execution_plan,
            "planner_debug": planner_debug,
            "messages": messages,
            "execution_trace": self._append_trace(state, "planner"),
        }

    def _tool_routing_node(self, state: AgentGraphState) -> dict[str, Any]:
        response = self.llm_client.complete_with_tools(
            messages=state["messages"],
            tools=self._tool_specs(),
            temperature=0,
        )
        pending_tool_calls = list(response.get("tool_calls") or [])
        max_steps_reached = bool(pending_tool_calls) and state.get("tool_steps", 0) >= MAX_TOOL_STEPS
        return {
            "latest_response": response,
            "pending_tool_calls": pending_tool_calls,
            "draft_answer": response.get("content") or "",
            "max_steps_reached": max_steps_reached,
            "execution_trace": self._append_trace(state, "tool_routing"),
        }

    def _tool_execution_node(self, state: AgentGraphState) -> dict[str, Any]:
        messages = list(state["messages"])
        latest_response = state["latest_response"]
        if latest_response.get("assistant_message"):
            messages.append(latest_response["assistant_message"])

        tool_outputs = list(state.get("tool_outputs", []))
        tool_steps = state.get("tool_steps", 0)
        for tool_call in state.get("pending_tool_calls", []):
            tool_output, tool_message = self._execute_tool_call(tool_call)
            tool_outputs.append(tool_output)
            tool_steps += 1
            messages.append(tool_message)

        return {
            "messages": messages,
            "tool_outputs": tool_outputs,
            "tool_steps": tool_steps,
            "pending_tool_calls": [],
            "execution_trace": self._append_trace(state, "tool_execution"),
        }

    def _finalizer_node(self, state: AgentGraphState) -> dict[str, Any]:
        answer = self._finalize_answer(
            user_query=state["user_query"],
            execution_plan=state.get("execution_plan", {}),
            draft_answer=state.get("draft_answer", ""),
            tool_outputs=state.get("tool_outputs", []),
            has_images=state.get("has_images", False),
        )
        return {"answer": answer, "execution_trace": self._append_trace(state, "finalizer")}

    def _risk_check_node(self, state: AgentGraphState) -> dict[str, Any]:
        assessment = safety_service.analyze_answer(state.get("answer", ""), state["user_query"])
        answer = state.get("answer", "")
        if assessment.get("disclaimer"):
            answer = safety_service.append_disclaimer(answer, assessment["disclaimer"])
        return {
            "answer": answer,
            "post_safety_assessment": assessment,
            "execution_trace": self._append_trace(state, "risk_check"),
        }

    def _escalation_node(self, state: AgentGraphState) -> dict[str, Any]:
        assessment = state.get("post_safety_assessment") or {}
        trigger_reason = "；".join(
            alert["message"]
            for alert in assessment.get("alerts", [])[:3]
            if alert.get("message")
        ) or "检测到高风险医疗问答场景，建议人工复核。"
        recommended_action = (
            assessment.get("recommended_actions") or ["建议尽快人工复核，必要时线下就医。"]
        )[0]
        event = escalation_service.create_manual_escalation_event(
            self.db,
            conversation_session_id=self.execution_context.conversation_session_id,
            patient_id=self.execution_context.verified_patient_id
            or self.execution_context.resolved_patient_id,
            risk_level=assessment.get("risk_level", "high"),
            trigger_reason=trigger_reason,
            recommended_action=recommended_action,
        )
        return {
            "manual_escalation": escalation_service.serialize_manual_escalation_event(event),
            "execution_trace": self._append_trace(state, "escalation"),
        }

    def _route_after_tool_routing(self, state: AgentGraphState) -> str:
        if state.get("pending_tool_calls") and not state.get("max_steps_reached"):
            return "tool_execution"
        return "finalizer"

    def _route_after_risk_check(self, state: AgentGraphState) -> str:
        assessment = state.get("post_safety_assessment") or {}
        if safety_service.should_escalate(assessment):
            return "escalation"
        return "finish"

    def _append_trace(self, state: AgentGraphState, node_name: str) -> list[str]:
        return [*state.get("execution_trace", []), node_name]

    def _build_result(
        self,
        answer: str,
        tool_outputs: List[Dict[str, Any]],
        planner_debug: Dict[str, Any],
        execution_plan: Dict[str, Any],
        debug_planner: bool,
        post_safety_assessment: Optional[Dict[str, Any]],
        manual_escalation: Optional[Dict[str, Any]],
        execution_trace: list[str],
    ) -> Dict[str, Any]:
        """统一构造 Agent 返回结果。"""

        result = {
            "answer": answer,
            "conversation_session_id": self.execution_context.conversation_session_id,
            "tool_outputs": tool_outputs,
            "planner_debug": (
                self._build_runtime_planner_debug(
                    planner_debug=planner_debug,
                    execution_plan=execution_plan,
                    tool_outputs=tool_outputs,
                    execution_trace=execution_trace,
                )
                if debug_planner
                else None
            ),
            "resolved_patient_id": self.execution_context.resolved_patient_id,
            "verified_patient_id": self.execution_context.verified_patient_id,
            "post_safety_assessment": post_safety_assessment,
        }
        if manual_escalation is not None:
            result["manual_escalation"] = manual_escalation
        return result

    def _build_execution_plan(
        self,
        user_query: str,
        has_images: bool,
        memory_context_summary: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """多次采样生成候选计划，再做一致性合并。"""

        candidates: List[Dict[str, Any]] = []
        raw_candidates: List[Dict[str, Any]] = []
        planner_user_prompt = (
            f"用户问题：{user_query}\n"
            f"是否包含图片：{'是' if has_images else '否'}\n"
            f"相关记忆摘要：{memory_context_summary or '无'}\n"
            "请输出最小可执行计划。"
        )
        for temperature in PLAN_TEMPERATURES:
            response = self.llm_client.complete(
                messages=[
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": planner_user_prompt},
                ],
                temperature=temperature,
            )
            parsed_candidate = self._parse_plan_candidate(response["content"])
            raw_candidates.append(
                {
                    "temperature": temperature,
                    "raw_content": response["content"],
                    "parsed_plan": parsed_candidate,
                }
            )
            candidates.append(parsed_candidate)
        merged_plan = self._merge_plan_candidates(candidates, has_images=has_images)
        return merged_plan, {
            "planner_prompt": PLANNER_PROMPT,
            "planner_memory_context_summary": memory_context_summary,
            "temperatures": PLAN_TEMPERATURES,
            "candidates": raw_candidates,
            "merged_plan": merged_plan,
        }

    def _parse_plan_candidate(self, content: str) -> Dict[str, Any]:
        """尽量把模型输出解析成稳定 JSON 计划。"""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {
            "objective": "基于用户问题规划查询与回答",
            "need_identity_verification": False,
            "image_reasoning": False,
            "tool_sequence": [],
            "steps": ["解析问题", "按需调用工具", "整理结论"],
            "final_answer_focus": ["直接回答问题", "标注依据和限制"],
        }

    def _merge_plan_candidates(
        self,
        candidates: List[Dict[str, Any]],
        has_images: bool,
    ) -> Dict[str, Any]:
        """对多候选计划做简单投票与去重合并。"""

        verification_votes = sum(
            1 for candidate in candidates if candidate.get("need_identity_verification")
        )
        image_votes = sum(1 for candidate in candidates if candidate.get("image_reasoning"))
        tool_scores: Dict[str, int] = {}
        merged_steps: List[str] = []
        merged_focus: List[str] = []

        for candidate in candidates:
            for tool_name in candidate.get("tool_sequence", []):
                if tool_registry_service.get_tool_definition(tool_name) is None:
                    continue
                tool_scores[tool_name] = tool_scores.get(tool_name, 0) + 1
            for step in candidate.get("steps", []):
                if step not in merged_steps:
                    merged_steps.append(step)
            for focus in candidate.get("final_answer_focus", []):
                if focus not in merged_focus:
                    merged_focus.append(focus)

        ranked_tools = [
            tool_name
            for tool_name, _ in sorted(
                tool_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        first_candidate = candidates[0] if candidates else {}
        return {
            "objective": first_candidate.get("objective", "完成患者问题回答"),
            "need_identity_verification": verification_votes >= 2,
            "image_reasoning": has_images or image_votes >= 2,
            "tool_sequence": ranked_tools,
            "steps": merged_steps[:5] or ["解析问题", "按需调用工具", "整理结论"],
            "final_answer_focus": merged_focus[:5] or ["直接回答问题", "说明依据与限制"],
        }

    def _format_execution_plan(self, execution_plan: Dict[str, Any]) -> str:
        """把执行计划转成注入模型的紧凑文本。"""

        return (
            "内部执行计划（已做多候选一致性筛选）：\n"
            f"- 目标：{execution_plan.get('objective', '完成患者问题回答')}\n"
            f"- 是否优先验权：{'是' if execution_plan.get('need_identity_verification') else '否'}\n"
            f"- 是否结合图片：{'是' if execution_plan.get('image_reasoning') else '否'}\n"
            f"- 推荐工具顺序：{', '.join(execution_plan.get('tool_sequence', [])) or '按需决定'}\n"
            f"- 关键步骤：{'；'.join(execution_plan.get('steps', []))}\n"
            f"- 回答重点：{'；'.join(execution_plan.get('final_answer_focus', []))}"
        )

    def _finalize_answer(
        self,
        user_query: str,
        execution_plan: Dict[str, Any],
        draft_answer: str,
        tool_outputs: List[Dict[str, Any]],
        has_images: bool,
    ) -> str:
        """在收集完工具结果后，由 Finalizer 做最终收敛。"""

        response = self.llm_client.complete(
            messages=[
                {"role": "system", "content": FINALIZER_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_query": user_query,
                            "has_images": has_images,
                            "execution_plan": execution_plan,
                            "draft_answer": draft_answer,
                            "tool_outputs": tool_outputs,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
        )
        return response["content"] or draft_answer

    def _build_memory_messages(
        self,
        memory_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """把短期记忆、长期画像和关键事件注入到模型 system context。"""

        if not memory_context:
            return []

        memory_lines: List[str] = []
        short_term_memories = memory_context.get("short_term_memories", [])
        if short_term_memories:
            formatted_memories = []
            for item in short_term_memories:
                role = item.get("role", "unknown")
                content = item.get("content", "")
                multimodal_payload = item.get("multimodal_payload")
                suffix = f" [多模态摘要: {multimodal_payload}]" if multimodal_payload else ""
                formatted_memories.append(f"{role}: {content}{suffix}")
            memory_lines.append("短期记忆：\n" + "\n".join(formatted_memories))

        user_profile = memory_context.get("user_profile")
        if user_profile:
            memory_lines.append(
                "长期用户画像：\n"
                f"- 用户画像摘要：{user_profile.get('profile_summary') or '无'}\n"
                f"- 稳定偏好：{user_profile.get('stable_preferences') or '无'}\n"
                f"- 关注主题：{user_profile.get('preferred_topics') or '无'}"
            )

        relevant_events = memory_context.get("relevant_events", [])
        if relevant_events:
            formatted_events = [
                f"{event.get('event_time')}: {event.get('title')} - {event.get('summary') or ''}"
                for event in relevant_events
            ]
            memory_lines.append("相关关键事件：\n" + "\n".join(formatted_events))

        risk_alerts = memory_context.get("risk_alerts", [])
        if risk_alerts:
            formatted_alerts = [
                f"{alert.get('risk_level', 'unknown')}: {alert.get('message', '')}"
                for alert in risk_alerts
            ]
            memory_lines.append("风险提示：\n" + "\n".join(formatted_alerts))

        if not memory_lines:
            return []
        return [{"role": "system", "content": "\n\n".join(memory_lines)}]

    def _build_memory_summary(self, memory_context: Optional[Dict[str, Any]]) -> str:
        """给 Planner 使用的紧凑记忆摘要。"""

        if not memory_context:
            return ""

        parts: list[str] = []
        short_term_memories = memory_context.get("short_term_memories", [])
        if short_term_memories:
            recent_turns = []
            for item in short_term_memories[-3:]:
                role = item.get("role", "unknown")
                content = (item.get("content", "") or "").strip()
                if content:
                    recent_turns.append(f"{role}:{content[:60]}")
            if recent_turns:
                parts.append("最近对话=" + " | ".join(recent_turns))

        user_profile = memory_context.get("user_profile")
        if user_profile:
            profile_summary = (user_profile.get("profile_summary") or "").strip()
            if profile_summary:
                parts.append(f"长期画像={profile_summary[:120]}")

        relevant_events = memory_context.get("relevant_events", [])
        if relevant_events:
            event_titles = [
                event.get("title", "").strip()
                for event in relevant_events[:3]
                if event.get("title")
            ]
            if event_titles:
                parts.append("相关事件=" + "；".join(event_titles))

        risk_alerts = memory_context.get("risk_alerts", [])
        if risk_alerts:
            risk_summaries = [
                f"{alert.get('risk_level')}:{alert.get('message', '')[:50]}"
                for alert in risk_alerts[:2]
            ]
            if risk_summaries:
                parts.append("风险提示=" + " | ".join(risk_summaries))

        return "；".join(parts)

    def _build_runtime_planner_debug(
        self,
        planner_debug: Dict[str, Any],
        execution_plan: Dict[str, Any],
        tool_outputs: List[Dict[str, Any]],
        execution_trace: list[str],
    ) -> Dict[str, Any]:
        """构造调试用 planner 输出，并补充 LangGraph 轨迹。"""

        return {
            **planner_debug,
            "workflow_framework": "langgraph",
            "graph_trace": execution_trace,
            "execution_plan_prompt": self._format_execution_plan(execution_plan),
            "conversation_session_id": self.execution_context.conversation_session_id,
            "resolved_patient_id": self.execution_context.resolved_patient_id,
            "verified_patient_id": self.execution_context.verified_patient_id,
            "executed_tools": [
                {
                    "tool_name": item["tool_name"],
                    "arguments": item["arguments"],
                    "access_granted": item["access_granted"],
                    "denial_reason": item.get("denial_reason"),
                }
                for item in tool_outputs
            ],
        }

    def _build_user_content(
        self,
        user_query: str,
        images: List[Dict[str, Any]],
    ) -> Any:
        """把文本与图片拼成模型可消费的多模态消息。"""

        if not images:
            return user_query

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_query}]
        for image in images:
            image_url = image.get("image_url")
            image_base64 = image.get("image_base64")
            mime_type = image.get("mime_type", "image/png")
            if image_url and not image_base64:
                image_base64, mime_type = self._try_load_local_image(image_url, mime_type)
            if image_base64:
                image_url = f"data:{mime_type};base64,{image_base64}"
            if image_url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )
        return content

    def _try_load_local_image(
        self,
        image_url: str,
        mime_type: str,
    ) -> tuple[Optional[str], str]:
        """兼容传入本地文件路径的测试和联调场景。"""

        image_path = Path(image_url).expanduser()
        if not image_path.is_file():
            return None, mime_type

        detected_mime_type, _ = mimetypes.guess_type(image_path.name)
        with image_path.open("rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded, detected_mime_type or mime_type

    def _build_tool_registry(self) -> Dict[str, ToolHandler]:
        """把工具名映射到具体业务处理函数。"""

        return {
            "verify_patient_identity": lambda **kwargs: mcp_tool_service.verify_patient(
                self.db, **kwargs
            ),
            "get_patient_profile": lambda **kwargs: mcp_tool_service.get_patient_profile(
                self.db, **kwargs
            ),
            "get_patient_medical_cases": lambda **kwargs: mcp_tool_service.get_patient_medical_cases(
                self.db, **kwargs
            ),
            "get_patient_visit_records": lambda **kwargs: mcp_tool_service.get_patient_visit_records(
                self.db, **kwargs
            ),
            "get_patient_medical_reports": lambda **kwargs: mcp_tool_service.get_patient_medical_reports(
                self.db, **kwargs
            ),
            "search_knowledge_base": lambda **kwargs: mcp_tool_service.search_knowledge_base(
                self.db, **kwargs
            ),
            "create_manual_escalation": lambda **kwargs: mcp_tool_service.create_manual_escalation(
                self.db, **kwargs
            ),
        }

    def _tool_specs(self) -> List[Dict[str, Any]]:
        """读取统一工具 schema，供 OpenAI tools 调用。"""

        return tool_registry_service.build_openai_tool_specs()

    def _execute_tool_call(
        self,
        tool_call: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """执行单次工具调用，并统一做鉴权和审计。"""

        tool_name = tool_call["name"]
        arguments = dict(tool_call["arguments"])
        definition = tool_registry_service.get_tool_definition(tool_name)
        handler = self.tools.get(tool_name)
        if definition is None or handler is None:
            result = {
                "found": False,
                "access_granted": False,
                "reason": "unsupported tool",
                "tool_name": tool_name,
            }
            return self._tool_output_payload(tool_name, arguments, result, False, "unsupported tool"), {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            }

        arguments, target_patient = self._normalize_tool_arguments(tool_name, arguments)
        denial_reason: Optional[str] = None

        if definition.policy.require_verified_identity:
            if target_patient is None:
                denial_reason = "patient target is required before reading sensitive data"
            elif self.execution_context.verified_patient_id is None:
                denial_reason = "identity verification required"
            elif target_patient.id != self.execution_context.verified_patient_id:
                denial_reason = "verified identity does not match target patient"

        if denial_reason is not None:
            result = mcp_tool_service.build_verification_required_response(
                tool_name=tool_name,
                patient=target_patient,
            )
            tool_audit_service.create_tool_audit_log(
                db=self.db,
                conversation_session_id=self.execution_context.conversation_session_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                access_granted=False,
                denial_reason=denial_reason,
                patient_id=target_patient.id if target_patient else None,
            )
            output = self._tool_output_payload(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                access_granted=False,
                denial_reason=denial_reason,
            )
            return output, {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": json.dumps(result, ensure_ascii=False),
            }

        result = handler(**arguments)
        access_granted = True
        if tool_name == "verify_patient_identity":
            access_granted = bool(result.get("verified"))
            denial_reason = None if access_granted else result.get("reason")
            patient_data = result.get("patient", {})
            patient_id = patient_data.get("id")
            if access_granted and patient_id is not None:
                self.execution_context.verified_patient_id = patient_id
                self.execution_context.resolved_patient_id = patient_id
        elif target_patient is not None:
            self.execution_context.resolved_patient_id = target_patient.id

        audit_patient_id = (
            target_patient.id
            if target_patient is not None
            else result.get("patient", {}).get("id")
        )
        tool_audit_service.create_tool_audit_log(
            db=self.db,
            conversation_session_id=self.execution_context.conversation_session_id,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            access_granted=access_granted,
            denial_reason=denial_reason,
            patient_id=audit_patient_id,
        )
        output = self._tool_output_payload(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            access_granted=access_granted,
            denial_reason=denial_reason,
        )
        return output, {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        }

    def _normalize_tool_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Any]:
        """补全工具参数，并解析目标患者实体。"""

        if tool_name == "verify_patient_identity":
            patient = mcp_tool_service.resolve_patient_from_identity(
                self.db,
                patient_code=arguments.get("patient_code"),
            )
            return arguments, patient

        if tool_name == "create_manual_escalation":
            arguments.setdefault(
                "conversation_session_id",
                self.execution_context.conversation_session_id,
            )
            arguments.setdefault(
                "patient_id",
                self.execution_context.verified_patient_id
                or self.execution_context.resolved_patient_id,
            )
            return arguments, None

        patient = mcp_tool_service.resolve_patient_from_identity(
            self.db,
            patient_id=arguments.get("patient_id"),
            patient_code=arguments.get("patient_code"),
        )
        if patient is None and self.execution_context.verified_patient_id is not None:
            arguments.setdefault("patient_id", self.execution_context.verified_patient_id)
            patient = mcp_tool_service.resolve_patient_from_identity(
                self.db,
                patient_id=arguments.get("patient_id"),
            )
        elif patient is None and self.execution_context.resolved_patient_id is not None:
            arguments.setdefault("patient_id", self.execution_context.resolved_patient_id)
            patient = mcp_tool_service.resolve_patient_from_identity(
                self.db,
                patient_id=arguments.get("patient_id"),
            )
        return arguments, patient

    def _tool_output_payload(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        access_granted: bool,
        denial_reason: Optional[str],
    ) -> Dict[str, Any]:
        """把工具执行结果转换成稳定返回结构。"""

        return {
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "access_granted": access_granted,
            "denial_reason": denial_reason,
        }
