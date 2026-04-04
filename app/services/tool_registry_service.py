"""
统一工具元数据注册表。
职责概览：
1. 维护工具名称、描述、参数 schema 和权限策略。
2. 让 Agent 主链路与独立 MCP Server 共用同一套工具定义。
3. 把“哪些工具需要先验权”收口到统一元数据层。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolPolicy:
    """工具访问策略。"""

    require_verified_identity: bool = False


@dataclass(frozen=True)
class ToolDefinition:
    """单个工具的统一定义。"""

    name: str
    description: str
    parameters: dict[str, Any]
    policy: ToolPolicy


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "verify_patient_identity": ToolDefinition(
        name="verify_patient_identity",
        description="校验患者身份，可通过 patient_code 搭配 phone 或 id_number 验证。",
        parameters={
            "type": "object",
            "properties": {
                "patient_code": {"type": "string"},
                "phone": {"type": "string"},
                "id_number": {"type": "string"},
            },
            "required": ["patient_code"],
        },
        policy=ToolPolicy(require_verified_identity=False),
    ),
    "get_patient_profile": ToolDefinition(
        name="get_patient_profile",
        description="获取患者基础身份信息。读取隐私数据前需要已完成身份验证。",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "patient_code": {"type": "string"},
            },
        },
        policy=ToolPolicy(require_verified_identity=True),
    ),
    "get_patient_medical_cases": ToolDefinition(
        name="get_patient_medical_cases",
        description="查询患者病例信息。读取隐私数据前需要已完成身份验证。",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "patient_code": {"type": "string"},
            },
        },
        policy=ToolPolicy(require_verified_identity=True),
    ),
    "get_patient_visit_records": ToolDefinition(
        name="get_patient_visit_records",
        description="查询患者就诊记录。读取隐私数据前需要已完成身份验证。",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "patient_code": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "description": "返回记录条数。查询最近一次时使用 1。",
                },
            },
        },
        policy=ToolPolicy(require_verified_identity=True),
    ),
    "get_patient_medical_reports": ToolDefinition(
        name="get_patient_medical_reports",
        description="查询患者的检验检查报告。读取隐私数据前需要已完成身份验证。",
        parameters={
            "type": "object",
            "properties": {
                "patient_id": {"type": "integer"},
                "patient_code": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
        policy=ToolPolicy(require_verified_identity=True),
    ),
    "search_knowledge_base": ToolDefinition(
        name="search_knowledge_base",
        description="搜索医院知识库或通用医疗说明文档，用于提供非隐私类解释依据。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tenant_code": {"type": "string"},
                "category": {"type": "string"},
                "top_n": {"type": "integer"},
            },
            "required": ["query"],
        },
        policy=ToolPolicy(require_verified_identity=False),
    ),
    "create_manual_escalation": ToolDefinition(
        name="create_manual_escalation",
        description="当问题属于高风险或需要人工接管时，创建转人工建议事件。",
        parameters={
            "type": "object",
            "properties": {
                "conversation_session_id": {"type": "string"},
                "patient_id": {"type": "integer"},
                "risk_level": {"type": "string"},
                "trigger_reason": {"type": "string"},
                "recommended_action": {"type": "string"},
            },
        },
        policy=ToolPolicy(require_verified_identity=False),
    ),
}


def get_tool_definition(name: str) -> ToolDefinition | None:
    """按工具名读取统一定义。"""

    return TOOL_DEFINITIONS.get(name)


def build_openai_tool_specs() -> list[dict[str, Any]]:
    """转换为 OpenAI-compatible tools 所需的 schema。"""

    return [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.parameters,
            },
        }
        for definition in TOOL_DEFINITIONS.values()
    ]
