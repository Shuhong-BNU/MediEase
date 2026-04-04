"""
独立运行的 MCP server 入口。

本文件把患者身份验证、患者资料、病例、就诊记录等能力通过 FastMCP 暴露出来。
它与主 Agent 链路共享同一批业务处理函数和工具元数据，便于后续扩展到独立工具进程模式。
"""

from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "The MCP server requires the `mcp` package and Python 3.10+. "
        "Your current environment cannot import it."
    ) from exc

from app.db.session import SessionLocal
from app.services import mcp_tool_service


mcp = FastMCP("patient-agent-mcp-server")


@mcp.tool()
def verify_patient_identity(
    patient_code: str,
    phone: Optional[str] = None,
    id_number: Optional[str] = None,
) -> dict:
    """校验患者身份，可使用 patient_code + phone 或 patient_code + id_number。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.verify_patient(
            db,
            patient_code=patient_code,
            phone=phone,
            id_number=id_number,
        )
    finally:
        db.close()


@mcp.tool()
def get_patient_profile(
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
) -> dict:
    """查询患者基础信息；独立调试时默认返回安全可展示字段。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.get_patient_profile(
            db,
            patient_id=patient_id,
            patient_code=patient_code,
        )
    finally:
        db.close()


@mcp.tool()
def get_patient_medical_cases(
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
) -> dict:
    """查询患者病例信息。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.get_patient_medical_cases(
            db,
            patient_id=patient_id,
            patient_code=patient_code,
        )
    finally:
        db.close()


@mcp.tool()
def get_patient_visit_records(
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """查询患者就诊记录；只查最近一次时传 `limit=1`。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.get_patient_visit_records(
            db,
            patient_id=patient_id,
            patient_code=patient_code,
            limit=limit,
        )
    finally:
        db.close()


@mcp.tool()
def get_patient_medical_reports(
    patient_id: Optional[int] = None,
    patient_code: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """查询患者检验检查报告。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.get_patient_medical_reports(
            db,
            patient_id=patient_id,
            patient_code=patient_code,
            limit=limit,
        )
    finally:
        db.close()


@mcp.tool()
def search_knowledge_base(
    query: str,
    tenant_code: Optional[str] = None,
    category: Optional[str] = None,
    top_n: int = 5,
) -> dict:
    """搜索知识库文档。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.search_knowledge_base(
            db,
            query=query,
            tenant_code=tenant_code,
            category=category,
            top_n=top_n,
        )
    finally:
        db.close()


@mcp.tool()
def create_manual_escalation(
    conversation_session_id: Optional[str] = None,
    patient_id: Optional[int] = None,
    risk_level: str = "medium",
    trigger_reason: str = "",
    recommended_action: str = "",
) -> dict:
    """创建转人工建议事件。"""

    db = SessionLocal()
    try:
        return mcp_tool_service.create_manual_escalation(
            db,
            conversation_session_id=conversation_session_id,
            patient_id=patient_id,
            risk_level=risk_level,
            trigger_reason=trigger_reason,
            recommended_action=recommended_action,
        )
    finally:
        db.close()


if __name__ == "__main__":
    mcp.run()
