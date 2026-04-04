"""
工具调用审计服务。

所有受控工具调用都会在这里落库，便于：
1. 追踪是否已验权。
2. 追踪读取了哪位患者的数据。
3. 追踪工具入参与返回摘要。
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import ToolAuditLog


def create_tool_audit_log(
    db: Session,
    conversation_session_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
    access_granted: bool,
    denial_reason: str | None = None,
    patient_id: int | None = None,
) -> ToolAuditLog:
    """写入一条工具调用审计日志。"""

    log = ToolAuditLog(
        conversation_session_id=conversation_session_id,
        patient_id=patient_id,
        tool_name=tool_name,
        arguments_json=json.dumps(arguments, ensure_ascii=False),
        result_summary_json=json.dumps(result, ensure_ascii=False),
        access_granted=access_granted,
        denial_reason=denial_reason,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log

