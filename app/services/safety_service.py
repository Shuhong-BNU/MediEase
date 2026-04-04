"""
风控与安全检查服务。
职责概览：
1. 在回答前后识别急症、停药、加药、确诊型高风险问法。
2. 生成统一的风险告警、建议动作和免责声明。
3. 为人工升级判断提供规则化输入。
"""

from __future__ import annotations

from typing import Any, Dict, List


HIGH_RISK_PATTERNS = {
    "急症症状": ["胸痛", "呼吸困难", "抽搐", "昏迷", "大出血", "严重过敏"],
    "停药决策": ["停药", "停掉", "不用吃了", "可以不吃"],
    "加药决策": ["加药", "加量", "自己加", "双倍吃"],
    "确诊型问法": ["是不是", "能不能确诊", "就是这个病吗", "是不是癌"],
}


DISCLAIMER = (
    "本系统提供的是辅助解释与信息整理，不替代医生面诊、诊断和处方。"
)


def analyze_user_query(query: str) -> dict[str, Any]:
    """对用户输入做前置风险分析。"""

    alerts = _collect_alerts(query, trigger_stage="pre")
    return _build_assessment(alerts)


def analyze_answer(answer: str, query: str) -> dict[str, Any]:
    """对最终回答做后置风险分析，避免给出过度确定的医疗建议。"""

    alerts = _collect_alerts(query, trigger_stage="post")
    if any(token in answer for token in ["立即停药", "可以自行加药", "基本可以确诊"]):
        alerts.append(
            {
                "risk_level": "high",
                "message": "回答中出现了可能过度确定的治疗或诊断表达，已追加保守提示。",
                "trigger_stage": "post",
            }
        )
    return _build_assessment(alerts)


def should_escalate(assessment: dict[str, Any]) -> bool:
    """高风险场景建议生成人工升级事件。"""

    return assessment["risk_level"] == "high"


def append_disclaimer(answer: str, disclaimer: str | None = None) -> str:
    """在主回答后附加统一免责声明。"""

    tail = disclaimer or DISCLAIMER
    if tail in answer:
        return answer
    return f"{answer}\n\n提示：{tail}"


def _collect_alerts(text: str, trigger_stage: str) -> List[Dict[str, str]]:
    """按规则聚合命中告警。"""

    alerts: list[dict[str, str]] = []
    for label, patterns in HIGH_RISK_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            risk_level = "high" if label != "确诊型问法" else "medium"
            alerts.append(
                {
                    "risk_level": risk_level,
                    "message": f"检测到{label}相关表达，回答需要保持保守并提示及时线下评估。",
                    "trigger_stage": trigger_stage,
                }
            )
    return alerts


def _build_assessment(alerts: List[Dict[str, str]]) -> dict[str, Any]:
    """归并告警并生成建议动作。"""

    risk_level = "low"
    if any(alert["risk_level"] == "high" for alert in alerts):
        risk_level = "high"
    elif any(alert["risk_level"] == "medium" for alert in alerts):
        risk_level = "medium"

    recommended_actions = [
        "需要时结合既往病历、检查报告和线下医生意见一起判断。"
    ]
    if risk_level == "high":
        recommended_actions.append("建议尽快线下就医，必要时直接前往急诊或联系人工客服。")
    elif risk_level == "medium":
        recommended_actions.append("建议尽快门诊复诊，不要仅凭单次在线回答自行调整治疗。")

    return {
        "risk_level": risk_level,
        "alerts": alerts,
        "recommended_actions": recommended_actions,
        "disclaimer": DISCLAIMER if risk_level != "low" else None,
    }
