"""
报告服务。
职责概览：
1. 管理检验检查报告的增删改查。
2. 提供规则驱动的报告解读能力，产出摘要、异常项、风险等级和建议动作。
3. 作为 Agent 报告工具、独立报告页面和后续 OCR 流程的共用业务层。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db.models import MedicalReport
from app.schemas.report import (
    MedicalReportCreate,
    MedicalReportUpdate,
    ReportAbnormalItem,
    ReportInterpretResponse,
)


ABNORMAL_RULES = [
    ("高", "结果偏高，建议结合参考范围与既往检查结果判断变化趋势。"),
    ("偏高", "结果偏高，建议结合参考范围与既往检查结果判断变化趋势。"),
    ("低", "结果偏低，建议结合症状、饮食和既往病史综合判断。"),
    ("偏低", "结果偏低，建议结合症状、饮食和既往病史综合判断。"),
    ("异常", "报告提示异常，建议尽快结合临床症状咨询医生。"),
    ("阳性", "结果提示阳性，需要结合具体项目进一步确认临床意义。"),
]

HIGH_RISK_KEYWORDS = ["急性", "严重", "危急", "胸痛", "呼吸困难", "出血", "梗死"]
MEDIUM_RISK_KEYWORDS = ["异常", "升高", "降低", "阳性", "结节", "积液"]


def create_medical_report(db: Session, payload: MedicalReportCreate) -> MedicalReport:
    """创建一条报告记录。"""

    data = payload.model_dump()
    if data.get("report_time") is None:
        data["report_time"] = datetime.utcnow()
    report = MedicalReport(**data)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def list_medical_reports(
    db: Session,
    patient_id: Optional[int] = None,
    query: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[MedicalReport]:
    """按患者和关键词筛选报告。"""

    stmt = select(MedicalReport).order_by(
        MedicalReport.report_time.desc(),
        MedicalReport.id.desc(),
    )
    if patient_id is not None:
        stmt = stmt.where(MedicalReport.patient_id == patient_id)
    if query:
        like_query = f"%{query}%"
        stmt = stmt.where(
            or_(
                MedicalReport.title.like(like_query),
                MedicalReport.raw_text.like(like_query),
                MedicalReport.abnormal_summary.like(like_query),
                MedicalReport.report_type.like(like_query),
            )
        )
    if limit is not None and limit > 0:
        stmt = stmt.limit(limit)
    return list(db.scalars(stmt).all())


def get_medical_report(db: Session, report_id: int) -> Optional[MedicalReport]:
    """按主键读取报告。"""

    return db.get(MedicalReport, report_id)


def update_medical_report(
    db: Session,
    report: MedicalReport,
    payload: MedicalReportUpdate,
) -> MedicalReport:
    """更新报告并返回最新结果。"""

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(report, field, value)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def interpret_report_text(
    report_text: str,
    title: Optional[str] = None,
    report_type: Optional[str] = None,
    extracted_text: Optional[str] = None,
) -> ReportInterpretResponse:
    """对报告文本做轻量规则解读。"""

    clean_text = (report_text or "").strip()
    if not clean_text:
        clean_text = "未提供足够的报告文本，暂时无法生成可靠解读。"

    abnormal_items = _extract_abnormal_items(clean_text)
    risk_level = _infer_risk_level(clean_text, abnormal_items)
    summary = _build_summary(clean_text, title=title, report_type=report_type)
    recommended_actions = _build_recommended_actions(risk_level, abnormal_items)
    disclaimer = (
        "以下内容仅用于帮助理解报告，不替代医生诊断；如有胸痛、呼吸困难、持续高热、明显出血等情况，请及时线下就医。"
    )
    return ReportInterpretResponse(
        summary=summary,
        abnormal_items=abnormal_items,
        risk_level=risk_level,
        recommended_actions=recommended_actions,
        disclaimer=disclaimer,
        extracted_text=extracted_text,
    )


def interpret_report_record(report: MedicalReport) -> ReportInterpretResponse:
    """基于已入库报告生成解读。"""

    return interpret_report_text(
        report_text=report.raw_text or report.abnormal_summary or "",
        title=report.title,
        report_type=report.report_type,
    )


def serialize_medical_report(report: MedicalReport) -> dict:
    """把 ORM 报告对象转成稳定字典，供工具调用与接口复用。"""

    return {
        "id": report.id,
        "patient_id": report.patient_id,
        "report_code": report.report_code,
        "report_type": report.report_type,
        "title": report.title,
        "department": report.department,
        "report_time": report.report_time.isoformat() if report.report_time else None,
        "raw_text": report.raw_text,
        "structured_data_json": report.structured_data_json,
        "abnormal_summary": report.abnormal_summary,
        "source_type": report.source_type,
    }


def _extract_abnormal_items(report_text: str) -> list[ReportAbnormalItem]:
    """扫描常见异常描述，并尽量附上所在行上下文。"""

    items: list[ReportAbnormalItem] = []
    for line in [item.strip() for item in report_text.splitlines() if item.strip()]:
        for keyword, explanation in ABNORMAL_RULES:
            if keyword in line:
                items.append(
                    ReportAbnormalItem(
                        label=_extract_label(line),
                        value=_extract_value(line),
                        explanation=explanation,
                    )
                )
                break
    return items[:8]


def _extract_label(line: str) -> str:
    """从报告行里抽取项目名，抽不到时退回原行摘要。"""

    if ":" in line:
        return line.split(":", 1)[0].strip()
    if "：" in line:
        return line.split("：", 1)[0].strip()
    return line[:24]


def _extract_value(line: str) -> Optional[str]:
    """提取指标值或异常判断。"""

    match = re.search(r"([-+]?\d+(?:\.\d+)?\s*[A-Za-z%/uULmg]*)", line)
    if match:
        return match.group(1).strip()
    for keyword, _ in ABNORMAL_RULES:
        if keyword in line:
            return keyword
    return None


def _infer_risk_level(
    report_text: str,
    abnormal_items: list[ReportAbnormalItem],
) -> str:
    """根据关键词和异常数量估算风险等级。"""

    if any(keyword in report_text for keyword in HIGH_RISK_KEYWORDS):
        return "high"
    if len(abnormal_items) >= 3 or any(
        keyword in report_text for keyword in MEDIUM_RISK_KEYWORDS
    ):
        return "medium"
    return "low"


def _build_summary(
    report_text: str,
    title: Optional[str],
    report_type: Optional[str],
) -> str:
    """生成面向患者的摘要。"""

    head = f"{title or '该份报告'}"
    if report_type:
        head += f"（{report_type}）"
    lines = [line.strip() for line in report_text.splitlines() if line.strip()]
    if not lines:
        return f"{head}当前只有少量文本，建议补充完整报告内容后再解读。"
    preview = "；".join(lines[:3])
    return f"{head}的重点信息主要集中在：{preview}。"


def _build_recommended_actions(
    risk_level: str,
    abnormal_items: list[ReportAbnormalItem],
) -> list[str]:
    """根据风险等级产出建议动作。"""

    actions = ["结合既往检查结果和当前症状一起看，不要只看单次指标。"]
    if abnormal_items:
        actions.append("把异常项目和参考范围整理给医生，有助于更快判断。")
    if risk_level == "high":
        actions.append("如果伴随明显不适或急症表现，请尽快线下就医或急诊评估。")
    elif risk_level == "medium":
        actions.append("建议尽快预约门诊复查，确认异常是否持续存在。")
    else:
        actions.append("如无明显不适，可按医生建议定期复查并持续观察。")
    return actions


def structured_data_to_json(data: Optional[dict]) -> Optional[str]:
    """便于外部调用方统一序列化结构化报告内容。"""

    if not data:
        return None
    return json.dumps(data, ensure_ascii=False)
