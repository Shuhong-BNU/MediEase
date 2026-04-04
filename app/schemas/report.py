"""
报告域 Schema。
职责概览：
1. 约束检验检查报告的 CRUD 输入输出。
2. 定义报告解读接口返回的摘要、异常项、风险等级和免责声明结构。
3. 供 API 层、服务层和前端报告页面共享稳定的数据契约。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class MedicalReportBase(BaseModel):
    """报告对象的公共字段。"""

    patient_id: int
    report_code: str
    report_type: str
    title: str
    department: Optional[str] = None
    report_time: Optional[datetime] = None
    raw_text: Optional[str] = None
    structured_data_json: Optional[str] = None
    abnormal_summary: Optional[str] = None
    source_type: str = "manual"


class MedicalReportCreate(MedicalReportBase):
    """创建报告时使用的输入结构。"""


class MedicalReportUpdate(BaseModel):
    """更新报告时允许变更的字段。"""

    report_type: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    report_time: Optional[datetime] = None
    raw_text: Optional[str] = None
    structured_data_json: Optional[str] = None
    abnormal_summary: Optional[str] = None
    source_type: Optional[str] = None


class MedicalReportRead(MedicalReportBase):
    """报告读取结构。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime


class ReportAbnormalItem(BaseModel):
    """单条异常指标或风险提示项。"""

    label: str
    value: Optional[str] = None
    explanation: str


class ReportInterpretRequest(BaseModel):
    """报告解读请求，可来自已入库报告、文本粘贴或 OCR 结果。"""

    report_id: Optional[int] = None
    patient_id: Optional[int] = None
    report_text: Optional[str] = None
    title: Optional[str] = None
    report_type: Optional[str] = None
    image_text: Optional[str] = None


class ReportInterpretResponse(BaseModel):
    """报告解读结果。"""

    summary: str
    abnormal_items: list[ReportAbnormalItem]
    risk_level: str
    recommended_actions: list[str]
    disclaimer: str
    extracted_text: Optional[str] = None


class ReportSearchRequest(BaseModel):
    """按患者和关键词筛选报告。"""

    patient_id: Optional[int] = None
    patient_code: Optional[str] = None
    query: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
