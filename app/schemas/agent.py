"""
Agent 查询接口的请求与响应模型。
职责概览：
1. 定义文本、图片、语音等多模态输入以及稳定会话 ID。
2. 定义工具输出、引用来源、风险告警、人工升级和语音结果等统一响应结构。
3. 作为 `/api/agent/query`、前端页面和自动化测试之间的共享契约。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class AgentImageInput(BaseModel):
    """用户上传的单张图片输入，支持 URL 或 Base64 二选一。"""

    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    mime_type: str = "image/png"

    @model_validator(mode="after")
    def validate_source(self) -> "AgentImageInput":
        """保证每张图片至少提供一种可消费的来源。"""

        if not self.image_url and not self.image_base64:
            raise ValueError("image_url or image_base64 is required")
        return self


class AgentQueryRequest(BaseModel):
    """主问答接口入参。"""

    query: str = ""
    images: List[AgentImageInput] = Field(default_factory=list)
    conversation_session_id: Optional[str] = None
    speech_input_text: Optional[str] = None
    speech_input_base64: Optional[str] = None
    speech_input_mime_type: Optional[str] = None
    debug_planner: bool = False
    enable_speech: bool = False
    speech_voice: str = "longanyang"
    speech_format: str = "mp3"

    @model_validator(mode="after")
    def validate_any_input(self) -> "AgentQueryRequest":
        """文本、图片、语音文件至少提供一种，避免空请求进入主链路。"""

        has_text = bool(self.query.strip())
        has_speech_text = bool((self.speech_input_text or "").strip())
        has_speech_file = bool((self.speech_input_base64 or "").strip())
        if has_text or has_speech_text or has_speech_file or self.images:
            return self
        raise ValueError(
            "query, speech_input_text, speech_input_base64 or images is required"
        )


class AgentToolOutput(BaseModel):
    """单次工具调用的稳定输出结构，兼容前端展示和测试断言。"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]
    access_granted: bool = True
    denial_reason: Optional[str] = None


class AgentCitation(BaseModel):
    """回答引用来源。"""

    source_type: str
    title: str
    snippet: str
    source_id: Optional[str] = None
    source_url: Optional[str] = None


class AgentRiskAlert(BaseModel):
    """风控告警。"""

    risk_level: str
    message: str
    trigger_stage: str


class AgentQueryResponse(BaseModel):
    """主问答接口返回体。"""

    answer: str
    conversation_session_id: str
    tool_outputs: List[AgentToolOutput]
    citations: List[AgentCitation] = Field(default_factory=list)
    risk_alerts: List[AgentRiskAlert] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    manual_escalation: Optional[Dict[str, Any]] = None
    disclaimer: Optional[str] = None
    ocr_text: Optional[str] = None
    planner_debug: Optional[Dict[str, Any]] = None
    speech_mime_type: Optional[str] = None
    speech_model: Optional[str] = None
    speech_voice: Optional[str] = None
    speech_file_path: Optional[str] = None
    speech_download_url: Optional[str] = None
