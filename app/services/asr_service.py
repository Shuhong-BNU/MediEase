"""
语音输入辅助服务。
职责概览：
1. 为 `/api/agent/query` 提供语音输入转文本的统一入口。
2. 当前优先复用前端或调用方提供的 `speech_input_text`，无转写能力时优雅降级。
3. 为后续接入真实 ASR 模型或云服务保留单点扩展位。
"""

from __future__ import annotations

from typing import Optional

from app.llm.qwen_client import QwenClient


def resolve_speech_input_text(
    llm_client: QwenClient,
    speech_input_text: Optional[str] = None,
    speech_input_base64: Optional[str] = None,
    speech_input_mime_type: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """解析语音输入；当前优先使用显式文本，其余情况降级返回空串和说明。"""

    if speech_input_text and speech_input_text.strip():
        return speech_input_text.strip(), None
    if speech_input_base64:
        # 当前原型未强依赖真实 ASR，先返回可解释的降级提示，避免主链路报错。
        return "", "当前环境未接入实时语音识别服务，已跳过语音文件转写。"
    return "", None
