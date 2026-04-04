"""
Qwen 语音合成客户端。

职责概览：
- 封装 DashScope TTS WebSocket 调用。
- 将文本回答转换为 base64 编码的 MP3 音频，供 API 层落盘后返回。

与其他模块的关系：
- `app.api.routes` 在用户开启语音播报时调用本模块。
- 本模块只负责调用与结果标准化，不参与文件保存。

关键技术点：
- 采用延迟导入，避免未安装 `dashscope` 时阻塞非语音路径。
- 将外部 SDK 的返回统一为稳定字典结构，方便测试和前端消费。
"""

import base64
import os
from typing import Dict, Optional


DEFAULT_SPEECH_MODEL = "cosyvoice-v3-flash"
DEFAULT_SPEECH_VOICE = "longanyang"
DEFAULT_WEBSOCKET_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"


class QwenSpeechClient:
    """对 DashScope 语音合成接口做最小可用封装。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        websocket_url: Optional[str] = None,
    ) -> None:
        try:
            import dashscope
            from dashscope.audio.tts_v2 import SpeechSynthesizer
            from dashscope.audio.tts_v2.speech_synthesizer import AudioFormat
        except ImportError as exc:
            raise ValueError(
                "dashscope package is required for speech synthesis. "
                "Run: pip install dashscope"
            ) from exc

        resolved_api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError("DASHSCOPE_API_KEY or QWEN_API_KEY is not configured")

        dashscope.api_key = resolved_api_key
        dashscope.base_websocket_api_url = (
            websocket_url
            or os.getenv("DASHSCOPE_WEBSOCKET_URL")
            or DEFAULT_WEBSOCKET_URL
        )

        self.model = model or os.getenv("QWEN_TTS_MODEL", DEFAULT_SPEECH_MODEL)
        self.voice = os.getenv("QWEN_TTS_VOICE", DEFAULT_SPEECH_VOICE)
        self.websocket_url = (
            websocket_url
            or os.getenv("DASHSCOPE_WEBSOCKET_URL")
            or DEFAULT_WEBSOCKET_URL
        )
        self._speech_synthesizer_cls = SpeechSynthesizer
        self._audio_format_cls = AudioFormat

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_SPEECH_VOICE,
        audio_format: str = "mp3",
    ) -> Dict[str, str]:
        """将文本转成 MP3，并返回可直接落盘的 base64 结果。"""

        normalized_format = audio_format.lower()
        if normalized_format != "mp3":
            raise ValueError("Only mp3 speech_format is currently supported")

        synthesizer = self._speech_synthesizer_cls(
            model=self.model,
            voice=voice or self.voice,
            format=self._audio_format_cls.MP3_22050HZ_MONO_256KBPS,
        )
        audio_bytes = synthesizer.call(text)
        if not audio_bytes:
            response = None
            try:
                response = synthesizer.get_response()
            except Exception:
                response = None
            raise ValueError(
                "Speech synthesis returned empty audio: "
                f"model={self.model}, voice={voice or self.voice}, "
                f"websocket_url={self.websocket_url}, response={response}"
            )

        return {
            "audio_base64": base64.b64encode(audio_bytes).decode("utf-8"),
            "mime_type": "audio/mp3",
            "model": self.model,
            "voice": voice or self.voice,
        }
