"""
Qwen / DashScope OpenAI-compatible 客户端封装。

本模块把文本、多模态和 tools 调用统一收口到一个轻量客户端里。
为了提升环境韧性，这里采用延迟导入 OpenAI SDK 的方式：只有真正实例化客户端时才要求依赖存在。
"""

import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-plus-latest"


class QwenClient:
    """Qwen 模型的最小封装，供 Agent 和 Planner 共用。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError(
                "openai package is required for QwenClient. Run: pip install openai"
            ) from exc

        resolved_api_key = api_key or os.getenv("QWEN_API_KEY")
        if not resolved_api_key:
            raise ValueError("QWEN_API_KEY is not configured")

        self.model = model or os.getenv("QWEN_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=base_url or os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL),
        )

    def complete_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0,
    ) -> Dict[str, Any]:
        """执行一次带工具定义的 chat completion。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
        )
        message = response.choices[0].message
        return {
            "content": message.content,
            "assistant_message": {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in (message.tool_calls or [])
                ],
            },
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": json.loads(call.function.arguments),
                }
                for call in (message.tool_calls or [])
            ],
            "raw_response": response.model_dump(),
        }

    def complete(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0,
    ) -> Dict[str, Any]:
        """执行一次普通 chat completion。"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        message = response.choices[0].message
        return {
            "content": message.content or "",
            "raw_response": response.model_dump(),
        }
