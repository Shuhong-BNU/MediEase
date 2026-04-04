"""
OCR 辅助服务。
职责概览：
1. 对上传的报告图片做“尽力而为”的文本提取。
2. 优先复用现有 Qwen 多模态能力，失败时优雅降级。
3. 让报告图片也能进入报告解读与 Agent 主链路。
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.llm.qwen_client import QwenClient


OCR_PROMPT = (
    "请从图片中尽可能提取与医学报告相关的文字内容，只输出提取出的纯文本，不要解释。"
)


def extract_text_from_images(
    llm_client: QwenClient,
    images: List[Dict[str, Any]],
) -> str:
    """调用多模态模型提取图片中的文本，失败时返回空字符串。"""

    if not images:
        return ""

    content: list[dict[str, Any]] = [{"type": "text", "text": OCR_PROMPT}]
    for image in images:
        image_url = image.get("image_url")
        image_base64 = image.get("image_base64")
        mime_type = image.get("mime_type", "image/png")
        if image_base64:
            image_url = f"data:{mime_type};base64,{image_base64}"
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

    try:
        response = llm_client.complete(
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return (response.get("content") or "").strip()
    except Exception:
        return ""
