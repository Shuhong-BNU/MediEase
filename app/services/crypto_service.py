"""
敏感字段加密辅助服务。
职责概览：
1. 为患者手机号、身份证号、地址等字段生成可逆的加密镜像列。
2. 使用轻量级的标准库方案，避免为当前原型额外引入重型依赖。
3. 为未来切换到“密文优先读取”保留统一入口。

说明：
- 当前实现属于工程过渡方案，重点是把“敏感字段存在加密存储方案”真正落到代码里。
- 明文字段仍然保留，兼容现有查询、身份校验和测试；后续可以逐步切换读取路径。
"""

from __future__ import annotations

import base64
import hashlib
import os
from itertools import cycle
from typing import Optional


def _secret_bytes() -> bytes:
    """从环境变量加载加密种子；本地未配置时使用开发默认值。"""

    secret = os.getenv("PATIENT_AGENT_CRYPTO_KEY", "patient-agent-dev-key")
    return hashlib.sha256(secret.encode("utf-8")).digest()


def _xor_bytes(raw: bytes, secret: bytes) -> bytes:
    """用固定密钥流做可逆异或，满足原型阶段“可加密、可回读”的需求。"""

    return bytes(value ^ mask for value, mask in zip(raw, cycle(secret)))


def encrypt_text(value: str) -> str:
    """将 UTF-8 文本编码后加密，并转成可持久化的 base64 字符串。"""

    encrypted = _xor_bytes(value.encode("utf-8"), _secret_bytes())
    return base64.urlsafe_b64encode(encrypted).decode("ascii")


def decrypt_text(value: str) -> str:
    """将密文恢复为明文，主要用于调试或后续密文读取迁移。"""

    raw = base64.urlsafe_b64decode(value.encode("ascii"))
    return _xor_bytes(raw, _secret_bytes()).decode("utf-8")


def encrypt_optional_text(value: Optional[str]) -> Optional[str]:
    """空值透传，非空值生成密文。"""

    if value is None or not value.strip():
        return None
    return encrypt_text(value)
