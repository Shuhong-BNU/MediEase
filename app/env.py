"""
环境变量加载工具。

职责概览：
- 读取项目根目录下的 `.env` 文件。
- 将文件中的键值对写入进程环境，但不覆盖已存在的系统环境变量。

设计说明：
- 这里故意保持轻量，不依赖 `python-dotenv`。
- 只处理最常见的 `KEY=VALUE` 形式，足够覆盖本项目的本地开发场景。
"""

from __future__ import annotations

import os
from pathlib import Path


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(env_file: Path = ENV_FILE) -> None:
    """读取 `.env` 并补充进当前进程环境。"""

    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        # 兼容最常见的单双引号包裹写法。
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)
