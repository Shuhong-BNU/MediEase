"""
数据库引擎与会话工厂。

本模块负责：
1. 解析项目数据目录并初始化 SQLite 数据库地址。
2. 提供全局 `engine` 与 `SessionLocal`。
3. 通过 `get_db()` 为 FastAPI 路由提供按请求生命周期管理的数据库会话。

它是 API 层、服务层和 Agent 执行层共享的数据库入口。
"""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_URL = f"sqlite:///{DATA_DIR / 'patient_agent.db'}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """为请求提供独立数据库会话，并在结束后自动关闭。"""

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

