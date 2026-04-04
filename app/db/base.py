"""
SQLAlchemy 声明式模型基类。

本文件只负责提供全项目共享的 Declarative Base，所有 ORM 模型都从这里继承。
它与 `app.db.models` 配套使用，由 `app.db.init_db` 在启动时统一建表。
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """全项目 ORM 模型的声明式基类。"""

