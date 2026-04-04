"""
应用入口。
职责概览：
1. 加载 `.env` 配置并初始化数据库。
2. 创建 FastAPI 应用，挂载 API 路由、静态资源和生成媒体目录。
3. 暴露 `/query`、`/chat`、`/report` 三个本地调试页面。

与其他模块的关系：
- `app.db.init_db` 负责建表和轻量迁移。
- `app.api.routes` 提供全部业务 API。
- `app.static` 提供查询、聊天和报告解读页。
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.db.init_db import init_db
from app.db.session import DATA_DIR
from app.env import load_env_file


load_env_file()
init_db()
STATIC_DIR = Path(__file__).resolve().parent / "static"

openapi_tags = [
    {
        "name": "Agent",
        "description": "通过 Qwen 模型与内部工具完成规划、检索和问答。",
    },
    {
        "name": "Patients",
        "description": "患者基础身份信息的创建、查询和更新。",
    },
    {
        "name": "Medical Cases",
        "description": "患者病例信息的创建、查询和更新。",
    },
    {
        "name": "Visit Records",
        "description": "患者就诊记录的创建、查询和更新。",
    },
    {
        "name": "Memory",
        "description": "长期偏好、画像和记忆事件的查询、更新与清理。",
    },
    {
        "name": "Reports",
        "description": "检验检查报告的录入、查询与解读。",
    },
    {
        "name": "Knowledge",
        "description": "知识库文档的维护与检索。",
    },
]

app = FastAPI(
    title="MediEase API",
    version="0.2.0",
    description="面向患者咨询场景的 Agent 后端服务。",
    openapi_tags=openapi_tags,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=DATA_DIR), name="media")
app.include_router(router, prefix="/api")


@app.get("/", include_in_schema=False)
def read_index() -> FileResponse:
    """返回默认查询页，方便本地直接打开项目。"""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/query", include_in_schema=False)
def read_query_page() -> FileResponse:
    """返回单轮查询调试页。"""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat", include_in_schema=False)
def read_chat_page() -> FileResponse:
    """返回多轮对话调试页。"""

    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/report", include_in_schema=False)
def read_report_page() -> FileResponse:
    """返回报告解读页。"""

    return FileResponse(STATIC_DIR / "report.html")
