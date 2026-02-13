"""
FastAPI 应用工厂
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from .v1.router import router as api_v1_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI 投研助手",
        description="全维度多Agent协作的AI投研分析系统",
        version="0.1.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API 路由
    app.include_router(api_v1_router)

    # 健康检查
    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    # SPA 静态文件
    dist_dir = Path(__file__).parent.parent.parent / "web" / "dist"
    if dist_dir.exists():
        # 静态资源
        assets_dir = dist_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # SPA fallback: 非 /api 开头的路径返回 index.html
        @app.get("/{full_path:path}")
        async def spa_fallback(full_path: str):
            # 避免拦截 API 路由
            if full_path.startswith("api/"):
                return {"error": "Not Found"}
            index = dist_dir / "index.html"
            if index.exists():
                return FileResponse(str(index))
            return {"message": "Frontend not built. Run: cd web && npm run build"}

    return app
