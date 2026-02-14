"""V1 路由聚合"""

from fastapi import APIRouter

from .endpoints import analysis, history, hotspot

router = APIRouter(prefix="/api/v1")

router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
router.include_router(history.router, prefix="/history", tags=["History"])
router.include_router(hotspot.router, prefix="/hotspot", tags=["Hotspot"])
