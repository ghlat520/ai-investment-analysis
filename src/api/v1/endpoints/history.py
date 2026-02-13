"""
历史记录端点

GET /history           - 历史列表
GET /history/{run_id}  - 详细结果
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from src.services.history_service import get_history_detail, list_history

from ..schemas.history import (
    HistoryDetailResponse,
    HistoryItem,
    HistoryListResponse,
)

router = APIRouter()


@router.get("/", response_model=HistoryListResponse, summary="查询历史分析记录")
def get_history_list(
    symbol: Optional[str] = Query(None, description="按股票代码筛选"),
    limit: int = Query(20, ge=1, le=100),
):
    items = list_history(symbol=symbol, limit=limit)
    return HistoryListResponse(
        total=len(items),
        items=[HistoryItem(**item) for item in items],
    )


@router.get("/{run_id}", response_model=HistoryDetailResponse, summary="查询单次分析详情")
def get_history(run_id: str):
    detail = get_history_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"run_id {run_id} 不存在")
    return HistoryDetailResponse(**detail)
