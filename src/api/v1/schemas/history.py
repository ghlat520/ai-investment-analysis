"""API Schemas - 历史记录相关"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class HistoryItem(BaseModel):
    run_id: str
    symbol: str
    final_score: int
    final_action: str
    confidence: float
    position_pct: Optional[int] = None
    market_regime: Optional[str] = None
    created_at: Optional[str] = None


class HistoryListResponse(BaseModel):
    total: int
    items: list[HistoryItem]


class HistoryDetailResponse(BaseModel):
    run_id: str
    symbol: str
    fusion: Optional[dict[str, Any]] = None
    signals: list[dict[str, Any]] = []
    report: Optional[str] = None
    created_at: Optional[str] = None
