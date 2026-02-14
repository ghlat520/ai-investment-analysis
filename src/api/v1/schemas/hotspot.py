"""API Schemas - 热点分析相关"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HotspotTaskAccepted(BaseModel):
    task_id: str
    status: str = "pending"
    message: str = ""


class ChainStockResponse(BaseModel):
    symbol: str
    name: str
    role: str = ""
    reason: str = ""


class AnalyzedThemeResponse(BaseModel):
    title: str
    investment_logic: str = ""
    actionability: str = "medium"
    upstream: list[ChainStockResponse] = []
    midstream: list[ChainStockResponse] = []
    downstream: list[ChainStockResponse] = []
    value_flow: str = ""


class ThemeResponse(BaseModel):
    title: str
    summary: str = ""
    relevance_score: int = 0
    catalyst: str = ""
    timeline: str = ""
    risk: str = ""


class HotspotTaskInfoResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    message: Optional[str] = None
    current_theme: int = 0
    total_themes: int = 0
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class BriefingResponse(BaseModel):
    task_id: str
    analysis_date: str = ""
    themes: list[ThemeResponse] = []
    analyzed_themes: list[AnalyzedThemeResponse] = []
    briefing: str = ""
    errors: list[str] = []


class DuplicateHotspotErrorResponse(BaseModel):
    error: str = "duplicate_hotspot"
    message: str
    existing_task_id: str


class HotspotHistoryResponse(BaseModel):
    total: int
    tasks: list[HotspotTaskInfoResponse]
