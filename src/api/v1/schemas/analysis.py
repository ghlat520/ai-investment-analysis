"""API Schemas - 分析相关"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    symbol: str = Field(..., description="股票代码，如 300054.SZ")
    market: str = Field("A", description="市场：A/HK/US")
    use_research: bool = Field(False, description="是否使用已上传的研报PDF")


class TaskAccepted(BaseModel):
    task_id: str
    status: str = "pending"
    message: str = ""


class AgentSignalResponse(BaseModel):
    agent_name: str
    signal_score: int
    confidence: float
    reasoning: str = ""
    key_factors: list[str] = []
    risks: list[str] = []
    llm_model: str = ""
    execution_time_ms: int = 0


class FusionDecisionResponse(BaseModel):
    final_score: int
    final_action: str
    confidence: float
    position_pct: int = 0
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    reasoning: str = ""
    signal_summary: dict[str, int] = {}
    conflicts: list[str] = []
    conflict_resolution: str = ""
    market_regime: str = "neutral"
    weights_used: dict[str, float] = {}
    bull_arguments: list[str] = []
    bear_arguments: list[str] = []
    divergence_points: list[str] = []
    target_prices: dict[str, float] = {}


class AnalysisResultResponse(BaseModel):
    run_id: Optional[str] = None
    symbol: str
    stock_name: Optional[str] = None
    fusion: Optional[FusionDecisionResponse] = None
    signals: list[AgentSignalResponse] = []
    report: str = ""
    errors: list[str] = []
    created_at: Optional[str] = None


class TaskInfoResponse(BaseModel):
    task_id: str
    symbol: str
    market: str = "A"
    stock_name: Optional[str] = None
    status: str
    progress: int = 0
    message: Optional[str] = None
    completed_agents: list[str] = []
    total_agents: int = 9
    run_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class TaskListResponse(BaseModel):
    total: int
    tasks: list[TaskInfoResponse]


class DuplicateTaskErrorResponse(BaseModel):
    error: str = "duplicate_task"
    message: str
    symbol: str
    existing_task_id: str
