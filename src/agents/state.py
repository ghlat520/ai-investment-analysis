"""
LangGraph 共享状态定义

所有Agent通过State读写数据，实现解耦协作。
State是LangGraph图的核心数据结构，贯穿整个分析流程。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Annotated, Any, Optional
from uuid import UUID, uuid4

from langgraph.graph import MessagesState


@dataclass(frozen=True)
class AgentSignal:
    """单个Agent的输出信号（不可变）"""

    agent_name: str
    signal_score: int  # -100 ~ +100
    confidence: float  # 0.0 ~ 1.0
    reasoning: str = ""
    key_factors: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    data_quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_model: str = ""
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    execution_time_ms: int = 0


@dataclass(frozen=True)
class FusionDecision:
    """决策融合Agent的输出"""

    final_score: int  # -100 ~ +100
    final_action: str  # 建仓/加仓/持有/减仓/清仓/观望
    confidence: float
    position_pct: int = 0  # 建议仓位 %
    stop_loss_pct: float = -8.0
    take_profit_pct: float = 15.0
    reasoning: str = ""
    signal_summary: dict[str, int] = field(default_factory=dict)
    conflicts: tuple[str, ...] = ()
    conflict_resolution: str = ""
    market_regime: str = "neutral"
    weights_used: dict[str, float] = field(default_factory=dict)


@dataclass
class StockData:
    """单只股票的全部数据（传递给Agent的输入）"""

    symbol: str
    name: str
    market: str  # A/HK/US

    # 行情数据（DataFrame序列化为dict列表）
    daily_quotes: list[dict[str, Any]] = field(default_factory=list)

    # 财务数据
    financial_data: list[dict[str, Any]] = field(default_factory=list)

    # 资金流向
    money_flow: list[dict[str, Any]] = field(default_factory=list)

    # 新闻
    news: list[dict[str, Any]] = field(default_factory=list)

    # 股票基本信息
    info: dict[str, Any] = field(default_factory=dict)


def _merge_signals(
    existing: list[AgentSignal], new: list[AgentSignal]
) -> list[AgentSignal]:
    """合并信号列表（LangGraph reducer）"""
    return existing + new


def _merge_errors(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


class AnalysisState(MessagesState):
    """LangGraph 分析流程的共享状态

    所有Agent节点读写此State，通过LangGraph的reducer机制
    自动合并并发写入。
    """

    # 运行标识
    run_id: UUID = field(default_factory=uuid4)
    analysis_date: str = ""  # YYYY-MM-DD

    # 输入：待分析的股票数据
    stock: Optional[StockData] = None

    # 各Agent产出的信号（通过reducer自动追加）
    signals: Annotated[list[AgentSignal], _merge_signals] = field(
        default_factory=list
    )

    # 决策融合结果
    fusion: Optional[FusionDecision] = None

    # 研报内容
    report: str = ""

    # 错误收集
    errors: Annotated[list[str], _merge_errors] = field(default_factory=list)

    # 元数据
    total_llm_cost_usd: float = 0.0
    total_tokens_used: int = 0
