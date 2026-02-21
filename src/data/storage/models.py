"""
SQLAlchemy ORM 模型定义

对应 docs/database.md 中的10张核心表。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StockInfo(Base):
    __tablename__ = "stock_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    market: Mapped[str] = mapped_column(String(10), nullable=False)
    exchange: Mapped[Optional[str]] = mapped_column(String(20))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    list_date: Mapped[Optional[date]] = mapped_column(Date)
    is_st: Mapped[bool] = mapped_column(Boolean, default=False)
    is_delisting: Mapped[bool] = mapped_column(Boolean, default=False)
    total_shares: Mapped[Optional[int]] = mapped_column(BigInteger)
    float_shares: Mapped[Optional[int]] = mapped_column(BigInteger)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class StockDailyQuote(Base):
    __tablename__ = "stock_daily_quotes"
    __table_args__ = (UniqueConstraint("time", "symbol"),)

    time: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    open: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    high: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    low: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    close: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    amount: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    turnover_rate: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    amplitude: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    change_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    change_amount: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))


class StockFinancialData(Base):
    __tablename__ = "stock_financial_data"
    __table_args__ = (UniqueConstraint("symbol", "report_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    report_type: Mapped[Optional[str]] = mapped_column(String(10))
    revenue: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    net_profit: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    net_profit_deducted: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    roe: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    roa: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    gross_margin: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    net_margin: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    revenue_yoy: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    profit_yoy: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    debt_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    current_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    quick_ratio: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    operating_cashflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    free_cashflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    pe_ttm: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    pb: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    ps_ttm: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    total_market_cap: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class StockMoneyFlow(Base):
    __tablename__ = "stock_money_flow"
    __table_args__ = (UniqueConstraint("symbol", "date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    main_net_inflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    main_net_inflow_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    huge_net_inflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    large_net_inflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    medium_net_inflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    small_net_inflow: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    northbound_holding: Mapped[Optional[int]] = mapped_column(BigInteger)
    northbound_holding_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    northbound_change: Mapped[Optional[int]] = mapped_column(BigInteger)
    margin_balance: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    short_balance: Mapped[Optional[float]] = mapped_column(Numeric(20, 4))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class StockNews(Base):
    __tablename__ = "stock_news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    url: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    search_dimension: Mapped[Optional[str]] = mapped_column(String(50))
    sentiment: Mapped[Optional[str]] = mapped_column(String(20))
    sentiment_score: Mapped[Optional[float]] = mapped_column(Numeric(4, 2))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_score: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    key_factors: Mapped[Optional[dict]] = mapped_column(JSON)
    risks: Mapped[Optional[dict]] = mapped_column(JSON)
    data_quality: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))
    extra_data: Mapped[Optional[dict]] = mapped_column("metadata", JSON)
    llm_model: Mapped[Optional[str]] = mapped_column(String(100))
    llm_tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    llm_cost: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class FusionDecisionRecord(Base):
    __tablename__ = "fusion_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    final_score: Mapped[int] = mapped_column(Integer, nullable=False)
    final_action: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    position_pct: Mapped[Optional[int]] = mapped_column(Integer)
    stop_loss_pct: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    take_profit_pct: Mapped[Optional[float]] = mapped_column(Numeric(6, 2))
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    signal_summary: Mapped[Optional[dict]] = mapped_column(JSON)
    conflicts: Mapped[Optional[dict]] = mapped_column(JSON)
    conflict_resolution: Mapped[Optional[str]] = mapped_column(Text)
    market_regime: Mapped[Optional[str]] = mapped_column(String(20))
    weights_used: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class InvestmentReport(Base):
    __tablename__ = "investment_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    report_format: Mapped[str] = mapped_column(String(20), default="markdown")
    report_content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    rating: Mapped[Optional[str]] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ScreeningResult(Base):
    __tablename__ = "screening_results"
    __table_args__ = (UniqueConstraint("run_date", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    composite_score: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))
    factor_scores: Mapped[Optional[dict]] = mapped_column(JSON)
    market: Mapped[Optional[str]] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class LLMCostLog(Base):
    __tablename__ = "llm_cost_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[Optional[str]] = mapped_column(String(36))
    agent_name: Mapped[Optional[str]] = mapped_column(String(50))
    llm_provider: Mapped[Optional[str]] = mapped_column(String(50))
    llm_model: Mapped[Optional[str]] = mapped_column(String(100))
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    total_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    cost_usd: Mapped[Optional[float]] = mapped_column(Numeric(8, 6))
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    is_cached: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class PredictionRecord(Base):
    """预测记录表 - 用于验证闭环"""
    __tablename__ = "prediction_records"
    __table_args__ = (UniqueConstraint("run_id", "symbol"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)

    # 预测时间
    prediction_date: Mapped[date] = mapped_column(Date, nullable=False)
    verification_date: Mapped[Optional[date]] = mapped_column(Date)  # 计划验证日期

    # 预测内容
    final_score: Mapped[int] = mapped_column(Integer, nullable=False)
    final_action: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)

    # 目标价预测
    target_price_conservative: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    target_price_base: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    target_price_optimistic: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))
    current_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))  # 预测时的价格

    # 分歧点（关键假设）
    divergence_points: Mapped[Optional[dict]] = mapped_column(JSON)  # [{point, verify_method, verify_date}]

    # Agent评分快照
    agent_scores: Mapped[Optional[dict]] = mapped_column(JSON)  # {agent_name: score}

    # 验证结果（后续更新）
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # 实际结果
    actual_price: Mapped[Optional[float]] = mapped_column(Numeric(12, 4))  # 验证时的价格
    actual_return_pct: Mapped[Optional[float]] = mapped_column(Numeric(8, 4))  # 实际收益率
    price_target_hit: Mapped[Optional[str]] = mapped_column(String(20))  # conservative/base/optimistic/miss

    # 预测准确性
    direction_correct: Mapped[Optional[bool]] = mapped_column(Boolean)  # 方向是否正确
    score_accuracy: Mapped[Optional[float]] = mapped_column(Numeric(4, 3))  # 评分准确性指标

    # Agent准确性归因
    agent_accuracy: Mapped[Optional[dict]] = mapped_column(JSON)  # {agent_name: {correct: bool, contribution: float}}

    # 错误分析
    error_analysis: Mapped[Optional[dict]] = mapped_column(JSON)  # {root_cause, lessons, suggested_fix}

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)
