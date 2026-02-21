"""
分析结果持久化

将采集的数据和分析结果保存到数据库。
P1优化：集成预测验证闭环。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import uuid4

from loguru import logger
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .database import get_database
from .models import (
    AnalysisResult,
    FusionDecisionRecord,
    InvestmentReport,
    PredictionRecord,
    StockDailyQuote,
    StockFinancialData,
    StockInfo,
)


def _upsert_stock_info(session, symbol: str, name: str, market: str) -> None:
    """插入/更新股票基本信息"""
    existing = session.query(StockInfo).filter_by(symbol=symbol).first()
    if existing:
        existing.name = name
        existing.market = market
        existing.updated_at = datetime.now()
    else:
        session.add(StockInfo(
            symbol=symbol,
            name=name,
            market=market,
            updated_at=datetime.now(),
        ))


def _save_daily_quotes(session, symbol: str, quotes: list[dict[str, Any]]) -> int:
    """保存日线行情（跳过已有日期）"""
    if not quotes:
        return 0

    existing_dates = {
        row.time
        for row in session.query(StockDailyQuote.time).filter_by(symbol=symbol).all()
    }

    count = 0
    for q in quotes:
        trade_date = q.get("date")
        if trade_date is None:
            continue
        if isinstance(trade_date, str):
            trade_date = date.fromisoformat(trade_date[:10])
        elif hasattr(trade_date, "date"):
            # pandas Timestamp -> date
            trade_date = trade_date.date()
        if trade_date in existing_dates:
            continue

        session.add(StockDailyQuote(
            time=trade_date,
            symbol=symbol,
            open=q.get("open"),
            high=q.get("high"),
            low=q.get("low"),
            close=q.get("close"),
            volume=q.get("volume"),
            amount=q.get("amount"),
            turnover_rate=q.get("turnover_rate"),
            amplitude=q.get("amplitude"),
            change_pct=q.get("change_pct"),
            change_amount=q.get("change_amount"),
        ))
        count += 1

    return count


def _save_financial_data(session, symbol: str, records: list[dict[str, Any]]) -> int:
    """保存财务数据（跳过已有报告期）"""
    if not records:
        return 0

    existing_dates = {
        row.report_date
        for row in session.query(StockFinancialData.report_date).filter_by(symbol=symbol).all()
    }

    count = 0
    for rec in records:
        report_date = rec.get("report_date")
        if report_date is None:
            continue
        if isinstance(report_date, str):
            report_date = date.fromisoformat(report_date[:10])
        if report_date in existing_dates:
            continue

        session.add(StockFinancialData(
            symbol=symbol,
            report_date=report_date,
            report_type=rec.get("report_type"),
            revenue=rec.get("revenue"),
            net_profit=rec.get("net_profit"),
            net_profit_deducted=rec.get("net_profit_deducted"),
            roe=rec.get("roe"),
            roa=rec.get("roa"),
            gross_margin=rec.get("gross_margin"),
            net_margin=rec.get("net_margin"),
            revenue_yoy=rec.get("revenue_yoy"),
            profit_yoy=rec.get("profit_yoy"),
            debt_ratio=rec.get("debt_ratio"),
            current_ratio=rec.get("current_ratio"),
            quick_ratio=rec.get("quick_ratio"),
            pe_ttm=rec.get("pe_ttm"),
            pb=rec.get("pb"),
            total_market_cap=rec.get("total_market_cap"),
            updated_at=datetime.now(),
        ))
        count += 1

    return count


def persist_analysis(state: dict[str, Any]) -> str:
    """将完整分析结果持久化到数据库

    P1优化：同时记录预测到 prediction_records 表，用于后续验证闭环。

    Returns: run_id
    """
    db = get_database()
    db.create_tables()

    stock = state.get("stock")
    signals = state.get("signals", [])
    fusion = state.get("fusion")
    report = state.get("report", "")
    run_id = str(uuid4())

    # 提取当前价格（用于预测验证）
    current_price = None
    if stock and stock.daily_quotes:
        # 取最新收盘价
        latest_quote = stock.daily_quotes[-1] if isinstance(stock.daily_quotes, list) else None
        if latest_quote:
            current_price = latest_quote.get("close")

    with db.session() as session:
        # 1. 股票基本信息
        if stock:
            _upsert_stock_info(session, stock.symbol, stock.name, stock.market)

            # 2. 行情数据
            q_count = _save_daily_quotes(session, stock.symbol, stock.daily_quotes)
            if q_count:
                logger.info(f"[持久化] 新增{q_count}条行情数据")

            # 3. 财务数据
            f_count = _save_financial_data(session, stock.symbol, stock.financial_data)
            if f_count:
                logger.info(f"[持久化] 新增{f_count}期财务数据")

        symbol = stock.symbol if stock else "UNKNOWN"

        # 4. Agent信号
        for signal in signals:
            session.add(AnalysisResult(
                run_id=run_id,
                symbol=symbol,
                agent_name=signal.agent_name,
                signal_score=signal.signal_score,
                confidence=signal.confidence,
                reasoning=signal.reasoning,
                key_factors=list(signal.key_factors) if signal.key_factors else None,
                risks=list(signal.risks) if signal.risks else None,
                data_quality=signal.data_quality,
                extra_data=signal.metadata if signal.metadata else None,
                llm_model=signal.llm_model or None,
                llm_tokens_used=signal.llm_tokens_used or None,
                llm_cost=signal.llm_cost_usd or None,
                execution_time_ms=signal.execution_time_ms or None,
                created_at=datetime.now(),
            ))

        # 5. 融合决策
        if fusion:
            session.add(FusionDecisionRecord(
                run_id=run_id,
                symbol=symbol,
                final_score=fusion.final_score,
                final_action=fusion.final_action,
                confidence=fusion.confidence,
                position_pct=fusion.position_pct,
                stop_loss_pct=fusion.stop_loss_pct,
                take_profit_pct=fusion.take_profit_pct,
                reasoning=fusion.reasoning,
                signal_summary=fusion.signal_summary if fusion.signal_summary else None,
                conflicts=list(fusion.conflicts) if fusion.conflicts else None,
                conflict_resolution=fusion.conflict_resolution or None,
                market_regime=fusion.market_regime,
                weights_used=fusion.weights_used if fusion.weights_used else None,
                created_at=datetime.now(),
            ))

            # 6. P1新增：记录预测（用于验证闭环）
            divergence_points = list(fusion.divergence_points) if fusion.divergence_points else []
            target_prices = fusion.target_prices if fusion.target_prices else {}
            agent_scores = fusion.signal_summary if fusion.signal_summary else {}

            prediction = PredictionRecord(
                run_id=run_id,
                symbol=symbol,
                prediction_date=date.today(),
                verification_date=date.today(),  # 将由ValidationService设置
                final_score=fusion.final_score,
                final_action=fusion.final_action,
                confidence=fusion.confidence,
                target_price_conservative=target_prices.get("conservative"),
                target_price_base=target_prices.get("base"),
                target_price_optimistic=target_prices.get("optimistic"),
                current_price=current_price,
                divergence_points=divergence_points if divergence_points else None,
                agent_scores=agent_scores if agent_scores else None,
                is_verified=False,
            )
            session.add(prediction)

        # 7. 研报
        if report:
            session.add(InvestmentReport(
                run_id=run_id,
                symbol=symbol,
                report_format="markdown",
                report_content=report,
                rating=str(fusion.final_score) if fusion else None,
                created_at=datetime.now(),
            ))

    logger.info(f"[持久化] 分析结果已保存 run_id={run_id[:8]}...")
    return run_id
