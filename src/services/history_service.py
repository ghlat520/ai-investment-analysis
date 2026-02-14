"""
历史记录查询服务

查询 DB 中的 analysis_results / fusion_decisions / investment_reports。
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


def list_history(
    symbol: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """查询融合决策历史列表

    Returns:
        [{"run_id", "symbol", "stock_name", "final_score", "final_action", "confidence", "created_at", ...}]
    """
    from src.data.storage.database import get_database
    from src.data.storage.models import FusionDecisionRecord, StockInfo

    db = get_database()
    db.create_tables()

    with db.session() as session:
        query = (
            session.query(FusionDecisionRecord, StockInfo.name)
            .outerjoin(StockInfo, FusionDecisionRecord.symbol == StockInfo.symbol)
            .order_by(FusionDecisionRecord.created_at.desc())
        )
        if symbol:
            query = query.filter(FusionDecisionRecord.symbol == symbol)
        rows = query.limit(limit).all()

        return [
            {
                "run_id": r.run_id,
                "symbol": r.symbol,
                "stock_name": name or r.symbol,
                "final_score": r.final_score,
                "final_action": r.final_action,
                "confidence": float(r.confidence) if r.confidence else 0,
                "position_pct": r.position_pct,
                "market_regime": r.market_regime,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r, name in rows
        ]


def get_history_detail(run_id: str) -> Optional[dict[str, Any]]:
    """查询单次分析的完整结果

    Returns:
        {"run_id", "symbol", "fusion", "signals", "report"} 或 None
    """
    from src.data.storage.database import get_database
    from src.data.storage.models import (
        AnalysisResult,
        FusionDecisionRecord,
        InvestmentReport,
        StockInfo,
    )

    db = get_database()
    db.create_tables()

    with db.session() as session:
        fusion = session.query(FusionDecisionRecord).filter_by(run_id=run_id).first()
        if not fusion:
            return None

        stock_info = session.query(StockInfo).filter_by(symbol=fusion.symbol).first()
        signals = session.query(AnalysisResult).filter_by(run_id=run_id).all()
        report = session.query(InvestmentReport).filter_by(run_id=run_id).first()

        return {
            "run_id": run_id,
            "symbol": fusion.symbol,
            "stock_name": stock_info.name if stock_info else fusion.symbol,
            "fusion": {
                "final_score": fusion.final_score,
                "final_action": fusion.final_action,
                "confidence": float(fusion.confidence) if fusion.confidence else 0,
                "position_pct": fusion.position_pct,
                "stop_loss_pct": float(fusion.stop_loss_pct) if fusion.stop_loss_pct else None,
                "take_profit_pct": float(fusion.take_profit_pct) if fusion.take_profit_pct else None,
                "reasoning": fusion.reasoning,
                "signal_summary": fusion.signal_summary,
                "conflicts": fusion.conflicts,
                "conflict_resolution": fusion.conflict_resolution,
                "market_regime": fusion.market_regime,
                "weights_used": fusion.weights_used,
            },
            "signals": [
                {
                    "agent_name": s.agent_name,
                    "signal_score": s.signal_score,
                    "confidence": float(s.confidence) if s.confidence else 0,
                    "reasoning": s.reasoning,
                    "key_factors": s.key_factors,
                    "risks": s.risks,
                    "data_quality": float(s.data_quality) if s.data_quality else None,
                    "extra_data": s.extra_data if s.extra_data else None,
                    "llm_model": s.llm_model,
                    "llm_tokens_used": s.llm_tokens_used,
                    "llm_cost": float(s.llm_cost) if s.llm_cost else None,
                    "execution_time_ms": s.execution_time_ms,
                }
                for s in signals
            ],
            "report": report.report_content if report else None,
            "created_at": fusion.created_at.isoformat() if fusion.created_at else None,
        }
