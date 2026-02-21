"""
趋势追踪服务

P3多时间维度分析核心功能：
1. 评分趋势追踪 - 记录和展示评分变化
2. 定期重分析调度 - 自动重跑追踪中的股票
3. 催化剂日历管理 - 管理未来事件时间线
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import and_, desc, func

from src.data.storage.database import get_database
from src.data.storage.models import (
    CatalystEvent,
    FusionDecisionRecord,
    ScoreHistory,
    StockInfo,
    TrackedStock,
)


class TrendService:
    """趋势追踪服务"""

    def __init__(self) -> None:
        self.db = get_database()
        self.db.create_tables()

    # ========== 追踪股票管理 ==========

    def add_tracked_stock(
        self,
        symbol: str,
        frequency: str = "weekly",
        priority: int = 5,
        notes: Optional[str] = None,
    ) -> TrackedStock:
        """添加股票到追踪列表"""
        with self.db.session() as session:
            existing = session.query(TrackedStock).filter_by(symbol=symbol).first()
            if existing:
                existing.track_enabled = True
                existing.reanalysis_frequency = frequency
                existing.priority = priority
                existing.notes = notes
                session.commit()
                logger.info(f"[TrendService] 更新追踪股票: {symbol}")
                return existing

            tracked = TrackedStock(
                symbol=symbol,
                reanalysis_frequency=frequency,
                priority=priority,
                notes=notes,
            )
            session.add(tracked)
            session.commit()
            logger.info(f"[TrendService] 添加追踪股票: {symbol}")
            return tracked

    def remove_tracked_stock(self, symbol: str) -> bool:
        """从追踪列表移除股票"""
        with self.db.session() as session:
            tracked = session.query(TrackedStock).filter_by(symbol=symbol).first()
            if tracked:
                tracked.track_enabled = False
                session.commit()
                logger.info(f"[TrendService] 停止追踪股票: {symbol}")
                return True
            return False

    def get_tracked_stocks(
        self,
        enabled_only: bool = True,
        frequency: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """获取追踪股票列表"""
        with self.db.session() as session:
            query = session.query(TrackedStock)
            if enabled_only:
                query = query.filter(TrackedStock.track_enabled == True)
            if frequency:
                query = query.filter(TrackedStock.reanalysis_frequency == frequency)

            tracked = query.order_by(desc(TrackedStock.priority)).all()

            result = []
            for t in tracked:
                stock_info = session.query(StockInfo).filter_by(symbol=t.symbol).first()
                result.append({
                    "symbol": t.symbol,
                    "name": stock_info.name if stock_info else t.symbol,
                    "track_enabled": t.track_enabled,
                    "reanalysis_frequency": t.reanalysis_frequency,
                    "priority": t.priority,
                    "last_score": t.last_score,
                    "last_action": t.last_action,
                    "last_analysis_date": t.last_analysis_date.isoformat() if t.last_analysis_date else None,
                    "score_trend": t.score_trend,
                    "score_change_7d": t.score_change_7d,
                    "score_change_30d": t.score_change_30d,
                    "notes": t.notes,
                })
            return result

    def get_stocks_for_reanalysis(self, frequency: str = "weekly") -> list[str]:
        """获取需要重分析的股票列表"""
        with self.db.session() as session:
            now = datetime.now()

            # 根据频率计算时间间隔
            if frequency == "daily":
                interval = timedelta(days=1)
            elif frequency == "weekly":
                interval = timedelta(weeks=1)
            elif frequency == "monthly":
                interval = timedelta(days=30)
            else:
                interval = timedelta(weeks=1)

            cutoff = now - interval

            query = session.query(TrackedStock).filter(
                and_(
                    TrackedStock.track_enabled == True,
                    TrackedStock.reanalysis_frequency == frequency,
                    (TrackedStock.last_analysis_date == None) | (TrackedStock.last_analysis_date < cutoff),
                )
            )

            return [t.symbol for t in query.order_by(desc(TrackedStock.priority)).all()]

    # ========== 评分历史记录 ==========

    def record_score_snapshot(
        self,
        symbol: str,
        final_score: int,
        final_action: str,
        confidence: float,
        agent_scores: Optional[dict] = None,
        close_price: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> ScoreHistory:
        """记录评分快照"""
        today = date.today()

        with self.db.session() as session:
            # 检查是否已有今天的记录
            existing = session.query(ScoreHistory).filter(
                and_(ScoreHistory.symbol == symbol, ScoreHistory.score_date == today)
            ).first()

            if existing:
                # 更新今天的记录
                existing.final_score = final_score
                existing.final_action = final_action
                existing.confidence = confidence
                existing.agent_scores = agent_scores
                existing.close_price = close_price
                existing.run_id = run_id
                session.commit()
                logger.debug(f"[TrendService] 更新评分快照: {symbol} score={final_score:+d}")
                return existing

            snapshot = ScoreHistory(
                symbol=symbol,
                score_date=today,
                final_score=final_score,
                final_action=final_action,
                confidence=confidence,
                agent_scores=agent_scores,
                close_price=close_price,
                run_id=run_id,
            )
            session.add(snapshot)
            session.commit()
            logger.debug(f"[TrendService] 记录评分快照: {symbol} score={final_score:+d}")
            return snapshot

    def update_tracked_stock_stats(self, symbol: str) -> None:
        """更新追踪股票的统计信息（趋势、变化）"""
        today = date.today()
        day_7_ago = today - timedelta(days=7)
        day_30_ago = today - timedelta(days=30)

        with self.db.session() as session:
            tracked = session.query(TrackedStock).filter_by(symbol=symbol).first()
            if not tracked:
                return

            # 获取当前评分
            current = session.query(ScoreHistory).filter(
                and_(ScoreHistory.symbol == symbol, ScoreHistory.score_date == today)
            ).first()

            if not current:
                return

            tracked.last_score = current.final_score
            tracked.last_action = current.final_action
            tracked.last_analysis_date = datetime.now()

            # 计算7天变化
            score_7d_ago = session.query(ScoreHistory).filter(
                and_(ScoreHistory.symbol == symbol, ScoreHistory.score_date <= day_7_ago)
            ).order_by(desc(ScoreHistory.score_date)).first()

            if score_7d_ago:
                tracked.score_change_7d = current.final_score - score_7d_ago.final_score
            else:
                tracked.score_change_7d = None

            # 计算30天变化
            score_30d_ago = session.query(ScoreHistory).filter(
                and_(ScoreHistory.symbol == symbol, ScoreHistory.score_date <= day_30_ago)
            ).order_by(desc(ScoreHistory.score_date)).first()

            if score_30d_ago:
                tracked.score_change_30d = current.final_score - score_30d_ago.final_score
            else:
                tracked.score_change_30d = None

            # 计算趋势
            history = session.query(ScoreHistory).filter(
                ScoreHistory.symbol == symbol
            ).order_by(desc(ScoreHistory.score_date)).limit(7).all()

            if len(history) >= 3:
                scores = [h.final_score for h in reversed(history)]
                # 简单线性趋势判断
                if scores[-1] > scores[0] + 5:
                    tracked.score_trend = "rising"
                elif scores[-1] < scores[0] - 5:
                    tracked.score_trend = "falling"
                else:
                    tracked.score_trend = "stable"
            else:
                tracked.score_trend = None

            session.commit()

    def get_score_history(
        self,
        symbol: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """获取评分历史"""
        start_date = date.today() - timedelta(days=days)

        with self.db.session() as session:
            history = session.query(ScoreHistory).filter(
                and_(ScoreHistory.symbol == symbol, ScoreHistory.score_date >= start_date)
            ).order_by(ScoreHistory.score_date).all()

            return [
                {
                    "date": h.score_date.isoformat(),
                    "final_score": h.final_score,
                    "final_action": h.final_action,
                    "confidence": float(h.confidence),
                    "agent_scores": h.agent_scores,
                    "close_price": float(h.close_price) if h.close_price else None,
                }
                for h in history
            ]

    def get_trend_summary(self, symbol: str) -> dict[str, Any]:
        """获取趋势摘要"""
        with self.db.session() as session:
            tracked = session.query(TrackedStock).filter_by(symbol=symbol).first()
            stock_info = session.query(StockInfo).filter_by(symbol=symbol).first()

            if not tracked:
                # 如果不在追踪列表，从历史数据计算
                history = session.query(ScoreHistory).filter(
                    ScoreHistory.symbol == symbol
                ).order_by(desc(ScoreHistory.score_date)).limit(30).all()

                if not history:
                    return {"symbol": symbol, "tracked": False, "history_count": 0}

                return {
                    "symbol": symbol,
                    "name": stock_info.name if stock_info else symbol,
                    "tracked": False,
                    "last_score": history[0].final_score,
                    "last_action": history[0].final_action,
                    "last_analysis_date": history[0].score_date.isoformat(),
                    "history_count": len(history),
                    "history": [
                        {"date": h.score_date.isoformat(), "score": h.final_score}
                        for h in reversed(history)
                    ],
                }

            # 获取最近30天历史
            history = self.get_score_history(symbol, days=30)

            return {
                "symbol": symbol,
                "name": stock_info.name if stock_info else symbol,
                "tracked": True,
                "track_enabled": tracked.track_enabled,
                "reanalysis_frequency": tracked.reanalysis_frequency,
                "priority": tracked.priority,
                "last_score": tracked.last_score,
                "last_action": tracked.last_action,
                "last_analysis_date": tracked.last_analysis_date.isoformat() if tracked.last_analysis_date else None,
                "score_trend": tracked.score_trend,
                "score_change_7d": tracked.score_change_7d,
                "score_change_30d": tracked.score_change_30d,
                "history_count": len(history),
                "history": history,
            }

    # ========== 催化剂日历 ==========

    def add_catalyst_event(
        self,
        symbol: str,
        event_type: str,
        event_name: str,
        event_date: Optional[date] = None,
        event_date_precision: str = "day",
        expected_impact: str = "neutral",
        impact_confidence: float = 0.5,
        impact_magnitude: str = "moderate",
        source: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CatalystEvent:
        """添加催化剂事件"""
        with self.db.session() as session:
            event = CatalystEvent(
                symbol=symbol,
                event_type=event_type,
                event_name=event_name,
                event_date=event_date,
                event_date_precision=event_date_precision,
                expected_impact=expected_impact,
                impact_confidence=impact_confidence,
                impact_magnitude=impact_magnitude,
                source=source,
                notes=notes,
            )
            session.add(event)
            session.commit()
            logger.info(f"[TrendService] 添加催化剂事件: {symbol} - {event_name}")
            return event

    def update_catalyst_status(
        self,
        event_id: int,
        status: str,
        actual_impact: Optional[str] = None,
    ) -> Optional[CatalystEvent]:
        """更新催化剂事件状态"""
        with self.db.session() as session:
            event = session.query(CatalystEvent).filter_by(id=event_id).first()
            if event:
                event.status = status
                if actual_impact:
                    event.actual_impact = actual_impact
                session.commit()
                logger.info(f"[TrendService] 更新催化剂状态: {event.symbol} - {event.event_name} -> {status}")
                return event
            return None

    def get_upcoming_catalysts(
        self,
        symbol: Optional[str] = None,
        days: int = 30,
        include_past: bool = False,
    ) -> list[dict[str, Any]]:
        """获取即将到来的催化剂事件"""
        today = date.today()
        end_date = today + timedelta(days=days)

        with self.db.session() as session:
            query = session.query(CatalystEvent).filter(
                CatalystEvent.status == "pending"
            )

            if symbol:
                query = query.filter(CatalystEvent.symbol == symbol)

            if not include_past:
                query = query.filter(
                    (CatalystEvent.event_date == None) |
                    (CatalystEvent.event_date >= today)
                )

            events = query.filter(
                (CatalystEvent.event_date == None) |
                (CatalystEvent.event_date <= end_date)
            ).order_by(CatalystEvent.event_date).all()

            result = []
            for e in events:
                stock_info = session.query(StockInfo).filter_by(symbol=e.symbol).first()

                # 计算距离天数
                days_until = None
                if e.event_date:
                    days_until = (e.event_date - today).days

                result.append({
                    "id": e.id,
                    "symbol": e.symbol,
                    "name": stock_info.name if stock_info else e.symbol,
                    "event_type": e.event_type,
                    "event_name": e.event_name,
                    "event_date": e.event_date.isoformat() if e.event_date else None,
                    "event_date_precision": e.event_date_precision,
                    "days_until": days_until,
                    "expected_impact": e.expected_impact,
                    "impact_confidence": float(e.impact_confidence),
                    "impact_magnitude": e.impact_magnitude,
                    "status": e.status,
                    "source": e.source,
                    "notes": e.notes,
                })
            return result

    def get_catalyst_calendar(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """获取催化剂日历（按日期分组）"""
        if not start_date:
            start_date = date.today()
        if not end_date:
            end_date = start_date + timedelta(days=30)

        events = self.get_upcoming_catalysts(include_past=False, days=90)

        calendar = {}
        for e in events:
            if e["event_date"]:
                event_date = e["event_date"]
                if start_date.isoformat() <= event_date <= end_date.isoformat():
                    if event_date not in calendar:
                        calendar[event_date] = []
                    calendar[event_date].append(e)

        return calendar

    # ========== 批量操作 ==========

    def sync_all_trends(self) -> dict[str, int]:
        """同步所有追踪股票的趋势数据"""
        with self.db.session() as session:
            tracked = session.query(TrackedStock).filter(
                TrackedStock.track_enabled == True
            ).all()

            updated = 0
            for t in tracked:
                try:
                    self.update_tracked_stock_stats(t.symbol)
                    updated += 1
                except Exception as e:
                    logger.error(f"[TrendService] 同步{t.symbol}失败: {e}")

            logger.info(f"[TrendService] 同步完成: {updated}/{len(tracked)}")
            return {"total": len(tracked), "updated": updated}

    def cleanup_old_history(self, keep_days: int = 365) -> int:
        """清理过期历史数据"""
        cutoff = date.today() - timedelta(days=keep_days)

        with self.db.session() as session:
            deleted = session.query(ScoreHistory).filter(
                ScoreHistory.score_date < cutoff
            ).delete()
            session.commit()
            logger.info(f"[TrendService] 清理{deleted}条过期历史数据")
            return deleted
