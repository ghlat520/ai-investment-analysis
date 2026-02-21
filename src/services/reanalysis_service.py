"""
重分析调度服务

P3多时间维度分析：定期重跑追踪中的股票分析。
支持每日/每周/每月频率的自动重分析。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger

from src.services.trend_service import TrendService


class ReanalysisService:
    """重分析调度服务"""

    def __init__(self) -> None:
        self.trend_service = TrendService()

    def get_reanalysis_queue(self) -> dict[str, list[str]]:
        """获取需要重分析的股票队列"""
        return {
            "daily": self.trend_service.get_stocks_for_reanalysis("daily"),
            "weekly": self.trend_service.get_stocks_for_reanalysis("weekly"),
            "monthly": self.trend_service.get_stocks_for_reanalysis("monthly"),
        }

    def run_reanalysis(
        self,
        symbols: Optional[list[str]] = None,
        frequency: Optional[str] = None,
        max_stocks: int = 20,
    ) -> dict[str, Any]:
        """执行重分析

        Args:
            symbols: 指定股票列表，None则自动获取
            frequency: 分析频率 (daily/weekly/monthly)
            max_stocks: 最大分析数量

        Returns:
            分析结果统计
        """
        if symbols is None:
            if frequency:
                symbols = self.trend_service.get_stocks_for_reanalysis(frequency)
            else:
                # 按优先级合并所有频率
                daily = self.trend_service.get_stocks_for_reanalysis("daily")
                weekly = self.trend_service.get_stocks_for_reanalysis("weekly")
                monthly = self.trend_service.get_stocks_for_reanalysis("monthly")
                symbols = daily + weekly + monthly

        symbols = symbols[:max_stocks]

        if not symbols:
            logger.info("[ReanalysisService] 没有需要重分析的股票")
            return {"total": 0, "success": 0, "failed": 0, "results": []}

        logger.info(f"[ReanalysisService] 开始重分析 {len(symbols)} 只股票")

        results = []
        success = 0
        failed = 0

        for symbol in symbols:
            try:
                result = self._analyze_single_stock(symbol)
                results.append(result)
                if result.get("success"):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"[ReanalysisService] 分析 {symbol} 失败: {e}")
                results.append({"symbol": symbol, "success": False, "error": str(e)})
                failed += 1

        summary = {
            "total": len(symbols),
            "success": success,
            "failed": failed,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"[ReanalysisService] 重分析完成: {success}/{len(symbols)}")
        return summary

    def _analyze_single_stock(self, symbol: str) -> dict[str, Any]:
        """分析单只股票并更新趋势数据"""
        from src.services.analysis_service import analyze_stock

        try:
            # 执行分析
            result = analyze_stock(symbol)

            if not result or not result.get("fusion"):
                return {"symbol": symbol, "success": False, "error": "分析结果为空"}

            fusion = result["fusion"]

            # 记录评分快照
            self.trend_service.record_score_snapshot(
                symbol=symbol,
                final_score=fusion.final_score,
                final_action=fusion.final_action,
                confidence=fusion.confidence,
                agent_scores=fusion.signal_summary if hasattr(fusion, "signal_summary") else None,
                run_id=result.get("run_id"),
            )

            # 更新追踪股票统计
            self.trend_service.update_tracked_stock_stats(symbol)

            return {
                "symbol": symbol,
                "success": True,
                "final_score": fusion.final_score,
                "final_action": fusion.final_action,
                "confidence": fusion.confidence,
                "run_id": result.get("run_id"),
            }

        except Exception as e:
            return {"symbol": symbol, "success": False, "error": str(e)}

    def schedule_reanalysis_job(
        self,
        frequency: str = "weekly",
        trigger_time: str = "18:00",
    ) -> None:
        """注册定时重分析任务（用于APScheduler）"""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = BackgroundScheduler()

        hour, minute = trigger_time.split(":")

        # 根据频率设置触发器
        if frequency == "daily":
            trigger = CronTrigger(day_of_week="mon-fri", hour=int(hour), minute=int(minute))
        elif frequency == "weekly":
            trigger = CronTrigger(day_of_week="fri", hour=int(hour), minute=int(minute))
        elif frequency == "monthly":
            trigger = CronTrigger(day=1, hour=int(hour), minute=int(minute))
        else:
            trigger = CronTrigger(day_of_week="fri", hour=int(hour), minute=int(minute))

        job_id = f"reanalysis_{frequency}"

        scheduler.add_job(
            self.run_reanalysis,
            trigger=trigger,
            id=job_id,
            name=f"定期重分析({frequency})",
            kwargs={"frequency": frequency},
            misfire_grace_time=3600,
            replace_existing=True,
        )

        logger.info(f"[ReanalysisService] 注册定时任务: {job_id} @ {trigger_time}")


def run_scheduled_reanalysis(frequency: str = "weekly") -> dict[str, Any]:
    """定时任务入口函数"""
    service = ReanalysisService()
    return service.run_reanalysis(frequency=frequency)
