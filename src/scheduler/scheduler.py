"""
分析调度器

每日收盘后自动执行：量化筛选 → AI深度分析 → 研报入库 → 通知推送。
工作日（周一至周五）定时触发，支持手动立即执行。
"""

from __future__ import annotations

import os
import signal
import time
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger


class AnalysisScheduler:
    """每日分析调度器"""

    def __init__(
        self,
        trigger_time: str = "17:30",
        timezone: str = "Asia/Shanghai",
        market: str = "A",
        top_n: int = 10,
    ) -> None:
        self._scheduler = BlockingScheduler(timezone=timezone)
        self._trigger_time = trigger_time
        self._market = market
        self._top_n = top_n
        self._running = False

        hour, minute = trigger_time.split(":")
        self._scheduler.add_job(
            self._run_daily_pipeline,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=int(hour),
                minute=int(minute),
            ),
            id="daily_analysis",
            name="每日分析流水线",
            misfire_grace_time=3600,  # 错过1小时内仍执行
        )

        # 盘前热点分析（每个交易日 08:30）
        self._scheduler.add_job(
            self._run_hotspot_pipeline,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=8,
                minute=30,
            ),
            id="hotspot_analysis",
            name="盘前热点分析",
            misfire_grace_time=3600,
        )

        logger.info(f"调度器初始化: 每周一至周五 {trigger_time} ({timezone}), top_n={top_n}")

    @classmethod
    def from_env(cls) -> "AnalysisScheduler":
        """从环境变量创建调度器"""
        return cls(
            trigger_time=os.environ.get("ANALYSIS_TRIGGER_TIME", "17:30"),
            timezone=os.environ.get("TIMEZONE", "Asia/Shanghai"),
            market=os.environ.get("ANALYSIS_MARKET", "A"),
            top_n=int(os.environ.get("ANALYSIS_TOP_N", "10")),
        )

    def _run_daily_pipeline(self) -> None:
        """执行每日分析流水线"""
        if self._running:
            logger.warning("上一次流水线仍在运行，跳过本次")
            return

        self._running = True
        start = time.time()
        logger.info("=" * 60)
        logger.info(f"开始每日分析流水线 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        logger.info("=" * 60)

        try:
            results = run_pipeline(market=self._market, top_n=self._top_n)

            # 通知推送
            if results:
                _send_notification(results)

            elapsed = time.time() - start
            logger.info(f"每日分析流水线完成, 耗时{elapsed:.1f}s, 分析{len(results)}只股票")
        except Exception as e:
            logger.error(f"流水线执行失败: {e}", exc_info=True)
        finally:
            self._running = False

    def _run_hotspot_pipeline(self) -> None:
        """执行盘前热点分析"""
        start = time.time()
        logger.info("=" * 60)
        logger.info(f"开始盘前热点分析 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        logger.info("=" * 60)

        try:
            from src.services.hotspot_service import run_hotspot

            result = run_hotspot()
            elapsed = time.time() - start
            themes = result.get("themes", [])
            logger.info(f"盘前热点分析完成, 耗时{elapsed:.1f}s, {len(themes)}个主题")

            # 通知推送
            briefing = result.get("briefing", "")
            if briefing:
                _send_notification([{
                    "symbol": "HOTSPOT",
                    "name": "盘前热点",
                    "fusion": type("F", (), {
                        "final_score": 0,
                        "final_action": "盘前纪要",
                        "confidence": 1.0,
                        "position_pct": 0,
                    })(),
                    "report": briefing,
                    "signals": [],
                    "screening_rank": 0,
                    "screening_score": 0,
                }])
        except Exception as e:
            logger.error(f"盘前热点分析失败: {e}", exc_info=True)

    def run_now(self) -> None:
        """立即执行一次（不等待调度时间）"""
        logger.info("手动触发分析流水线")
        self._run_daily_pipeline()

    def start(self) -> None:
        """启动调度器（阻塞）"""
        # 注册信号处理
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_signal)

        jobs = self._scheduler.get_jobs()
        if jobs:
            logger.info(f"调度器启动, 任务数: {len(jobs)}")
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("调度器收到退出信号")

    def _handle_signal(self, signum: int, frame) -> None:
        logger.info(f"收到信号 {signum}, 优雅关闭调度器...")
        self._scheduler.shutdown(wait=False)

    def stop(self) -> None:
        """停止调度器"""
        self._scheduler.shutdown()
        logger.info("调度器已停止")


def run_pipeline(market: str = "A", top_n: int = 10) -> list[dict]:
    """执行完整分析流水线: 筛选 → 分析 → 入库

    返回: [{symbol, name, fusion, report, signals}, ...]
    """
    from src.agents.graph import compile_analysis_graph
    from src.data.storage.persist import persist_analysis
    from src.market.adapter import AShareAdapter
    from src.screening.engine import ScreeningEngine

    # Step 1: 量化筛选
    logger.info("[1/3] 量化筛选...")
    adapter = AShareAdapter()
    source_manager = adapter.create_source_manager()
    engine = ScreeningEngine(source_manager)
    candidates = engine.run(market=market)

    if not candidates:
        logger.warning("筛选无结果")
        return []

    top_candidates = candidates[:top_n]
    logger.info(f"筛选完成: {len(candidates)}只 → Top {len(top_candidates)}")

    # Step 2: AI深度分析
    logger.info("[2/3] AI深度分析...")
    from src.cli import _collect_stock_data

    graph = compile_analysis_graph()
    results = []

    for i, c in enumerate(top_candidates, 1):
        logger.info(f"  [{i}/{len(top_candidates)}] {c.name}({c.symbol})")
        try:
            stock_data = _collect_stock_data(c.symbol, market)
            result = graph.invoke({"stock": stock_data})

            fusion = result.get("fusion")
            report = result.get("report", "")

            if fusion:
                results.append({
                    "symbol": c.symbol,
                    "name": c.name,
                    "fusion": fusion,
                    "report": report,
                    "signals": result.get("signals", []),
                    "screening_rank": c.rank,
                    "screening_score": c.composite_score,
                })
                logger.info(
                    f"    → {fusion.final_score:+d} {fusion.final_action} "
                    f"(confidence={fusion.confidence:.0%})"
                )

            # 入库
            try:
                persist_analysis(result)
            except Exception as e:
                logger.warning(f"    持久化失败: {e}")

        except Exception as e:
            logger.error(f"    分析失败: {e}")

    logger.info(f"[3/3] 分析完成: {len(results)}/{len(top_candidates)}只成功")
    return results


def _send_notification(results: list[dict]) -> None:
    """汇总分析结果并推送通知"""
    from src.notification.manager import NotificationManager

    mgr = NotificationManager.from_env()
    if not mgr.has_channels:
        logger.info("未配置通知渠道，跳过推送")
        return

    title = f"AI投研日报 ({datetime.now().strftime('%m-%d')})"
    content = _build_daily_summary(results)

    send_results = mgr.send(title, content)
    for ch, ok in send_results.items():
        if ok:
            logger.info(f"通知推送成功: {ch}")
        else:
            logger.warning(f"通知推送失败: {ch}")


def _build_daily_summary(results: list[dict]) -> str:
    """构建每日摘要消息"""
    if not results:
        return "今日无分析结果"

    lines = [f"共分析 **{len(results)}** 只股票\n"]

    # 按评分排序
    sorted_results = sorted(results, key=lambda r: r["fusion"].final_score, reverse=True)

    # 看多信号
    bullish = [r for r in sorted_results if r["fusion"].final_score >= 30]
    if bullish:
        lines.append("### 看多信号")
        for r in bullish:
            f = r["fusion"]
            lines.append(
                f"- **{r['name']}**({r['symbol']}): "
                f"{f.final_score:+d}分 | {f.final_action} | "
                f"仓位{f.position_pct}% | 置信{f.confidence:.0%}"
            )
        lines.append("")

    # 中性
    neutral = [r for r in sorted_results if -30 < r["fusion"].final_score < 30]
    if neutral:
        lines.append("### 中性/观望")
        for r in neutral:
            f = r["fusion"]
            lines.append(f"- {r['name']}({r['symbol']}): {f.final_score:+d}分 {f.final_action}")
        lines.append("")

    # 看空
    bearish = [r for r in sorted_results if r["fusion"].final_score <= -30]
    if bearish:
        lines.append("### 看空信号")
        for r in bearish:
            f = r["fusion"]
            lines.append(f"- {r['name']}({r['symbol']}): {f.final_score:+d}分 {f.final_action}")
        lines.append("")

    # LLM成本
    total_tokens = sum(
        sum(s.llm_tokens_used for s in r.get("signals", []))
        for r in results
    )
    total_cost = sum(
        sum(s.llm_cost_usd for s in r.get("signals", []))
        for r in results
    )
    if total_tokens > 0:
        lines.append(f"*LLM: {total_tokens} tokens, ${total_cost:.4f}*")

    return "\n".join(lines)
