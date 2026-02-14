"""
CLI 入口

命令：
  ai-invest analyze --stock 000001.SZ    分析单只股票
  ai-invest screen                       量化筛选
  ai-invest run                          运行完整流水线（筛选→分析→通知）
  ai-invest scheduler                    启动每日调度器
  ai-invest notify --test                测试通知渠道
  ai-invest cost                         查看LLM成本
  ai-invest init-db                      初始化数据库
  ai-invest history                      查看历史记录
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import click
from loguru import logger


def _setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, format="{time:HH:mm:ss} | {level:<7} | {message}")


@click.group()
@click.option("--debug", is_flag=True, help="启用DEBUG日志")
def main(debug: bool) -> None:
    """AI投研助手系统 CLI"""
    _setup_logging("DEBUG" if debug else "INFO")


def _collect_stock_data(symbol: str, market: str):
    """采集单只股票的全部数据（委托给 service 层）"""
    from src.services.analysis_service import collect_stock_data
    return collect_stock_data(symbol, market)


# ─── analyze ───────────────────────────────────────────────

@main.command()
@click.option("--stock", required=True, help="股票代码（如 000001.SZ）")
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--notify", "do_notify", is_flag=True, help="分析后推送通知")
def analyze(stock: str, market: str, do_notify: bool) -> None:
    """分析单只股票"""
    from src.agents.graph import compile_analysis_graph

    logger.info(f"开始分析: {stock} (market={market})")
    t0 = time.time()

    stock_data = _collect_stock_data(stock, market)

    logger.info(f"[分析] 启动 LangGraph 分析流水线...")
    graph = compile_analysis_graph()
    result = graph.invoke({"stock": stock_data})

    elapsed = time.time() - t0

    # 输出报告
    report = result.get("report", "无报告生成")
    click.echo("\n" + "=" * 60)
    click.echo(report)
    click.echo("=" * 60)

    # 统计信息
    signals = result.get("signals", [])
    total_llm_cost = sum(s.llm_cost_usd for s in signals)
    total_llm_tokens = sum(s.llm_tokens_used for s in signals)
    click.echo(f"\n耗时: {elapsed:.1f}s | Agent数: {len(signals)}")
    if total_llm_tokens > 0:
        click.echo(f"LLM: {total_llm_tokens} tokens, ${total_llm_cost:.4f}")
    for s in signals:
        llm_tag = f" [{s.llm_model}]" if s.llm_model else " [code-only]"
        click.echo(f"  {s.agent_name}: {s.signal_score:+d} (confidence={s.confidence:.0%}, {s.execution_time_ms}ms){llm_tag}")

    errors = result.get("errors", [])
    if errors:
        click.echo(f"\n错误: {errors}")

    # 持久化
    try:
        from src.data.storage.persist import persist_analysis
        run_id = persist_analysis(result)
        click.echo(f"\n结果已保存 (run_id={run_id[:8]}...)")
    except Exception as e:
        logger.warning(f"持久化失败: {e}")

    # 通知推送
    if do_notify:
        _notify_single(result)


def _notify_single(result: dict) -> None:
    """推送单只股票分析结果"""
    from src.notification.manager import NotificationManager

    mgr = NotificationManager.from_env()
    if not mgr.has_channels:
        click.echo("未配置通知渠道（设置 .env 中的 WECHAT_WEBHOOK_URL 等）")
        return

    stock = result.get("stock")
    fusion = result.get("fusion")
    if not stock or not fusion:
        return

    title = f"{stock.name}({stock.symbol}) {fusion.final_action} ({fusion.final_score:+d})"
    content = result.get("report", fusion.reasoning)
    send_results = mgr.send(title, content)
    for ch, ok in send_results.items():
        click.echo(f"  通知[{ch}]: {'成功' if ok else '失败'}")


# ─── screen ────────────────────────────────────────────────

@main.command()
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--top-n", default=50, help="输出Top N")
def screen(market: str, top_n: int) -> None:
    """量化筛选全市场"""
    from src.market.adapter import AShareAdapter
    from src.screening.engine import ScreeningEngine

    logger.info(f"开始量化筛选: market={market} top_n={top_n}")
    t0 = time.time()

    adapter = AShareAdapter()
    source_manager = adapter.create_source_manager()
    engine = ScreeningEngine(source_manager)
    results = engine.run(market=market)

    elapsed = time.time() - t0

    if not results:
        click.echo("筛选无结果")
        return

    click.echo(f"\n{'排名':>4} {'代码':<12} {'名称':<8} {'行业':<10} {'综合':>6} {'质量':>5} {'成长':>5} {'估值':>5} {'动量':>5}")
    click.echo("-" * 72)
    for r in results[:top_n]:
        fs = r.factor_scores
        industry = (r.industry or "")[:8]
        click.echo(
            f"{r.rank:>4} {r.symbol:<12} {r.name:<8} {industry:<10}"
            f" {r.composite_score:>6.1f}"
            f" {fs.get('quality', 0):>5.1f}"
            f" {fs.get('growth', 0):>5.1f}"
            f" {fs.get('valuation', 0):>5.1f}"
            f" {fs.get('momentum', 0):>5.1f}"
        )

    click.echo(f"\n耗时: {elapsed:.1f}s | 共{len(results)}只")


# ─── run ───────────────────────────────────────────────────

@main.command()
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--top-n", default=5, help="筛选Top N后深度分析")
@click.option("--notify", "do_notify", is_flag=True, help="完成后推送通知")
def run(market: str, top_n: int, do_notify: bool) -> None:
    """运行完整流水线：筛选 → AI深度分析 → 通知"""
    from src.scheduler.scheduler import run_pipeline, _send_notification

    t0 = time.time()
    click.echo(f"=== 完整流水线: 筛选 → Top {top_n} 深度分析 ===\n")

    results = run_pipeline(market=market, top_n=top_n)

    elapsed = time.time() - t0
    click.echo(f"\n=== 完成 ===")
    click.echo(f"耗时: {elapsed:.1f}s | 分析: {len(results)}只")

    if results:
        click.echo(f"\n{'排名':>4} {'代码':<12} {'名称':<8} {'评分':>6} {'操作':<8} {'置信度':>6}")
        click.echo("-" * 50)
        for r in results:
            f = r["fusion"]
            click.echo(f"{r['screening_rank']:>4} {r['symbol']:<12} {r['name']:<8} {f.final_score:>+6d} {f.final_action:<8} {f.confidence:>6.0%}")

    # 通知
    if do_notify and results:
        click.echo("\n推送通知...")
        _send_notification(results)


# ─── scheduler ─────────────────────────────────────────────

@main.command()
@click.option("--time", "trigger_time", default="17:30", help="每日触发时间 (HH:MM)")
@click.option("--top-n", default=10, help="每日分析Top N")
@click.option("--run-now", is_flag=True, help="启动后立即执行一次")
def scheduler(trigger_time: str, top_n: int, run_now: bool) -> None:
    """启动每日调度器（周一至周五定时运行）"""
    from src.scheduler.scheduler import AnalysisScheduler

    sched = AnalysisScheduler(trigger_time=trigger_time, top_n=top_n)

    if run_now:
        sched.run_now()

    click.echo(f"调度器运行中... 每周一至周五 {trigger_time} 自动执行 (Ctrl+C 退出)")
    sched.start()


# ─── notify ────────────────────────────────────────────────

@main.command()
@click.option("--test", "do_test", is_flag=True, help="发送测试消息")
def notify(do_test: bool) -> None:
    """查看/测试通知渠道配置"""
    from src.notification.manager import NotificationManager

    mgr = NotificationManager.from_env()

    if not mgr.has_channels:
        click.echo("未配置任何通知渠道")
        click.echo("请在 .env 中配置以下任一变量：")
        click.echo("  WECHAT_WEBHOOK_URL  — 企业微信")
        click.echo("  FEISHU_WEBHOOK_URL  — 飞书")
        click.echo("  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID — Telegram")
        click.echo("  EMAIL_SENDER + EMAIL_PASSWORD + EMAIL_RECEIVERS — 邮件")
        return

    click.echo(f"已配置渠道: {', '.join(mgr.channel_names)}")

    if do_test:
        click.echo("发送测试消息...")
        results = mgr.send(
            "AI投研助手 测试通知",
            "这是一条测试消息。\n\n如果您收到此消息，说明通知渠道配置正确。"
        )
        for ch, ok in results.items():
            click.echo(f"  {ch}: {'成功' if ok else '失败'}")


# ─── cost ──────────────────────────────────────────────────

@main.command()
def cost() -> None:
    """查看LLM成本统计"""
    from src.llm.router import get_llm_router

    router = get_llm_router()
    stats = router.stats
    click.echo(f"LLM调用统计:")
    click.echo(f"  总成本: ${stats['total_cost_usd']:.4f}")
    click.echo(f"  总Token: {stats['total_tokens']}")
    click.echo(f"  调用次数: {stats['call_count']}")


# ─── init-db ───────────────────────────────────────────────

@main.command("init-db")
def init_db() -> None:
    """初始化数据库（创建表结构）"""
    from src.data.storage.database import get_database

    db = get_database()
    db.create_tables()
    click.echo("数据库初始化完成")


# ─── history ───────────────────────────────────────────────

@main.command()
@click.option("--stock", default=None, help="按股票代码筛选")
@click.option("--limit", "n", default=10, help="显示最近N条")
def history(stock: str | None, n: int) -> None:
    """查看历史分析记录"""
    from src.data.storage.database import get_database
    from src.data.storage.models import FusionDecisionRecord

    db = get_database()
    db.create_tables()

    with db.session() as session:
        query = session.query(FusionDecisionRecord).order_by(
            FusionDecisionRecord.created_at.desc()
        )
        if stock:
            query = query.filter_by(symbol=stock)
        records = query.limit(n).all()

    if not records:
        click.echo("暂无分析记录")
        return

    click.echo(f"\n{'日期':<20} {'代码':<12} {'评分':>6} {'操作':<8} {'置信度':>6}")
    click.echo("-" * 56)
    for r in records:
        dt = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "N/A"
        click.echo(f"{dt:<20} {r.symbol:<12} {r.final_score:>+6d} {r.final_action:<8} {r.confidence:>6.0%}")


# ─── hotspot ──────────────────────────────────────────────

@main.command()
@click.option("--output", "-o", default=None, help="输出研报到文件")
def hotspot(output: str | None) -> None:
    """盘前热点分析：热点新闻 → 产业链 → 核心标的"""
    import time as _time

    logger.info("开始盘前热点分析...")
    t0 = _time.time()

    from src.services.hotspot_service import run_hotspot_streaming

    def on_stage(node_name: str, node_output: dict):
        if node_name == "collect_and_rank":
            concepts = node_output.get("ranked_concepts", [])
            click.echo(f"  数据采集完成，热门概念: {len(concepts)}个")
        elif node_name == "extract_themes":
            themes = node_output.get("themes", [])
            click.echo(f"  主题提取完成: {len(themes)}个")
            for t in themes:
                click.echo(f"    - {t.title} (相关度:{t.relevance_score})")
        elif node_name == "analyze_themes_batch":
            analyzed = node_output.get("analyzed_themes", [])
            click.echo(f"  产业链分析完成: {len(analyzed)}个主题")
        elif node_name == "generate_briefing":
            click.echo("  研报生成完成")

    result = run_hotspot_streaming(on_stage_done=on_stage)

    elapsed = _time.time() - t0

    # 输出研报
    briefing = result.get("briefing", "无研报生成")
    click.echo("\n" + "=" * 60)
    click.echo(briefing)
    click.echo("=" * 60)
    click.echo(f"\n耗时: {elapsed:.1f}s")

    errors = result.get("errors", [])
    if errors:
        click.echo(f"错误: {errors}")

    # 保存到文件
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(briefing)
        click.echo(f"研报已保存: {output}")


# ─── serve ────────────────────────────────────────────────

@main.command()
@click.option("--host", default="0.0.0.0", help="监听地址")
@click.option("--port", default=8000, help="监听端口")
@click.option("--reload", "do_reload", is_flag=True, help="开发模式（自动重载）")
def serve(host: str, port: int, do_reload: bool) -> None:
    """启动 Web API 服务"""
    import uvicorn

    click.echo(f"启动 API 服务: http://{host}:{port}")
    click.echo("API 文档: http://localhost:{}/docs".format(port))
    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=do_reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
