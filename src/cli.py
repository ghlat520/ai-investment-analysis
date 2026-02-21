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


def _collect_stock_data(
    symbol: str,
    market: str,
    research_dir: str | None = None,
    auto_research: bool = True,
):
    """采集单只股票的全部数据（委托给 service 层）"""
    from src.services.analysis_service import collect_stock_data
    return collect_stock_data(
        symbol, market, research_dir=research_dir, auto_research=auto_research,
    )


# ─── analyze ───────────────────────────────────────────────

@main.command()
@click.option("--stock", required=True, help="股票代码（如 000001.SZ）")
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--notify", "do_notify", is_flag=True, help="分析后推送通知")
@click.option("--research-dir", default=None, help="研报PDF目录路径（手动指定）")
@click.option("--auto-research/--no-auto-research", default=True, help="自动从东财采集最新研报（默认开启）")
def analyze(stock: str, market: str, do_notify: bool, research_dir: str | None, auto_research: bool) -> None:
    """分析单只股票"""
    from src.agents.graph import compile_analysis_graph

    logger.info(f"开始分析: {stock} (market={market})")
    if research_dir:
        logger.info(f"[研报] 手动研报目录: {research_dir}")
    elif auto_research:
        logger.info("[研报] 自动采集模式")
    t0 = time.time()

    stock_data = _collect_stock_data(stock, market, research_dir=research_dir, auto_research=auto_research)

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


# ─── growth-screen ────────────────────────────────────────

@main.command("growth-screen")
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--min-periods", default=3, help="最少连续增长期数")
@click.option("--top-n", default=50, help="输出Top N")
def growth_screen(market: str, min_periods: int, top_n: int) -> None:
    """连续季度增长筛选：找出连续多季度业绩同比增长的股票"""
    from src.market.adapter import AShareAdapter
    from src.screening.engine import ScreeningEngine

    logger.info(f"连续增长筛选: market={market} min_periods={min_periods} top_n={top_n}")
    t0 = time.time()

    adapter = AShareAdapter()
    source_manager = adapter.create_source_manager()
    engine = ScreeningEngine(source_manager)
    results = engine.run_growth_screen(market=market, min_periods=min_periods, top_n=top_n)

    elapsed = time.time() - t0

    if not results:
        click.echo("筛选无结果（无股票满足连续增长条件）")
        return

    click.echo(f"\n{'排名':>4} {'代码':<12} {'名称':<8} {'综合':>6} {'连续增长':>8} {'质量':>5} {'估值':>5}")
    click.echo("-" * 60)
    for r in results[:top_n]:
        fs = r.factor_scores
        click.echo(
            f"{r.rank:>4} {r.symbol:<12} {r.name:<8}"
            f" {r.composite_score:>6.1f}"
            f" {fs.get('consecutive_growth', 0):>8.1f}"
            f" {fs.get('quality', 0):>5.1f}"
            f" {fs.get('valuation', 0):>5.1f}"
        )

    click.echo(f"\n耗时: {elapsed:.1f}s | 共{len(results)}只")


# ─── batch ───────────────────────────────────────────────

@main.command()
@click.option("--stocks", default=None, help="逗号分隔的股票代码（如 300054.SZ,600150.SH）")
@click.option("--file", "file_path", default=None, help="股票列表文件（每行一个代码）")
@click.option("--resume", is_flag=True, help="断点续传（从上次中断处继续）")
@click.option("--output-dir", default=None, help="输出目录（默认 output/batch）")
@click.option("--research-dir", default=None, help="研报PDF目录")
@click.option("--cooldown", default=5, help="股间冷却秒数")
def batch(
    stocks: str | None,
    file_path: str | None,
    resume: bool,
    output_dir: str | None,
    research_dir: str | None,
    cooldown: int,
) -> None:
    """批量分析多只股票（串行执行，支持断点续传）"""
    from pathlib import Path
    from src.services.batch_service import batch_analyze, parse_stock_list

    if not stocks and not file_path:
        click.echo("错误: 必须指定 --stocks 或 --file")
        click.echo("示例: ai-invest batch --stocks 300054.SZ,600150.SH")
        click.echo("示例: ai-invest batch --file watchlist.txt")
        return

    symbols = parse_stock_list(stocks_str=stocks, file_path=file_path)
    if not symbols:
        click.echo("错误: 未找到有效的股票代码")
        return

    click.echo(f"=== 批量分析: {len(symbols)}只股票 ===")
    click.echo(f"股票列表: {', '.join(symbols)}")
    click.echo(f"预计耗时: ~{len(symbols) * 11}分钟")
    click.echo("")

    out_dir = Path(output_dir) if output_dir else None

    def on_done(symbol: str, idx: int, total: int, result: dict):
        score = result.get("score", 0)
        action = result.get("action", "")
        name = result.get("name", "")
        click.echo(f"  [{idx}/{total}] {name}({symbol}): {score:+d} {action}")

    t0 = time.time()
    result = batch_analyze(
        symbols=symbols,
        output_dir=out_dir,
        resume=resume,
        research_dir=research_dir,
        on_stock_done=on_done,
        cooldown=cooldown,
    )
    elapsed = time.time() - t0

    click.echo(f"\n{'='*60}")
    click.echo(result["summary"])
    click.echo(f"{'='*60}")
    click.echo(f"\n总耗时: {elapsed/60:.1f}分钟")

    out_path = out_dir or Path("output/batch")
    click.echo(f"输出目录: {out_path}")
    click.echo(f"汇总报告: {out_path}/summary.md")


# ─── pipeline ────────────────────────────────────────────

@main.command()
@click.option("--hotspot", "use_hotspot", is_flag=True, help="从盘前热点选股")
@click.option("--screen", "use_screen", is_flag=True, help="从量化筛选选股")
@click.option("--top-n", default=5, help="选取Top N进行深度分析")
@click.option("--output-dir", default=None, help="输出目录")
def pipeline(use_hotspot: bool, use_screen: bool, top_n: int, output_dir: str | None) -> None:
    """端到端投研流水线: 选股 → 批量深度分析 → 汇总报告"""
    from pathlib import Path
    from src.services.batch_service import batch_analyze

    if not use_hotspot and not use_screen:
        click.echo("错误: 必须指定 --hotspot 或 --screen")
        click.echo("示例: ai-invest pipeline --hotspot --top-n 5")
        return

    symbols: list[str] = []

    if use_hotspot:
        click.echo("=== Phase 1: 盘前热点分析 ===")
        from src.services.hotspot_service import run_hotspot_streaming

        result = run_hotspot_streaming()
        # 从分析结果中提取推荐股票代码
        analyzed_themes = result.get("analyzed_themes", [])
        for theme in analyzed_themes:
            if hasattr(theme, "key_stocks"):
                for stock in theme.key_stocks:
                    code = getattr(stock, "code", "") or stock.get("code", "")
                    if code and code not in symbols:
                        symbols.append(code)
        click.echo(f"  热点选出 {len(symbols)} 只候选股票")

    if use_screen:
        click.echo("=== Phase 1: 量化筛选 ===")
        from src.market.adapter import AShareAdapter
        from src.screening.engine import ScreeningEngine

        adapter = AShareAdapter()
        source_manager = adapter.create_source_manager()
        engine = ScreeningEngine(source_manager)
        results = engine.run(market="A")
        symbols = [r.symbol for r in results[:top_n]]
        click.echo(f"  筛选出 Top {len(symbols)} 只股票")

    if not symbols:
        click.echo("未选出有效股票，流水线结束")
        return

    symbols = symbols[:top_n]
    click.echo(f"\n=== Phase 2: 批量深度分析 ({len(symbols)}只) ===")
    click.echo(f"股票: {', '.join(symbols)}")

    out_dir = Path(output_dir) if output_dir else None

    def on_done(symbol: str, idx: int, total: int, result: dict):
        click.echo(
            f"  [{idx}/{total}] {result.get('name', '')}({symbol}): "
            f"{result.get('score', 0):+d} {result.get('action', '')}"
        )

    t0 = time.time()
    result = batch_analyze(symbols=symbols, output_dir=out_dir, on_stock_done=on_done)
    elapsed = time.time() - t0

    click.echo(f"\n{'='*60}")
    click.echo(result["summary"])
    click.echo(f"{'='*60}")
    click.echo(f"\n总耗时: {elapsed/60:.1f}分钟")


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


# ─── validate ──────────────────────────────────────────────

@main.command()
@click.option("--days", default=30, help="验证最近N天到期的预测")
@click.option("--stock", default=None, help="只验证特定股票")
@click.option("--auto-verify", is_flag=True, help="自动获取最新价格并验证")
def validate(days: int, stock: str | None, auto_verify: bool) -> None:
    """预测验证：检查待验证的预测并执行验证"""
    from src.services.validation_service import get_validation_service

    svc = get_validation_service()

    # 1. 显示待验证的预测
    pending = svc.get_pending_verifications(days_overdue=days)

    if not pending:
        click.echo(f"最近{days}天内无待验证的预测记录")
        return

    click.echo(f"\n=== 待验证预测 ({len(pending)}条) ===")
    click.echo(f"{'ID':>5} {'代码':<12} {'预测日':<12} {'验证日':<12} {'评分':>6} {'操作':<8}")
    click.echo("-" * 60)

    for r in pending:
        click.echo(
            f"{r.id:>5} {r.symbol:<12} {str(r.prediction_date):<12} "
            f"{str(r.verification_date):<12} {r.final_score:>+6d} {r.final_action:<8}"
        )

    if not auto_verify:
        click.echo(f"\n提示: 使用 --auto-verify 自动获取最新价格并验证")
        return

    # 2. 自动验证
    click.echo(f"\n=== 开始自动验证 ===")

    from src.data.sources.manager import DataSourceManager
    from src.data.collectors.quote_collector import QuoteCollector

    source_mgr = DataSourceManager()
    quote_collector = QuoteCollector(source_mgr)

    verified_count = 0
    for r in pending:
        try:
            # 获取最新价格
            quotes = quote_collector.collect_daily_quotes(r.symbol, days=5)
            if not quotes:
                click.echo(f"  [{r.symbol}] 无法获取最新价格，跳过")
                continue

            latest_price = float(quotes[-1].get("close", 0))
            if latest_price <= 0:
                click.echo(f"  [{r.symbol}] 价格数据异常，跳过")
                continue

            # 执行验证
            result = svc.verify_prediction(r.id, latest_price)

            direction_icon = "✓" if result["direction_correct"] else "✗"
            click.echo(
                f"  [{r.symbol}] 预测{r.final_score:+d} → 实际{return['actual_return_pct']:+.2f}% "
                f"方向:{direction_icon} 目标价:{result['price_target_hit']}"
            )
            verified_count += 1

        except Exception as e:
            click.echo(f"  [{r.symbol}] 验证失败: {e}")

    click.echo(f"\n验证完成: {verified_count}/{len(pending)}")


@main.command("validation-stats")
@click.option("--days", default=90, help="统计最近N天的验证结果")
@click.option("--stock", default=None, help="只统计特定股票")
@click.option("--agent", default=None, help="只统计特定Agent")
def validation_stats(days: int, stock: str | None, agent: str | None) -> None:
    """预测准确性统计：分析历史验证结果"""
    from src.services.validation_service import get_validation_service

    svc = get_validation_service()
    stats = svc.get_accuracy_statistics(symbol=stock, days=days, agent_name=agent)

    if stats.get("total", 0) == 0:
        click.echo("无已验证的预测记录")
        return

    click.echo(f"\n=== 预测准确性统计 (最近{days}天) ===")
    click.echo(f"总验证数: {stats['total']}")
    click.echo(f"方向准确率: {stats['direction_accuracy']:.1f}%")
    click.echo(f"平均收益率: {stats['avg_return']:+.2f}%")

    # 按评分区间统计
    click.echo(f"\n=== 按评分区间统计 ===")
    click.echo(f"{'区间':<20} {'数量':>6} {'准确率':>8} {'平均收益':>10}")
    click.echo("-" * 50)

    for bin_name, bin_data in stats["score_bins"].items():
        if bin_data["count"] > 0:
            click.echo(
                f"{bin_name:<20} {bin_data['count']:>6} "
                f"{bin_data.get('accuracy', 0):>7.1f}% "
                f"{bin_data['avg_return']:>+9.2f}%"
            )

    # Agent统计
    if stats.get("agent_stats"):
        click.echo(f"\n=== Agent准确性统计 ===")
        click.echo(f"{'Agent':<20} {'验证数':>6} {'准确率':>8} {'贡献度':>10}")
        click.echo("-" * 50)

        for an, an_data in sorted(
            stats["agent_stats"].items(),
            key=lambda x: x[1].get("accuracy", 0),
            reverse=True
        ):
            click.echo(
                f"{an:<20} {an_data['count']:>6} "
                f"{an_data.get('accuracy', 0):>7.1f}% "
                f"{an_data['total_contribution']:>+9.1f}"
            )


@main.command("error-analysis")
@click.option("--record-id", required=True, type=int, help="预测记录ID")
def error_analysis(record_id: int) -> None:
    """错误分析：分析单个预测错误的原因"""
    from src.services.validation_service import get_validation_service

    svc = get_validation_service()
    report = svc.generate_error_report(record_id)

    if "error" in report:
        click.echo(f"错误: {report['error']}")
        return

    if report.get("status"):
        click.echo(report["status"])
        return

    click.echo(f"\n=== 错误分析报告 ===")
    click.echo(f"股票: {report['symbol']}")
    click.echo(f"预测日期: {report['prediction_date']}")
    click.echo(f"预测评分: {report['final_score']:+d}")
    click.echo(f"实际收益: {report['actual_return_pct']:+.2f}%")
    click.echo(f"错误类型: {report['error_type']}")

    if report.get("wrong_agents"):
        click.echo(f"\n判断错误的Agent: {', '.join(report['wrong_agents'])}")

    if report.get("suggested_fixes"):
        click.echo(f"\n=== 建议修复措施 ===")
        for fix in report["suggested_fixes"]:
            click.echo(f"  - {fix}")


if __name__ == "__main__":
    main()
