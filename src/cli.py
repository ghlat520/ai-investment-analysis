"""
CLI 入口

命令：
  ai-invest analyze --stock 000001.SZ    分析单只股票
  ai-invest screen                       量化筛选
  ai-invest run                          运行完整流水线
  ai-invest cost                         查看LLM成本
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
    """采集单只股票的全部数据"""
    from src.agents.state import StockData
    from src.data.sources import DataSourceManager
    from src.data.sources.akshare_source import AKShareSource
    from src.data.sources.efinance_source import EFinanceSource
    from src.data.sources.yfinance_source import YFinanceSource

    source_mgr = DataSourceManager()
    source_mgr.register(EFinanceSource())
    source_mgr.register(AKShareSource())
    source_mgr.register(YFinanceSource())

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365)).isoformat()
    stock_name = symbol  # 默认用代码

    # 1. 行情数据
    logger.info(f"[采集] 行情数据...")
    quotes_data = []
    try:
        quotes = source_mgr.fetch_daily_quotes(symbol, start_date, end_date)
        if not quotes.empty:
            # 尝试从行情数据中获取股票名称
            for col in ["股票名称", "name"]:
                if col in quotes.columns and quotes[col].iloc[0]:
                    stock_name = str(quotes[col].iloc[0])
                    break
            quotes_data = quotes.to_dict("records")
            logger.info(f"[采集] 行情: {len(quotes_data)}条 ({stock_name})")
    except Exception as e:
        logger.error(f"[采集] 行情失败: {e}")

    # 2. 财务数据
    logger.info(f"[采集] 财务数据...")
    financial_data = []
    try:
        financial = source_mgr.fetch_financial_data(symbol)
        if not financial.empty:
            financial_data = financial.to_dict("records")
            logger.info(f"[采集] 财务: {len(financial_data)}期")
    except Exception as e:
        logger.warning(f"[采集] 财务数据失败: {e}")

    # 3. 估值数据（PE/PB历史）
    logger.info(f"[采集] 估值数据...")
    valuation_data = []
    try:
        valuation = source_mgr.fetch_valuation(symbol)
        if not valuation.empty:
            valuation_data = valuation.to_dict("records")
            logger.info(f"[采集] 估值: {len(valuation_data)}天")
    except Exception as e:
        logger.warning(f"[采集] 估值数据失败: {e}")

    # 合并估值数据到财务数据中：取最新PE/PB附加到每期财报
    if valuation_data and financial_data:
        # 取估值序列的最新值作为当前估值
        latest_val = valuation_data[-1]
        for rec in financial_data:
            if "pe_ttm" not in rec or rec.get("pe_ttm") is None:
                rec["pe_ttm"] = latest_val.get("pe_ttm")
            if "pb" not in rec or rec.get("pb") is None:
                rec["pb"] = latest_val.get("pb")
            if "total_market_cap" not in rec or rec.get("total_market_cap") is None:
                rec["total_market_cap"] = latest_val.get("total_market_cap")

    # 如果有估值历史但没财务数据，构造一条最小记录供估值Agent使用
    if valuation_data and not financial_data:
        latest_val = valuation_data[-1]
        financial_data = [{
            "report_date": date.today(),
            "pe_ttm": latest_val.get("pe_ttm"),
            "pb": latest_val.get("pb"),
            "total_market_cap": latest_val.get("total_market_cap"),
        }]

    # 构建 StockData
    stock_data = StockData(
        symbol=symbol,
        name=stock_name,
        market=market,
        daily_quotes=quotes_data,
        financial_data=financial_data,
        info={
            "valuation_history": valuation_data,  # 传递完整历史给估值Agent
        },
    )
    return stock_data


@main.command()
@click.option("--stock", required=True, help="股票代码（如 000001.SZ）")
@click.option("--market", default="A", help="市场（A/HK/US）")
def analyze(stock: str, market: str) -> None:
    """分析单只股票"""
    from src.agents.graph import compile_analysis_graph

    logger.info(f"开始分析: {stock} (market={market})")
    t0 = time.time()

    # 数据采集
    stock_data = _collect_stock_data(stock, market)

    # 运行分析图
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
    click.echo(f"\n耗时: {elapsed:.1f}s | Agent数: {len(signals)}")
    for s in signals:
        click.echo(f"  {s.agent_name}: {s.signal_score:+d} (confidence={s.confidence:.0%}, {s.execution_time_ms}ms)")

    errors = result.get("errors", [])
    if errors:
        click.echo(f"\n错误: {errors}")

    # 持久化到数据库
    try:
        from src.data.storage.persist import persist_analysis
        run_id = persist_analysis(result)
        click.echo(f"\n结果已保存 (run_id={run_id[:8]}...)")
    except Exception as e:
        logger.warning(f"持久化失败: {e}")


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


@main.command("init-db")
def init_db() -> None:
    """初始化数据库（创建表结构）"""
    from src.data.storage.database import get_database

    db = get_database()
    db.create_tables()
    click.echo("数据库初始化完成")


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


@main.command()
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--top-n", default=5, help="筛选Top N后深度分析")
def run(market: str, top_n: int) -> None:
    """运行完整流水线：筛选 → AI深度分析"""
    from src.agents.graph import compile_analysis_graph
    from src.data.storage.persist import persist_analysis
    from src.market.adapter import AShareAdapter
    from src.screening.engine import ScreeningEngine

    t0 = time.time()

    # Phase 1: 量化筛选
    click.echo(f"=== Phase 1: 量化筛选 ===")
    adapter = AShareAdapter()
    source_manager = adapter.create_source_manager()
    engine = ScreeningEngine(source_manager)
    candidates = engine.run(market=market)

    if not candidates:
        click.echo("筛选无结果")
        return

    top_candidates = candidates[:top_n]
    click.echo(f"筛选完成: {len(candidates)}只候选, 取Top {len(top_candidates)}深度分析\n")

    # Phase 2: AI深度分析
    click.echo(f"=== Phase 2: AI深度分析 ===")
    graph = compile_analysis_graph()

    for i, c in enumerate(top_candidates, 1):
        click.echo(f"\n[{i}/{len(top_candidates)}] {c.name}({c.symbol})...")
        try:
            stock_data = _collect_stock_data(c.symbol, market)
            result = graph.invoke({"stock": stock_data})

            fusion = result.get("fusion")
            if fusion:
                click.echo(
                    f"  评分: {fusion.final_score:+d} | "
                    f"建议: {fusion.final_action} | "
                    f"置信度: {fusion.confidence:.0%}"
                )

            # 持久化
            try:
                persist_analysis(result)
            except Exception as e:
                logger.warning(f"持久化失败: {e}")

        except Exception as e:
            click.echo(f"  分析失败: {e}")

    elapsed = time.time() - t0
    click.echo(f"\n=== 完成 ===")
    click.echo(f"耗时: {elapsed:.1f}s | 筛选: {len(candidates)}只 → 深度分析: {len(top_candidates)}只")


if __name__ == "__main__":
    main()
