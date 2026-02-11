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


@main.command()
@click.option("--stock", required=True, help="股票代码（如 000001.SZ）")
@click.option("--market", default="A", help="市场（A/HK/US）")
def analyze(stock: str, market: str) -> None:
    """分析单只股票"""
    from src.agents.graph import compile_analysis_graph
    from src.agents.state import AnalysisState, StockData
    from src.market.adapter import AShareAdapter

    logger.info(f"开始分析: {stock} (market={market})")

    # 创建数据源
    adapter = AShareAdapter()
    source_manager = adapter.create_source_manager()

    # 采集数据
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365)).isoformat()

    try:
        quotes = source_manager.fetch_daily_quotes(stock, start_date, end_date)
        quotes_data = quotes.to_dict("records") if not quotes.empty else []
    except Exception as e:
        logger.error(f"行情数据采集失败: {e}")
        quotes_data = []

    try:
        financial = source_manager.fetch_financial_data(stock)
        financial_data = financial.to_dict("records") if not financial.empty else []
    except Exception:
        financial_data = []

    # 构建Stock数据
    stock_data = StockData(
        symbol=stock,
        name=stock,  # TODO: 从stock_info获取
        market=market,
        daily_quotes=quotes_data,
        financial_data=financial_data,
    )

    # 运行分析图
    graph = compile_analysis_graph()
    initial_state = {"stock": stock_data, "messages": []}
    result = graph.invoke(initial_state)

    # 输出报告
    report = result.get("report", "无报告生成")
    click.echo("\n" + "=" * 60)
    click.echo(report)
    click.echo("=" * 60)

    # 输出错误
    errors = result.get("errors", [])
    if errors:
        click.echo(f"\n⚠️ 错误: {errors}")


@main.command()
@click.option("--market", default="A", help="市场（A/HK/US）")
@click.option("--top-n", default=50, help="输出Top N")
def screen(market: str, top_n: int) -> None:
    """量化筛选全市场"""
    from src.market.adapter import AShareAdapter
    from src.screening.engine import ScreeningEngine

    logger.info(f"开始量化筛选: market={market} top_n={top_n}")

    adapter = AShareAdapter()
    source_manager = adapter.create_source_manager()
    engine = ScreeningEngine(source_manager)
    results = engine.run(market=market)

    click.echo(f"\n{'排名':>4} {'代码':<12} {'名称':<10} {'评分':>8}")
    click.echo("-" * 40)
    for r in results[:top_n]:
        click.echo(f"{r.rank:>4} {r.symbol:<12} {r.name:<10} {r.composite_score:>8.2f}")


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


if __name__ == "__main__":
    main()
