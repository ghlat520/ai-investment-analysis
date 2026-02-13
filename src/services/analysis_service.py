"""
分析服务层

从 cli.py 提取的共享分析逻辑，供 CLI 和 API 复用。
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any, Callable, Optional

from loguru import logger


def collect_stock_data(symbol: str, market: str = "A"):
    """采集单只股票的全部数据

    Returns:
        StockData 实例
    """
    from src.agents.state import StockData
    from src.data.sources import DataSourceManager
    from src.data.sources.akshare_source import AKShareSource
    from src.data.sources.baostock_source import BaoStockSource
    from src.data.sources.efinance_source import EFinanceSource
    from src.data.sources.yfinance_source import YFinanceSource

    source_mgr = DataSourceManager()
    source_mgr.register(BaoStockSource())   # TCP 协议，不受 HTTP 代理影响
    source_mgr.register(EFinanceSource())
    source_mgr.register(AKShareSource())
    source_mgr.register(YFinanceSource())

    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365)).isoformat()
    stock_name = symbol

    # 1. 行情数据
    logger.info("[采集] 行情数据...")
    quotes_data: list[dict] = []
    try:
        quotes = source_mgr.fetch_daily_quotes(symbol, start_date, end_date)
        if not quotes.empty:
            for col in ["股票名称", "name"]:
                if col in quotes.columns and quotes[col].iloc[0]:
                    stock_name = str(quotes[col].iloc[0])
                    break
            quotes_data = quotes.to_dict("records")
            logger.info(f"[采集] 行情: {len(quotes_data)}条 ({stock_name})")
    except Exception as e:
        logger.error(f"[采集] 行情失败: {e}")

    # 2. 财务数据
    logger.info("[采集] 财务数据...")
    financial_data: list[dict] = []
    try:
        financial = source_mgr.fetch_financial_data(symbol)
        if not financial.empty:
            financial_data = financial.to_dict("records")
            logger.info(f"[采集] 财务: {len(financial_data)}期")
    except Exception as e:
        logger.warning(f"[采集] 财务数据失败: {e}")

    # 3. 估值数据
    logger.info("[采集] 估值数据...")
    valuation_data: list[dict] = []
    try:
        valuation = source_mgr.fetch_valuation(symbol)
        if not valuation.empty:
            valuation_data = valuation.to_dict("records")
            logger.info(f"[采集] 估值: {len(valuation_data)}天")
    except Exception as e:
        logger.warning(f"[采集] 估值数据失败: {e}")

    # 4. 资金流向
    logger.info("[采集] 资金流向...")
    money_flow_data: list[dict] = []
    try:
        money_flow = source_mgr.fetch_money_flow(symbol, days=20)
        if not money_flow.empty:
            money_flow_data = money_flow.to_dict("records")
            logger.info(f"[采集] 资金流向: {len(money_flow_data)}天")
    except Exception as e:
        logger.warning(f"[采集] 资金流向失败: {e}")

    # 5. 个股新闻
    logger.info("[采集] 个股新闻...")
    news_data: list[dict] = []
    try:
        news_data = source_mgr.fetch_stock_news(symbol, limit=20)
        logger.info(f"[采集] 新闻: {len(news_data)}条")
    except Exception as e:
        logger.warning(f"[采集] 新闻获取失败: {e}")

    # 合并估值数据到财务数据
    if valuation_data and financial_data:
        latest_val = valuation_data[-1]
        for rec in financial_data:
            if "pe_ttm" not in rec or rec.get("pe_ttm") is None:
                rec["pe_ttm"] = latest_val.get("pe_ttm")
            if "pb" not in rec or rec.get("pb") is None:
                rec["pb"] = latest_val.get("pb")
            if "total_market_cap" not in rec or rec.get("total_market_cap") is None:
                rec["total_market_cap"] = latest_val.get("total_market_cap")

    if valuation_data and not financial_data:
        latest_val = valuation_data[-1]
        financial_data = [{
            "report_date": date.today(),
            "pe_ttm": latest_val.get("pe_ttm"),
            "pb": latest_val.get("pb"),
            "total_market_cap": latest_val.get("total_market_cap"),
        }]

    return StockData(
        symbol=symbol,
        name=stock_name,
        market=market,
        daily_quotes=quotes_data,
        financial_data=financial_data,
        money_flow=money_flow_data,
        news=news_data,
        info={"valuation_history": valuation_data},
    )


def run_analysis(stock_data) -> dict[str, Any]:
    """运行完整分析流水线（阻塞式）

    Returns:
        LangGraph 最终 state dict
    """
    from src.agents.graph import compile_analysis_graph

    graph = compile_analysis_graph()
    return graph.invoke({"stock": stock_data})


def run_analysis_streaming(
    stock_data,
    on_agent_done: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """运行分析流水线（流式，逐 agent 回调）

    Args:
        stock_data: StockData 实例
        on_agent_done: 回调函数 (node_name, node_output)

    Returns:
        最终累积 state dict
    """
    from src.agents.graph import compile_analysis_graph

    graph = compile_analysis_graph()
    final_state: dict[str, Any] = {"stock": stock_data}

    for event in graph.stream({"stock": stock_data}):
        # event = {"node_name": {"key": value}}
        for node_name, node_output in event.items():
            # 累积到 final_state
            if "signals" in node_output:
                final_state.setdefault("signals", [])
                final_state["signals"].extend(node_output["signals"])
            if "fusion" in node_output:
                final_state["fusion"] = node_output["fusion"]
            if "report" in node_output:
                final_state["report"] = node_output["report"]
            if "errors" in node_output:
                final_state.setdefault("errors", [])
                final_state["errors"].extend(node_output["errors"])
            if "analysis_date" in node_output:
                final_state["analysis_date"] = node_output["analysis_date"]

            if on_agent_done:
                on_agent_done(node_name, node_output)

    return final_state
