"""
分析服务层

从 cli.py 提取的共享分析逻辑，供 CLI 和 API 复用。
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any, Callable, Optional

from loguru import logger


_MAJOR_EVENT_KEYWORDS = (
    "收购", "并购", "重组", "重大资产", "借壳", "合并",
    "分拆", "剥离", "战略入股", "定向增发", "配股",
    "股权转让", "实控人变更", "破产", "退市",
)


def _detect_data_lag_warnings(
    news: list[dict],
    biz_comp: list[dict],
    warnings: list[str],
    stock_name: str,
) -> None:
    """检测新闻中的重大事件与财报数据时效性矛盾，生成风险提示"""
    if not news:
        return

    # 扫描新闻标题，匹配重大事件关键词
    matched_events: list[str] = []
    for item in news:
        title = item.get("title", "")
        for kw in _MAJOR_EVENT_KEYWORDS:
            if kw in title:
                matched_events.append(f"{kw}: {title[:60]}")
                break  # 同一条新闻只匹配一次

    if not matched_events:
        return

    # 取 business_composition 最新报告日期
    biz_date = "未知"
    if biz_comp:
        dates = [str(item.get("report_date", ""))[:10] for item in biz_comp if item.get("report_date")]
        if dates:
            biz_date = max(dates)

    event_list = "; ".join(matched_events[:3])
    warnings.append(
        f"⚠️ 重大事件预警: 近期新闻含[{event_list}]，"
        f"但分业务营收数据截至{biz_date}，可能未反映最新业务变化。"
        f"券商一致预期EPS可能已调整，但业务构成数据滞后，估值模型选择和分业务分析需审慎解读。"
    )
    logger.warning(f"[数据质量] {stock_name} 检测到重大事件与财报数据滞后矛盾: {event_list}")


def _load_segment_model(symbol: str) -> dict | None:
    """从config/segment_models.yaml加载分业务线预测模型（如果有）"""
    from pathlib import Path
    config_path = Path(__file__).parent.parent.parent / "config" / "segment_models.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        profiles = data.get("profiles", {})
        # 支持 300054.SZ 和 300054 两种格式匹配
        model = profiles.get(symbol) or profiles.get(symbol.split(".")[0])
        return model
    except Exception as e:
        logger.debug(f"[segment_model] YAML加载失败: {e}")
        return None


def collect_stock_data(
    symbol: str,
    market: str = "A",
    research_dir: str | None = None,
    auto_research: bool = True,
):
    """采集单只股票的全部数据

    Args:
        symbol: 股票代码
        market: 市场（A/HK/US）
        research_dir: 研报 PDF 目录路径（可选，手动指定）
        auto_research: 自动从东财采集最新研报（默认开启，research_dir 优先）

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

    # 0. 股票名称+行业（先查DB缓存，再查实时行情）
    industry_name = ""
    try:
        from src.data.storage.database import get_database
        from src.data.storage.models import StockInfo
        db = get_database()
        with db.session() as sess:
            info = sess.query(StockInfo).filter_by(symbol=symbol).first()
            if info and info.name and info.name != symbol and not info.name.endswith(('.SZ', '.SH', '.HK')):
                stock_name = info.name
            if info and info.industry:
                industry_name = info.industry
    except Exception:
        pass

    if stock_name == symbol or not industry_name:
        # DB没有完整信息，尝试从akshare获取
        try:
            import akshare as ak
            code = symbol.split(".")[0]
            info_df = ak.stock_individual_info_em(symbol=code)
            if info_df is not None and not info_df.empty:
                info_map = dict(zip(info_df["item"], info_df["value"]))
                if stock_name == symbol:
                    name_val = info_map.get("股票简称", "")
                    if name_val:
                        stock_name = str(name_val)
                        logger.info(f"[采集] 股票名称: {stock_name}")
                if not industry_name:
                    ind_val = info_map.get("行业", "")
                    if ind_val:
                        industry_name = str(ind_val).strip()
                        logger.info(f"[采集] 行业: {industry_name}")
        except Exception as e:
            logger.debug(f"[采集] 个股信息查询失败: {e}")

    if stock_name == symbol:
        # 最后尝试从实时行情获取名称
        try:
            import akshare as ak
            code = symbol.split(".")[0]
            spot = ak.stock_zh_a_spot_em()
            row = spot[spot["代码"] == code]
            if not row.empty:
                stock_name = str(row["名称"].iloc[0])
                logger.info(f"[采集] 股票名称(spot): {stock_name}")
        except Exception as e:
            logger.debug(f"[采集] 名称查询失败: {e}")

    # 1. 行情数据
    logger.info("[采集] 行情数据...")
    quotes_data: list[dict] = []
    try:
        quotes = source_mgr.fetch_daily_quotes(symbol, start_date, end_date)
        if not quotes.empty:
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

    # 6. 分业务营收构成
    logger.info("[采集] 分业务营收构成...")
    business_comp_data: list[dict] = []
    try:
        business_comp = source_mgr.fetch_business_composition(symbol)
        if not business_comp.empty:
            business_comp_data = business_comp.to_dict("records")
            logger.info(f"[采集] 分业务构成: {len(business_comp_data)}项")
    except Exception as e:
        logger.debug(f"[采集] 分业务构成获取失败（非致命）: {e}")

    # 7. 券商盈利预测
    logger.info("[采集] 券商盈利预测...")
    profit_forecast_data: list[dict] = []
    try:
        profit_forecast = source_mgr.fetch_profit_forecast(symbol)
        if not profit_forecast.empty:
            profit_forecast_data = profit_forecast.to_dict("records")
            logger.info(f"[采集] 券商预测: {len(profit_forecast_data)}条")
    except Exception as e:
        logger.debug(f"[采集] 券商盈利预测获取失败（非致命）: {e}")

    # 8. 股东人数变化
    logger.info("[采集] 股东人数...")
    shareholder_count: list[dict] = []
    try:
        df = source_mgr.fetch_shareholder_count(symbol)
        if not df.empty:
            shareholder_count = df.to_dict("records")
            logger.info(f"[采集] 股东人数: {len(shareholder_count)}期")
    except Exception as e:
        logger.debug(f"[采集] 股东人数获取失败（非致命）: {e}")

    # 9. 北向资金持股
    logger.info("[采集] 北向资金持股...")
    northbound: list[dict] = []
    try:
        df = source_mgr.fetch_northbound_holdings(symbol)
        if not df.empty:
            northbound = df.to_dict("records")
            logger.info(f"[采集] 北向持股: {len(northbound)}条")
    except Exception as e:
        logger.debug(f"[采集] 北向持股获取失败（非致命）: {e}")

    # 10. 分红历史
    logger.info("[采集] 分红历史...")
    dividend_history: list[dict] = []
    try:
        df = source_mgr.fetch_dividend_history(symbol)
        if not df.empty:
            dividend_history = df.to_dict("records")
            logger.info(f"[采集] 分红历史: {len(dividend_history)}期")
    except Exception as e:
        logger.debug(f"[采集] 分红历史获取失败（非致命）: {e}")

    # 11. 龙虎榜+大宗交易
    logger.info("[采集] 龙虎榜+大宗交易...")
    dragon_tiger: list[dict] = []
    try:
        df = source_mgr.fetch_dragon_tiger(symbol, days=90)
        if not df.empty:
            dragon_tiger = df.to_dict("records")
            logger.info(f"[采集] 龙虎榜/大宗: {len(dragon_tiger)}条")
    except Exception as e:
        logger.debug(f"[采集] 龙虎榜/大宗获取失败（非致命）: {e}")

    # 12. 融资融券数据
    logger.info("[采集] 融资融券...")
    margin_data: list[dict] = []
    try:
        df = source_mgr.fetch_margin_data(symbol)
        if not df.empty:
            margin_data = df.to_dict("records")
            logger.info(f"[采集] 融资融券: {len(margin_data)}条")
    except Exception as e:
        logger.debug(f"[采集] 融资融券获取失败（非致命）: {e}")

    # 13. 人气排名（检查个股是否在Top100）
    logger.info("[采集] 人气排名...")
    hot_rank_info: dict = {}
    try:
        hot_df = source_mgr.fetch_hot_rank()
        if not hot_df.empty:
            code = symbol.split(".")[0]
            match = hot_df[hot_df["symbol"].str.contains(code, na=False)]
            if not match.empty:
                row = match.iloc[0]
                hot_rank_info = {
                    "rank": int(row.get("rank", 0)),
                    "name": str(row.get("name", "")),
                    "price": float(row.get("price", 0) or 0),
                    "change_pct": float(row.get("change_pct", 0) or 0),
                }
                logger.info(f"[采集] 人气排名: 第{hot_rank_info['rank']}名")
            else:
                logger.debug(f"[采集] {symbol} 不在人气Top100中")
    except Exception as e:
        logger.debug(f"[采集] 人气排名获取失败（非致命）: {e}")

    # 14. 业绩预告（检查个股是否有预告）
    logger.info("[采集] 业绩预告...")
    performance_forecast: dict = {}
    try:
        forecast_df = source_mgr.fetch_performance_forecast()
        if not forecast_df.empty:
            code = symbol.split(".")[0]
            match = forecast_df[forecast_df["symbol"].str.contains(code, na=False)]
            if not match.empty:
                row = match.iloc[0]
                performance_forecast = {
                    "forecast_type": str(row.get("forecast_type", "")),
                    "change_pct": float(row.get("change_pct", 0) or 0),
                    "forecast_content": str(row.get("forecast_content", "")),
                    "announce_date": str(row.get("announce_date", "")),
                }
                logger.info(f"[采集] 业绩预告: {performance_forecast['forecast_type']}")
            else:
                logger.debug(f"[采集] {symbol} 无业绩预告")
    except Exception as e:
        logger.debug(f"[采集] 业绩预告获取失败（非致命）: {e}")

    # 15. 同行业个股对比
    logger.info("[采集] 同行业个股...")
    industry_peers: list[dict] = []
    if industry_name:
        try:
            df = source_mgr.fetch_industry_peers(symbol, industry=industry_name)
            if not df.empty:
                industry_peers = df.to_dict("records")
                logger.info(f"[采集] 同行业个股: {len(industry_peers)}只 (行业: {industry_name})")
        except Exception as e:
            logger.debug(f"[采集] 同行业个股获取失败（非致命）: {e}")
    else:
        logger.debug("[采集] 无行业信息，跳过同行业个股")

    # 16. 个股研报（机构评级+EPS预测+机构参与度）
    logger.info("[采集] 个股研报...")
    research_report_data: dict = {}
    try:
        research_report_data = source_mgr.fetch_research_reports(symbol)
        if research_report_data:
            coverage = research_report_data.get("coverage_count", 0)
            rating = research_report_data.get("latest_rating", "未知")
            logger.info(f"[采集] 个股研报: {coverage}家机构覆盖, 最新评级={rating}")
    except Exception as e:
        logger.debug(f"[采集] 个股研报获取失败（非致命）: {e}")

    # 13. YAML覆写（如果有精细预测）
    segment_model = _load_segment_model(symbol)
    if segment_model:
        logger.info(f"[采集] 加载YAML分业务预测模型: {segment_model.get('name', symbol)}")

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

    # 6. 数据新鲜度检查
    data_warnings = []
    if quotes_data:
        latest_quote_date = str(quotes_data[-1].get("date", ""))[:10]
        try:
            from datetime import datetime
            latest_dt = datetime.strptime(latest_quote_date, "%Y-%m-%d").date()
            stale_days = (date.today() - latest_dt).days
            if stale_days > 5:
                data_warnings.append(f"行情数据过期: 最新日期{latest_quote_date}，距今{stale_days}天")
                logger.warning(f"[数据质量] 行情数据可能过期: {latest_quote_date} (距今{stale_days}天)")
            elif stale_days > 3:
                data_warnings.append(f"行情数据略旧: {latest_quote_date}，距今{stale_days}天（可能含节假日）")
                logger.info(f"[数据质量] 行情数据距今{stale_days}天: {latest_quote_date}")
        except (ValueError, TypeError):
            pass

    if not valuation_data:
        data_warnings.append("估值历史数据缺失，百分位计算精度降低")

    # 重大事件 vs 财报数据时效性矛盾检测
    _detect_data_lag_warnings(
        news_data, business_comp_data, data_warnings, stock_name,
    )

    # 17. 研报PDF采集与解析
    research_summaries: dict = {}
    if not research_dir and auto_research:
        # 自动从东财采集最新研报
        try:
            from src.research.fetcher import auto_fetch_research

            logger.info("[采集] 自动采集最新券商研报...")
            fetched_dir, report_meta = auto_fetch_research(symbol, top_n=3)
            if fetched_dir:
                research_dir = fetched_dir
            # 用采集的元数据补充 research_reports（如果原有数据为空）
            if report_meta and not research_report_data:
                research_report_data = report_meta
        except Exception as e:
            logger.warning(f"[采集] 研报自动采集失败（非致命）: {e}")

    if research_dir:
        try:
            from src.research.extractor import extract_research_summaries, parse_research_pdfs

            logger.info(f"[采集] 解析研报PDF: {research_dir}")
            pdf_text = parse_research_pdfs(research_dir)
            if pdf_text:
                research_summaries = extract_research_summaries(pdf_text, symbol, stock_name)
        except Exception as e:
            logger.warning(f"[采集] 研报解析失败（非致命）: {e}")

    return StockData(
        symbol=symbol,
        name=stock_name,
        market=market,
        daily_quotes=quotes_data,
        financial_data=financial_data,
        money_flow=money_flow_data,
        news=news_data,
        info={
            "valuation_history": valuation_data,
            "data_warnings": data_warnings,
            "business_composition": business_comp_data,
            "profit_forecast": profit_forecast_data,
            "segment_model": segment_model,
            "research_summaries": research_summaries,
            "shareholder_count": shareholder_count,
            "northbound_holdings": northbound,
            "dividend_history": dividend_history,
            "dragon_tiger": dragon_tiger,
            "research_reports": research_report_data,
            "margin_data": margin_data,
            "hot_rank": hot_rank_info,
            "performance_forecast": performance_forecast,
            "industry": industry_name,
            "industry_peers": industry_peers,
        },
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
