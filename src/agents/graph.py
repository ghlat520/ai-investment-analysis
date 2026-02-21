"""
LangGraph 图编排

定义分析流程的DAG：
DataLoader → [技术面|基本面|估值|资金面|情绪面|护城河|商业模式|行业|产业链|竞争格局|管理层] (fan-out并行) → 决策融合 → 研报生成

11-Agent完整版：
- 量化优先: technical, money_flow (LLM ±15微调)
- 混合模式: fundamental, valuation, sentiment (LLM ±40深度增强)
- LLM主导: moat, business_model, industry, supply_chain, competition, management (LLM直接评分)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.graph import END, StateGraph
from loguru import logger

from .state import AnalysisState


def _load_agent_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f).get("agents", {})
    return {}


# ---------- Node functions ----------
# LangGraph 的 state 是 dict，通过 state["key"] 访问


def data_loader(state: dict[str, Any]) -> dict[str, Any]:
    """加载股票数据到State（入口节点）"""
    stock = state.get("stock")
    symbol = stock.symbol if stock else "N/A"
    logger.info(f"[DataLoader] Loading data for {symbol}")
    if stock is None:
        return {"errors": ["No stock data provided"]}
    return {"analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def technical_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """技术面分析Agent"""
    from .analysts.technical import analyze_technical

    stock = state["stock"]
    logger.info(f"[TechnicalAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_technical(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[TechnicalAnalyst] Error: {e}")
        return {"errors": [f"TechnicalAnalyst error: {e}"]}


def fundamental_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """基本面分析Agent"""
    from .analysts.fundamental import analyze_fundamental

    stock = state["stock"]
    logger.info(f"[FundamentalAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_fundamental(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[FundamentalAnalyst] Error: {e}")
        return {"errors": [f"FundamentalAnalyst error: {e}"]}


def valuation_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """估值分析Agent"""
    from .analysts.valuation import analyze_valuation

    stock = state["stock"]
    logger.info(f"[ValuationAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_valuation(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[ValuationAnalyst] Error: {e}")
        return {"errors": [f"ValuationAnalyst error: {e}"]}


def money_flow_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """资金面分析Agent"""
    from .analysts.money_flow import analyze_money_flow

    stock = state["stock"]
    logger.info(f"[MoneyFlowAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_money_flow(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[MoneyFlowAnalyst] Error: {e}")
        return {"errors": [f"MoneyFlowAnalyst error: {e}"]}


def sentiment_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """情绪面分析Agent"""
    from .analysts.sentiment import analyze_sentiment

    stock = state["stock"]
    logger.info(f"[SentimentAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_sentiment(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[SentimentAnalyst] Error: {e}")
        return {"errors": [f"SentimentAnalyst error: {e}"]}


def moat_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """护城河分析Agent"""
    from .analysts.moat import analyze_moat

    stock = state["stock"]
    logger.info(f"[MoatAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_moat(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[MoatAnalyst] Error: {e}")
        return {"errors": [f"MoatAnalyst error: {e}"]}


def business_model_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """商业模式分析Agent"""
    from .analysts.business_model import analyze_business_model

    stock = state["stock"]
    logger.info(f"[BusinessModelAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_business_model(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[BusinessModelAnalyst] Error: {e}")
        return {"errors": [f"BusinessModelAnalyst error: {e}"]}


def industry_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """产业链分析Agent"""
    from .analysts.industry import analyze_industry

    stock = state["stock"]
    logger.info(f"[IndustryAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_industry(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[IndustryAnalyst] Error: {e}")
        return {"errors": [f"IndustryAnalyst error: {e}"]}


def supply_chain_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """产业链（供应链）分析Agent"""
    from .analysts.supply_chain import analyze_supply_chain

    stock = state["stock"]
    logger.info(f"[SupplyChainAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_supply_chain(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[SupplyChainAnalyst] Error: {e}")
        return {"errors": [f"SupplyChainAnalyst error: {e}"]}


def competition_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """竞争格局分析Agent"""
    from .analysts.competition import analyze_competition

    stock = state["stock"]
    logger.info(f"[CompetitionAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_competition(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[CompetitionAnalyst] Error: {e}")
        return {"errors": [f"CompetitionAnalyst error: {e}"]}


def management_analyst(state: dict[str, Any]) -> dict[str, Any]:
    """管理层质量分析Agent"""
    from .analysts.management import analyze_management

    stock = state["stock"]
    logger.info(f"[ManagementAnalyst] Analyzing {stock.symbol}")
    try:
        signal = analyze_management(stock)
        return {"signals": [signal]}
    except Exception as e:
        logger.error(f"[ManagementAnalyst] Error: {e}")
        return {"errors": [f"ManagementAnalyst error: {e}"]}


def fusion_agent(state: dict[str, Any]) -> dict[str, Any]:
    """决策融合Agent"""
    from .fusion.engine import fuse_signals

    signals = state.get("signals", [])
    stock = state.get("stock")
    logger.info(f"[FusionAgent] Fusing {len(signals)} signals")
    try:
        decision = fuse_signals(signals, stock)
        return {"fusion": decision}
    except Exception as e:
        logger.error(f"[FusionAgent] Error: {e}")
        return {"errors": [f"FusionAgent error: {e}"]}


def report_agent(state: dict[str, Any]) -> dict[str, Any]:
    """研报生成Agent"""
    from ..report.generator import generate_report

    stock = state.get("stock")
    logger.info(f"[ReportAgent] Generating report for {stock.symbol if stock else 'N/A'}")
    try:
        report = generate_report(state)
        return {"report": report}
    except Exception as e:
        logger.error(f"[ReportAgent] Error: {e}")
        return {"errors": [f"ReportAgent error: {e}"]}


# ---------- Graph construction ----------


def build_analysis_graph() -> StateGraph:
    """构建分析流程图

    结构:
        data_loader
            ↓
        ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐  (fan-out: 并行)
        技术 基本 估值 资金 情绪 护城河 商业 行业 产业链 竞争 管理层
        └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘  (fan-in: 汇聚)
            ↓
        fusion
            ↓
        report
            ↓
          END
    """
    config = _load_agent_config()

    # Agent名 → node函数 映射
    analyst_nodes = {
        "technical": technical_analyst,
        "fundamental": fundamental_analyst,
        "valuation": valuation_analyst,
        "money_flow": money_flow_analyst,
        "sentiment": sentiment_analyst,
        "moat": moat_analyst,
        "business_model": business_model_analyst,
        "industry": industry_analyst,
        "supply_chain": supply_chain_analyst,
        "competition": competition_analyst,
        "management": management_analyst,
    }

    graph = StateGraph(AnalysisState)

    # 添加节点
    graph.add_node("data_loader", data_loader)
    for name, func in analyst_nodes.items():
        graph.add_node(name, func)
    graph.add_node("fusion", fusion_agent)
    graph.add_node("report", report_agent)

    # 入口 → DataLoader
    graph.set_entry_point("data_loader")

    # DataLoader → 并行分析（fan-out）
    enabled_analysts = []
    for name in analyst_nodes:
        agent_cfg = config.get(name, {})
        if agent_cfg.get("enabled", True):
            enabled_analysts.append(name)

    for analyst in enabled_analysts:
        graph.add_edge("data_loader", analyst)

    # 所有分析Agent → 融合（fan-in）
    for analyst in enabled_analysts:
        graph.add_edge(analyst, "fusion")

    # 融合 → 研报 → 结束
    graph.add_edge("fusion", "report")
    graph.add_edge("report", END)

    return graph


def compile_analysis_graph():
    """编译并返回可执行的分析图"""
    graph = build_analysis_graph()
    return graph.compile()
