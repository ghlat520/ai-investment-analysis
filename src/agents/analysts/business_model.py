"""
商业模式分析Agent

拆解公司的赚钱逻辑、定价权、脆弱点。
LLM-first模式 (Mode B): LLM做主角，代码仅提供基础数据和验证。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _build_business_context(stock: StockData) -> str:
    """构建商业模式分析的基础数据上下文"""
    lines = []

    # 基本信息
    info = stock.info or {}
    if info.get("industry"):
        lines.append(f"- 所属行业: {info['industry']}")
    if info.get("main_business"):
        lines.append(f"- 主营业务: {info['main_business']}")
    if info.get("market_cap"):
        lines.append(f"- 总市值: {info['market_cap']/1e8:.1f}亿" if isinstance(info['market_cap'], (int, float)) else f"- 总市值: {info['market_cap']}")

    # 从财务数据提取商业模式相关指标
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        latest = df.iloc[-1] if not df.empty else {}

        # 营收和利润
        if latest.get("revenue") is not None and not (isinstance(latest.get("revenue"), float) and np.isnan(latest["revenue"])):
            lines.append(f"- 营收: {latest['revenue']/1e8:.2f}亿")
        if latest.get("net_profit") is not None and not (isinstance(latest.get("net_profit"), float) and np.isnan(latest["net_profit"])):
            lines.append(f"- 净利润: {latest['net_profit']/1e8:.2f}亿")

        # 毛利率（定价权指标）
        if "gross_margin" in df.columns:
            gm_vals = df["gross_margin"].dropna()
            if len(gm_vals) >= 1:
                lines.append(f"- 毛利率: {gm_vals.iloc[-1]:.1f}%")
            if len(gm_vals) >= 4:
                lines.append(f"- 毛利率趋势(近4期): {' → '.join(f'{v:.1f}%' for v in gm_vals.tail(4))}")

        # 净利率
        if "net_margin" in df.columns:
            nm_vals = df["net_margin"].dropna()
            if len(nm_vals) >= 1:
                lines.append(f"- 净利率: {nm_vals.iloc[-1]:.1f}%")

        # 营收增速
        if "revenue_yoy" in df.columns:
            rev_vals = df["revenue_yoy"].dropna()
            if len(rev_vals) >= 1:
                lines.append(f"- 营收增速: {rev_vals.iloc[-1]:.1f}%")

        # ROE
        if "roe" in df.columns:
            roe_vals = df["roe"].dropna()
            if len(roe_vals) >= 1:
                lines.append(f"- ROE: {roe_vals.iloc[-1]:.1f}%")

    if not lines:
        lines.append("基础数据有限，请基于你对该公司和行业的知识进行分析")

    return "\n".join(lines)


def analyze_business_model(stock: StockData) -> AgentSignal:
    """商业模式分析主函数

    LLM-first模式：LLM直接给出-100~+100评分，代码仅做数据准备和边界验证。
    LLM不可用时返回中性评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    business_context = _build_business_context(stock)

    from src.research.extractor import get_research_context

    llm_result = llm_deep_analyze(
        agent_name="business_model",
        template_name="business_model.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "business_context": business_context,
            "research_context": get_research_context(stock, "business_model"),
        },
    )

    score = llm_result["score"]
    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度
    if llm_result["enhanced"]:
        confidence = 0.7
    else:
        confidence = 0.2
        score = 0

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="business_model",
        signal_score=max(-100, min(100, score)),
        confidence=round(confidence, 3),
        reasoning=reasoning,
        key_factors=tuple(factors),
        risks=tuple(risks),
        data_quality=confidence,
        llm_model=llm_result["llm_model"],
        llm_tokens_used=llm_result["llm_tokens"],
        llm_cost_usd=llm_result["llm_cost"],
        metadata={
            "llm_enhanced": llm_result["enhanced"],
            "llm_mode": "primary",
            "raw_response": llm_result.get("raw_response", {}),
        },
        execution_time_ms=elapsed_ms,
    )
