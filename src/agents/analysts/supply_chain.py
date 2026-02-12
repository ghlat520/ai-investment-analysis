"""
产业链分析Agent

梳理上下游关系、价值链利润分布、传导逻辑、关联投资机会。
LLM-first模式 (Mode B): LLM做主角，代码仅提供基础数据和验证。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _build_supply_chain_context(stock: StockData) -> str:
    """构建产业链分析的基础数据上下文"""
    lines = []

    # 基本信息
    info = stock.info or {}
    if info.get("industry"):
        lines.append(f"- 所属行业: {info['industry']}")
    if info.get("sector"):
        lines.append(f"- 所属板块: {info['sector']}")
    if info.get("main_business"):
        lines.append(f"- 主营业务: {info['main_business']}")
    if info.get("market_cap"):
        lines.append(
            f"- 总市值: {info['market_cap']/1e8:.1f}亿"
            if isinstance(info["market_cap"], (int, float))
            else f"- 总市值: {info['market_cap']}"
        )

    # 从财务数据提取产业链相关指标
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        # 毛利率趋势（议价能力指标）
        if "gross_margin" in df.columns:
            gm_vals = df["gross_margin"].dropna()
            if len(gm_vals) >= 2:
                lines.append(f"- 毛利率趋势: {' → '.join(f'{v:.1f}%' for v in gm_vals.tail(4))}")
                gm_change = gm_vals.iloc[-1] - gm_vals.iloc[-2]
                lines.append(f"- 毛利率变化: {gm_change:+.1f}pp")

        # 营收增速趋势（行业景气度）
        if "revenue_yoy" in df.columns:
            rev_vals = df["revenue_yoy"].dropna()
            if len(rev_vals) >= 1:
                lines.append(f"- 营收增速: {rev_vals.iloc[-1]:.1f}%")
            if len(rev_vals) >= 2:
                lines.append(f"- 营收增速趋势: {' → '.join(f'{v:.1f}%' for v in rev_vals.tail(4))}")

        # 净利率（价值链位置指标）
        if "net_margin" in df.columns:
            nm_vals = df["net_margin"].dropna()
            if len(nm_vals) >= 1:
                lines.append(f"- 净利率: {nm_vals.iloc[-1]:.1f}%")

        # 应收/应付比（议价地位指标）
        latest = df.iloc[-1] if not df.empty else {}
        ar = latest.get("accounts_receivable")
        ap = latest.get("accounts_payable")
        if ar is not None and ap is not None:
            if not (isinstance(ar, float) and np.isnan(ar)) and not (isinstance(ap, float) and np.isnan(ap)):
                if ap > 0:
                    ratio = ar / ap
                    lines.append(f"- 应收/应付比: {ratio:.2f} ({'下游话语权强' if ratio > 1.5 else '上游话语权强' if ratio < 0.5 else '较平衡'})")

    if not lines:
        lines.append("基础数据有限，请基于你对该公司和行业的知识进行分析")

    return "\n".join(lines)


def analyze_supply_chain(stock: StockData) -> AgentSignal:
    """产业链分析主函数

    LLM-first模式：LLM直接给出-100~+100评分，代码仅做数据准备和边界验证。
    LLM不可用时返回中性评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    supply_chain_context = _build_supply_chain_context(stock)

    llm_result = llm_deep_analyze(
        agent_name="supply_chain",
        template_name="supply_chain.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "supply_chain_context": supply_chain_context,
        },
    )

    score = llm_result["score"]
    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度：LLM可用时较高
    if llm_result["enhanced"]:
        confidence = 0.65
    else:
        confidence = 0.2
        score = 0  # LLM不可用时中性

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="supply_chain",
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
