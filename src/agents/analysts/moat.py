"""
护城河分析Agent

评估竞争壁垒的强度和持久性（Morningstar五维框架）。
LLM-first模式 (Mode B): LLM做主角，代码仅提供基础数据和验证。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _build_moat_context(stock: StockData) -> str:
    """构建护城河分析的基础数据上下文"""
    lines = []

    # 基本信息
    info = stock.info or {}
    if info.get("industry"):
        lines.append(f"- 所属行业: {info['industry']}")
    if info.get("sector"):
        lines.append(f"- 所属板块: {info['sector']}")
    if info.get("market_cap"):
        lines.append(f"- 总市值: {info['market_cap']/1e8:.1f}亿" if isinstance(info['market_cap'], (int, float)) else f"- 总市值: {info['market_cap']}")

    # 从财务数据提取护城河相关指标
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        # 毛利率趋势（护城河指标）
        if "gross_margin" in df.columns:
            gm_vals = df["gross_margin"].dropna()
            if len(gm_vals) >= 2:
                lines.append(f"- 毛利率趋势: {' → '.join(f'{v:.1f}%' for v in gm_vals.tail(4))}")
                gm_trend = "上升" if gm_vals.iloc[-1] > gm_vals.iloc[-2] else "下降"
                lines.append(f"- 毛利率方向: {gm_trend}")

        # ROE持续性（护城河质量）
        if "roe" in df.columns:
            roe_vals = df["roe"].dropna()
            if len(roe_vals) >= 2:
                lines.append(f"- ROE趋势: {' → '.join(f'{v:.1f}%' for v in roe_vals.tail(4))}")
                high_roe_years = (roe_vals > 15).sum()
                lines.append(f"- ROE>15%的期数: {high_roe_years}/{len(roe_vals)}")

        # 营收增速
        if "revenue_yoy" in df.columns:
            rev_vals = df["revenue_yoy"].dropna()
            if len(rev_vals) >= 1:
                lines.append(f"- 营收增速: {rev_vals.iloc[-1]:.1f}%")

        # 净利率（定价权指标）
        if "net_margin" in df.columns:
            nm_vals = df["net_margin"].dropna()
            if len(nm_vals) >= 1:
                lines.append(f"- 净利率: {nm_vals.iloc[-1]:.1f}%")

    if not lines:
        lines.append("基础数据有限，请基于你对该公司和行业的知识进行分析")

    return "\n".join(lines)


def analyze_moat(stock: StockData) -> AgentSignal:
    """护城河分析主函数

    LLM-first模式：LLM直接给出-100~+100评分，代码仅做数据准备和边界验证。
    LLM不可用时返回中性评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    moat_context = _build_moat_context(stock)

    llm_result = llm_deep_analyze(
        agent_name="moat",
        template_name="moat.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "moat_context": moat_context,
        },
    )

    score = llm_result["score"]
    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度：LLM可用时较高（定性分析依赖LLM能力）
    if llm_result["enhanced"]:
        confidence = 0.7
    else:
        confidence = 0.2
        score = 0  # LLM不可用时中性

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="moat",
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
