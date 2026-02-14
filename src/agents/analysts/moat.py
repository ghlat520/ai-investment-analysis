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


def _calc_moat_anchor(stock: StockData) -> tuple[int, str]:
    """计算护城河量化锚点分数和说明

    基于财务数据的持续性指标，为 LLM 评分提供参考锚点。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    Returns:
        (anchor_score, anchor_explanation)
    """
    if not stock.financial_data:
        return 0, "无财务数据，无法计算锚点"

    df = pd.DataFrame(stock.financial_data)
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date")

    score = 0
    details = []

    # 1. 毛利率水平与稳定性（定价权/成本优势代理指标）
    if "gross_margin" in df.columns:
        gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
        if len(gm) >= 3:
            gm_avg = float(gm.tail(6).mean())
            gm_std = float(gm.tail(6).std())
            # 水平分
            if gm_avg >= 50:
                score += 15
                details.append(f"毛利率均值{gm_avg:.1f}%≥50%(强定价权→+15)")
            elif gm_avg >= 30:
                score += 8
                details.append(f"毛利率均值{gm_avg:.1f}%≥30%(中等定价权→+8)")
            elif gm_avg >= 15:
                score += 0
                details.append(f"毛利率均值{gm_avg:.1f}%(一般)")
            else:
                score -= 10
                details.append(f"毛利率均值{gm_avg:.1f}%<15%(弱定价权→-10)")
            # 稳定性分
            if gm_std < 3:
                score += 10
                details.append(f"毛利率标准差{gm_std:.1f}%<3%(非常稳定→+10)")
            elif gm_std < 8:
                score += 5
                details.append(f"毛利率标准差{gm_std:.1f}%(较稳定→+5)")
            else:
                score -= 5
                details.append(f"毛利率标准差{gm_std:.1f}%≥8%(波动大→-5)")

    # 2. ROE持续性（护城河结果验证）
    if "roe" in df.columns:
        roe = pd.to_numeric(df["roe"], errors="coerce").dropna()
        if len(roe) >= 3:
            high_roe_ratio = float((roe > 15).sum()) / len(roe)
            roe_avg = float(roe.tail(6).mean())
            if high_roe_ratio >= 0.7 and roe_avg >= 15:
                score += 15
                details.append(f"ROE>15%占比{high_roe_ratio*100:.0f}%，均值{roe_avg:.1f}%(持续高回报→+15)")
            elif high_roe_ratio >= 0.4:
                score += 5
                details.append(f"ROE>15%占比{high_roe_ratio*100:.0f}%(中等→+5)")
            elif roe_avg < 5:
                score -= 10
                details.append(f"ROE均值{roe_avg:.1f}%<5%(无竞争优势迹象→-10)")
            else:
                details.append(f"ROE均值{roe_avg:.1f}%(一般)")

    # 3. 净利率水平（盈利质量）
    if "net_margin" in df.columns:
        nm = pd.to_numeric(df["net_margin"], errors="coerce").dropna()
        if len(nm) >= 2:
            nm_avg = float(nm.tail(4).mean())
            if nm_avg >= 20:
                score += 10
                details.append(f"净利率均值{nm_avg:.1f}%≥20%(强盈利→+10)")
            elif nm_avg >= 10:
                score += 5
                details.append(f"净利率均值{nm_avg:.1f}%≥10%(中等→+5)")
            elif nm_avg < 3:
                score -= 10
                details.append(f"净利率均值{nm_avg:.1f}%<3%(微利→-10)")

    # 4. 毛利率趋势方向（护城河趋势代理）
    if "gross_margin" in df.columns:
        gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
        if len(gm) >= 4:
            recent_half = gm.tail(len(gm) // 2).mean()
            earlier_half = gm.head(len(gm) // 2).mean()
            if recent_half > earlier_half + 2:
                score += 5
                details.append(f"毛利率趋势上升({earlier_half:.1f}%→{recent_half:.1f}%→+5)")
            elif recent_half < earlier_half - 2:
                score -= 5
                details.append(f"毛利率趋势下降({earlier_half:.1f}%→{recent_half:.1f}%→-5)")

    anchor = max(-60, min(60, score))
    explanation = "；".join(details) if details else "数据不足"
    return anchor, explanation


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
            gm_vals = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
            if len(gm_vals) >= 2:
                lines.append(f"- 毛利率趋势: {' → '.join(f'{v:.1f}%' for v in gm_vals.tail(4))}")
                gm_avg = float(gm_vals.tail(6).mean())
                gm_std = float(gm_vals.tail(6).std())
                lines.append(f"- 毛利率均值: {gm_avg:.1f}%，标准差: {gm_std:.1f}%")

        # ROE持续性（护城河质量）
        if "roe" in df.columns:
            roe_vals = pd.to_numeric(df["roe"], errors="coerce").dropna()
            if len(roe_vals) >= 2:
                lines.append(f"- ROE趋势: {' → '.join(f'{v:.1f}%' for v in roe_vals.tail(4))}")
                high_roe_years = (roe_vals > 15).sum()
                lines.append(f"- ROE>15%的期数: {high_roe_years}/{len(roe_vals)}")
                lines.append(f"- ROE均值: {float(roe_vals.tail(6).mean()):.1f}%")

        # 营收增速
        if "revenue_yoy" in df.columns:
            rev_vals = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
            if len(rev_vals) >= 1:
                lines.append(f"- 营收增速: {rev_vals.iloc[-1]:.1f}%")

        # 净利率（定价权指标）
        if "net_margin" in df.columns:
            nm_vals = pd.to_numeric(df["net_margin"], errors="coerce").dropna()
            if len(nm_vals) >= 1:
                lines.append(f"- 净利率: {nm_vals.iloc[-1]:.1f}%")
                if len(nm_vals) >= 2:
                    lines.append(f"- 净利率均值: {float(nm_vals.tail(4).mean()):.1f}%")

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
    anchor_score, anchor_explanation = _calc_moat_anchor(stock)

    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    llm_result = llm_deep_analyze(
        agent_name="moat",
        template_name="moat.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "moat_context": moat_context,
            "anchor_score": str(anchor_score),
            "anchor_explanation": anchor_explanation,
            "anchor_score_low": str(anchor_low),
            "anchor_score_high": str(anchor_high),
        },
    )

    score = llm_result["score"]

    # 锚点约束：LLM 评分偏离锚点超过 30 分时，拉回到锚点 ±30
    if abs(score - anchor_score) > 30:
        clamped = max(anchor_score - 30, min(anchor_score + 30, score))
        logger.info(f"[moat] 评分{score}偏离锚点{anchor_score}超过30分，修正为{clamped}")
        score = clamped

    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度：LLM可用时较高（定性分析依赖LLM能力）
    if llm_result["enhanced"]:
        confidence = 0.7
    else:
        confidence = 0.2
        score = anchor_score  # LLM不可用时使用锚点分数

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
            "anchor_score": anchor_score,
            "anchor_explanation": anchor_explanation,
            "llm_raw_score": llm_result["score"],
            "raw_response": llm_result.get("raw_response", {}),
        },
        execution_time_ms=elapsed_ms,
    )
