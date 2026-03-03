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


def _calc_moat_width(stock: StockData) -> dict:
    """R10: 护城河宽度量化指标

    返回量化的护城河宽度评估:
    - brand_premium: 品牌溢价指标（毛利率 vs 行业中位数）
    - scale_effect: 规模效应指标（营收增长 vs 成本增长比率）
    - moat_trend: 护城河变化趋势（近3-5年关键指标变化方向）
    - moat_width_score: 综合护城河宽度评分 (-30 ~ +30)
    """
    result = {
        "brand_premium": None,
        "scale_effect": None,
        "moat_trend": None,
        "moat_width_score": 0,
        "details": [],
    }

    if not stock.financial_data:
        return result

    df = pd.DataFrame(stock.financial_data)
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date")

    info = stock.info or {}
    width_score = 0
    details = []

    # 1. 品牌溢价: 毛利率 vs 行业
    if "gross_margin" in df.columns:
        gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
        if len(gm) >= 2:
            gm_avg = float(gm.tail(4).mean())
            # 尝试获取行业中位数
            peers = info.get("industry_peers", [])
            peer_gm = [float(p.get("gross_margin", 0) or 0) for p in peers if p.get("gross_margin")]
            if peer_gm:
                industry_median_gm = sorted(peer_gm)[len(peer_gm) // 2]
                premium = gm_avg - industry_median_gm
                result["brand_premium"] = {
                    "company_gm": round(gm_avg, 1),
                    "industry_median_gm": round(industry_median_gm, 1),
                    "premium": round(premium, 1),
                }
                if premium > 15:
                    width_score += 10
                    details.append(f"品牌溢价强(毛利率{gm_avg:.1f}%>行业{industry_median_gm:.1f}%+15pp)")
                elif premium > 5:
                    width_score += 5
                    details.append(f"品牌溢价中等(毛利率超行业{premium:.1f}pp)")
                elif premium < -5:
                    width_score -= 5
                    details.append(f"毛利率低于行业{abs(premium):.1f}pp")

    # 2. 规模效应: 营收增速 vs 成本增速
    if "revenue" in df.columns and len(df) >= 3:
        rev = pd.to_numeric(df["revenue"], errors="coerce").dropna()
        if len(rev) >= 3:
            rev_cagr = (rev.iloc[-1] / rev.iloc[-3]) ** (1 / 2) - 1 if rev.iloc[-3] > 0 else 0
            # 检查成本是否增长更慢
            if "gross_margin" in df.columns:
                gm_series = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
                if len(gm_series) >= 3:
                    gm_improvement = gm_series.iloc[-1] - gm_series.iloc[-3]
                    result["scale_effect"] = {
                        "revenue_cagr_2y": round(rev_cagr * 100, 1),
                        "gm_improvement_2y": round(gm_improvement, 1),
                    }
                    if rev_cagr > 0.1 and gm_improvement > 1:
                        width_score += 8
                        details.append(f"规模效应显现(营收CAGR {rev_cagr*100:.0f}%，毛利率提升{gm_improvement:.1f}pp)")
                    elif rev_cagr > 0 and gm_improvement > 0:
                        width_score += 3
                        details.append("轻微规模效应")

    # 3. 护城河变化趋势（近3-5年）
    if len(df) >= 4:
        trend_indicators = {}

        # 毛利率趋势
        if "gross_margin" in df.columns:
            gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
            if len(gm) >= 4:
                early = gm.head(len(gm) // 2).mean()
                late = gm.tail(len(gm) // 2).mean()
                trend_indicators["gross_margin"] = "上升" if late > early + 1 else "下降" if late < early - 1 else "稳定"

        # ROE趋势
        if "roe" in df.columns:
            roe = pd.to_numeric(df["roe"], errors="coerce").dropna()
            if len(roe) >= 4:
                early = roe.head(len(roe) // 2).mean()
                late = roe.tail(len(roe) // 2).mean()
                trend_indicators["roe"] = "上升" if late > early + 1 else "下降" if late < early - 1 else "稳定"

        # 净利率趋势
        if "net_margin" in df.columns:
            nm = pd.to_numeric(df["net_margin"], errors="coerce").dropna()
            if len(nm) >= 4:
                early = nm.head(len(nm) // 2).mean()
                late = nm.tail(len(nm) // 2).mean()
                trend_indicators["net_margin"] = "上升" if late > early + 1 else "下降" if late < early - 1 else "稳定"

        result["moat_trend"] = trend_indicators
        # 趋势评分
        improving = sum(1 for v in trend_indicators.values() if v == "上升")
        declining = sum(1 for v in trend_indicators.values() if v == "下降")
        if improving >= 2 and declining == 0:
            width_score += 8
            details.append("护城河趋势加宽(关键指标均上升)")
        elif declining >= 2:
            width_score -= 8
            details.append("护城河趋势收窄(关键指标在下降)")

    result["moat_width_score"] = max(-30, min(30, width_score))
    result["details"] = details
    return result


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

    # R10: 护城河宽度量化
    moat_width = _calc_moat_width(stock)
    moat_width_score = moat_width.get("moat_width_score", 0)

    # 护城河宽度影响锚点（±15范围内）
    anchor_score = max(-60, min(60, anchor_score + moat_width_score // 2))

    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    from src.research.extractor import get_research_context

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
            "research_context": get_research_context(stock, "moat"),
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
            # R10: 护城河宽度量化
            "moat_width": moat_width,
        },
        execution_time_ms=elapsed_ms,
    )
