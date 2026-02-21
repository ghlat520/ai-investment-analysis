"""
商业模式分析Agent

拆解公司的赚钱逻辑、定价权、脆弱点。
LLM-first模式 (Mode B): LLM做主角，代码仅提供基础数据和验证。
P0优化：添加量化锚点约束，LLM评分必须在锚点±30范围内。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _calc_business_model_anchor(stock: StockData) -> tuple[int, str]:
    """计算商业模式量化锚点分数和说明

    基于财务数据的商业模式质量指标，为 LLM 评分提供参考锚点。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    评估维度：
    1. 毛利率水平与稳定性（定价权代理指标）
    2. ROE 持续性（商业模式质量验证）
    3. 净利率水平（盈利质量）
    4. 营收稳定性（增长质量）

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

    # 1. 毛利率水平与稳定性（定价权的核心代理指标）
    if "gross_margin" in df.columns:
        gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
        if len(gm) >= 3:
            gm_avg = float(gm.tail(6).mean())
            gm_std = float(gm.tail(6).std())
            # 水平分（定价权强度）
            if gm_avg >= 50:
                score += 15
                details.append(f"毛利率均值{gm_avg:.1f}%≥50%(强定价权→+15)")
            elif gm_avg >= 35:
                score += 10
                details.append(f"毛利率均值{gm_avg:.1f}%≥35%(中等定价权→+10)")
            elif gm_avg >= 20:
                score += 3
                details.append(f"毛利率均值{gm_avg:.1f}%(一般定价权→+3)")
            else:
                score -= 10
                details.append(f"毛利率均值{gm_avg:.1f}%<20%(弱定价权→-10)")
            # 稳定性分（商业模式成熟度）
            if gm_std < 3:
                score += 10
                details.append(f"毛利率标准差{gm_std:.1f}%<3%(商业模式成熟→+10)")
            elif gm_std < 8:
                score += 5
                details.append(f"毛利率标准差{gm_std:.1f}%(较稳定→+5)")
            else:
                score -= 8
                details.append(f"毛利率标准差{gm_std:.1f}%≥8%(商业模式不稳定→-8)")

    # 2. ROE 持续性（商业模式质量的终极验证）
    if "roe" in df.columns:
        roe = pd.to_numeric(df["roe"], errors="coerce").dropna()
        if len(roe) >= 3:
            high_roe_ratio = float((roe > 15).sum()) / len(roe)
            roe_avg = float(roe.tail(6).mean())
            if high_roe_ratio >= 0.7 and roe_avg >= 18:
                score += 15
                details.append(f"ROE>15%占比{high_roe_ratio*100:.0f}%，均值{roe_avg:.1f}%(优秀商业模式→+15)")
            elif high_roe_ratio >= 0.5 and roe_avg >= 12:
                score += 8
                details.append(f"ROE>15%占比{high_roe_ratio*100:.0f}%(良好→+8)")
            elif roe_avg < 8:
                score -= 10
                details.append(f"ROE均值{roe_avg:.1f}%<8%(商业模式存疑→-10)")
            else:
                details.append(f"ROE均值{roe_avg:.1f}%(一般)")

    # 3. 净利率水平（盈利质量，区分"营收大户"和"利润大户"）
    if "net_margin" in df.columns:
        nm = pd.to_numeric(df["net_margin"], errors="coerce").dropna()
        if len(nm) >= 2:
            nm_avg = float(nm.tail(4).mean())
            if nm_avg >= 18:
                score += 12
                details.append(f"净利率均值{nm_avg:.1f}%≥18%(高利润率商业模式→+12)")
            elif nm_avg >= 10:
                score += 6
                details.append(f"净利率均值{nm_avg:.1f}%≥10%(中等→+6)")
            elif nm_avg >= 5:
                score += 0
                details.append(f"净利率均值{nm_avg:.1f}%(一般)")
            else:
                score -= 8
                details.append(f"净利率均值{nm_avg:.1f}%<5%(微利模式→-8)")

    # 4. 营收稳定性（增长质量）
    if "revenue_yoy" in df.columns:
        rev_yoy = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
        if len(rev_yoy) >= 4:
            rev_avg = float(rev_yoy.tail(4).mean())
            rev_std = float(rev_yoy.tail(4).std())
            # 增长水平
            if rev_avg > 20:
                score += 8
                details.append(f"营收增速均值{rev_avg:.1f}%>20%(高增长→+8)")
            elif rev_avg > 10:
                score += 4
                details.append(f"营收增速均值{rev_avg:.1f}%(稳健增长→+4)")
            elif rev_avg < 0:
                score -= 8
                details.append(f"营收增速均值{rev_avg:.1f}%<0%(萎缩→-8)")
            # 增长稳定性
            if not np.isnan(rev_std) and rev_std > 30:
                score -= 5
                details.append(f"营收增速波动大(std={rev_std:.1f}%)→-5")

    anchor = max(-60, min(60, score))
    explanation = "；".join(details) if details else "数据不足"
    return anchor, explanation


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
    P0优化：添加量化锚点约束，LLM评分必须在锚点±30范围内。
    LLM不可用时返回锚点评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    business_context = _build_business_context(stock)
    anchor_score, anchor_explanation = _calc_business_model_anchor(stock)

    # 锚点约束范围
    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    from src.research.extractor import get_research_context

    llm_result = llm_deep_analyze(
        agent_name="business_model",
        template_name="business_model.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "business_context": business_context,
            "anchor_score": str(anchor_score),
            "anchor_explanation": anchor_explanation,
            "anchor_score_low": str(anchor_low),
            "anchor_score_high": str(anchor_high),
            "research_context": get_research_context(stock, "business_model"),
        },
    )

    score = llm_result["score"]

    # 锚点约束：LLM 评分偏离锚点超过 30 分时，拉回到锚点 ±30
    if abs(score - anchor_score) > 30:
        clamped = max(anchor_score - 30, min(anchor_score + 30, score))
        logger.info(f"[business_model] 评分{score}偏离锚点{anchor_score}超过30分，修正为{clamped}")
        score = clamped

    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度：LLM可用时较高
    if llm_result["enhanced"]:
        confidence = 0.7
    else:
        confidence = 0.25
        score = anchor_score  # LLM不可用时使用锚点分数

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
            "anchor_score": anchor_score,
            "anchor_explanation": anchor_explanation,
            "llm_raw_score": llm_result["score"],
            "raw_response": llm_result.get("raw_response", {}),
        },
        execution_time_ms=elapsed_ms,
    )
