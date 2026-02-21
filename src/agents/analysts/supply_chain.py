"""
产业链分析Agent

梳理上下游关系、价值链利润分布、传导逻辑、关联投资机会。
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


def _calc_supply_chain_anchor(stock: StockData) -> tuple[int, str]:
    """计算产业链地位量化锚点分数和说明

    基于财务数据评估公司在产业链中的议价能力和地位，为 LLM 评分提供参考锚点。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    评估维度：
    1. 应收/应付比（议价能力核心指标）
    2. 毛利率趋势（产业链地位代理指标）
    3. 存货周转（下游需求强度）
    4. 现金流质量（产业链地位结果验证）

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

    latest = df.iloc[-1] if not df.empty else {}

    # 1. 应收/应付比（议价能力核心指标）
    ar = latest.get("accounts_receivable")
    ap = latest.get("accounts_payable")
    if ar is not None and ap is not None:
        if not (isinstance(ar, float) and np.isnan(ar)) and not (isinstance(ap, float) and np.isnan(ap)):
            if ap > 0 and ar >= 0:
                ar_ap_ratio = ar / ap
                if ar_ap_ratio < 0.5:
                    score += 15
                    details.append(f"应收/应付比{ar_ap_ratio:.2f}<0.5(对上游强势→+15)")
                elif ar_ap_ratio < 1.0:
                    score += 8
                    details.append(f"应收/应付比{ar_ap_ratio:.2f}(议价能力较强→+8)")
                elif ar_ap_ratio < 1.5:
                    score += 0
                    details.append(f"应收/应付比{ar_ap_ratio:.2f}(平衡)")
                elif ar_ap_ratio < 2.5:
                    score -= 10
                    details.append(f"应收/应付比{ar_ap_ratio:.2f}(下游占款多→-10)")
                else:
                    score -= 18
                    details.append(f"应收/应付比{ar_ap_ratio:.2f}>2.5(严重被下游占款→-18)")

    # 2. 毛利率趋势（产业链地位代理指标）
    if "gross_margin" in df.columns:
        gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
        if len(gm) >= 4:
            gm_avg = float(gm.tail(4).mean())
            gm_recent = float(gm.tail(2).mean())
            gm_earlier = float(gm.head(len(gm) // 2).mean()) if len(gm) >= 4 else gm_recent

            # 毛利率水平
            if gm_avg >= 40:
                score += 10
                details.append(f"毛利率均值{gm_avg:.1f}%≥40%(产业链优势地位→+10)")
            elif gm_avg >= 25:
                score += 5
                details.append(f"毛利率均值{gm_avg:.1f}%(中等地位→+5)")
            elif gm_avg < 15:
                score -= 8
                details.append(f"毛利率均值{gm_avg:.1f}%<15%(产业链弱势→-8)")

            # 毛利率趋势
            if gm_recent > gm_earlier + 3:
                score += 8
                details.append(f"毛利率上升({gm_earlier:.1f}%→{gm_recent:.1f}%)(议价能力增强→+8)")
            elif gm_recent < gm_earlier - 3:
                score -= 8
                details.append(f"毛利率下降({gm_earlier:.1f}%→{gm_recent:.1f}%)(议价能力减弱→-8)")

    # 3. 存货/营收比（下游需求强度）
    inventory = latest.get("inventory")
    revenue = latest.get("revenue")
    if inventory is not None and revenue is not None:
        if not (isinstance(inventory, float) and np.isnan(inventory)) and not (isinstance(revenue, float) and np.isnan(revenue)):
            if revenue > 0 and inventory >= 0:
                inv_ratio = inventory / revenue * 100  # 转为百分比
                if inv_ratio < 5:
                    score += 10
                    details.append(f"存货/营收比{inv_ratio:.1f}%<5%(产品供不应求→+10)")
                elif inv_ratio < 15:
                    score += 5
                    details.append(f"存货/营收比{inv_ratio:.1f}%(需求正常→+5)")
                elif inv_ratio < 30:
                    score -= 5
                    details.append(f"存货/营收比{inv_ratio:.1f}%(库存偏高→-5)")
                else:
                    score -= 12
                    details.append(f"存货/营收比{inv_ratio:.1f}%≥30%(库存积压严重→-12)")

    # 4. 现金流质量（产业链地位结果验证）
    if "ocf_to_profit" in df.columns:
        ocf_ratio = pd.to_numeric(df["ocf_to_profit"], errors="coerce").dropna()
        if len(ocf_ratio) >= 2:
            ocf_avg = float(ocf_ratio.tail(4).mean())
            if ocf_avg >= 1.2:
                score += 10
                details.append(f"经营现金流/净利润{ocf_avg:.2f}≥1.2(现金回收能力强→+10)")
            elif ocf_avg >= 0.8:
                score += 3
                details.append(f"经营现金流/净利润{ocf_avg:.2f}(正常→+3)")
            elif ocf_avg < 0.5:
                score -= 10
                details.append(f"经营现金流/净利润{ocf_avg:.2f}<0.5(现金回收困难→-10)")

    anchor = max(-60, min(60, score))
    explanation = "；".join(details) if details else "数据不足"
    return anchor, explanation


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
    P0优化：添加量化锚点约束，LLM评分必须在锚点±30范围内。
    LLM不可用时返回锚点评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    supply_chain_context = _build_supply_chain_context(stock)
    anchor_score, anchor_explanation = _calc_supply_chain_anchor(stock)

    # 锚点约束范围
    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    from src.research.extractor import get_research_context

    llm_result = llm_deep_analyze(
        agent_name="supply_chain",
        template_name="supply_chain.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "supply_chain_context": supply_chain_context,
            "anchor_score": str(anchor_score),
            "anchor_explanation": anchor_explanation,
            "anchor_score_low": str(anchor_low),
            "anchor_score_high": str(anchor_high),
            "research_context": get_research_context(stock, "supply_chain"),
        },
    )

    score = llm_result["score"]

    # 锚点约束：LLM 评分偏离锚点超过 30 分时，拉回到锚点 ±30
    if abs(score - anchor_score) > 30:
        clamped = max(anchor_score - 30, min(anchor_score + 30, score))
        logger.info(f"[supply_chain] 评分{score}偏离锚点{anchor_score}超过30分，修正为{clamped}")
        score = clamped

    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度：LLM可用时较高
    if llm_result["enhanced"]:
        confidence = 0.65
    else:
        confidence = 0.25
        score = anchor_score  # LLM不可用时使用锚点分数

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
            "anchor_score": anchor_score,
            "anchor_explanation": anchor_explanation,
            "llm_raw_score": llm_result["score"],
            "raw_response": llm_result.get("raw_response", {}),
        },
        execution_time_ms=elapsed_ms,
    )
