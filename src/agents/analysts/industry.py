"""
产业链分析Agent

分析行业周期、供需格局、政策方向、竞争格局。
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


def _calc_industry_anchor(stock: StockData) -> tuple[int, str]:
    """计算行业景气度量化锚点分数和说明

    基于财务数据和行业对比，为 LLM 评分提供参考锚点。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    评估维度：
    1. 相对行业PE位置（市场对行业前景的定价）
    2. 营收增速趋势（行业景气度）
    3. 毛利率趋势（供需格局）
    4. 同行业相对表现

    Returns:
        (anchor_score, anchor_explanation)
    """
    score = 0
    details = []

    info = stock.info or {}

    # 1. 相对行业PE位置
    company_pe = info.get("pe_ttm")
    peers = info.get("industry_peers", [])
    if company_pe and peers:
        pe_vals = [float(p.get("pe", 0) or 0) for p in peers if p.get("pe") and float(p.get("pe", 0) or 0) > 0]
        if pe_vals and company_pe > 0:
            industry_pe_median = float(sorted(pe_vals)[len(pe_vals) // 2])
            pe_ratio = company_pe / industry_pe_median if industry_pe_median > 0 else 1
            if pe_ratio > 1.5:
                # PE显著高于行业，可能是高增长预期或高估
                score += 5
                details.append(f"PE({company_pe:.1f})高于行业中位数({industry_pe_median:.1f}){pe_ratio:.1f}倍(市场预期高→+5)")
            elif pe_ratio < 0.7:
                # PE显著低于行业，可能是被低估或行业拖累
                score -= 5
                details.append(f"PE({company_pe:.1f})低于行业中位数({industry_pe_median:.1f}){pe_ratio:.1f}倍(可能被低估或行业低迷→-5)")
            else:
                details.append(f"PE({company_pe:.1f})接近行业中位数({industry_pe_median:.1f})")

    # 2. 营收增速趋势（行业景气度核心指标）
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        if "revenue_yoy" in df.columns:
            rev_yoy = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
            if len(rev_yoy) >= 3:
                rev_avg = float(rev_yoy.tail(4).mean())
                rev_recent = float(rev_yoy.iloc[-1])
                rev_prev = float(rev_yoy.iloc[-2]) if len(rev_yoy) >= 2 else rev_recent

                # 增速水平
                if rev_avg > 25:
                    score += 15
                    details.append(f"营收增速均值{rev_avg:.1f}%>25%(高景气→+15)")
                elif rev_avg > 15:
                    score += 10
                    details.append(f"营收增速均值{rev_avg:.1f}%>15%(景气向上→+10)")
                elif rev_avg > 5:
                    score += 3
                    details.append(f"营收增速均值{rev_avg:.1f}%(温和增长→+3)")
                elif rev_avg < 0:
                    score -= 12
                    details.append(f"营收增速均值{rev_avg:.1f}%<0%(行业萎缩→-12)")

                # 增速趋势（加速/减速）
                if rev_recent > rev_prev + 10:
                    score += 8
                    details.append(f"增速加速({rev_prev:.1f}%→{rev_recent:.1f}%)→+8")
                elif rev_recent < rev_prev - 10:
                    score -= 8
                    details.append(f"增速放缓({rev_prev:.1f}%→{rev_recent:.1f}%)→-8")

    # 3. 毛利率趋势（供需格局代理指标）
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        if "gross_margin" in df.columns:
            gm = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
            if len(gm) >= 4:
                gm_recent = float(gm.tail(2).mean())
                gm_earlier = float(gm.head(len(gm) // 2).mean()) if len(gm) >= 4 else gm_recent

                if gm_recent > gm_earlier + 3:
                    score += 10
                    details.append(f"毛利率上升({gm_earlier:.1f}%→{gm_recent:.1f}%)(供需改善→+10)")
                elif gm_recent < gm_earlier - 3:
                    score -= 10
                    details.append(f"毛利率下降({gm_earlier:.1f}%→{gm_recent:.1f}%)(竞争加剧→-10)")

    # 4. 同行业相对表现
    if peers:
        company_change = info.get("change_pct", 0) or 0
        peer_changes = [float(p.get("change_pct", 0) or 0) for p in peers if p.get("change_pct")]
        if peer_changes:
            avg_peer_change = sum(peer_changes) / len(peer_changes)
            rel_perf = float(company_change) - avg_peer_change
            if rel_perf > 3:
                score += 5
                details.append(f"当日表现优于行业均值{rel_perf:+.1f}pp→+5")
            elif rel_perf < -3:
                score -= 5
                details.append(f"当日表现弱于行业均值{rel_perf:.1f}pp→-5")

    anchor = max(-60, min(60, score))
    explanation = "；".join(details) if details else "数据不足"
    return anchor, explanation


def _build_industry_context(stock: StockData) -> str:
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
        lines.append(f"- 总市值: {info['market_cap']/1e8:.1f}亿" if isinstance(info['market_cap'], (int, float)) else f"- 总市值: {info['market_cap']}")

    # 从财务数据提取行业相关指标
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        # 营收增速趋势（行业景气度指标）
        if "revenue_yoy" in df.columns:
            rev_vals = df["revenue_yoy"].dropna()
            if len(rev_vals) >= 2:
                lines.append(f"- 营收增速趋势: {' → '.join(f'{v:.1f}%' for v in rev_vals.tail(4))}")
                trend = "加速" if rev_vals.iloc[-1] > rev_vals.iloc[-2] else "减速"
                lines.append(f"- 增长方向: {trend}")

        # 毛利率趋势（行业供需指标）
        if "gross_margin" in df.columns:
            gm_vals = df["gross_margin"].dropna()
            if len(gm_vals) >= 2:
                lines.append(f"- 毛利率趋势: {' → '.join(f'{v:.1f}%' for v in gm_vals.tail(4))}")

        # 净利增速
        if "profit_yoy" in df.columns:
            pf_vals = df["profit_yoy"].dropna()
            if len(pf_vals) >= 1:
                lines.append(f"- 净利增速: {pf_vals.iloc[-1]:.1f}%")

    # 同行业个股对比
    peers = info.get("industry_peers", [])
    if peers:
        industry = info.get("industry", "")
        lines.append(f"\n### 同行业个股 ({industry}, {len(peers)}只)")
        lines.append("代码 | 名称 | 涨跌% | PE | PB | 换手率%")
        lines.append("---|---|---|---|---|---")
        for p in peers[:15]:  # 取前15
            sym = p.get("symbol", "")
            nm = p.get("name", "")
            chg = float(p.get("change_pct", 0) or 0)
            pe = p.get("pe", "-")
            pb = p.get("pb", "-")
            tr = p.get("turnover_rate", "-")
            lines.append(f"{sym} | {nm} | {chg:+.1f} | {pe} | {pb} | {tr}")

        # 行业均值
        pe_vals = [float(p.get("pe", 0) or 0) for p in peers if p.get("pe") and float(p.get("pe", 0) or 0) > 0]
        pb_vals = [float(p.get("pb", 0) or 0) for p in peers if p.get("pb") and float(p.get("pb", 0) or 0) > 0]
        if pe_vals:
            lines.append(f"\n行业PE均值: {sum(pe_vals)/len(pe_vals):.1f}, 中位数: {sorted(pe_vals)[len(pe_vals)//2]:.1f}")
        if pb_vals:
            lines.append(f"行业PB均值: {sum(pb_vals)/len(pb_vals):.1f}, 中位数: {sorted(pb_vals)[len(pb_vals)//2]:.1f}")

    if not lines:
        lines.append("基础数据有限，请基于你对该行业的知识进行分析")

    return "\n".join(lines)


def analyze_industry(stock: StockData) -> AgentSignal:
    """产业链分析主函数

    LLM-first模式：LLM直接给出-100~+100评分，代码仅做数据准备和边界验证。
    P0优化：添加量化锚点约束，LLM评分必须在锚点±30范围内。
    LLM不可用时返回锚点评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    industry_context = _build_industry_context(stock)
    anchor_score, anchor_explanation = _calc_industry_anchor(stock)

    # 锚点约束范围
    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    from src.research.extractor import get_research_context

    llm_result = llm_deep_analyze(
        agent_name="industry",
        template_name="industry.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "industry_context": industry_context,
            "anchor_score": str(anchor_score),
            "anchor_explanation": anchor_explanation,
            "anchor_score_low": str(anchor_low),
            "anchor_score_high": str(anchor_high),
            "research_context": get_research_context(stock, "industry"),
        },
    )

    score = llm_result["score"]

    # 锚点约束：LLM 评分偏离锚点超过 30 分时，拉回到锚点 ±30
    if abs(score - anchor_score) > 30:
        clamped = max(anchor_score - 30, min(anchor_score + 30, score))
        logger.info(f"[industry] 评分{score}偏离锚点{anchor_score}超过30分，修正为{clamped}")
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
        agent_name="industry",
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
