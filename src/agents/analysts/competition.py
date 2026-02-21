"""
竞争格局分析Agent

分析行业竞争态势、市场份额变化、竞争优势评估。
LLM-first模式 (Mode B): LLM做主角，代码仅提供基础数据和验证。
P1新增：覆盖之前9维分析中的盲区。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _calc_competition_anchor(stock: StockData) -> tuple[int, str]:
    """计算竞争格局量化锚点分数和说明

    基于财务数据和行业对比，评估公司的竞争地位。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    评估维度：
    1. 相对行业ROE位置（盈利能力竞争力）
    2. 相对行业毛利率位置（成本/定价竞争力）
    3. 市值排名（市场认可度）
    4. 营收增速相对行业（增长竞争力）

    Returns:
        (anchor_score, anchor_explanation)
    """
    score = 0
    details = []

    info = stock.info or {}
    peers = info.get("industry_peers", [])

    # 1. 相对行业ROE位置
    company_roe = None
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")
        if "roe" in df.columns:
            roe_vals = pd.to_numeric(df["roe"], errors="coerce").dropna()
            if len(roe_vals) >= 1:
                company_roe = float(roe_vals.iloc[-1])

    if company_roe is not None and peers:
        # 计算行业ROE分布
        peer_roes = []
        for p in peers:
            roe = p.get("roe")
            if roe and not (isinstance(roe, float) and np.isnan(roe)):
                try:
                    peer_roes.append(float(roe))
                except (ValueError, TypeError):
                    pass

        if peer_roes:
            peer_roes.sort()
            # 计算百分位
            rank = sum(1 for r in peer_roes if r < company_roe)
            percentile = rank / len(peer_roes) * 100

            if percentile >= 80:
                score += 15
                details.append(f"ROE排名行业前20%(第{percentile:.0f}百分位，{company_roe:.1f}%)→+15")
            elif percentile >= 60:
                score += 10
                details.append(f"ROE排名行业前40%(第{percentile:.0f}百分位，{company_roe:.1f}%)→+10")
            elif percentile >= 40:
                score += 3
                details.append(f"ROE排名行业中游(第{percentile:.0f}百分位，{company_roe:.1f}%)→+3")
            elif percentile < 20:
                score -= 12
                details.append(f"ROE排名行业后20%(第{percentile:.0f}百分位，{company_roe:.1f}%)→-12")

    # 2. 相对行业毛利率位置
    company_gm = None
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "gross_margin" in df.columns:
            gm_vals = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
            if len(gm_vals) >= 1:
                company_gm = float(gm_vals.iloc[-1])

    if company_gm is not None:
        # 毛利率水平直接评估
        if company_gm >= 45:
            score += 12
            details.append(f"毛利率{company_gm:.1f}%≥45%(成本/定价优势→+12)")
        elif company_gm >= 30:
            score += 6
            details.append(f"毛利率{company_gm:.1f}%(中等→+6)")
        elif company_gm < 15:
            score -= 10
            details.append(f"毛利率{company_gm:.1f}%<15%(成本劣势→-10)")

    # 3. 市值排名（市场认可度）
    company_mv = info.get("market_cap")
    if company_mv and peers:
        peer_mvs = []
        for p in peers:
            mv = p.get("market_cap") or p.get("total_mv")
            if mv and not (isinstance(mv, float) and np.isnan(mv)):
                try:
                    peer_mvs.append(float(mv))
                except (ValueError, TypeError):
                    pass

        if peer_mvs:
            # 计算市值排名
            larger = sum(1 for mv in peer_mvs if mv > company_mv)
            rank = larger + 1  # 排名（1=最大）
            total = len(peer_mvs) + 1

            if rank <= 3:
                score += 10
                details.append(f"市值行业第{rank}名/共{total}只(龙头→+10)")
            elif rank <= total // 3:
                score += 5
                details.append(f"市值行业前1/3(第{rank}名→+5)")
            elif rank > total * 2 // 3:
                score -= 5
                details.append(f"市值行业后1/3(第{rank}名→-5)")

    # 4. 营收增速相对行业（增长竞争力）
    company_rev_yoy = None
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "revenue_yoy" in df.columns:
            rev_vals = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
            if len(rev_vals) >= 1:
                company_rev_yoy = float(rev_vals.iloc[-1])

    if company_rev_yoy is not None and peers:
        peer_rev_yoys = []
        for p in peers:
            rev = p.get("revenue_yoy") or p.get("rev_yoy")
            if rev and not (isinstance(rev, float) and np.isnan(rev)):
                try:
                    peer_rev_yoys.append(float(rev))
                except (ValueError, TypeError):
                    pass

        if peer_rev_yoys:
            avg_peer_rev = sum(peer_rev_yoys) / len(peer_rev_yoys)
            rel_growth = company_rev_yoy - avg_peer_rev

            if rel_growth > 15:
                score += 12
                details.append(f"营收增速{company_rev_yoy:.1f}%高于行业均值{avg_peer_rev:.1f}%达{rel_growth:.1f}pp→+12")
            elif rel_growth > 5:
                score += 6
                details.append(f"营收增速{company_rev_yoy:.1f}%略高于行业均值→+6")
            elif rel_growth < -10:
                score -= 10
                details.append(f"营收增速{company_rev_yoy:.1f}%低于行业均值{rel_growth:.1f}pp→-10")

    anchor = max(-60, min(60, score))
    explanation = "；".join(details) if details else "数据不足"
    return anchor, explanation


def _build_competition_context(stock: StockData) -> str:
    """构建竞争格局分析的基础数据上下文"""
    lines = []

    info = stock.info or {}

    # 基本信息
    if info.get("industry"):
        lines.append(f"- 所属行业: {info['industry']}")
    if info.get("sector"):
        lines.append(f"- 所属板块: {info['sector']}")
    if info.get("main_business"):
        lines.append(f"- 主营业务: {info['main_business']}")
    if info.get("market_cap"):
        mv = info['market_cap']
        if isinstance(mv, (int, float)):
            lines.append(f"- 总市值: {mv/1e8:.1f}亿")

    # 从财务数据提取竞争力相关指标
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        # ROE（盈利能力竞争力）
        if "roe" in df.columns:
            roe_vals = pd.to_numeric(df["roe"], errors="coerce").dropna()
            if len(roe_vals) >= 2:
                lines.append(f"- ROE趋势: {' → '.join(f'{v:.1f}%' for v in roe_vals.tail(4))}")

        # 毛利率（成本/定价竞争力）
        if "gross_margin" in df.columns:
            gm_vals = pd.to_numeric(df["gross_margin"], errors="coerce").dropna()
            if len(gm_vals) >= 2:
                lines.append(f"- 毛利率趋势: {' → '.join(f'{v:.1f}%' for v in gm_vals.tail(4))}")

        # 营收增速（增长竞争力）
        if "revenue_yoy" in df.columns:
            rev_vals = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
            if len(rev_vals) >= 1:
                lines.append(f"- 营收增速: {rev_vals.iloc[-1]:.1f}%")

    # 同行业个股对比（核心数据）
    peers = info.get("industry_peers", [])
    if peers:
        industry = info.get("industry", "")
        lines.append(f"\n### 同行业竞争对手 ({industry}, {len(peers)}只)")
        lines.append("代码 | 名称 | 市值亿 | ROE% | 毛利率% | 营收增速% | PE")
        lines.append("---|---|---|---|---|---|---")

        # 按市值排序
        sorted_peers = sorted(
            peers,
            key=lambda p: float(p.get("market_cap") or p.get("total_mv") or 0),
            reverse=True
        )

        for p in sorted_peers[:12]:  # 取前12
            sym = p.get("symbol", "")
            nm = p.get("name", "")
            mv = p.get("market_cap") or p.get("total_mv")
            mv_str = f"{float(mv)/1e8:.0f}" if mv else "-"
            roe = p.get("roe", "-")
            roe_str = f"{float(roe):.1f}" if roe and not (isinstance(roe, float) and np.isnan(roe)) else "-"
            gm = p.get("gross_margin", "-")
            gm_str = f"{float(gm):.1f}" if gm and not (isinstance(gm, float) and np.isnan(gm)) else "-"
            rev = p.get("revenue_yoy") or p.get("rev_yoy", "-")
            rev_str = f"{float(rev):.1f}" if rev and not (isinstance(rev, float) and np.isnan(rev)) else "-"
            pe = p.get("pe", "-")
            pe_str = f"{float(pe):.1f}" if pe and not (isinstance(pe, float) and np.isnan(pe)) else "-"
            lines.append(f"{sym} | {nm} | {mv_str} | {roe_str} | {gm_str} | {rev_str} | {pe_str}")

        # 行业统计
        roe_vals = [float(p.get("roe", 0) or 0) for p in peers if p.get("roe") and not (isinstance(p.get("roe"), float) and np.isnan(p.get("roe")))]
        gm_vals = [float(p.get("gross_margin", 0) or 0) for p in peers if p.get("gross_margin") and not (isinstance(p.get("gross_margin"), float) and np.isnan(p.get("gross_margin")))]

        if roe_vals:
            lines.append(f"\n行业ROE: 均值{sum(roe_vals)/len(roe_vals):.1f}%, 中位数{sorted(roe_vals)[len(roe_vals)//2]:.1f}%")
        if gm_vals:
            lines.append(f"行业毛利率: 均值{sum(gm_vals)/len(gm_vals):.1f}%, 中位数{sorted(gm_vals)[len(gm_vals)//2]:.1f}%")

    if not lines:
        lines.append("基础数据有限，请基于你对该公司和行业的知识进行分析")

    return "\n".join(lines)


def analyze_competition(stock: StockData) -> AgentSignal:
    """竞争格局分析主函数

    LLM-first模式：LLM直接给出-100~+100评分，代码仅做数据准备和边界验证。
    P1新增：添加量化锚点约束，LLM评分必须在锚点±30范围内。
    LLM不可用时返回锚点评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    competition_context = _build_competition_context(stock)
    anchor_score, anchor_explanation = _calc_competition_anchor(stock)

    # 锚点约束范围
    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    from src.research.extractor import get_research_context

    llm_result = llm_deep_analyze(
        agent_name="competition",
        template_name="competition.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "competition_context": competition_context,
            "anchor_score": str(anchor_score),
            "anchor_explanation": anchor_explanation,
            "anchor_score_low": str(anchor_low),
            "anchor_score_high": str(anchor_high),
            "research_context": get_research_context(stock, "competition"),
        },
    )

    score = llm_result["score"]

    # 锚点约束：LLM 评分偏离锚点超过 30 分时，拉回到锚点 ±30
    if abs(score - anchor_score) > 30:
        clamped = max(anchor_score - 30, min(anchor_score + 30, score))
        logger.info(f"[competition] 评分{score}偏离锚点{anchor_score}超过30分，修正为{clamped}")
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
        agent_name="competition",
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
