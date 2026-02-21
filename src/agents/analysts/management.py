"""
管理层质量评估Agent

分析管理层能力、诚信度、激励机制。
LLM-first模式 (Mode B): LLM做主角，代码仅提供基础数据和验证。
P2新增：覆盖投资分析中的重要维度。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _calc_management_anchor(stock: StockData) -> tuple[int, str]:
    """计算管理层质量量化锚点分数和说明

    基于财务数据和公司治理指标，评估管理层质量。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    评估维度：
    1. ROE稳定性（管理层经营能力的代理指标）
    2. 分红稳定性（股东回报意愿）
    3. 现金流质量（财务透明度代理）
    4. 负债率控制（风险意识）

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

    # 1. ROE稳定性（管理层经营能力）
    if "roe" in df.columns:
        roe = pd.to_numeric(df["roe"], errors="coerce").dropna()
        if len(roe) >= 4:
            roe_avg = float(roe.tail(8).mean())
            roe_std = float(roe.tail(8).std())

            # ROE水平
            if roe_avg >= 15:
                score += 15
                details.append(f"ROE均值{roe_avg:.1f}%≥15%(优秀经营能力→+15)")
            elif roe_avg >= 10:
                score += 8
                details.append(f"ROE均值{roe_avg:.1f}%≥10%(良好→+8)")
            elif roe_avg < 5:
                score -= 10
                details.append(f"ROE均值{roe_avg:.1f}%<5%(经营能力存疑→-10)")

            # ROE稳定性（管理层稳定性代理）
            if roe_std < 3:
                score += 10
                details.append(f"ROE标准差{roe_std:.1f}%<3%(经营稳定→+10)")
            elif roe_std > 10:
                score -= 8
                details.append(f"ROE标准差{roe_std:.1f}%>10%(经营波动大→-8)")

    # 2. 分红稳定性（股东回报意愿）
    info = stock.info or {}
    dividend_yield = info.get("dividend_yield", 0)
    if dividend_yield:
        try:
            div_yield = float(dividend_yield)
            if div_yield >= 2:
                score += 10
                details.append(f"股息率{div_yield:.2f}%≥2%(重视股东回报→+10)")
            elif div_yield >= 1:
                score += 5
                details.append(f"股息率{div_yield:.2f}%(有分红→+5)")
        except (ValueError, TypeError):
            pass

    # 3. 现金流质量（财务透明度代理）
    if "ocf_to_profit" in df.columns:
        ocf_ratio = pd.to_numeric(df["ocf_to_profit"], errors="coerce").dropna()
        if len(ocf_ratio) >= 3:
            ocf_avg = float(ocf_ratio.tail(4).mean())
            if ocf_avg >= 1.0:
                score += 12
                details.append(f"经营现金流/净利润{ocf_avg:.2f}≥1.0(盈利质量高→+12)")
            elif ocf_avg >= 0.7:
                score += 5
                details.append(f"经营现金流/净利润{ocf_avg:.2f}(正常→+5)")
            elif ocf_avg < 0.3:
                score -= 12
                details.append(f"经营现金流/净利润{ocf_avg:.2f}<0.3(盈利质量存疑→-12)")

    # 4. 负债率控制（风险意识）
    if "debt_ratio" in df.columns:
        debt = pd.to_numeric(df["debt_ratio"], errors="coerce").dropna()
        if len(debt) >= 2:
            debt_recent = float(debt.iloc[-1])
            debt_avg = float(debt.tail(4).mean())

            if debt_recent <= 40:
                score += 8
                details.append(f"资产负债率{debt_recent:.1f}%≤40%(财务稳健→+8)")
            elif debt_recent <= 60:
                score += 3
                details.append(f"资产负债率{debt_recent:.1f}%(适中→+3)")
            elif debt_recent > 75:
                score -= 10
                details.append(f"资产负债率{debt_recent:.1f}%>75%(财务风险高→-10)")

            # 负债率上升趋势
            if len(debt) >= 4:
                debt_trend = float(debt.tail(2).mean()) - float(debt.head(2).mean())
                if debt_trend > 10:
                    score -= 8
                    details.append(f"负债率上升{debt_trend:+.1f}pp(财务激进→-8)")
                elif debt_trend < -5:
                    score += 5
                    details.append(f"负债率下降{debt_trend:.1f}pp(去杠杆→+5)")

    # 5. 营收增长稳定性（战略执行力）
    if "revenue_yoy" in df.columns:
        rev_yoy = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
        if len(rev_yoy) >= 4:
            rev_avg = float(rev_yoy.tail(4).mean())
            positive_ratio = float((rev_yoy > 0).sum()) / len(rev_yoy)

            if positive_ratio >= 0.8 and rev_avg > 10:
                score += 8
                details.append(f"营收增长期占比{positive_ratio*100:.0f}%，均值{rev_avg:.1f}%(战略执行强→+8)")
            elif positive_ratio < 0.5:
                score -= 8
                details.append(f"营收增长期占比{positive_ratio*100:.0f}%<50%(战略执行弱→-8)")

    anchor = max(-60, min(60, score))
    explanation = "；".join(details) if details else "数据不足"
    return anchor, explanation


def _build_management_context(stock: StockData) -> str:
    """构建管理层分析的基础数据上下文"""
    lines = []

    info = stock.info or {}

    # 基本信息
    if info.get("industry"):
        lines.append(f"- 所属行业: {info['industry']}")
    if info.get("sector"):
        lines.append(f"- 所属板块: {info['sector']}")
    if info.get("market_cap"):
        mv = info['market_cap']
        if isinstance(mv, (int, float)):
            lines.append(f"- 总市值: {mv/1e8:.1f}亿")

    # 分红信息
    if info.get("dividend_yield"):
        lines.append(f"- 股息率: {info['dividend_yield']}%")
    if info.get("dividend_2025"):
        lines.append(f"- 分红方案: {info['dividend_2025']}")

    # 从财务数据提取管理层相关指标
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        # ROE趋势（管理层经营能力）
        if "roe" in df.columns:
            roe_vals = pd.to_numeric(df["roe"], errors="coerce").dropna()
            if len(roe_vals) >= 2:
                lines.append(f"- ROE趋势: {' → '.join(f'{v:.1f}%' for v in roe_vals.tail(4))}")

        # 负债率趋势（财务稳健性）
        if "debt_ratio" in df.columns:
            debt_vals = pd.to_numeric(df["debt_ratio"], errors="coerce").dropna()
            if len(debt_vals) >= 2:
                lines.append(f"- 资产负债率趋势: {' → '.join(f'{v:.1f}%' for v in debt_vals.tail(4))}")

        # 现金流质量
        if "ocf_to_profit" in df.columns:
            ocf_vals = pd.to_numeric(df["ocf_to_profit"], errors="coerce").dropna()
            if len(ocf_vals) >= 1:
                lines.append(f"- 经营现金流/净利润: {' → '.join(f'{v:.2f}' for v in ocf_vals.tail(4))}")

        # 营收增长
        if "revenue_yoy" in df.columns:
            rev_vals = pd.to_numeric(df["revenue_yoy"], errors="coerce").dropna()
            if len(rev_vals) >= 1:
                lines.append(f"- 营收增速趋势: {' → '.join(f'{v:.1f}%' for v in rev_vals.tail(4))}")

    # 高管信息（如果有）
    if info.get("executives"):
        lines.append(f"\n### 高管团队")
        for exec_info in info["executives"][:5]:
            name = exec_info.get("name", "")
            title = exec_info.get("title", "")
            lines.append(f"- {name}: {title}")

    # 减持记录（如果有）
    if info.get("insider_selling"):
        lines.append(f"\n### 近期减持记录")
        for sale in info["insider_selling"][:5]:
            name = sale.get("name", "")
            amount = sale.get("amount", "")
            date = sale.get("date", "")
            lines.append(f"- {name}: {amount} ({date})")

    if not lines:
        lines.append("基础数据有限，请基于你对该公司和管理层的知识进行分析")

    return "\n".join(lines)


def analyze_management(stock: StockData) -> AgentSignal:
    """管理层质量分析主函数

    LLM-first模式：LLM直接给出-100~+100评分，代码仅做数据准备和边界验证。
    P2新增：添加量化锚点约束，LLM评分必须在锚点±30范围内。
    LLM不可用时返回锚点评分。
    """
    start = time.time()

    from datetime import date
    from ..llm_enhance import llm_deep_analyze

    management_context = _build_management_context(stock)
    anchor_score, anchor_explanation = _calc_management_anchor(stock)

    # 锚点约束范围
    anchor_low = max(-100, anchor_score - 30)
    anchor_high = min(100, anchor_score + 30)

    from src.research.extractor import get_research_context

    llm_result = llm_deep_analyze(
        agent_name="management",
        template_name="management.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "management_context": management_context,
            "anchor_score": str(anchor_score),
            "anchor_explanation": anchor_explanation,
            "anchor_score_low": str(anchor_low),
            "anchor_score_high": str(anchor_high),
            "research_context": get_research_context(stock, "management"),
        },
    )

    score = llm_result["score"]

    # 锚点约束：LLM 评分偏离锚点超过 30 分时，拉回到锚点 ±30
    if abs(score - anchor_score) > 30:
        clamped = max(anchor_score - 30, min(anchor_score + 30, score))
        logger.info(f"[management] 评分{score}偏离锚点{anchor_score}超过30分，修正为{clamped}")
        score = clamped

    reasoning = llm_result["reasoning"]
    factors = llm_result["extra_factors"]
    risks = llm_result["extra_risks"]

    # 置信度：LLM可用时较高
    if llm_result["enhanced"]:
        confidence = 0.6  # 管理层评估主观性强，置信度略低
    else:
        confidence = 0.2
        score = anchor_score  # LLM不可用时使用锚点分数

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="management",
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
