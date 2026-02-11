"""
基本面分析Agent

分析财务质量、盈利能力、成长性。
代码/LLM = 60/40: 财务指标代码计算，综合评估LLM辅助。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _to_dataframe(data: list[dict[str, Any]]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.sort_values("report_date").reset_index(drop=True)
    return df


def _score_profitability(df: pd.DataFrame) -> tuple[int, list[str]]:
    """盈利能力评分 (权重30%)"""
    if df.empty or "roe" not in df.columns:
        return 0, ["无盈利数据"]

    latest = df.iloc[-1]
    factors = []
    score = 0

    # ROE
    roe = latest.get("roe")
    if roe is not None and not np.isnan(roe):
        if roe > 20:
            score += 15
            factors.append(f"ROE={roe:.1f}% 优秀")
        elif roe > 12:
            score += 8
            factors.append(f"ROE={roe:.1f}% 良好")
        elif roe > 6:
            score += 0
            factors.append(f"ROE={roe:.1f}% 一般")
        else:
            score -= 10
            factors.append(f"ROE={roe:.1f}% 较差")

    # 毛利率
    gm = latest.get("gross_margin")
    if gm is not None and not np.isnan(gm):
        if gm > 40:
            score += 8
            factors.append(f"毛利率={gm:.1f}% 高")
        elif gm > 20:
            score += 3
            factors.append(f"毛利率={gm:.1f}% 中")
        else:
            score -= 5
            factors.append(f"毛利率={gm:.1f}% 低")

    # 净利率
    nm = latest.get("net_margin")
    if nm is not None and not np.isnan(nm):
        if nm > 15:
            score += 7
            factors.append(f"净利率={nm:.1f}% 高")
        elif nm > 5:
            score += 2
            factors.append(f"净利率={nm:.1f}% 中")
        else:
            score -= 5
            factors.append(f"净利率={nm:.1f}% 低")

    return score, factors


def _score_growth(df: pd.DataFrame) -> tuple[int, list[str]]:
    """成长性评分 (权重25%)"""
    if df.empty:
        return 0, ["无成长数据"]

    latest = df.iloc[-1]
    factors = []
    score = 0

    # 营收增速
    rev_yoy = latest.get("revenue_yoy")
    if rev_yoy is not None and not np.isnan(rev_yoy):
        if rev_yoy > 30:
            score += 12
            factors.append(f"营收增速={rev_yoy:.1f}% 高增长")
        elif rev_yoy > 10:
            score += 5
            factors.append(f"营收增速={rev_yoy:.1f}% 稳定增长")
        elif rev_yoy > 0:
            score += 0
            factors.append(f"营收增速={rev_yoy:.1f}% 低增长")
        else:
            score -= 10
            factors.append(f"营收增速={rev_yoy:.1f}% 负增长")

    # 净利增速
    profit_yoy = latest.get("profit_yoy")
    if profit_yoy is not None and not np.isnan(profit_yoy):
        if profit_yoy > 30:
            score += 12
            factors.append(f"净利增速={profit_yoy:.1f}% 高增长")
        elif profit_yoy > 10:
            score += 5
            factors.append(f"净利增速={profit_yoy:.1f}% 稳定增长")
        elif profit_yoy > 0:
            score += 0
            factors.append(f"净利增速={profit_yoy:.1f}% 低增长")
        else:
            score -= 10
            factors.append(f"净利增速={profit_yoy:.1f}% 负增长")

    return score, factors


def _score_financial_health(df: pd.DataFrame) -> tuple[int, list[str]]:
    """财务健康评分 (权重20%)"""
    if df.empty:
        return 0, ["无财务健康数据"]

    latest = df.iloc[-1]
    factors = []
    score = 0

    # 资产负债率
    dr = latest.get("debt_ratio")
    if dr is not None and not np.isnan(dr):
        if dr < 30:
            score += 8
            factors.append(f"资产负债率={dr:.1f}% 低")
        elif dr < 60:
            score += 3
            factors.append(f"资产负债率={dr:.1f}% 适中")
        else:
            score -= 8
            factors.append(f"资产负债率={dr:.1f}% 偏高")

    # 流动比率
    cr = latest.get("current_ratio")
    if cr is not None and not np.isnan(cr):
        if cr > 2.0:
            score += 5
            factors.append(f"流动比率={cr:.2f} 充足")
        elif cr > 1.0:
            score += 2
            factors.append(f"流动比率={cr:.2f} 适中")
        else:
            score -= 8
            factors.append(f"流动比率={cr:.2f} 偏低")

    return score, factors


def _score_cash_quality(df: pd.DataFrame) -> tuple[int, list[str]]:
    """现金流质量评分 (权重15%)"""
    if df.empty:
        return 0, ["无现金流数据"]

    latest = df.iloc[-1]
    factors = []
    score = 0

    ocf = latest.get("operating_cashflow")
    np_ = latest.get("net_profit")

    if ocf is not None and np_ is not None and not np.isnan(ocf) and not np.isnan(np_) and np_ > 0:
        ratio = ocf / np_
        if ratio > 1.0:
            score += 10
            factors.append(f"经营现金流/净利润={ratio:.2f} 质量优")
        elif ratio > 0.5:
            score += 3
            factors.append(f"经营现金流/净利润={ratio:.2f} 质量一般")
        else:
            score -= 8
            factors.append(f"经营现金流/净利润={ratio:.2f} 质量差")
    else:
        factors.append("现金流数据不完整")

    return score, factors


def analyze_fundamental(stock: StockData) -> AgentSignal:
    """基本面分析主函数"""
    start = time.time()

    df = _to_dataframe(stock.financial_data)
    if df.empty:
        return AgentSignal(
            agent_name="fundamental",
            signal_score=0,
            confidence=0.0,
            reasoning="无财务数据，无法进行基本面分析",
            data_quality=0.0,
        )

    all_factors = []
    all_risks = []
    total_score = 0

    # 各维度评分
    prof_score, prof_factors = _score_profitability(df)
    total_score += prof_score
    all_factors.extend(prof_factors)

    grow_score, grow_factors = _score_growth(df)
    total_score += grow_score
    all_factors.extend(grow_factors)

    health_score, health_factors = _score_financial_health(df)
    total_score += health_score
    all_factors.extend(health_factors)

    cash_score, cash_factors = _score_cash_quality(df)
    total_score += cash_score
    all_factors.extend(cash_factors)

    # 映射到 -100 ~ +100
    # 理论最大 = 15+8+7+12+12+8+5+10 = 77, 最小约 -64
    signal_score = max(-100, min(100, int(total_score * 100 / 77)))

    # 置信度：基于财报期数
    num_reports = len(df)
    confidence = min(1.0, num_reports / 8)  # 8个季度算完全可信

    # 风险检测
    latest = df.iloc[-1]
    if latest.get("debt_ratio", 0) > 70:
        all_risks.append("资产负债率超过70%，偿债风险较高")
    if latest.get("profit_yoy", 0) < -30:
        all_risks.append("净利润同比大幅下降超30%")

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="fundamental",
        signal_score=signal_score,
        confidence=round(confidence, 3),
        reasoning=f"基本面综合评分{signal_score}。" + "；".join(all_factors),
        key_factors=tuple(all_factors),
        risks=tuple(all_risks),
        data_quality=confidence,
        metadata={
            "total_raw_score": total_score,
            "component_scores": {
                "profitability": prof_score,
                "growth": grow_score,
                "financial_health": health_score,
                "cash_quality": cash_score,
            },
            "num_reports": num_reports,
        },
        execution_time_ms=elapsed_ms,
    )
