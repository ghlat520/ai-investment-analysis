"""
估值分析Agent

评估当前估值水平：PE/PB分位、PEG、行业对比。
代码/LLM = 70/30: 估值指标代码计算，综合评估LLM辅助。
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
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _calc_percentile(values: pd.Series, current: float) -> float:
    """计算当前值在历史序列中的百分位"""
    valid = values.dropna()
    if len(valid) < 10:
        return 0.5  # 数据不足返回中位
    return float((valid < current).sum() / len(valid))


def _score_pe_percentile(pe_ttm: float, pe_history: pd.Series) -> tuple[int, str]:
    """PE分位评分"""
    if np.isnan(pe_ttm) or pe_ttm <= 0:
        return 0, "PE为负或无效"

    pct = _calc_percentile(pe_history, pe_ttm)

    if pct < 0.1:
        return 20, f"PE={pe_ttm:.1f}，处于{pct*100:.0f}%分位（极度低估）"
    if pct < 0.3:
        return 12, f"PE={pe_ttm:.1f}，处于{pct*100:.0f}%分位（低估）"
    if pct < 0.7:
        return 0, f"PE={pe_ttm:.1f}，处于{pct*100:.0f}%分位（合理）"
    if pct < 0.9:
        return -12, f"PE={pe_ttm:.1f}，处于{pct*100:.0f}%分位（偏高）"
    return -20, f"PE={pe_ttm:.1f}，处于{pct*100:.0f}%分位（严重高估）"


def _score_pb_percentile(pb: float, pb_history: pd.Series) -> tuple[int, str]:
    """PB分位评分"""
    if np.isnan(pb) or pb <= 0:
        return 0, "PB为负或无效"

    pct = _calc_percentile(pb_history, pb)

    if pct < 0.1:
        return 15, f"PB={pb:.2f}，处于{pct*100:.0f}%分位（极度低估）"
    if pct < 0.3:
        return 8, f"PB={pb:.2f}，处于{pct*100:.0f}%分位（低估）"
    if pct < 0.7:
        return 0, f"PB={pb:.2f}，处于{pct*100:.0f}%分位（合理）"
    if pct < 0.9:
        return -8, f"PB={pb:.2f}，处于{pct*100:.0f}%分位（偏高）"
    return -15, f"PB={pb:.2f}，处于{pct*100:.0f}%分位（严重高估）"


def _score_peg(pe_ttm: float, profit_growth: float) -> tuple[int, str]:
    """PEG评分"""
    if np.isnan(pe_ttm) or pe_ttm <= 0 or np.isnan(profit_growth) or profit_growth <= 0:
        return 0, "PEG无法计算（PE或增速不可用）"

    peg = pe_ttm / profit_growth

    if peg < 0.5:
        return 15, f"PEG={peg:.2f} 极低（强烈低估）"
    if peg < 1.0:
        return 8, f"PEG={peg:.2f} 偏低（低估）"
    if peg < 1.5:
        return 0, f"PEG={peg:.2f} 合理"
    if peg < 2.5:
        return -8, f"PEG={peg:.2f} 偏高"
    return -15, f"PEG={peg:.2f} 过高（高估）"


def analyze_valuation(stock: StockData) -> AgentSignal:
    """估值分析主函数"""
    start = time.time()

    fin_df = _to_dataframe(stock.financial_data)
    quote_df = _to_dataframe(stock.daily_quotes)

    if fin_df.empty:
        return AgentSignal(
            agent_name="valuation",
            signal_score=0,
            confidence=0.0,
            reasoning="无财务/估值数据",
            data_quality=0.0,
        )

    all_factors = []
    all_risks = []
    total_score = 0

    latest = fin_df.iloc[-1]

    # PE分位评分
    pe_ttm = latest.get("pe_ttm", np.nan)
    pe_history = fin_df["pe_ttm"].dropna() if "pe_ttm" in fin_df.columns else pd.Series(dtype=float)
    pe_score, pe_desc = _score_pe_percentile(pe_ttm, pe_history)
    total_score += pe_score
    all_factors.append(pe_desc)

    # PB分位评分
    pb = latest.get("pb", np.nan)
    pb_history = fin_df["pb"].dropna() if "pb" in fin_df.columns else pd.Series(dtype=float)
    pb_score, pb_desc = _score_pb_percentile(pb, pb_history)
    total_score += pb_score
    all_factors.append(pb_desc)

    # PEG评分
    profit_yoy = latest.get("profit_yoy", np.nan)
    peg_score, peg_desc = _score_peg(pe_ttm, profit_yoy)
    total_score += peg_score
    all_factors.append(peg_desc)

    # 映射到 -100 ~ +100
    # 最大 = 20+15+15 = 50, 最小 = -50
    signal_score = max(-100, min(100, int(total_score * 100 / 50)))

    # 置信度
    num_reports = len(fin_df)
    confidence = min(1.0, num_reports / 8)

    # 风险
    if pe_score < -10:
        all_risks.append("PE处于历史高位，估值压力大")
    if pb_score < -10:
        all_risks.append("PB处于历史高位")

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="valuation",
        signal_score=signal_score,
        confidence=round(confidence, 3),
        reasoning=f"估值综合评分{signal_score}。" + "；".join(all_factors),
        key_factors=tuple(all_factors),
        risks=tuple(all_risks),
        data_quality=confidence,
        metadata={
            "pe_ttm": float(pe_ttm) if not np.isnan(pe_ttm) else None,
            "pb": float(pb) if not np.isnan(pb) else None,
            "pe_percentile": _calc_percentile(pe_history, pe_ttm) if not np.isnan(pe_ttm) else None,
            "pb_percentile": _calc_percentile(pb_history, pb) if not np.isnan(pb) else None,
            "component_scores": {
                "pe": pe_score,
                "pb": pb_score,
                "peg": peg_score,
            },
        },
        execution_time_ms=elapsed_ms,
    )
