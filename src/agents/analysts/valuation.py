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


def _build_valuation_summary(
    pe_ttm: float, pb: float, pe_pct: float | None, pb_pct: float | None,
    profit_yoy: float, val_days: int,
) -> str:
    """构建估值数据摘要供LLM阅读"""
    lines = []
    if not np.isnan(pe_ttm):
        lines.append(f"- PE(TTM): {pe_ttm:.2f}")
        if pe_pct is not None:
            lines.append(f"- PE历史分位: {pe_pct*100:.1f}%（{val_days}天数据）")
    if not np.isnan(pb):
        lines.append(f"- PB: {pb:.2f}")
        if pb_pct is not None:
            lines.append(f"- PB历史分位: {pb_pct*100:.1f}%")
    if not np.isnan(profit_yoy):
        lines.append(f"- 净利润同比增速: {profit_yoy:.1f}%")
        if not np.isnan(pe_ttm) and profit_yoy > 0:
            peg = pe_ttm / profit_yoy
            lines.append(f"- PEG: {peg:.2f}")
    return "\n".join(lines) if lines else "无估值数据"


def _build_valuation_trend(val_df: pd.DataFrame) -> str:
    """构建估值历史趋势（近6个月月度快照）"""
    if val_df.empty or "date" not in val_df.columns:
        return "无趋势数据"

    # 每月取最后一条，最近6个月
    df = val_df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month").last().tail(6)

    lines = ["月份 | PE(TTM) | PB"]
    lines.append("---|---|---")
    for idx, row in monthly.iterrows():
        pe = f"{row.get('pe_ttm', 0):.1f}" if row.get("pe_ttm") is not None else "-"
        pb = f"{row.get('pb', 0):.2f}" if row.get("pb") is not None else "-"
        lines.append(f"{idx} | {pe} | {pb}")
    return "\n".join(lines)


def analyze_valuation(stock: StockData) -> AgentSignal:
    """估值分析主函数

    优先使用 stock.info["valuation_history"] 中的3年日频PE/PB历史数据，
    比财报中的季度数据更精确地计算百分位。
    70%代码量化评分 + 30%LLM定性分析（行业对比、估值陷阱、驱动因素）。
    """
    start = time.time()

    fin_df = _to_dataframe(stock.financial_data)
    val_df = _to_dataframe(stock.info.get("valuation_history", []))

    if fin_df.empty and val_df.empty:
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

    # 确定当前PE/PB值：优先用估值历史最新值，其次用财报数据
    if not val_df.empty:
        latest_val = val_df.iloc[-1]
        pe_ttm = float(latest_val.get("pe_ttm", np.nan))
        pb = float(latest_val.get("pb", np.nan))
        pe_history = pd.to_numeric(val_df["pe_ttm"], errors="coerce").dropna() if "pe_ttm" in val_df.columns else pd.Series(dtype=float)
        pb_history = pd.to_numeric(val_df["pb"], errors="coerce").dropna() if "pb" in val_df.columns else pd.Series(dtype=float)
        val_days = len(val_df)
    elif not fin_df.empty:
        latest_val = fin_df.iloc[-1]
        pe_ttm = float(latest_val.get("pe_ttm", np.nan))
        pb = float(latest_val.get("pb", np.nan))
        pe_history = pd.to_numeric(fin_df["pe_ttm"], errors="coerce").dropna() if "pe_ttm" in fin_df.columns else pd.Series(dtype=float)
        pb_history = pd.to_numeric(fin_df["pb"], errors="coerce").dropna() if "pb" in fin_df.columns else pd.Series(dtype=float)
        val_days = 0
    else:
        pe_ttm = np.nan
        pb = np.nan
        pe_history = pd.Series(dtype=float)
        pb_history = pd.Series(dtype=float)
        val_days = 0

    # PE分位评分
    pe_score, pe_desc = _score_pe_percentile(pe_ttm, pe_history)
    total_score += pe_score
    all_factors.append(pe_desc)

    # PB分位评分
    pb_score, pb_desc = _score_pb_percentile(pb, pb_history)
    total_score += pb_score
    all_factors.append(pb_desc)

    # PEG评分（需要财报中的利润增速）
    profit_yoy = np.nan
    if not fin_df.empty:
        profit_yoy = float(fin_df.iloc[-1].get("profit_yoy", np.nan))
    peg_score, peg_desc = _score_peg(pe_ttm, profit_yoy)
    total_score += peg_score
    all_factors.append(peg_desc)

    # 代码评分（-100 ~ +100）
    code_score = max(-100, min(100, int(total_score * 100 / 50)))

    # 置信度：日频估值数据充足则高置信度
    if val_days > 500:
        confidence = 0.9
    elif val_days > 200:
        confidence = 0.7
    elif val_days > 0:
        confidence = 0.5
    else:
        confidence = min(1.0, len(fin_df) / 8) * 0.5

    # 风险
    if pe_score < -10:
        all_risks.append("PE处于历史高位，估值压力大")
    if pb_score < -10:
        all_risks.append("PB处于历史高位")

    pe_pct = _calc_percentile(pe_history, pe_ttm) if not np.isnan(pe_ttm) and len(pe_history) > 0 else None
    pb_pct = _calc_percentile(pb_history, pb) if not np.isnan(pb) and len(pb_history) > 0 else None

    # --- LLM增强（可选）---
    from ..llm_enhance import llm_enhance
    from datetime import date

    llm_result = llm_enhance(
        agent_name="valuation",
        template_name="valuation.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "valuation_summary": _build_valuation_summary(
                pe_ttm, pb, pe_pct, pb_pct, profit_yoy, val_days,
            ),
            "valuation_trend": _build_valuation_trend(val_df),
        },
        code_score=code_score,
        code_reasoning=f"估值综合评分{code_score}（{val_days}天历史数据）。" + "；".join(all_factors),
        code_factors=all_factors,
        code_risks=all_risks,
    )

    # 合并LLM结果
    signal_score = max(-100, min(100, code_score + llm_result["score_adjustment"]))
    reasoning = llm_result["reasoning"]
    final_factors = all_factors + llm_result["extra_factors"]
    final_risks = list(all_risks) + llm_result["extra_risks"]

    if llm_result["enhanced"]:
        confidence = min(1.0, confidence + 0.05)

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="valuation",
        signal_score=signal_score,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        key_factors=tuple(final_factors),
        risks=tuple(final_risks),
        data_quality=confidence,
        llm_model=llm_result["llm_model"],
        llm_tokens_used=llm_result["llm_tokens"],
        llm_cost_usd=llm_result["llm_cost"],
        metadata={
            "pe_ttm": float(pe_ttm) if not np.isnan(pe_ttm) else None,
            "pb": float(pb) if not np.isnan(pb) else None,
            "pe_percentile": pe_pct,
            "pb_percentile": pb_pct,
            "valuation_history_days": val_days,
            "code_score": code_score,
            "llm_adjustment": llm_result["score_adjustment"],
            "llm_enhanced": llm_result["enhanced"],
            "component_scores": {
                "pe": pe_score,
                "pb": pb_score,
                "peg": peg_score,
            },
        },
        execution_time_ms=elapsed_ms,
    )
