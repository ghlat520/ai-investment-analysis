"""
技术面分析Agent

复用 daily_stock_analysis 的 StockTrendAnalyzer 核心逻辑：
- MA多头/空头排列
- MACD金叉/死叉/背离
- RSI超买超卖
- 量能分析
- 乖离率

代码/LLM = 80/20: 大部分指标纯代码计算，仅形态识别用LLM。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _to_dataframe(quotes: list[dict[str, Any]]) -> pd.DataFrame:
    """将行情数据转为DataFrame"""
    if not quotes:
        return pd.DataFrame()
    df = pd.DataFrame(quotes)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


def _calc_ma(close: pd.Series, periods: list[int]) -> dict[str, pd.Series]:
    """计算多周期均线"""
    return {f"ma{p}": close.rolling(p).mean() for p in periods}


def _score_ma_arrangement(df: pd.DataFrame, ma_cols: dict[str, pd.Series]) -> tuple[int, str]:
    """均线排列评分

    多头排列(MA5>MA10>MA20>MA60): +30
    空头排列: -30
    缠绕: 0
    """
    if len(df) < 60:
        return 0, "数据不足"

    last = {k: v.iloc[-1] for k, v in ma_cols.items() if not np.isnan(v.iloc[-1])}
    if len(last) < 4:
        return 0, "均线数据不足"

    vals = [last.get(f"ma{p}", 0) for p in [5, 10, 20, 60]]
    if all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        return 30, "多头排列"
    if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
        return -30, "空头排列"
    return 0, "均线缠绕"


def _score_macd(df: pd.DataFrame) -> tuple[int, str]:
    """MACD评分"""
    close = df["close"]
    if len(close) < 35:
        return 0, "数据不足"

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    macd_hist = (dif - dea) * 2

    if len(macd_hist) < 3:
        return 0, "数据不足"

    curr_hist = macd_hist.iloc[-1]
    prev_hist = macd_hist.iloc[-2]

    # 金叉/死叉
    if prev_hist < 0 and curr_hist > 0:
        return 20, "MACD金叉"
    if prev_hist > 0 and curr_hist < 0:
        return -20, "MACD死叉"

    # 趋势
    if curr_hist > 0 and curr_hist > prev_hist:
        return 10, "MACD多头增强"
    if curr_hist < 0 and curr_hist < prev_hist:
        return -10, "MACD空头增强"

    return 0, "MACD中性"


def _score_rsi(df: pd.DataFrame, period: int = 14) -> tuple[int, str]:
    """RSI评分"""
    close = df["close"]
    if len(close) < period + 1:
        return 0, "数据不足"

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1]

    if np.isnan(rsi_val):
        return 0, "RSI计算异常"

    if rsi_val > 80:
        return -15, f"RSI={rsi_val:.1f} 严重超买"
    if rsi_val > 70:
        return -10, f"RSI={rsi_val:.1f} 超买"
    if rsi_val < 20:
        return 15, f"RSI={rsi_val:.1f} 严重超卖"
    if rsi_val < 30:
        return 10, f"RSI={rsi_val:.1f} 超卖"
    return 0, f"RSI={rsi_val:.1f} 中性"


def _score_volume(df: pd.DataFrame) -> tuple[int, str]:
    """量能评分"""
    if "volume" not in df.columns or len(df) < 20:
        return 0, "无量能数据"

    vol = df["volume"]
    avg_vol_5 = vol.iloc[-5:].mean()
    avg_vol_20 = vol.iloc[-20:].mean()

    if avg_vol_20 == 0:
        return 0, "成交量为零"

    vol_ratio = avg_vol_5 / avg_vol_20

    if vol_ratio > 2.0:
        return 10, f"显著放量(量比{vol_ratio:.1f})"
    if vol_ratio > 1.3:
        return 5, f"温和放量(量比{vol_ratio:.1f})"
    if vol_ratio < 0.5:
        return -5, f"显著缩量(量比{vol_ratio:.1f})"
    return 0, f"量能平稳(量比{vol_ratio:.1f})"


def _score_bias(df: pd.DataFrame) -> tuple[int, str]:
    """乖离率评分（价格偏离20日均线程度）"""
    close = df["close"]
    if len(close) < 20:
        return 0, "数据不足"

    ma20 = close.rolling(20).mean().iloc[-1]
    curr = close.iloc[-1]
    if ma20 == 0:
        return 0, "均线为零"

    bias = (curr - ma20) / ma20 * 100

    if bias > 15:
        return -10, f"乖离率{bias:.1f}%过高"
    if bias < -15:
        return 10, f"乖离率{bias:.1f}%过低"
    return 0, f"乖离率{bias:.1f}%正常"


def _build_indicators_summary(df: pd.DataFrame, ma_cols: dict, component_scores: dict) -> str:
    """构建指标摘要文本供LLM阅读"""
    close = df["close"].iloc[-1]
    lines = [f"最新价: {close:.2f}"]

    for p in [5, 10, 20, 60]:
        key = f"ma{p}"
        if key in ma_cols and not np.isnan(ma_cols[key].iloc[-1]):
            lines.append(f"MA{p}: {ma_cols[key].iloc[-1]:.2f}")

    for name, score in component_scores.items():
        lines.append(f"{name}评分: {score:+d}")

    if "change_pct" in df.columns:
        lines.append(f"今日涨跌幅: {df['change_pct'].iloc[-1]:.2f}%")

    return "\n".join(lines)


def _build_recent_quotes(df: pd.DataFrame, n: int = 10) -> str:
    """构建近期行情表格供LLM阅读"""
    recent = df.tail(n)
    lines = ["日期 | 收盘 | 涨跌幅% | 成交量"]
    lines.append("---|---|---|---")
    for _, row in recent.iterrows():
        dt = row.get("date", "")
        if hasattr(dt, "strftime"):
            dt = dt.strftime("%m-%d")
        close = row.get("close", 0)
        chg = row.get("change_pct", 0)
        vol = row.get("volume", 0)
        vol_str = f"{vol/10000:.0f}万" if vol > 10000 else str(int(vol))
        lines.append(f"{dt} | {close:.2f} | {chg:+.2f} | {vol_str}")
    return "\n".join(lines)


def analyze_technical(stock: StockData) -> AgentSignal:
    """技术面分析主函数

    80%代码量化评分 + 20%LLM定性分析（形态识别、支撑阻力、操作建议）。
    LLM不可用时自动降级为纯代码分析。
    """
    start = time.time()

    df = _to_dataframe(stock.daily_quotes)
    if df.empty or "close" not in df.columns:
        return AgentSignal(
            agent_name="technical",
            signal_score=0,
            confidence=0.0,
            reasoning="无行情数据，无法进行技术面分析",
            data_quality=0.0,
        )

    # 计算均线
    ma_cols = _calc_ma(df["close"], [5, 10, 20, 60, 120, 250])
    for name, series in ma_cols.items():
        df[name] = series

    # 各维度评分
    scores = []
    factors = []

    ma_score, ma_desc = _score_ma_arrangement(df, ma_cols)
    scores.append(ma_score)
    factors.append(f"均线: {ma_desc}")

    macd_score, macd_desc = _score_macd(df)
    scores.append(macd_score)
    factors.append(f"MACD: {macd_desc}")

    rsi_score, rsi_desc = _score_rsi(df)
    scores.append(rsi_score)
    factors.append(f"RSI: {rsi_desc}")

    vol_score, vol_desc = _score_volume(df)
    scores.append(vol_score)
    factors.append(f"量能: {vol_desc}")

    bias_score, bias_desc = _score_bias(df)
    scores.append(bias_score)
    factors.append(f"乖离率: {bias_desc}")

    # 代码评分（-100 ~ +100）
    total = sum(scores)
    code_score = max(-100, min(100, int(total * 100 / 85)))

    component_scores = {
        "ma": ma_score, "macd": macd_score, "rsi": rsi_score,
        "volume": vol_score, "bias": bias_score,
    }

    # 风险提示
    risks = []
    if rsi_score < -10:
        risks.append("RSI超买区间，注意回调风险")
    if bias_score < -5:
        risks.append("乖离率过高，短期可能回归均线")

    # --- LLM增强（可选）---
    from ..llm_enhance import llm_enhance
    from datetime import date

    llm_result = llm_enhance(
        agent_name="technical",
        template_name="technical.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "indicators_summary": _build_indicators_summary(df, ma_cols, component_scores),
            "recent_quotes": _build_recent_quotes(df),
        },
        code_score=code_score,
        code_reasoning=f"技术面综合评分{code_score}。" + "；".join(factors),
        code_factors=factors,
        code_risks=risks,
    )

    # 合并LLM结果
    signal_score = max(-100, min(100, code_score + llm_result["score_adjustment"]))
    reasoning = llm_result["reasoning"]
    all_factors = factors + llm_result["extra_factors"]
    all_risks = list(risks) + llm_result["extra_risks"]

    # 置信度
    data_days = len(df)
    confidence = min(1.0, data_days / 250)
    if llm_result["enhanced"]:
        confidence = min(1.0, confidence + 0.05)

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="technical",
        signal_score=signal_score,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        key_factors=tuple(all_factors),
        risks=tuple(all_risks),
        data_quality=confidence,
        llm_model=llm_result["llm_model"],
        llm_tokens_used=llm_result["llm_tokens"],
        llm_cost_usd=llm_result["llm_cost"],
        metadata={
            "total_raw_score": total,
            "code_score": code_score,
            "llm_adjustment": llm_result["score_adjustment"],
            "llm_enhanced": llm_result["enhanced"],
            "component_scores": component_scores,
            "data_days": data_days,
            "latest_close": float(df["close"].iloc[-1]),
        },
        execution_time_ms=elapsed_ms,
    )
