"""
资金面分析Agent

分析主力资金流向、大单/小单博弈、连续流入/流出趋势。
数据来源：akshare stock_individual_fund_flow（东方财富资金流向）。

代码/LLM = 80/20: 资金流向大部分指标纯代码计算，仅趋势解读用LLM。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


# ─── 列名映射（akshare中文列 → 标准英文列）───────────────────

_COL_MAP = {
    "日期": "date",
    "收盘价": "close",
    "涨跌幅": "change_pct",
    "主力净流入-净额": "main_net",
    "主力净流入-净占比": "main_pct",
    "超大单净流入-净额": "xl_net",
    "超大单净流入-净占比": "xl_pct",
    "大单净流入-净额": "lg_net",
    "大单净流入-净占比": "lg_pct",
    "中单净流入-净额": "md_net",
    "中单净流入-净占比": "md_pct",
    "小单净流入-净额": "sm_net",
    "小单净流入-净占比": "sm_pct",
}


def _to_dataframe(data: list[dict[str, Any]]) -> pd.DataFrame:
    """将资金流向数据转为标准化DataFrame"""
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df = df.rename(columns=_COL_MAP)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # 数值列强制转换
    for col in ["main_net", "main_pct", "xl_net", "lg_net", "md_net", "sm_net",
                "xl_pct", "lg_pct", "md_pct", "sm_pct", "close", "change_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ─── 评分函数 ──────────────────────────────────────────────


def _score_main_force_trend(df: pd.DataFrame) -> tuple[int, str]:
    """主力资金趋势评分

    近5日主力持续净流入: +25
    近5日主力持续净流出: -25
    近3日流入加速: +15
    近3日流出加速: -15
    """
    if "main_net" not in df.columns or len(df) < 5:
        return 0, "主力数据不足"

    recent_5 = df["main_net"].iloc[-5:]
    recent_3 = df["main_net"].iloc[-3:]

    # 连续流入/流出
    if (recent_5 > 0).all():
        return 25, f"主力连续5日净流入，累计{recent_5.sum()/1e4:.0f}万"
    if (recent_5 < 0).all():
        return -25, f"主力连续5日净流出，累计{abs(recent_5.sum())/1e4:.0f}万"

    if (recent_3 > 0).all():
        # 检查是否加速
        if recent_3.iloc[-1] > recent_3.iloc[0]:
            return 20, f"主力连续3日净流入且加速"
        return 15, f"主力连续3日净流入"
    if (recent_3 < 0).all():
        if abs(recent_3.iloc[-1]) > abs(recent_3.iloc[0]):
            return -20, f"主力连续3日净流出且加速"
        return -15, f"主力连续3日净流出"

    # 近5日净额
    net_5 = recent_5.sum()
    if net_5 > 0:
        return 5, f"近5日主力净流入{net_5/1e4:.0f}万"
    if net_5 < 0:
        return -5, f"近5日主力净流出{abs(net_5)/1e4:.0f}万"
    return 0, "主力资金中性"


def _score_order_divergence(df: pd.DataFrame) -> tuple[int, str]:
    """大单/小单博弈评分

    大单流入 + 小单流出 = 主力吸筹: +20
    大单流出 + 小单流入 = 主力出货: -20
    """
    if not {"lg_net", "sm_net"}.issubset(df.columns) or len(df) < 3:
        return 0, "订单数据不足"

    recent_3 = df.iloc[-3:]
    lg_sum = recent_3["lg_net"].sum()
    sm_sum = recent_3["sm_net"].sum()

    if lg_sum > 0 and sm_sum < 0:
        return 20, f"大单流入+小单流出（主力吸筹信号）"
    if lg_sum < 0 and sm_sum > 0:
        return -20, f"大单流出+小单流入（主力出货信号）"
    if lg_sum > 0 and sm_sum > 0:
        return 10, "大小单同步流入"
    if lg_sum < 0 and sm_sum < 0:
        return -10, "大小单同步流出"
    return 0, "大小单博弈均衡"


def _score_xl_orders(df: pd.DataFrame) -> tuple[int, str]:
    """超大单评分（机构行为信号）

    超大单持续净流入是最强的机构买入信号。
    """
    if "xl_net" not in df.columns or len(df) < 3:
        return 0, "超大单数据不足"

    recent_3 = df["xl_net"].iloc[-3:]
    xl_sum = recent_3.sum()

    if (recent_3 > 0).all():
        return 15, f"超大单连续3日净流入（强机构买入）"
    if (recent_3 < 0).all():
        return -15, f"超大单连续3日净流出（机构抛售）"
    if xl_sum > 0:
        return 5, f"近3日超大单净流入{xl_sum/1e4:.0f}万"
    if xl_sum < 0:
        return -5, f"近3日超大单净流出{abs(xl_sum)/1e4:.0f}万"
    return 0, "超大单中性"


def _score_flow_price_correlation(df: pd.DataFrame) -> tuple[int, str]:
    """资金流向与价格走势相关性评分

    量价齐升（资金流入+股价涨）= 健康上涨: +10
    量价背离（资金流出+股价涨）= 诱多风险: -15
    底部吸筹（资金流入+股价跌）= 逆势建仓: +10
    """
    if not {"main_net", "change_pct"}.issubset(df.columns) or len(df) < 3:
        return 0, "相关性数据不足"

    recent_3 = df.iloc[-3:]
    main_sum = recent_3["main_net"].sum()
    price_change = recent_3["change_pct"].sum()

    if main_sum > 0 and price_change > 0:
        return 10, "量价齐升（资金流入+股价涨）"
    if main_sum < 0 and price_change > 0:
        return -15, "量价背离（资金流出+股价涨，诱多风险）"
    if main_sum > 0 and price_change < 0:
        return 10, "底部吸筹信号（资金流入+股价跌）"
    if main_sum < 0 and price_change < 0:
        return -10, "资金出逃（资金流出+股价跌）"
    return 0, "量价关系中性"


def _score_flow_intensity(df: pd.DataFrame) -> tuple[int, str]:
    """资金流入强度评分

    主力净占比连续3日 > 5%: +10
    主力净占比连续3日 < -5%: -10
    """
    if "main_pct" not in df.columns or len(df) < 3:
        return 0, "占比数据不足"

    recent_3 = df["main_pct"].iloc[-3:]
    avg_pct = recent_3.mean()

    if avg_pct > 10:
        return 15, f"主力净占比极高({avg_pct:.1f}%)"
    if avg_pct > 5:
        return 10, f"主力净占比较高({avg_pct:.1f}%)"
    if avg_pct < -10:
        return -15, f"主力净占比极低({avg_pct:.1f}%)"
    if avg_pct < -5:
        return -10, f"主力净占比偏低({avg_pct:.1f}%)"
    return 0, f"主力净占比{avg_pct:.1f}%"


# ─── 摘要构建 ──────────────────────────────────────────────


def _build_flow_summary(df: pd.DataFrame, component_scores: dict) -> str:
    """构建资金流向摘要供LLM阅读"""
    lines = []

    # 近5日汇总
    recent_5 = df.tail(5)
    if "main_net" in recent_5.columns:
        total_main = recent_5["main_net"].sum()
        lines.append(f"近5日主力净流入: {total_main/1e4:.0f}万元")
    if "xl_net" in recent_5.columns:
        total_xl = recent_5["xl_net"].sum()
        lines.append(f"近5日超大单净流入: {total_xl/1e4:.0f}万元")
    if "lg_net" in recent_5.columns:
        total_lg = recent_5["lg_net"].sum()
        lines.append(f"近5日大单净流入: {total_lg/1e4:.0f}万元")
    if "sm_net" in recent_5.columns:
        total_sm = recent_5["sm_net"].sum()
        lines.append(f"近5日小单净流入: {total_sm/1e4:.0f}万元")

    lines.append("")
    for name, score in component_scores.items():
        lines.append(f"{name}评分: {score:+d}")

    return "\n".join(lines)


def _build_flow_table(df: pd.DataFrame, n: int = 10) -> str:
    """构建近期资金流向表格供LLM阅读"""
    recent = df.tail(n)
    lines = ["日期 | 收盘 | 涨跌% | 主力净流入(万) | 主力占比% | 超大单(万) | 大单(万)"]
    lines.append("---|---|---|---|---|---|---")

    for _, row in recent.iterrows():
        dt = row.get("date", "")
        if hasattr(dt, "strftime"):
            dt = dt.strftime("%m-%d")
        close = row.get("close", 0)
        chg = row.get("change_pct", 0)
        main = row.get("main_net", 0) / 1e4 if pd.notna(row.get("main_net")) else 0
        main_pct = row.get("main_pct", 0) if pd.notna(row.get("main_pct")) else 0
        xl = row.get("xl_net", 0) / 1e4 if pd.notna(row.get("xl_net")) else 0
        lg = row.get("lg_net", 0) / 1e4 if pd.notna(row.get("lg_net")) else 0
        lines.append(
            f"{dt} | {close:.2f} | {chg:+.2f} | {main:+.0f} | {main_pct:+.1f} | {xl:+.0f} | {lg:+.0f}"
        )
    return "\n".join(lines)


# ─── 主函数 ────────────────────────────────────────────────


def analyze_money_flow(stock: StockData) -> AgentSignal:
    """资金面分析主函数

    80%代码量化评分 + 20%LLM趋势解读。
    LLM不可用时自动降级为纯代码分析。
    """
    start = time.time()

    df = _to_dataframe(stock.money_flow)
    if df.empty or "main_net" not in df.columns:
        return AgentSignal(
            agent_name="money_flow",
            signal_score=0,
            confidence=0.0,
            reasoning="无资金流向数据，无法进行资金面分析",
            data_quality=0.0,
        )

    # 各维度评分
    scores = []
    factors = []

    main_score, main_desc = _score_main_force_trend(df)
    scores.append(main_score)
    factors.append(f"主力趋势: {main_desc}")

    div_score, div_desc = _score_order_divergence(df)
    scores.append(div_score)
    factors.append(f"大小单博弈: {div_desc}")

    xl_score, xl_desc = _score_xl_orders(df)
    scores.append(xl_score)
    factors.append(f"超大单: {xl_desc}")

    corr_score, corr_desc = _score_flow_price_correlation(df)
    scores.append(corr_score)
    factors.append(f"量价相关: {corr_desc}")

    intensity_score, intensity_desc = _score_flow_intensity(df)
    scores.append(intensity_score)
    factors.append(f"流入强度: {intensity_desc}")

    # 代码评分（-100 ~ +100）
    total = sum(scores)
    # 满分约85（25+20+15+15+15），归一化到100
    code_score = max(-100, min(100, int(total * 100 / 85)))

    component_scores = {
        "main_trend": main_score,
        "order_divergence": div_score,
        "xl_orders": xl_score,
        "flow_price_corr": corr_score,
        "flow_intensity": intensity_score,
    }

    # 风险提示
    risks = []
    if corr_score < -10:
        risks.append("量价背离，注意诱多风险")
    if main_score < -15:
        risks.append("主力持续出逃")
    if div_score < -15:
        risks.append("大单出货+散户接盘")

    # --- LLM增强（可选）---
    from ..llm_enhance import llm_enhance
    from datetime import date

    llm_result = llm_enhance(
        agent_name="money_flow",
        template_name="money_flow.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "flow_summary": _build_flow_summary(df, component_scores),
            "flow_table": _build_flow_table(df),
        },
        code_score=code_score,
        code_reasoning=f"资金面综合评分{code_score}。" + "；".join(factors),
        code_factors=factors,
        code_risks=risks,
    )

    # 合并LLM结果
    signal_score = max(-100, min(100, code_score + llm_result["score_adjustment"]))
    reasoning = llm_result["reasoning"]
    all_factors = factors + llm_result["extra_factors"]
    all_risks = list(risks) + llm_result["extra_risks"]

    # 置信度（数据天数越多越可靠）
    data_days = len(df)
    confidence = min(1.0, data_days / 20)
    if llm_result["enhanced"]:
        confidence = min(1.0, confidence + 0.05)

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="money_flow",
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
        },
        execution_time_ms=elapsed_ms,
    )
