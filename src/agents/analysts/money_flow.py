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


def _score_smart_money(stock: StockData) -> tuple[int, str]:
    """大资金画像评分 [-25, +25]

    综合北向资金、股东人数变化、龙虎榜+大宗交易三个维度。
    """
    from datetime import date, timedelta

    score = 0
    reasons = []

    # A. 北向资金（-10 ~ +10）
    nb = stock.info.get("northbound_holdings", [])
    if nb:
        latest = nb[-1] if isinstance(nb, list) else nb
        hold_ratio = float(latest.get("hold_ratio_float", 0) or 0)
        # 检查数据时效性：超过30天的change_shares不可信，只用持股比例
        nb_date = str(latest.get("date", ""))[:10]
        try:
            from datetime import datetime
            nb_dt = datetime.strptime(nb_date, "%Y-%m-%d").date()
            stale_days = (date.today() - nb_dt).days
        except (ValueError, TypeError):
            stale_days = 999

        if stale_days <= 30:
            # 数据新鲜，可用增减持方向
            change = float(latest.get("change_shares", 0) or 0)
            if change > 0:
                score += 5
                reasons.append("北向增持")
                if hold_ratio > 5:
                    score += 5
                    reasons.append(f"北向持股{hold_ratio:.1f}%")
            elif change < 0:
                score -= 5
                reasons.append("北向减持")
                if hold_ratio < 1:
                    score -= 5
                    reasons.append("北向持仓极低")
        else:
            # 数据过期，只参考持股比例存量
            if hold_ratio > 5:
                score += 3
                reasons.append(f"北向持股{hold_ratio:.1f}%(数据截至{nb_date})")
            elif hold_ratio < 0.5:
                score -= 3
                reasons.append(f"北向持仓极低{hold_ratio:.1f}%(数据截至{nb_date})")

    # B. 股东人数趋势（-10 ~ +10）
    sc = stock.info.get("shareholder_count", [])
    if len(sc) >= 2:
        recent = sc[-2:]
        pct_changes = [float(r.get("change_pct", 0) or 0) for r in recent]
        if all(p < 0 for p in pct_changes):
            score += 10
            reasons.append("连续2期股东减少(筹码集中)")
        elif all(p > 0 for p in pct_changes):
            score -= 10
            reasons.append("连续2期股东增加(筹码分散)")
        elif pct_changes[-1] < -5:
            score += 5
            reasons.append("最新期股东大幅减少")

    # C. 龙虎榜+大宗（-5 ~ +5）
    dt = stock.info.get("dragon_tiger", [])
    if dt:
        lhb = [d for d in dt if d.get("source") == "lhb"]
        dzjy = [d for d in dt if d.get("source") == "dzjy"]
        if lhb:
            net_buy = sum(float(d.get("net_buy_amount", 0) or 0) for d in lhb)
            if net_buy > 0:
                score += 3
                reasons.append("龙虎榜净买入")
            else:
                score -= 3
                reasons.append("龙虎榜净卖出")
        if dzjy:
            avg_premium = sum(float(d.get("premium_rate", 0) or 0) for d in dzjy) / len(dzjy)
            if avg_premium > 0:
                score += 2
                reasons.append("大宗溢价成交")
            elif avg_premium < -5:
                score -= 2
                reasons.append("大宗折价>5%")

    return max(-25, min(25, score)), "; ".join(reasons) if reasons else "无大资金数据"


def _score_margin(stock: StockData) -> tuple[int, str]:
    """融资融券评分 [-15, +15]

    融资余额增长 = 杠杆资金看多；融券余额增长 = 对冲/看空。
    """
    margin = stock.info.get("margin_data", [])
    if not margin:
        return 0, "无融资融券数据"

    score = 0
    reasons = []

    latest = margin[-1] if isinstance(margin, list) else margin
    margin_balance = float(latest.get("margin_balance", 0) or 0)
    short_balance = float(latest.get("short_balance", 0) or 0)

    # 融资余额趋势（需要多期数据）
    if len(margin) >= 2:
        prev = margin[-2]
        prev_margin = float(prev.get("margin_balance", 0) or 0)
        prev_short = float(prev.get("short_balance", 0) or 0)

        if prev_margin > 0:
            margin_chg = (margin_balance - prev_margin) / prev_margin * 100
            if margin_chg > 5:
                score += 8
                reasons.append(f"融资余额增{margin_chg:.1f}%(杠杆看多)")
            elif margin_chg > 0:
                score += 3
                reasons.append(f"融资余额微增{margin_chg:.1f}%")
            elif margin_chg < -5:
                score -= 8
                reasons.append(f"融资余额降{margin_chg:.1f}%(杠杆降低)")
            elif margin_chg < 0:
                score -= 3
                reasons.append(f"融资余额微降{margin_chg:.1f}%")

        if prev_short > 0:
            short_chg = (short_balance - prev_short) / prev_short * 100
            if short_chg > 10:
                score -= 5
                reasons.append(f"融券余额增{short_chg:.1f}%(空方加码)")
            elif short_chg < -10:
                score += 5
                reasons.append(f"融券余额降{short_chg:.1f}%(空方撤退)")
    else:
        # 只有单期数据，判断融资/融券比
        if margin_balance > 0 and short_balance > 0:
            ratio = margin_balance / short_balance
            if ratio > 10:
                score += 5
                reasons.append(f"融资/融券比={ratio:.0f}(多方主导)")
            elif ratio < 2:
                score -= 5
                reasons.append(f"融资/融券比={ratio:.1f}(空方活跃)")

    return max(-15, min(15, score)), "; ".join(reasons) if reasons else "融资融券中性"


def _build_smart_money_context(stock: StockData) -> str:
    """构建大资金画像上下文供LLM阅读"""
    lines = []

    # 北向
    nb = stock.info.get("northbound_holdings", [])
    if nb:
        from datetime import date as _date
        latest = nb[-1] if isinstance(nb, list) else nb
        hold_shares = float(latest.get("hold_shares", 0) or 0)
        hold_ratio = float(latest.get("hold_ratio_float", 0) or 0)
        nb_date = str(latest.get("date", ""))[:10]
        # 检查时效性
        try:
            from datetime import datetime
            stale_days = (_date.today() - datetime.strptime(nb_date, "%Y-%m-%d").date()).days
        except (ValueError, TypeError):
            stale_days = 999
        stale_tag = f" (数据截至{nb_date}，已过期{stale_days}天)" if stale_days > 30 else ""
        change = float(latest.get("change_shares", 0) or 0)
        lines.append(
            f"北向持股: {hold_shares/1e4:.0f}万股, "
            f"占流通股{hold_ratio:.2f}%{stale_tag}"
        )
        if stale_days <= 30:
            lines.append(f"  最新日增减: {change/1e4:.0f}万股")
            if len(nb) >= 5:
                recent_5 = nb[-5:]
                total_change = sum(float(r.get("change_shares", 0) or 0) for r in recent_5)
                lines.append(f"  近5日北向累计增减: {total_change/1e4:.0f}万股")

    # 股东人数
    sc = stock.info.get("shareholder_count", [])
    if sc:
        lines.append("股东人数变化:")
        for r in sc[-3:]:
            lines.append(
                f"  {r.get('date', '')}: 股东{r.get('count', 0)}户, "
                f"增减{float(r.get('change_pct', 0) or 0):+.1f}%"
            )

    # 龙虎榜+大宗
    dt = stock.info.get("dragon_tiger", [])
    if dt:
        lines.append("龙虎榜/大宗交易:")
        for d in dt[-5:]:
            src = "龙虎榜" if d.get("source") == "lhb" else "大宗交易"
            net = float(d.get("net_buy_amount", 0) or 0)
            line = f"  {d.get('date', '')} {src}: 净买{net/1e4:.0f}万"
            if d.get("reason"):
                line += f" ({d['reason'][:30]})"
            if d.get("premium_rate") is not None and d.get("source") == "dzjy":
                line += f" 折溢率{float(d.get('premium_rate', 0) or 0):+.1f}%"
            lines.append(line)

    # 融资融券
    margin = stock.info.get("margin_data", [])
    if margin:
        lines.append("融资融券:")
        for r in margin[-3:]:
            mb = float(r.get("margin_balance", 0) or 0)
            sb = float(r.get("short_balance", 0) or 0)
            lines.append(
                f"  {r.get('date', '')}: 融资余额{mb/1e8:.2f}亿, 融券余额{sb/1e8:.2f}亿"
            )

    return "\n".join(lines) if lines else "无大资金数据"


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

    # 大资金画像（北向+股东+龙虎榜）
    smart_score, smart_desc = _score_smart_money(stock)
    scores.append(smart_score)
    factors.append(f"大资金画像: {smart_desc}")

    # 融资融券
    margin_score, margin_desc = _score_margin(stock)
    scores.append(margin_score)
    factors.append(f"融资融券: {margin_desc}")

    # 代码评分（-100 ~ +100）
    total = sum(scores)
    # 满分约125（25+20+15+15+15+25+15），归一化到100
    code_score = max(-100, min(100, int(total * 100 / 125)))

    component_scores = {
        "main_trend": main_score,
        "order_divergence": div_score,
        "xl_orders": xl_score,
        "flow_price_corr": corr_score,
        "flow_intensity": intensity_score,
        "smart_money": smart_score,
        "margin": margin_score,
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

    smart_money_context = _build_smart_money_context(stock)

    llm_result = llm_enhance(
        agent_name="money_flow",
        template_name="money_flow.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "flow_summary": _build_flow_summary(df, component_scores),
            "flow_table": _build_flow_table(df),
            "smart_money_context": smart_money_context,
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
