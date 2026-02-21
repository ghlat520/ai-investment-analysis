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
    """PE分位评分（仅在正PE区间内计算百分位，排除亏损期负PE污染）"""
    if np.isnan(pe_ttm) or pe_ttm <= 0:
        return 0, "PE为负或无效"

    # 过滤掉负PE（亏损期），仅在盈利期内计算百分位
    positive_pe = pe_history[pe_history > 0]
    pct = _calc_percentile(positive_pe, pe_ttm)

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


def _calc_from_yaml(yaml_model: dict) -> dict:
    """从YAML精细预测模型计算各年净利润/EPS/目标价"""
    segments = yaml_model.get("segments", [])
    period_costs = yaml_model.get("period_costs", {})
    total_shares_b = yaml_model.get("total_shares_billion", 1.0)
    pe_assumption = yaml_model.get("pe_assumption", {})

    # 收集所有预测年份
    years = set()
    for seg in segments:
        years.update(seg.get("projections", {}).keys())
    years = sorted(years)

    yearly = []
    for year in years:
        gross_profit = 0.0
        segment_details = []
        for seg in segments:
            rev = seg.get("projections", {}).get(year, 0)
            margin = seg.get("margin_pct", 0) / 100.0
            gp = rev * margin
            gross_profit += gp
            segment_details.append({
                "name": seg["name"],
                "revenue": rev,
                "margin_pct": seg.get("margin_pct", 0),
                "gross_profit": round(gp, 2),
            })

        cost = period_costs.get(year, 0)
        net_profit = gross_profit - cost
        # net_profit单位=亿元, total_shares_b单位=十亿股, 需转为亿股
        total_shares_yi = total_shares_b * 10
        eps = net_profit / total_shares_yi if total_shares_yi > 0 else 0

        # PE取值
        pe = _get_segment_weighted_pe(segments, year, pe_assumption)
        target_price = eps * pe

        yearly.append({
            "year": year,
            "total_revenue": round(sum(s["revenue"] for s in segment_details), 2),
            "gross_profit": round(gross_profit, 2),
            "period_cost": cost,
            "net_profit": round(net_profit, 2),
            "eps": round(eps, 2),
            "pe": pe,
            "target_price": round(target_price, 2),
            "segments": segment_details,
        })

    return {
        "source": "yaml_model",
        "data_source": yaml_model.get("data_source", "用户自有研究"),
        "last_updated": yaml_model.get("last_updated", ""),
        "total_shares_billion": total_shares_b,
        "yearly_forecast": yearly,
    }


def _get_segment_weighted_pe(segments: list, year: int, pe_assumption: dict) -> float:
    """计算分业务加权PE"""
    method = pe_assumption.get("method", "fixed")
    if method == "segment_weighted":
        segments_pe = pe_assumption.get("segments_pe", {})
        total_rev = 0.0
        weighted_pe = 0.0
        for seg in segments:
            rev = seg.get("projections", {}).get(year, 0)
            # 匹配PE：按segment名称关键词匹配
            seg_pe = _match_segment_pe(seg["name"], segments_pe)
            weighted_pe += rev * seg_pe
            total_rev += rev
        return round(weighted_pe / total_rev, 1) if total_rev > 0 else 30.0
    return pe_assumption.get("pe", 30.0)


def _match_segment_pe(seg_name: str, segments_pe: dict) -> float:
    """根据业务线名称匹配PE倍数"""
    for keyword, pe in segments_pe.items():
        if keyword in seg_name:
            return pe
    # 默认PE
    return 20.0


def _build_auto_context(biz_comp: list[dict], forecast: list[dict]) -> dict:
    """用API数据构建分业务上下文（无YAML时的自动模式）"""
    result = {"source": "auto_api", "yearly_forecast": []}
    parts = []

    if biz_comp:
        parts.append("### 分业务/分产品营收构成（最新报告期）")
        parts.append("| 产品/业务 | 营收(元) | 占比 | 毛利率 | 分类 |")
        parts.append("|----------|---------|------|--------|------|")
        for item in biz_comp[:15]:
            parts.append(
                f"| {item.get('product', '-')} "
                f"| {item.get('revenue', '-')} "
                f"| {item.get('revenue_pct', '-')}% "
                f"| {item.get('gross_margin', '-')}% "
                f"| {item.get('classification', '-')} |"
            )

    if forecast:
        parts.append("\n### 券商盈利预测（一致预期EPS）")
        parts.append("| 年度 | 机构数 | EPS最低 | EPS均值 | EPS最高 | 行业均值 |")
        parts.append("|------|-------|--------|--------|--------|---------|")
        for item in forecast[:5]:
            parts.append(
                f"| {item.get('year', '-')} "
                f"| {item.get('num_institutions', '-')} "
                f"| {item.get('eps_min', '-')} "
                f"| {item.get('eps_mean', '-')} "
                f"| {item.get('eps_max', '-')} "
                f"| {item.get('industry_avg', '-')} |"
            )

    result["context_text"] = "\n".join(parts) if parts else ""
    return result


def _build_segment_forecast_context(stock: StockData) -> tuple[str, dict]:
    """构建分业务线前瞻估值上下文

    Returns:
        (context_text, segment_result) — 文本注入LLM + 结构化数据存metadata
    """
    yaml_model = stock.info.get("segment_model")
    biz_comp = stock.info.get("business_composition", [])
    forecast = stock.info.get("profit_forecast", [])

    if yaml_model:
        # YAML精确计算模式
        segment_result = _calc_from_yaml(yaml_model)
        yearly = segment_result.get("yearly_forecast", [])

        lines = [
            "### 分业务线前瞻盈利预测（YAML精细模型，代码已算好）",
            f"数据来源: {segment_result.get('data_source', '用户研究')}",
            f"总股本: {segment_result.get('total_shares_billion', 0)}亿股",
            "",
            "| 年份 | 总营收(亿) | 毛利(亿) | 期间费用(亿) | 净利润(亿) | EPS(元) | PE | 目标价(元) |",
            "|------|----------|---------|------------|----------|---------|----|---------:|",
        ]
        for y in yearly:
            lines.append(
                f"| {y['year']} | {y['total_revenue']} | {y['gross_profit']} "
                f"| {y['period_cost']} | {y['net_profit']} | {y['eps']} "
                f"| {y['pe']} | {y['target_price']} |"
            )

        # 各业务线明细（取最后一年展示）
        if yearly:
            last = yearly[-1]
            lines.append(f"\n#### {last['year']}年各业务线明细")
            lines.append("| 业务线 | 营收(亿) | 毛利率 | 毛利(亿) |")
            lines.append("|--------|---------|--------|---------|")
            for seg in last["segments"]:
                lines.append(
                    f"| {seg['name']} | {seg['revenue']} | {seg['margin_pct']}% | {seg['gross_profit']} |"
                )

        lines.append("\n**【铁律】以上净利润/EPS/目标价已由代码精确计算。你必须直接引用这些目标价填入target_prices字段（第1年=conservative，第2年=base，最后1年=optimistic），严禁自行用PEG/DCF等公式重新计算目标价。你的任务仅限于：(1)验证各业务线增速假设是否合理 (2)识别假设过于乐观/保守的风险 (3)与券商一致预期交叉对比偏差。**")
        return "\n".join(lines), segment_result

    elif biz_comp or forecast:
        # 自动API模式
        segment_result = _build_auto_context(biz_comp, forecast)
        return segment_result.get("context_text", ""), segment_result

    return "", {}


def _merge_precision_warnings(
    llm_warnings: list,
    segment_result: dict,
    data_warnings: list,
    stock: "StockData",
) -> list:
    """合并 LLM 输出的 precision_warnings 与代码侧兜底检测。

    即使 LLM 漏填，代码也会补上关键降级警告。
    """
    warnings = list(llm_warnings) if llm_warnings else []
    existing_types = {w.get("type") for w in warnings if isinstance(w, dict)}

    # 兜底1：无分业务数据
    if not segment_result and "missing_segment_data" not in existing_types:
        warnings.append({
            "type": "missing_segment_data",
            "severity": "high",
            "message": "缺少分业务营收构成及前瞻预测数据，估值采用整体法",
            "impact": "无法区分高增长与低增长业务线的差异化价值，整体PE/PEG可能高估或低估部分业务板块",
            "suggestion": "关注公司年报中的分业务披露，或等待YAML精细模型配置后重新分析",
        })

    # 兜底2：数据滞后
    for w in data_warnings:
        if "行情数据距今" in w and "stale_data" not in existing_types:
            warnings.append({
                "type": "stale_data",
                "severity": "medium",
                "message": w,
                "impact": "估值基于非最新行情，目标价与实际偏差可能较大",
                "suggestion": "待开盘后获取最新行情再重新分析",
            })
            existing_types.add("stale_data")

    # 兜底3：无券商研报覆盖
    research = stock.info.get("research_reports", {})
    has_research = bool(research.get("reports") or research.get("institution_count"))
    if not has_research and "missing_research" not in existing_types:
        warnings.append({
            "type": "missing_research",
            "severity": "medium",
            "message": "无券商研报覆盖，缺少专业机构估值交叉验证",
            "impact": "目标价仅基于量化模型推算，无法与卖方一致预期对比",
            "suggestion": "参考同行业可比公司的券商覆盖情况辅助判断",
        })

    return warnings


def _calc_auto_target_prices(
    pe_ttm: float,
    pe_history: pd.Series,
    profit_yoy: float,
    current_price: float,
    forecast_eps: list[dict],
) -> dict | None:
    """无YAML时，代码用历史PE分位+前瞻EPS自动计算目标价

    核心思想：用真实历史PE区间锚定，不依赖LLM猜PE。
    - 保守PE = 历史正PE的25分位
    - 中性PE = 历史正PE的50分位（中位数）
    - 乐观PE = 历史正PE的75分位
    - EPS优先用券商一致预期均值，否则用TTM EPS×(1+增速)外推
    """
    if np.isnan(pe_ttm) or pe_ttm <= 0 or current_price <= 0:
        return None

    positive_pe = pe_history[pe_history > 0].dropna()
    if len(positive_pe) < 30:
        return None  # 历史数据不足，不做自动计算

    pe_conservative = float(positive_pe.quantile(0.25))
    pe_neutral = float(positive_pe.quantile(0.50))
    pe_optimistic = float(positive_pe.quantile(0.75))

    # EPS: 优先券商一致预期，否则TTM外推
    ttm_eps = current_price / pe_ttm
    forward_eps = ttm_eps  # 默认

    if forecast_eps:
        # 取最近预测年的均值EPS
        for fc in forecast_eps:
            eps_mean = fc.get("eps_mean")
            if eps_mean and not np.isnan(float(eps_mean)) and float(eps_mean) > 0:
                forward_eps = float(eps_mean)
                break
    elif not np.isnan(profit_yoy) and profit_yoy > 0:
        forward_eps = ttm_eps * (1 + profit_yoy / 100)

    # 三情景目标价
    tp_conservative = round(forward_eps * pe_conservative, 2)
    tp_neutral = round(forward_eps * pe_neutral, 2)
    tp_optimistic = round(forward_eps * pe_optimistic, 2)
    tp_weighted = round(
        tp_conservative * 0.3 + tp_neutral * 0.4 + tp_optimistic * 0.3, 2
    )

    return {
        "conservative": tp_conservative,
        "base": tp_neutral,
        "optimistic": tp_optimistic,
        "probability_weighted": tp_weighted,
        "_meta": {
            "method": "auto_pe_percentile",
            "pe_25pct": round(pe_conservative, 1),
            "pe_50pct": round(pe_neutral, 1),
            "pe_75pct": round(pe_optimistic, 1),
            "forward_eps": round(forward_eps, 3),
            "ttm_eps": round(ttm_eps, 3),
            "eps_source": "券商一致预期" if forecast_eps else "TTM外推",
        },
    }


def _build_dividend_context(stock: StockData) -> str:
    """构建分红回报上下文供LLM阅读"""
    div = stock.info.get("dividend_history", [])
    if not div:
        return ""
    lines = ["### 历史分红"]
    for d in div[-5:]:  # 最近5年
        yield_str = f"股息率{float(d.get('dividend_yield', 0) or 0):.2f}%" if d.get("dividend_yield") else ""
        payout = d.get("payout_ratio")
        payout_str = f"派息率{payout}%" if pd.notna(payout) else ""
        lines.append(
            f"  {d.get('report_date', '')}: "
            f"每股{float(d.get('div_per_share', 0) or 0):.2f}元 "
            f"{yield_str} {payout_str}"
        )
    # 计算近3年平均股息率
    recent_yields = [float(d.get("dividend_yield", 0) or 0) for d in div[-3:] if d.get("dividend_yield")]
    if recent_yields:
        avg_yield = sum(recent_yields) / len(recent_yields)
        lines.append(f"  近{len(recent_yields)}年平均股息率: {avg_yield:.2f}%")
    return "\n".join(lines)


def _build_peer_valuation_context(stock: StockData) -> str:
    """构建同行业估值对比上下文"""
    peers = stock.info.get("industry_peers", [])
    if not peers:
        return ""

    industry = stock.info.get("industry", "")
    pe_vals = [float(p.get("pe", 0) or 0) for p in peers if p.get("pe") and float(p.get("pe", 0) or 0) > 0]
    pb_vals = [float(p.get("pb", 0) or 0) for p in peers if p.get("pb") and float(p.get("pb", 0) or 0) > 0]

    lines = [f"\n### 同行业估值对比 ({industry}, {len(peers)}家)"]
    if pe_vals:
        pe_avg = sum(pe_vals) / len(pe_vals)
        pe_med = sorted(pe_vals)[len(pe_vals) // 2]
        lines.append(f"- 行业PE: 均值{pe_avg:.1f}, 中位数{pe_med:.1f}")
    if pb_vals:
        pb_avg = sum(pb_vals) / len(pb_vals)
        pb_med = sorted(pb_vals)[len(pb_vals) // 2]
        lines.append(f"- 行业PB: 均值{pb_avg:.1f}, 中位数{pb_med:.1f}")

    # Top5 by market cap
    lines.append("\n同行TOP5:")
    lines.append("名称 | PE | PB | 涨跌%")
    lines.append("---|---|---|---")
    for p in peers[:5]:
        nm = p.get("name", "")
        pe = p.get("pe", "-")
        pb = p.get("pb", "-")
        chg = float(p.get("change_pct", 0) or 0)
        lines.append(f"{nm} | {pe} | {pb} | {chg:+.1f}")

    return "\n".join(lines)


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

    # PEG评分（使用近2-3年平均增速，减少单期波动）
    profit_yoy = np.nan
    if not fin_df.empty:
        # 取最近几期年报的profit_yoy做平滑
        if "report_type" in fin_df.columns:
            annual = fin_df[fin_df["report_type"].astype(str).str.contains("年报", na=False)]
            growth_series = pd.to_numeric(annual["profit_yoy"], errors="coerce").dropna().tail(3)
        else:
            growth_series = pd.to_numeric(fin_df["profit_yoy"], errors="coerce").dropna().tail(3)

        if len(growth_series) >= 2:
            # 用中位数（比均值更抗突变）
            profit_yoy = float(growth_series.median())
        elif len(growth_series) == 1:
            profit_yoy = float(growth_series.iloc[0])
        else:
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

    # 百分位计算：PE仅用正值（排除亏损期），PB仅用正值
    positive_pe = pe_history[pe_history > 0] if len(pe_history) > 0 else pe_history
    positive_pb = pb_history[pb_history > 0] if len(pb_history) > 0 else pb_history
    pe_pct = _calc_percentile(positive_pe, pe_ttm) if not np.isnan(pe_ttm) and pe_ttm > 0 and len(positive_pe) > 0 else None
    pb_pct = _calc_percentile(positive_pb, pb) if not np.isnan(pb) and pb > 0 and len(positive_pb) > 0 else None

    # 分业务线前瞻估值上下文
    segment_context, segment_result = _build_segment_forecast_context(stock)

    # 数据质量警告（注入LLM上下文，使其在分析中标红提示）
    data_warnings = stock.info.get("data_warnings", [])
    warnings_text = ""
    if data_warnings:
        warnings_text = "\n### ⚠️ 数据质量风险提示（必须在分析中明确标注）\n"
        for w in data_warnings:
            warnings_text += f"- {w}\n"
        warnings_text += "\n请在reasoning和key_risks中明确提及以上数据局限性，提醒投资者注意。\n"

    # --- LLM增强（可选，扩大范围至±40）---
    from ..llm_enhance import llm_enhance
    from src.research.extractor import get_research_context
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
            "dividend_context": _build_dividend_context(stock),
            "segment_forecast_context": segment_context + _build_peer_valuation_context(stock) + warnings_text,
            "research_context": get_research_context(stock, "valuation"),
        },
        code_score=code_score,
        code_reasoning=f"估值综合评分{code_score}（{val_days}天历史数据）。" + "；".join(all_factors),
        code_factors=all_factors,
        code_risks=all_risks,
    )

    # 提取LLM丰富字段（V2: 模型选择、情景分析、因子表等）
    raw = llm_result.get("raw_response") if isinstance(llm_result.get("raw_response"), dict) else {}

    # === 代码精算目标价覆写LLM（代码计算 > LLM猜测）===
    # 核心原则：数值计算必须由代码完成，LLM只做定性判断
    # 优先级：YAML精细模型 > 自动PE分位模型 > LLM生成（仅作参考，不信任）
    #
    # 设计依据（第一性原理）：
    # 1. LLM没有精确数值计算能力，目标价必须是代码计算
    # 2. LLM目标价误差可达2-3倍（如54元 vs 138元），不可作为决策依据
    # 3. 历史覆写条件（LLM<现价80%才覆写）存在漏洞，允许LLM在现价附近随意猜测

    code_targets = None  # 代码计算的目标价
    target_price_source = "none"  # 来源标记

    if segment_result and segment_result.get("source") == "yaml_model":
        # 路径1: YAML精细模型（最高优先级，用户自有研究）
        yearly = segment_result.get("yearly_forecast", [])
        if len(yearly) >= 2:
            code_targets = {
                "conservative": yearly[0]["target_price"],
                "base": yearly[1]["target_price"],
                "optimistic": yearly[-1]["target_price"],
                "probability_weighted": round(
                    yearly[0]["target_price"] * 0.3
                    + yearly[1]["target_price"] * 0.4
                    + yearly[-1]["target_price"] * 0.3, 2
                ),
            }
            target_price_source = "yaml_model"
            logger.info(
                f"[valuation] 目标价来源=YAML精细模型: "
                f"保守={code_targets['conservative']}, 中性={code_targets['base']}, "
                f"乐观={code_targets['optimistic']}"
            )

    if not code_targets:
        # 路径2: 自动PE分位模型（YAML不可用时的兜底）
        current_price = 0.0
        if stock.daily_quotes:
            last_quote = stock.daily_quotes[-1]
            current_price = float(last_quote.get("close", 0) or 0)

        forecast_eps = stock.info.get("profit_forecast", [])
        auto_targets = _calc_auto_target_prices(
            pe_ttm, pe_history, profit_yoy, current_price, forecast_eps,
        )
        if auto_targets:
            meta = auto_targets.pop("_meta", {})
            code_targets = auto_targets
            target_price_source = "auto_pe_percentile"
            logger.info(
                f"[valuation] 目标价来源=自动PE分位: "
                f"保守={auto_targets['conservative']}, "
                f"中性={auto_targets['base']}, "
                f"乐观={auto_targets['optimistic']} "
                f"[PE区间={meta.get('pe_25pct')}/{meta.get('pe_50pct')}/{meta.get('pe_75pct')}, "
                f"EPS={meta.get('forward_eps')}({meta.get('eps_source')})]"
            )

    # 强制覆写：代码计算的目标价始终优先于LLM猜测
    if code_targets:
        llm_targets = raw.get("target_prices", {})
        if llm_targets:
            # 保留LLM原始值到metadata供对比分析，但不作为最终输出
            logger.warning(
                f"[valuation] ⚠️ LLM目标价被代码覆写（LLM无精确计算能力）"
                f"\n  LLM猜测: 保守={llm_targets.get('conservative')}, "
                f"中性={llm_targets.get('base')}, 乐观={llm_targets.get('optimistic')}"
                f"\n  代码计算: 保守={code_targets['conservative']}, "
                f"中性={code_targets['base']}, 乐观={code_targets['optimistic']}"
            )
            # 存储LLM原始值供审计
            raw["_llm_original_target_prices"] = llm_targets
        raw["target_prices"] = code_targets
        raw["_target_price_source"] = target_price_source
    else:
        # 代码也无法计算时，记录警告，保留LLM值但标记为低置信度
        logger.warning(
            "[valuation] ⚠️ 代码无法计算目标价（数据不足），使用LLM值但置信度降低"
        )
        raw["_target_price_source"] = "llm_fallback"

    # 合并LLM结果
    signal_score = max(-100, min(100, code_score + llm_result["score_adjustment"]))
    reasoning = llm_result["reasoning"]
    final_factors = all_factors + llm_result["extra_factors"]
    final_risks = list(all_risks) + llm_result["extra_risks"]

    # 数据质量警告追加到风险列表（确保即使LLM忽略也能出现）
    for w in data_warnings:
        if "⚠️" in w:
            final_risks.append(w)

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
            # V2新增字段（从LLM raw_response提取）
            "raw_response": raw,
            "model_selection": raw.get("model_selection"),
            "factor_table": raw.get("factor_table"),
            "scenario_analysis": raw.get("scenario_analysis"),
            "pe_implied_growth": raw.get("pe_implied_growth"),
            "primary_valuation": raw.get("primary_valuation"),
            "secondary_valuation": raw.get("secondary_valuation"),
            "trap_detection": raw.get("trap_detection"),
            "target_prices": raw.get("target_prices", {}),
            "segment_forecast": segment_result if segment_result else None,
            "precision_warnings": _merge_precision_warnings(
                raw.get("precision_warnings", []),
                segment_result,
                data_warnings,
                stock,
            ),
        },
        execution_time_ms=elapsed_ms,
    )
