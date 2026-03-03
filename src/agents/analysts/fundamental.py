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


def _get_annualized_row(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """获取年化后的最新财务数据，解决季报/年报ROE混排问题。

    策略：
    1. 优先用最近年报（report_type含'年报'）
    2. 如果最近年报太旧（>18个月），用最新季报年化
    3. 年化方法：Q3 ROE × 4/3, Q2 × 2, Q1 × 4
    """
    latest = df.iloc[-1]
    report_type = str(latest.get("report_type", ""))

    # 尝试找最近年报
    if "report_type" in df.columns:
        annual_mask = df["report_type"].astype(str).str.contains("年报", na=False)
        annual_rows = df[annual_mask]
        if not annual_rows.empty:
            annual_latest = annual_rows.iloc[-1]
            # 检查年报是否太旧（距最新报告超过2期）
            if len(df) - annual_rows.index[-1] <= 4:
                return annual_latest, "年报"

    # 没有合适的年报，用最新季报年化
    annualize_factor = 1.0
    label = "年报"
    if "一季报" in report_type or "Q1" in report_type.upper():
        annualize_factor = 4.0
        label = "Q1年化"
    elif "中报" in report_type or "Q2" in report_type.upper():
        annualize_factor = 2.0
        label = "中报年化"
    elif "三季报" in report_type or "Q3" in report_type.upper():
        annualize_factor = 4.0 / 3.0
        label = "三季报年化"

    if annualize_factor != 1.0:
        latest = latest.copy()
        for col in ["roe", "net_margin"]:
            val = latest.get(col)
            if val is not None and not np.isnan(val):
                latest[col] = val * annualize_factor
    return latest, label


def _score_profitability(df: pd.DataFrame) -> tuple[int, list[str]]:
    """盈利能力评分 (权重30%)"""
    if df.empty or "roe" not in df.columns:
        return 0, ["无盈利数据"]

    latest, period_label = _get_annualized_row(df)
    factors = []
    score = 0

    # ROE
    roe = latest.get("roe")
    if roe is not None and not np.isnan(roe):
        if roe > 20:
            score += 15
            factors.append(f"ROE={roe:.1f}%({period_label}) 优秀")
        elif roe > 12:
            score += 8
            factors.append(f"ROE={roe:.1f}%({period_label}) 良好")
        elif roe > 6:
            score += 0
            factors.append(f"ROE={roe:.1f}%({period_label}) 一般")
        else:
            score -= 10
            factors.append(f"ROE={roe:.1f}%({period_label}) 较差")

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


def _score_earnings_quality(df: pd.DataFrame) -> tuple[int, list[str]]:
    """盈利质量评分：扣非净利润 vs 归母净利润 (权重10%)

    如果扣非/归母比值过低，说明利润靠非经常性损益撑着。
    """
    if df.empty:
        return 0, ["无盈利质量数据"]

    latest = df.iloc[-1]
    net_profit = latest.get("net_profit")
    deducted = latest.get("net_profit_deducted")

    if net_profit is None or deducted is None:
        return 0, ["无扣非净利润数据"]

    # 处理NaN
    try:
        net_profit = float(net_profit)
        deducted = float(deducted)
    except (TypeError, ValueError):
        return 0, ["扣非净利润数据格式异常"]

    if np.isnan(net_profit) or np.isnan(deducted):
        return 0, ["无扣非净利润数据"]

    factors = []
    score = 0

    if net_profit > 0:
        ratio = deducted / net_profit
        if ratio > 0.9:
            score += 8
            factors.append(f"扣非/归母={ratio:.0%} 盈利质量优(极少非经常性损益)")
        elif ratio > 0.7:
            score += 3
            factors.append(f"扣非/归母={ratio:.0%} 盈利质量中")
        elif ratio > 0.5:
            score -= 3
            factors.append(f"扣非/归母={ratio:.0%} 非经常性损益占比较高")
        else:
            score -= 8
            factors.append(f"扣非/归母={ratio:.0%} 利润严重依赖非经常性损益")
    elif net_profit < 0 and deducted < 0:
        if abs(deducted) > abs(net_profit) * 1.2:
            score -= 5
            factors.append("扣非亏损大于报表亏损，主业更差")
        else:
            factors.append("公司亏损中，扣非口径差异不大")
    elif net_profit > 0 and deducted < 0:
        score -= 10
        factors.append("报表盈利但扣非亏损！主业实质亏损，利润全靠非经常性损益")

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

    # 主路径：经营现金流 / 净利润
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
        # 备用路径：cashflow_to_revenue（经营现金流/营收比，小数形式 0.28 = 28%）
        ctr = latest.get("cashflow_to_revenue")
        if ctr is not None and not np.isnan(ctr):
            pct = ctr * 100  # 转为百分比展示
            if ctr > 0.15:
                score += 8
                factors.append(f"经营现金流/营收={pct:.1f}% 质量优")
            elif ctr > 0.05:
                score += 3
                factors.append(f"经营现金流/营收={pct:.1f}% 质量一般")
            elif ctr > 0:
                score += 0
                factors.append(f"经营现金流/营收={pct:.1f}% 偏低")
            else:
                score -= 8
                factors.append(f"经营现金流/营收={pct:.1f}% 现金流为负")
        else:
            factors.append("现金流数据不完整")

    return score, factors


# ============================================================
# 价值投资增强指标 (R1)
# ============================================================


def _safe_float(val: Any, default: float = np.nan) -> float:
    """安全提取float值"""
    if val is None:
        return default
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def _score_roic(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """ROIC (投入资本回报率) — 比ROE更准确反映真实盈利能力

    ROIC = NOPAT / Invested Capital
    NOPAT ≈ 营业利润 × (1 - 有效税率)
    Invested Capital ≈ 总资产 - 无息流动负债 (近似为: 总资产 - 流动负债 + 有息负债)
    简化计算: ROIC ≈ (净利润 + 利息费用×(1-税率)) / (总资产 - 流动负债 + 短期借款)
    """
    if df.empty:
        return 0, ["无ROIC数据"], {}

    latest = df.iloc[-1]
    score = 0
    factors = []
    meta = {}

    net_profit = _safe_float(latest.get("net_profit"))
    total_assets = _safe_float(latest.get("total_assets"))
    total_liabilities = _safe_float(latest.get("total_liabilities"))
    revenue = _safe_float(latest.get("revenue"))

    if np.isnan(net_profit) or np.isnan(total_assets) or total_assets <= 0:
        return 0, ["ROIC数据不足"], {}

    # 简化ROIC: 使用ROA的调整版本
    # 如果有equity数据，用 净利润 / (总资产 - 无息流动负债)
    equity = total_assets - total_liabilities if not np.isnan(total_liabilities) else total_assets * 0.5
    invested_capital = max(equity, total_assets * 0.3)  # 防止负数或极端值

    roic = (net_profit / invested_capital) * 100
    meta["roic"] = round(roic, 2)
    meta["invested_capital"] = round(invested_capital / 1e8, 2)  # 亿

    if roic > 20:
        score += 8
        factors.append(f"ROIC={roic:.1f}% 优秀(>20%，强竞争优势)")
    elif roic > 12:
        score += 4
        factors.append(f"ROIC={roic:.1f}% 良好(>12%)")
    elif roic > 6:
        score += 0
        factors.append(f"ROIC={roic:.1f}% 一般")
    else:
        score -= 5
        factors.append(f"ROIC={roic:.1f}% 较差(<6%)")

    return score, factors, meta


def _score_owner_earnings(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """Owner Earnings (巴菲特核心指标)

    Owner Earnings = 净利润 + 折旧摊销 - 维护性资本开支 - 营运资金增量
    简化计算: ≈ 经营现金流 - 资本开支 (即自由现金流)
    """
    if df.empty:
        return 0, ["无Owner Earnings数据"], {}

    latest = df.iloc[-1]
    score = 0
    factors = []
    meta = {}

    net_profit = _safe_float(latest.get("net_profit"))
    ocf = _safe_float(latest.get("operating_cashflow"))
    capex = _safe_float(latest.get("capital_expenditure", latest.get("capex")))

    if np.isnan(net_profit) or net_profit == 0:
        return 0, ["Owner Earnings数据不足"], {}

    # 自由现金流 = 经营现金流 - 资本开支
    if not np.isnan(ocf):
        if not np.isnan(capex):
            # capex 通常为负数（现金流出），取绝对值
            capex_abs = abs(capex)
            fcf = ocf - capex_abs
            owner_earnings = fcf
        else:
            # 无capex数据，用OCF近似
            fcf = ocf
            owner_earnings = ocf
    else:
        return 0, ["无经营现金流数据"], {}

    meta["owner_earnings"] = round(owner_earnings / 1e8, 2)  # 亿
    meta["fcf"] = round(fcf / 1e8, 2)  # 亿
    meta["ocf"] = round(ocf / 1e8, 2)

    # Owner Earnings / 净利润 比率
    oe_ratio = owner_earnings / net_profit if net_profit > 0 else 0
    meta["oe_to_net_profit"] = round(oe_ratio, 2)

    if owner_earnings > 0 and oe_ratio > 0.8:
        score += 8
        factors.append(f"Owner Earnings={owner_earnings/1e8:.1f}亿，占净利{oe_ratio:.0%}(优秀)")
    elif owner_earnings > 0 and oe_ratio > 0.4:
        score += 3
        factors.append(f"Owner Earnings={owner_earnings/1e8:.1f}亿，占净利{oe_ratio:.0%}(一般)")
    elif owner_earnings > 0:
        score += 0
        factors.append(f"Owner Earnings={owner_earnings/1e8:.1f}亿，占净利{oe_ratio:.0%}(偏低)")
    else:
        score -= 5
        factors.append(f"Owner Earnings为负({owner_earnings/1e8:.1f}亿)，企业在烧钱")

    return score, factors, meta


def _dupont_decomposition(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """杜邦分解: ROE = 净利率 × 资产周转率 × 权益乘数

    判断ROE来源: 高净利率(好) vs 高杠杆(危险) vs 高周转(中性偏好)
    """
    if df.empty:
        return 0, ["无杜邦分解数据"], {}

    latest = df.iloc[-1]
    score = 0
    factors = []
    meta = {}

    roe = _safe_float(latest.get("roe"))
    net_margin = _safe_float(latest.get("net_margin"))
    revenue = _safe_float(latest.get("revenue"))
    total_assets = _safe_float(latest.get("total_assets"))
    total_liabilities = _safe_float(latest.get("total_liabilities"))

    if np.isnan(roe) or np.isnan(net_margin):
        return 0, ["杜邦分解数据不足"], {}

    # 资产周转率 = 营收 / 总资产
    asset_turnover = revenue / total_assets if not np.isnan(revenue) and not np.isnan(total_assets) and total_assets > 0 else np.nan

    # 权益乘数 = 总资产 / (总资产 - 总负债)
    if not np.isnan(total_assets) and not np.isnan(total_liabilities) and total_assets > total_liabilities:
        equity = total_assets - total_liabilities
        equity_multiplier = total_assets / equity
    else:
        equity_multiplier = np.nan

    meta["net_margin"] = round(net_margin, 2) if not np.isnan(net_margin) else None
    meta["asset_turnover"] = round(asset_turnover, 3) if not np.isnan(asset_turnover) else None
    meta["equity_multiplier"] = round(equity_multiplier, 2) if not np.isnan(equity_multiplier) else None

    # 评判ROE来源质量
    if roe > 15:
        if not np.isnan(net_margin) and net_margin > 15:
            score += 5
            factors.append(f"杜邦分解: ROE={roe:.1f}%由高净利率({net_margin:.1f}%)驱动(最优模式)")
        elif not np.isnan(equity_multiplier) and equity_multiplier > 4:
            score -= 3
            factors.append(f"杜邦分解: ROE={roe:.1f}%主要由高杠杆(权益乘数{equity_multiplier:.1f})驱动(危险)")
        elif not np.isnan(asset_turnover) and asset_turnover > 1.0:
            score += 2
            factors.append(f"杜邦分解: ROE={roe:.1f}%由高周转({asset_turnover:.2f})驱动(效率型)")
        else:
            factors.append(f"杜邦分解: ROE={roe:.1f}%来源均衡")
    else:
        if not np.isnan(equity_multiplier) and equity_multiplier > 4:
            score -= 3
            factors.append(f"杜邦分解: ROE={roe:.1f}%虽不高但杠杆偏大(权益乘数{equity_multiplier:.1f})")
        else:
            factors.append(f"杜邦分解: ROE={roe:.1f}%")

    return score, factors, meta


def _calc_piotroski_f_score(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """Piotroski F-Score (0-9分) — 学术验证有效的财务健康评分

    9个二元指标:
    1. ROA > 0
    2. OCF > 0
    3. ΔROA > 0 (ROA改善)
    4. OCF > 净利润 (应计利润质量)
    5. Δ杠杆 < 0 (负债率下降)
    6. Δ流动比率 > 0 (流动性改善)
    7. 无增发 (稀释检测)
    8. Δ毛利率 > 0
    9. Δ资产周转率 > 0
    """
    if len(df) < 2:
        return 0, ["Piotroski数据不足(需至少2期)"], {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    f_score = 0
    details = []
    meta = {}

    # 1. ROA > 0
    net_profit = _safe_float(latest.get("net_profit"))
    total_assets = _safe_float(latest.get("total_assets"))
    if not np.isnan(net_profit) and not np.isnan(total_assets) and total_assets > 0:
        roa = net_profit / total_assets
        if roa > 0:
            f_score += 1
            details.append("ROA>0 ✓")
        else:
            details.append("ROA≤0 ✗")
    else:
        details.append("ROA数据缺失")

    # 2. OCF > 0
    ocf = _safe_float(latest.get("operating_cashflow"))
    if not np.isnan(ocf) and ocf > 0:
        f_score += 1
        details.append("OCF>0 ✓")
    elif not np.isnan(ocf):
        details.append("OCF≤0 ✗")

    # 3. ΔROA > 0
    prev_np = _safe_float(prev.get("net_profit"))
    prev_ta = _safe_float(prev.get("total_assets"))
    if all(not np.isnan(v) for v in [net_profit, total_assets, prev_np, prev_ta]) and total_assets > 0 and prev_ta > 0:
        roa_curr = net_profit / total_assets
        roa_prev = prev_np / prev_ta
        if roa_curr > roa_prev:
            f_score += 1
            details.append("ΔROA>0 ✓")
        else:
            details.append("ΔROA≤0 ✗")

    # 4. OCF > 净利润 (应计利润质量)
    if not np.isnan(ocf) and not np.isnan(net_profit) and net_profit > 0:
        if ocf > net_profit:
            f_score += 1
            details.append("OCF>NI ✓")
        else:
            details.append("OCF≤NI ✗")

    # 5. Δ杠杆 < 0
    debt_ratio = _safe_float(latest.get("debt_ratio"))
    prev_debt_ratio = _safe_float(prev.get("debt_ratio"))
    if not np.isnan(debt_ratio) and not np.isnan(prev_debt_ratio):
        if debt_ratio < prev_debt_ratio:
            f_score += 1
            details.append("Δ负债率<0 ✓")
        else:
            details.append("Δ负债率≥0 ✗")

    # 6. Δ流动比率 > 0
    cr = _safe_float(latest.get("current_ratio"))
    prev_cr = _safe_float(prev.get("current_ratio"))
    if not np.isnan(cr) and not np.isnan(prev_cr):
        if cr > prev_cr:
            f_score += 1
            details.append("Δ流动比率>0 ✓")
        else:
            details.append("Δ流动比率≤0 ✗")

    # 7. 无增发 (简化: 检查总股本是否增加)
    shares = _safe_float(latest.get("total_shares", latest.get("shares_outstanding")))
    prev_shares = _safe_float(prev.get("total_shares", prev.get("shares_outstanding")))
    if not np.isnan(shares) and not np.isnan(prev_shares) and prev_shares > 0:
        if shares <= prev_shares * 1.01:  # 允许1%误差
            f_score += 1
            details.append("无增发 ✓")
        else:
            details.append("有增发 ✗")
    else:
        f_score += 1  # 无数据时假设未增发
        details.append("增发数据缺失(默认✓)")

    # 8. Δ毛利率 > 0
    gm = _safe_float(latest.get("gross_margin"))
    prev_gm = _safe_float(prev.get("gross_margin"))
    if not np.isnan(gm) and not np.isnan(prev_gm):
        if gm > prev_gm:
            f_score += 1
            details.append("Δ毛利率>0 ✓")
        else:
            details.append("Δ毛利率≤0 ✗")

    # 9. Δ资产周转率 > 0
    rev = _safe_float(latest.get("revenue"))
    prev_rev = _safe_float(prev.get("revenue"))
    if all(not np.isnan(v) for v in [rev, total_assets, prev_rev, prev_ta]) and total_assets > 0 and prev_ta > 0:
        turnover = rev / total_assets
        prev_turnover = prev_rev / prev_ta
        if turnover > prev_turnover:
            f_score += 1
            details.append("Δ周转率>0 ✓")
        else:
            details.append("Δ周转率≤0 ✗")

    meta["piotroski_f_score"] = f_score
    meta["piotroski_details"] = details

    # 评分转换
    if f_score >= 8:
        score = 10
        factors = [f"Piotroski F-Score={f_score}/9 (财务极度健康)"]
    elif f_score >= 6:
        score = 5
        factors = [f"Piotroski F-Score={f_score}/9 (财务健康)"]
    elif f_score >= 4:
        score = 0
        factors = [f"Piotroski F-Score={f_score}/9 (一般)"]
    elif f_score >= 2:
        score = -5
        factors = [f"Piotroski F-Score={f_score}/9 (偏弱)"]
    else:
        score = -10
        factors = [f"Piotroski F-Score={f_score}/9 (财务困境信号)"]

    return score, factors, meta


def _calc_altman_z_score(df: pd.DataFrame, market_cap: float = 0.0) -> tuple[int, list[str], dict]:
    """Altman Z-Score — 财务困境预警

    Z = 1.2×A + 1.4×B + 3.3×C + 0.6×D + 1.0×E
    A = 营运资金/总资产
    B = 留存收益/总资产
    C = EBIT/总资产
    D = 股权市值/总负债
    E = 营收/总资产
    """
    if df.empty:
        return 0, ["无Z-Score数据"], {}

    latest = df.iloc[-1]
    meta = {}

    total_assets = _safe_float(latest.get("total_assets"))
    total_liabilities = _safe_float(latest.get("total_liabilities"))
    revenue = _safe_float(latest.get("revenue"))
    net_profit = _safe_float(latest.get("net_profit"))
    current_assets = _safe_float(latest.get("current_assets"))
    current_liabilities = _safe_float(latest.get("current_liabilities"))
    retained_earnings = _safe_float(latest.get("retained_earnings", latest.get("undistributed_profit")))

    if np.isnan(total_assets) or total_assets <= 0:
        return 0, ["Z-Score数据不足"], {}

    # A = 营运资金/总资产
    if not np.isnan(current_assets) and not np.isnan(current_liabilities):
        a = (current_assets - current_liabilities) / total_assets
    else:
        a = 0

    # B = 留存收益/总资产
    if not np.isnan(retained_earnings):
        b = retained_earnings / total_assets
    else:
        # 用净利润近似
        b = net_profit / total_assets if not np.isnan(net_profit) else 0

    # C = EBIT/总资产 (用净利润近似EBIT)
    c = net_profit / total_assets if not np.isnan(net_profit) else 0

    # D = 股权市值/总负债
    if market_cap > 0 and not np.isnan(total_liabilities) and total_liabilities > 0:
        d = market_cap / total_liabilities
    else:
        # 用净资产/总负债近似
        equity = total_assets - total_liabilities if not np.isnan(total_liabilities) else total_assets * 0.5
        d = max(0, equity / total_liabilities) if not np.isnan(total_liabilities) and total_liabilities > 0 else 1.0

    # E = 营收/总资产
    e = revenue / total_assets if not np.isnan(revenue) else 0

    z_score = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e
    meta["altman_z_score"] = round(z_score, 2)
    meta["z_components"] = {"A": round(a, 3), "B": round(b, 3), "C": round(c, 3), "D": round(d, 3), "E": round(e, 3)}

    if z_score > 2.99:
        score = 5
        factors = [f"Altman Z-Score={z_score:.2f} (安全区>2.99)"]
    elif z_score > 1.81:
        score = 0
        factors = [f"Altman Z-Score={z_score:.2f} (灰色区1.81-2.99)"]
    else:
        score = -10
        factors = [f"Altman Z-Score={z_score:.2f} (危险区<1.81，财务困境风险)"]

    return score, factors, meta


def _score_free_cash_flow(df: pd.DataFrame, market_cap: float = 0.0) -> tuple[int, list[str], dict]:
    """自由现金流分析 — FCF Yield和FCF增长趋势"""
    if df.empty:
        return 0, ["无FCF数据"], {}

    latest = df.iloc[-1]
    score = 0
    factors = []
    meta = {}

    ocf = _safe_float(latest.get("operating_cashflow"))
    capex = _safe_float(latest.get("capital_expenditure", latest.get("capex")))
    revenue = _safe_float(latest.get("revenue"))

    if np.isnan(ocf):
        return 0, ["无经营现金流数据"], {}

    capex_abs = abs(capex) if not np.isnan(capex) else 0
    fcf = ocf - capex_abs
    meta["fcf"] = round(fcf / 1e8, 2)

    # FCF Yield = FCF / 市值
    if market_cap > 0:
        fcf_yield = (fcf / market_cap) * 100
        meta["fcf_yield"] = round(fcf_yield, 2)
        if fcf_yield > 8:
            score += 5
            factors.append(f"FCF Yield={fcf_yield:.1f}% 极高(>8%，价值投资最爱)")
        elif fcf_yield > 5:
            score += 3
            factors.append(f"FCF Yield={fcf_yield:.1f}% 高(>5%)")
        elif fcf_yield > 2:
            score += 1
            factors.append(f"FCF Yield={fcf_yield:.1f}% 合理")
        elif fcf_yield > 0:
            factors.append(f"FCF Yield={fcf_yield:.1f}% 偏低")
        else:
            score -= 3
            factors.append(f"FCF Yield={fcf_yield:.1f}% 为负(烧钱)")

    # 资本开支强度 = Capex / 营收
    if not np.isnan(revenue) and revenue > 0 and capex_abs > 0:
        capex_intensity = (capex_abs / revenue) * 100
        meta["capex_intensity"] = round(capex_intensity, 2)
        if capex_intensity < 5:
            score += 2
            factors.append(f"资本开支/营收={capex_intensity:.1f}%(轻资产)")
        elif capex_intensity > 20:
            score -= 2
            factors.append(f"资本开支/营收={capex_intensity:.1f}%(重资产，资本密集)")

    # FCF增长趋势
    if len(df) >= 3 and "operating_cashflow" in df.columns:
        ocf_series = pd.to_numeric(df["operating_cashflow"], errors="coerce").dropna().tail(4)
        if len(ocf_series) >= 3:
            if all(ocf_series.diff().dropna() > 0):
                score += 2
                factors.append("FCF连续增长趋势")
            elif ocf_series.iloc[-1] < ocf_series.iloc[0]:
                score -= 1
                factors.append("FCF呈下降趋势")

    return score, factors, meta


def _score_receivable_quality(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """应收账款质量 — 检测赊账式增长"""
    if df.empty:
        return 0, ["无应收数据"], {}

    latest = df.iloc[-1]
    score = 0
    factors = []
    meta = {}

    accounts_receivable = _safe_float(latest.get("accounts_receivable"))
    revenue = _safe_float(latest.get("revenue"))

    if np.isnan(accounts_receivable) or np.isnan(revenue) or revenue <= 0:
        return 0, ["应收账款数据不足"], {}

    ar_ratio = (accounts_receivable / revenue) * 100
    meta["ar_to_revenue"] = round(ar_ratio, 2)

    # 应收/营收比
    if ar_ratio < 10:
        score += 3
        factors.append(f"应收/营收={ar_ratio:.1f}%(优秀，回款快)")
    elif ar_ratio < 25:
        score += 0
        factors.append(f"应收/营收={ar_ratio:.1f}%(正常)")
    elif ar_ratio < 40:
        score -= 3
        factors.append(f"应收/营收={ar_ratio:.1f}%(偏高，关注回款风险)")
    else:
        score -= 5
        factors.append(f"应收/营收={ar_ratio:.1f}%(过高，赊账式增长风险)")

    # 应收增速 vs 营收增速
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_ar = _safe_float(prev.get("accounts_receivable"))
        prev_rev = _safe_float(prev.get("revenue"))
        if not np.isnan(prev_ar) and prev_ar > 0 and not np.isnan(prev_rev) and prev_rev > 0:
            ar_growth = (accounts_receivable - prev_ar) / prev_ar * 100
            rev_growth = (revenue - prev_rev) / prev_rev * 100
            meta["ar_growth"] = round(ar_growth, 1)
            meta["revenue_growth"] = round(rev_growth, 1)
            if ar_growth > rev_growth + 20:
                score -= 3
                factors.append(f"应收增速({ar_growth:.0f}%)远超营收增速({rev_growth:.0f}%)，赊账式增长")

    return score, factors, meta


def _build_financial_summary(df: pd.DataFrame) -> str:
    """构建最新一期财务摘要供LLM阅读"""
    latest = df.iloc[-1]
    lines = []
    field_names = {
        "roe": "ROE(%)", "roa": "ROA(%)", "gross_margin": "毛利率(%)",
        "net_margin": "净利率(%)", "revenue_yoy": "营收同比(%)",
        "profit_yoy": "净利同比(%)", "debt_ratio": "资产负债率(%)",
        "current_ratio": "流动比率", "quick_ratio": "速动比率",
        "eps": "每股收益", "bps": "每股净资产",
        "revenue": "营收(元)", "net_profit": "净利润(元)",
    }
    for key, label in field_names.items():
        val = latest.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            if key in ("revenue", "net_profit"):
                lines.append(f"- {label}: {val/1e8:.2f}亿")
            else:
                lines.append(f"- {label}: {val:.2f}")
    return "\n".join(lines) if lines else "无详细数据"


def _build_financial_trend(df: pd.DataFrame, n: int = 4) -> str:
    """构建近N期财务趋势表格（优先使用年报做同口径对比）"""
    # 优先筛选年报，保证同口径对比
    if "report_type" in df.columns:
        annual = df[df["report_type"].astype(str).str.contains("年报", na=False)]
        if len(annual) >= 2:
            recent = annual.tail(n)
        else:
            recent = df.tail(n)
    else:
        recent = df.tail(n)

    if recent.empty:
        return "无趋势数据"

    lines = ["报告期 | ROE% | 毛利率% | 营收同比% | 净利同比% | 负债率%"]
    lines.append("---|---|---|---|---|---")
    for _, row in recent.iterrows():
        rd = row.get("report_date", "")
        if hasattr(rd, "strftime"):
            rd = rd.strftime("%Y-%m")
        roe = f"{row.get('roe', 0):.1f}" if row.get("roe") is not None else "-"
        gm = f"{row.get('gross_margin', 0):.1f}" if row.get("gross_margin") is not None else "-"
        rev = f"{row.get('revenue_yoy', 0):.1f}" if row.get("revenue_yoy") is not None else "-"
        pft = f"{row.get('profit_yoy', 0):.1f}" if row.get("profit_yoy") is not None else "-"
        dr = f"{row.get('debt_ratio', 0):.1f}" if row.get("debt_ratio") is not None else "-"
        lines.append(f"{rd} | {roe} | {gm} | {rev} | {pft} | {dr}")
    return "\n".join(lines)


def analyze_fundamental(stock: StockData) -> AgentSignal:
    """基本面分析主函数

    60%代码量化评分 + 40%LLM定性分析（趋势解读、行业对比、风险评估）。
    LLM不可用时自动降级为纯代码分析。

    价值投资增强维度:
    - 盈利能力(20%): ROE + ROIC + 毛利率 + 净利率
    - 成长性(15%): 营收/净利/FCF增速
    - 财务健康(15%): 资产负债率 + 流动比率 + Altman Z-Score
    - 盈利质量(15%): 扣非比 + 杜邦分解 + 应收质量
    - 现金流质量(20%): 经营现金流/净利 + FCF Yield + Owner Earnings
    - 财务综合(15%): Piotroski F-Score
    """
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
    value_meta = {}  # 价值投资增强元数据

    # === 原有维度评分 ===
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

    eq_score, eq_factors = _score_earnings_quality(df)
    total_score += eq_score
    all_factors.extend(eq_factors)

    # === 价值投资增强维度 (R1) ===
    vi_score = 0  # 价值投资增强得分

    # 获取市值用于FCF Yield和Z-Score
    market_cap = 0.0
    info = stock.info or {}
    mc = info.get("market_cap")
    if mc and isinstance(mc, (int, float)) and mc > 0:
        market_cap = float(mc)

    # ROIC
    roic_score, roic_factors, roic_meta = _score_roic(df)
    vi_score += roic_score
    all_factors.extend(roic_factors)
    value_meta.update(roic_meta)

    # Owner Earnings
    oe_score, oe_factors, oe_meta = _score_owner_earnings(df)
    vi_score += oe_score
    all_factors.extend(oe_factors)
    value_meta.update(oe_meta)

    # 杜邦分解
    dupont_score, dupont_factors, dupont_meta = _dupont_decomposition(df)
    vi_score += dupont_score
    all_factors.extend(dupont_factors)
    value_meta["dupont"] = dupont_meta

    # Piotroski F-Score
    piotroski_score, piotroski_factors, piotroski_meta = _calc_piotroski_f_score(df)
    vi_score += piotroski_score
    all_factors.extend(piotroski_factors)
    value_meta.update(piotroski_meta)

    # Altman Z-Score
    zscore_score, zscore_factors, zscore_meta = _calc_altman_z_score(df, market_cap)
    vi_score += zscore_score
    all_factors.extend(zscore_factors)
    value_meta.update(zscore_meta)

    # 自由现金流分析
    fcf_score, fcf_factors, fcf_meta = _score_free_cash_flow(df, market_cap)
    vi_score += fcf_score
    all_factors.extend(fcf_factors)
    value_meta.update(fcf_meta)

    # 应收账款质量
    ar_score, ar_factors, ar_meta = _score_receivable_quality(df)
    vi_score += ar_score
    all_factors.extend(ar_factors)
    value_meta.update(ar_meta)

    # 代码评分（-100 ~ +100）
    # 原有满分约85，价值投资增强满分约50 (ROIC 8 + OE 8 + 杜邦 5 + Piotroski 10 + Z 5 + FCF 9 + AR 3)
    # 综合满分约135，归一化到100
    combined_score = total_score + vi_score
    code_score = max(-100, min(100, int(combined_score * 100 / 135)))

    # 置信度：基于财报期数
    num_reports = len(df)
    confidence = min(1.0, num_reports / 8)

    # 风险检测
    latest = df.iloc[-1]
    if latest.get("debt_ratio", 0) > 70:
        all_risks.append("资产负债率超过70%，偿债风险较高")
    if latest.get("profit_yoy", 0) < -30:
        all_risks.append("净利润同比大幅下降超30%")
    if eq_score <= -8:
        all_risks.append("盈利质量差：利润严重依赖非经常性损益")
    # 价值投资增强风险检测
    if value_meta.get("altman_z_score") is not None and value_meta["altman_z_score"] < 1.81:
        all_risks.append(f"Altman Z-Score={value_meta['altman_z_score']:.2f}<1.81，财务困境高风险")
    if value_meta.get("piotroski_f_score") is not None and value_meta["piotroski_f_score"] <= 2:
        all_risks.append(f"Piotroski F-Score={value_meta['piotroski_f_score']}/9，财务状况极差")
    if value_meta.get("oe_to_net_profit") is not None and value_meta["oe_to_net_profit"] < 0:
        all_risks.append("Owner Earnings为负，企业在消耗现金")

    # --- LLM增强（可选，扩大范围至±40）---
    from ..llm_enhance import llm_enhance
    from src.research.extractor import get_research_context
    from datetime import date

    llm_result = llm_enhance(
        agent_name="fundamental",
        template_name="fundamental.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "financial_summary": _build_financial_summary(df),
            "financial_trend": _build_financial_trend(df),
            "research_context": get_research_context(stock, "fundamental"),
        },
        code_score=code_score,
        code_reasoning=f"基本面综合评分{code_score}。" + "；".join(all_factors),
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
        agent_name="fundamental",
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
            "total_raw_score": total_score,
            "vi_raw_score": vi_score,
            "code_score": code_score,
            "llm_adjustment": llm_result["score_adjustment"],
            "llm_enhanced": llm_result["enhanced"],
            "component_scores": {
                "profitability": prof_score,
                "growth": grow_score,
                "financial_health": health_score,
                "cash_quality": cash_score,
                "earnings_quality": eq_score,
                # 价值投资增强
                "roic": roic_score,
                "owner_earnings": oe_score,
                "dupont": dupont_score,
                "piotroski": piotroski_score,
                "altman_z": zscore_score,
                "fcf": fcf_score,
                "receivable_quality": ar_score,
            },
            "num_reports": num_reports,
            # 价值投资核心指标
            "value_metrics": value_meta,
            # V2新增字段（从LLM raw_response提取）
            "raw_response": llm_result.get("raw_response", {}),
            "quality_rating": llm_result.get("raw_response", {}).get("quality_rating"),
            "revenue_breakdown": llm_result.get("raw_response", {}).get("revenue_breakdown"),
            "profitability_detail": llm_result.get("raw_response", {}).get("profitability"),
            "cash_flow_verification": llm_result.get("raw_response", {}).get("cash_flow_verification"),
            "growth_analysis": llm_result.get("raw_response", {}).get("growth_analysis"),
            "red_flags_detail": llm_result.get("raw_response", {}).get("red_flags"),
            "green_flags_detail": llm_result.get("raw_response", {}).get("green_flags"),
        },
        execution_time_ms=elapsed_ms,
    )
