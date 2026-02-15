"""
量化筛选引擎

Layer 1: 纯代码，全市场覆盖 5000+ 标的
输出 Top N 候选标的给 Layer 2 (AI Agent分析)

数据策略：
- 主数据源：stock_yjbb_em 批量财报（~5000只，always works）
- 辅助数据源：stock_zh_a_spot_em 实时行情（PE/PB/市值/涨跌幅）
- 多期数据：拉取最近4期季报，计算连续增长因子
- 优雅降级：spot数据不可用时，仅用财报因子筛选
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from src.data.sources import DataSourceManager


@dataclass
class ScreeningResult:
    """筛选结果"""

    symbol: str
    name: str
    rank: int
    composite_score: float
    factor_scores: dict[str, float]
    industry: str = ""


class ScreeningEngine:
    """量化筛选引擎"""

    def __init__(self, source_manager: DataSourceManager, config_path: Optional[str] = None) -> None:
        self._source = source_manager
        self._config = self._load_config(config_path)

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict[str, Any]:
        path = Path(config_path) if config_path else Path(__file__).parent.parent.parent / "config" / "screening.yaml"
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f).get("screening", {})
        return {"top_n": 50}

    def run(self, market: str = "A", include_growth: bool = False) -> list[ScreeningResult]:
        """执行筛选流程

        Args:
            market: 市场
            include_growth: 是否包含连续增长因子（需额外拉取多期数据，+30s）
        """
        top_n = self._config.get("top_n", 50)
        logger.info(f"开始量化筛选 market={market} top_n={top_n} growth={include_growth}")

        # Step 1: 获取批量财务数据（主数据源）
        fin_df = self._fetch_batch_financial()
        if fin_df.empty:
            logger.error("获取批量财务数据失败")
            return []
        logger.info(f"批量财务数据: {len(fin_df)}只")

        # Step 1.5: 多期数据（连续增长因子，可选）
        multi_period: dict[str, pd.DataFrame] = {}
        if include_growth:
            multi_period = self._fetch_multi_period_financial(n_periods=4)

        # Step 2: 尝试获取实时行情（辅助数据源，可能失败）
        spot_df = self._fetch_spot_data(market)
        if not spot_df.empty:
            logger.info(f"实时行情数据: {len(spot_df)}只")
            # 合并spot数据到financial数据
            fin_df = fin_df.merge(
                spot_df[["symbol", "price", "change_pct", "pe", "pb",
                         "total_market_cap", "float_market_cap",
                         "change_pct_60d", "change_pct_ytd",
                         "volume_ratio", "turnover_rate", "amount"]],
                on="symbol",
                how="left",
            )
            has_spot = True
        else:
            logger.warning("实时行情不可用，仅用财报因子筛选")
            has_spot = False

        # Step 3: 前置过滤
        filtered = self._pre_filter(fin_df, has_spot)
        logger.info(f"前置过滤后: {len(filtered)}")

        if filtered.empty:
            return []

        # Step 4: 计算因子评分
        scored = self._score_all(filtered, has_spot, multi_period=multi_period)

        # Step 5: 排名输出
        scored = scored.sort_values("composite_score", ascending=False).head(top_n)
        scored["rank"] = range(1, len(scored) + 1)

        results = []
        for _, row in scored.iterrows():
            results.append(
                ScreeningResult(
                    symbol=row["symbol"],
                    name=row.get("name", ""),
                    rank=row["rank"],
                    composite_score=round(row["composite_score"], 2),
                    factor_scores=row.get("factor_scores", {}),
                    industry=row.get("industry", ""),
                )
            )

        logger.info(f"筛选完成: Top {len(results)}")
        return results

    def _fetch_batch_financial(self) -> pd.DataFrame:
        """获取批量财务数据"""
        try:
            return self._source.fetch_batch_financial()
        except Exception as e:
            logger.error(f"批量财务数据获取失败: {e}")
            return pd.DataFrame()

    @staticmethod
    def _get_recent_report_dates(n: int = 4) -> list[str]:
        """获取最近N个季报期的日期字符串（YYYYMMDD）"""
        from datetime import date

        today = date.today()
        # 所有季报期：Q1=0331, Q2=0630, Q3=0930, Q4=1231
        quarters = [(today.year, "0331"), (today.year, "0630"),
                     (today.year, "0930"), (today.year, "1231")]
        # 往前推2年确保够用
        for y in range(today.year - 1, today.year - 3, -1):
            quarters.extend([(y, "0331"), (y, "0630"), (y, "0930"), (y, "1231")])

        # 过滤掉未来的日期，按时间降序
        dates = []
        for year, mmdd in quarters:
            d = f"{year}{mmdd}"
            if int(d) <= int(today.strftime("%Y%m%d")):
                dates.append(d)
        dates.sort(reverse=True)
        return dates[:n]

    def _fetch_multi_period_financial(self, n_periods: int = 4) -> dict[str, pd.DataFrame]:
        """获取最近N期批量财务数据

        Returns:
            {report_date: DataFrame} 按时间降序
        """
        dates = self._get_recent_report_dates(n_periods)
        result = {}
        for d in dates:
            try:
                df = self._source.fetch_batch_financial(report_date=d)
                if not df.empty:
                    result[d] = df
                    logger.info(f"[多期财报] {d}: {len(df)}只")
                _time.sleep(1)  # API冷却
            except Exception as e:
                logger.warning(f"[多期财报] {d} 获取失败: {e}")
        return result

    def _fetch_spot_data(self, market: str) -> pd.DataFrame:
        """获取实时行情（允许失败）"""
        try:
            return self._source.fetch_spot_data(market=market)
        except Exception:
            return pd.DataFrame()

    def _pre_filter(self, df: pd.DataFrame, has_spot: bool) -> pd.DataFrame:
        """前置过滤"""
        filters = self._config.get("filters", {})
        before = len(df)

        # 排除ST / *ST
        if filters.get("exclude_st", True) and "name" in df.columns:
            df = df[~df["name"].str.contains(r"ST|\*ST", na=False, regex=True)]
            logger.debug(f"排除ST: {before} → {len(df)}")

        # 排除亏损股（EPS <= 0 或 ROE < -20%）
        if "eps" in df.columns:
            df = df[pd.to_numeric(df["eps"], errors="coerce") > 0]
            logger.debug(f"排除亏损: → {len(df)}")

        # 排除ROE异常（> 100% 通常是数据异常）
        if "roe" in df.columns:
            roe = pd.to_numeric(df["roe"], errors="coerce")
            df = df[(roe > -20) & (roe < 100)]
            logger.debug(f"排除ROE异常: → {len(df)}")

        # 市值过滤（需要spot数据）
        if has_spot and "total_market_cap" in df.columns:
            min_cap = filters.get("min_market_cap", 2e9)
            cap = pd.to_numeric(df["total_market_cap"], errors="coerce")
            df = df[cap >= min_cap]
            logger.debug(f"排除小市值(<{min_cap/1e8:.0f}亿): → {len(df)}")

        # 成交额过滤（排除流动性极差的）
        if has_spot and "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce")
            df = df[amt >= 5e6]  # 成交额 >= 500万
            logger.debug(f"排除低流动性: → {len(df)}")

        return df.reset_index(drop=True)

    def _score_all(
        self, df: pd.DataFrame, has_spot: bool,
        multi_period: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """计算所有因子评分"""
        factors_config = self._config.get("factors", {})
        df = df.copy()

        # 计算各因子
        quality = self._calc_quality_score(df)
        growth = self._calc_growth_score(df)

        if has_spot:
            valuation = self._calc_valuation_score(df)
            momentum = self._calc_momentum_score(df)
        else:
            valuation = pd.Series(50.0, index=df.index)  # 中性默认
            momentum = pd.Series(50.0, index=df.index)

        # 连续增长因子（可选）
        if multi_period:
            consec_growth = self._calc_consecutive_growth_score(df, multi_period)
        else:
            consec_growth = None

        # 权重（从配置读取，或使用默认值）
        w_quality = factors_config.get("quality_score", {}).get("weight", 0.30)
        w_growth = factors_config.get("growth_score", {}).get("weight", 0.25) if "growth_score" in factors_config else 0.25
        w_valuation = factors_config.get("valuation_score", {}).get("weight", 0.25)
        w_momentum = factors_config.get("momentum_score", {}).get("weight", 0.20)

        if not has_spot:
            total_w = w_quality + w_growth
            w_quality = w_quality / total_w * 0.85
            w_growth = w_growth / total_w * 0.85
            w_valuation = 0.075
            w_momentum = 0.075

        if consec_growth is not None:
            # 有连续增长因子时，从其他因子匀出15%权重
            w_consec = 0.15
            scale = 1.0 - w_consec
            df["composite_score"] = (
                quality * w_quality * scale +
                growth * w_growth * scale +
                valuation * w_valuation * scale +
                momentum * w_momentum * scale +
                consec_growth * w_consec
            )
        else:
            df["composite_score"] = (
                quality * w_quality +
                growth * w_growth +
                valuation * w_valuation +
                momentum * w_momentum
            )

        # 保存各因子得分
        factor_scores_list = []
        for i in range(len(df)):
            fs = {
                "quality": round(float(quality.iloc[i]), 1),
                "growth": round(float(growth.iloc[i]), 1),
                "valuation": round(float(valuation.iloc[i]), 1),
                "momentum": round(float(momentum.iloc[i]), 1),
            }
            if consec_growth is not None:
                fs["consecutive"] = round(float(consec_growth.iloc[i]), 1)
            factor_scores_list.append(fs)
        df["factor_scores"] = factor_scores_list

        return df

    def _calc_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """质量因子：ROE + 毛利率 + 现金流质量

        评分 0-100，使用绝对阈值+百分位混合打分。
        """
        score = pd.Series(0.0, index=df.index)

        # ROE 评分 (0-40分): 绝对阈值打分
        if "roe" in df.columns:
            roe = pd.to_numeric(df["roe"], errors="coerce").fillna(0)
            # ROE 8%→16分, 15%→30分, 20%→40分
            roe_score = np.clip(roe / 20 * 40, 0, 40)
            score += roe_score

        # 毛利率 评分 (0-30分): 百分位排名
        if "gross_margin" in df.columns:
            gm = pd.to_numeric(df["gross_margin"], errors="coerce").fillna(0)
            gm_score = gm.rank(pct=True) * 30
            score += gm_score

        # 每股经营现金流 评分 (0-30分): 现金流/EPS 质量
        if "cashflow_per_share" in df.columns and "eps" in df.columns:
            cf = pd.to_numeric(df["cashflow_per_share"], errors="coerce").fillna(0)
            eps = pd.to_numeric(df["eps"], errors="coerce").fillna(0.01)
            cf_ratio = pd.Series(np.where(eps > 0, cf / eps, 0), index=df.index)
            cf_score = cf_ratio.rank(pct=True) * 30
            score += cf_score

        return np.clip(score, 0, 100)

    def _calc_growth_score(self, df: pd.DataFrame) -> pd.Series:
        """成长因子：营收增速 + 利润增速

        评分 0-100，使用百分位排名获得更好的区分度。
        """
        score = pd.Series(0.0, index=df.index)

        # 营收同比增速 (0-50分): 使用百分位排名
        if "revenue_yoy" in df.columns:
            rev_g = pd.to_numeric(df["revenue_yoy"], errors="coerce").fillna(0)
            rev_g = np.clip(rev_g, -100, 300)
            # 百分位排名：排名越高 → 分数越高
            rev_score = rev_g.rank(pct=True) * 50
            score += rev_score

        # 净利润同比增速 (0-50分): 使用百分位排名
        if "profit_yoy" in df.columns:
            pft_g = pd.to_numeric(df["profit_yoy"], errors="coerce").fillna(0)
            pft_g = np.clip(pft_g, -100, 300)
            pft_score = pft_g.rank(pct=True) * 50
            score += pft_score

        return np.clip(score, 0, 100)

    def _calc_valuation_score(self, df: pd.DataFrame) -> pd.Series:
        """估值因子：PE/PB 越低评分越高（逆序百分位）

        评分 0-100，低估 = 高分。
        """
        score = pd.Series(50.0, index=df.index)

        if "pe" in df.columns and "pb" in df.columns:
            pe = pd.to_numeric(df["pe"], errors="coerce")
            pb = pd.to_numeric(df["pb"], errors="coerce")

            # PE 评分 (0-60): 过滤无效值，使用逆百分位
            pe_valid = pe[(pe > 0) & (pe < 300)]
            if len(pe_valid) > 100:
                # 逆百分位：PE 越低，rank 越高
                pe_rank = pe.rank(pct=True, ascending=True)  # 低PE → 低百分位
                pe_score = (1 - pe_rank) * 60  # 反转：低PE → 高分
                pe_score = pe_score.fillna(30)  # 无效PE给中性分
                score = pe_score
            else:
                pe_score = pd.Series(30.0, index=df.index)
                score = pe_score

            # PB 评分 (0-40): 低PB = 高分
            pb_valid = pb[(pb > 0) & (pb < 30)]
            if len(pb_valid) > 100:
                pb_rank = pb.rank(pct=True, ascending=True)
                pb_score = (1 - pb_rank) * 40
                pb_score = pb_score.fillna(20)
                score = score + pb_score

        return np.clip(score, 0, 100)

    def _calc_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """动量因子：60日涨幅 + 年初至今涨幅 + 量比

        评分 0-100，适度正动量 = 高分。
        """
        score = pd.Series(50.0, index=df.index)

        # 60日涨幅 (0-40分): 适度上涨最优，暴涨反而减分
        if "change_pct_60d" in df.columns:
            chg60 = pd.to_numeric(df["change_pct_60d"], errors="coerce").fillna(0)
            # 最优区间 10%-40%，过高过低都扣分
            # 使用钟形打分：20% 涨幅 → 满分
            m60_score = 40 * np.exp(-((chg60 - 20) ** 2) / (2 * 25 ** 2))
            score = m60_score

        # 年初至今涨幅 (0-30分)
        if "change_pct_ytd" in df.columns:
            chg_ytd = pd.to_numeric(df["change_pct_ytd"], errors="coerce").fillna(0)
            ytd_score = 30 * np.exp(-((chg_ytd - 15) ** 2) / (2 * 30 ** 2))
            score = score + ytd_score

        # 量比 (0-30分): 1.5-3.0 最优
        if "volume_ratio" in df.columns:
            vr = pd.to_numeric(df["volume_ratio"], errors="coerce").fillna(1)
            vr_score = 30 * np.exp(-((vr - 2.0) ** 2) / (2 * 1.5 ** 2))
            score = score + vr_score

        return np.clip(score, 0, 100)

    def _calc_consecutive_growth_score(
        self, df: pd.DataFrame, multi_period: dict[str, pd.DataFrame],
    ) -> pd.Series:
        """连续增长因子：基于多期季报数据

        评分逻辑（0-100）：
        - 连续N期利润同比正增长：N×15分（最高45分）
        - 连续N期营收同比正增长：N×10分（最高30分）
        - 增速递增加分：最新期 > 上期 → +10分
        - 环比正增长加分：利润环比>0 → +5/期（最高15分）
        """
        if not multi_period:
            return pd.Series(50.0, index=df.index)

        # 按时间降序排列的季报期
        sorted_dates = sorted(multi_period.keys(), reverse=True)

        # 构建每只股票的多期增速矩阵
        # {symbol: [{profit_yoy, revenue_yoy, profit_qoq, revenue_qoq}, ...]}
        symbol_growth: dict[str, list[dict]] = {}
        for d in sorted_dates:
            period_df = multi_period[d]
            for _, row in period_df.iterrows():
                sym = row.get("symbol", "")
                if not sym:
                    continue
                if sym not in symbol_growth:
                    symbol_growth[sym] = []
                symbol_growth[sym].append({
                    "date": d,
                    "profit_yoy": _safe_float(row.get("profit_yoy")),
                    "revenue_yoy": _safe_float(row.get("revenue_yoy")),
                })

        # 为 df 中每只股票计算得分
        scores = []
        for _, row in df.iterrows():
            sym = row.get("symbol", "")
            periods = symbol_growth.get(sym, [])
            scores.append(_score_consecutive_growth(periods))

        return pd.Series(scores, index=df.index, dtype=float)

    def run_growth_screen(
        self, market: str = "A", min_periods: int = 3, top_n: int = 50,
    ) -> list[ScreeningResult]:
        """专项筛选：连续季度增长股票

        与 run() 的区别：以连续增长因子为核心权重。

        Args:
            market: 市场
            min_periods: 最少连续增长的期数
            top_n: 输出Top N
        """
        logger.info(f"开始连续增长筛选 min_periods={min_periods} top_n={top_n}")

        # Step 1: 获取多期财报数据
        multi_period = self._fetch_multi_period_financial(n_periods=min_periods + 1)
        if not multi_period:
            logger.error("获取多期财务数据失败")
            return []

        # Step 2: 以数据量最大的一期为基础（最新期可能只有少量股票披露）
        sorted_dates = sorted(multi_period.keys(), reverse=True)
        base_date = sorted_dates[0]
        for d in sorted_dates:
            if len(multi_period[d]) >= 1000:
                base_date = d
                break
        fin_df = multi_period[base_date].copy()
        logger.info(f"基础数据: {base_date}, {len(fin_df)}只")

        # Step 3: 实时行情（辅助）
        spot_df = self._fetch_spot_data(market)
        has_spot = False
        if not spot_df.empty:
            logger.info(f"实时行情: {len(spot_df)}只")
            fin_df = fin_df.merge(
                spot_df[["symbol", "price", "change_pct", "pe", "pb",
                         "total_market_cap", "float_market_cap",
                         "change_pct_60d", "change_pct_ytd",
                         "volume_ratio", "turnover_rate", "amount"]],
                on="symbol", how="left",
            )
            has_spot = True

        # Step 4: 前置过滤
        filtered = self._pre_filter(fin_df, has_spot)
        logger.info(f"前置过滤后: {len(filtered)}")
        if filtered.empty:
            return []

        # Step 5: 计算连续增长因子
        consec_score = self._calc_consecutive_growth_score(filtered, multi_period)

        # Step 6: 其他因子
        quality = self._calc_quality_score(filtered)
        if has_spot:
            valuation = self._calc_valuation_score(filtered)
        else:
            valuation = pd.Series(50.0, index=filtered.index)

        # 连续增长为核心权重
        filtered = filtered.copy()
        filtered["composite_score"] = (
            consec_score * 0.50 +   # 连续增长（核心）
            quality * 0.30 +        # 质量（辅助）
            valuation * 0.20        # 估值（辅助）
        )

        # 过滤：连续增长分 >= 60才有意义
        filtered = filtered[consec_score >= 60]
        logger.info(f"连续增长过滤后: {len(filtered)}")

        if filtered.empty:
            return []

        # 保存因子得分
        factor_scores_list = []
        for i in filtered.index:
            factor_scores_list.append({
                "consecutive_growth": round(float(consec_score.loc[i]), 1),
                "quality": round(float(quality.loc[i]), 1),
                "valuation": round(float(valuation.loc[i]), 1),
            })
        filtered["factor_scores"] = factor_scores_list

        # 排名
        filtered = filtered.sort_values("composite_score", ascending=False).head(top_n)
        filtered["rank"] = range(1, len(filtered) + 1)

        results = []
        for _, row in filtered.iterrows():
            results.append(
                ScreeningResult(
                    symbol=row["symbol"],
                    name=row.get("name", ""),
                    rank=row["rank"],
                    composite_score=round(row["composite_score"], 2),
                    factor_scores=row.get("factor_scores", {}),
                    industry=row.get("industry", ""),
                )
            )

        logger.info(f"连续增长筛选完成: {len(results)}只")
        return results


def _safe_float(val) -> float:
    """安全转float"""
    try:
        v = float(val)
        return v if not np.isnan(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _score_consecutive_growth(periods: list[dict]) -> float:
    """计算单只股票的连续增长得分

    Args:
        periods: 按时间降序排列的多期增速数据

    Returns:
        0-100分
    """
    if len(periods) < 2:
        return 30.0  # 数据不足，中性偏低

    score = 0.0

    # 1. 连续利润同比正增长（每期15分，最高45分）
    profit_streak = 0
    for p in periods:
        if p["profit_yoy"] > 0:
            profit_streak += 1
        else:
            break
    score += min(profit_streak * 15, 45)

    # 2. 连续营收同比正增长（每期10分，最高30分）
    revenue_streak = 0
    for p in periods:
        if p["revenue_yoy"] > 0:
            revenue_streak += 1
        else:
            break
    score += min(revenue_streak * 10, 30)

    # 3. 利润增速递增（最新 > 上期 → +10）
    if len(periods) >= 2 and periods[0]["profit_yoy"] > periods[1]["profit_yoy"] > 0:
        score += 10

    # 4. 营收增速递增（最新 > 上期 → +5）
    if len(periods) >= 2 and periods[0]["revenue_yoy"] > periods[1]["revenue_yoy"] > 0:
        score += 5

    # 5. 最新期增速高（利润同比>30% → +10）
    if periods[0]["profit_yoy"] > 30:
        score += 10

    return min(score, 100.0)
