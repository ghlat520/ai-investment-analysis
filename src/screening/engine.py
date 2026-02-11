"""
量化筛选引擎

Layer 1: 纯代码，全市场覆盖 5000+ 标的
输出 Top N 候选标的给 Layer 2 (AI Agent分析)

数据策略：
- 主数据源：stock_yjbb_em 批量财报（~5000只，always works）
- 辅助数据源：stock_zh_a_spot_em 实时行情（PE/PB/市值/涨跌幅）
- 优雅降级：spot数据不可用时，仅用财报因子筛选
"""

from __future__ import annotations

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

    def run(self, market: str = "A") -> list[ScreeningResult]:
        """执行筛选流程"""
        top_n = self._config.get("top_n", 50)
        logger.info(f"开始量化筛选 market={market} top_n={top_n}")

        # Step 1: 获取批量财务数据（主数据源）
        fin_df = self._fetch_batch_financial()
        if fin_df.empty:
            logger.error("获取批量财务数据失败")
            return []
        logger.info(f"批量财务数据: {len(fin_df)}只")

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
        scored = self._score_all(filtered, has_spot)

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

    def _score_all(self, df: pd.DataFrame, has_spot: bool) -> pd.DataFrame:
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

        # 权重（从配置读取，或使用默认值）
        w_quality = factors_config.get("quality_score", {}).get("weight", 0.30)
        w_growth = factors_config.get("growth_score", {}).get("weight", 0.25) if "growth_score" in factors_config else 0.25
        w_valuation = factors_config.get("valuation_score", {}).get("weight", 0.25)
        w_momentum = factors_config.get("momentum_score", {}).get("weight", 0.20)

        if not has_spot:
            # 无spot数据时，重新分配权重给quality和growth
            total_w = w_quality + w_growth
            w_quality = w_quality / total_w * 0.85
            w_growth = w_growth / total_w * 0.85
            w_valuation = 0.075
            w_momentum = 0.075

        df["composite_score"] = (
            quality * w_quality +
            growth * w_growth +
            valuation * w_valuation +
            momentum * w_momentum
        )

        # 保存各因子得分
        factor_scores_list = []
        for i in range(len(df)):
            factor_scores_list.append({
                "quality": round(float(quality.iloc[i]), 1),
                "growth": round(float(growth.iloc[i]), 1),
                "valuation": round(float(valuation.iloc[i]), 1),
                "momentum": round(float(momentum.iloc[i]), 1),
            })
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
