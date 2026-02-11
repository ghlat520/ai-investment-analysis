"""
量化筛选引擎

Layer 1: 纯代码，全市场覆盖 5000+ 标的
输出 Top N 候选标的给 Layer 2 (AI Agent分析)

流程：
1. 获取全市场股票列表
2. 前置过滤（排除ST/停牌/市值过小）
3. 计算多因子评分
4. 排名输出 Top N
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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

        # Step 1: 获取股票列表
        stock_list = self._source.fetch_stock_list(market=market)
        if stock_list.empty:
            logger.error("获取股票列表失败")
            return []
        logger.info(f"全市场股票数: {len(stock_list)}")

        # Step 2: 前置过滤
        filtered = self._pre_filter(stock_list)
        logger.info(f"前置过滤后: {len(filtered)}")

        # Step 3: 计算因子评分
        scored = self._score_all(filtered)

        # Step 4: 排名输出
        scored = scored.sort_values("composite_score", ascending=False).head(top_n)
        scored["rank"] = range(1, len(scored) + 1)

        results = []
        for _, row in scored.iterrows():
            results.append(
                ScreeningResult(
                    symbol=row["symbol"],
                    name=row.get("name", ""),
                    rank=row["rank"],
                    composite_score=row["composite_score"],
                    factor_scores=row.get("factor_scores", {}),
                )
            )

        logger.info(f"筛选完成: Top {len(results)}")
        return results

    def _pre_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """前置过滤"""
        filters = self._config.get("filters", {})

        # 排除ST
        if filters.get("exclude_st", True) and "name" in df.columns:
            before = len(df)
            df = df[~df["name"].str.contains("ST|\\*ST", na=False)]
            logger.debug(f"排除ST: {before} → {len(df)}")

        return df

    def _score_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """为所有股票计算因子评分

        Phase 1: 简化版，仅返回随机分数占位。
        后续替换为真实因子计算（需要行情+财务数据）。
        """
        factors_config = self._config.get("factors", {})

        # Phase 1 占位：随机评分（后续替换为真实计算）
        rng = np.random.default_rng(42)
        df = df.copy()
        df["composite_score"] = 0.0
        factor_scores_list = []

        for _, row in df.iterrows():
            scores = {}
            total = 0.0
            for factor_name, factor_cfg in factors_config.items():
                weight = factor_cfg.get("weight", 0.2)
                # TODO: 替换为真实因子计算
                score = rng.uniform(0, 100)
                scores[factor_name] = round(score, 2)
                total += score * weight
            factor_scores_list.append(scores)
            df.at[_, "composite_score"] = round(total, 2)

        df["factor_scores"] = factor_scores_list
        return df
