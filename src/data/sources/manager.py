"""
数据源管理器

Auto-fallback: 按优先级尝试多个数据源，失败自动切换。
复用 daily_stock_analysis/data_provider/base.py 的 DataFetcherManager 模式。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd
from loguru import logger

from .base import BaseDataSource, DataSourceError


class AllSourcesFailedError(DataSourceError):
    """所有数据源均失败"""


class DataSourceManager:
    """数据源管理器（auto-fallback）"""

    def __init__(self) -> None:
        self._sources: list[BaseDataSource] = []

    def register(self, source: BaseDataSource) -> None:
        """注册数据源"""
        self._sources.append(source)
        self._sources.sort(key=lambda s: s.priority)
        logger.info(f"注册数据源: {source.name} (priority={source.priority})")

    def _call_with_fallback(
        self, method_name: str, **kwargs: Any
    ) -> pd.DataFrame:
        """通用fallback调用"""
        errors = []
        for source in self._sources:
            if not source.is_available():
                logger.debug(f"[{source.name}] 冷却中，跳过")
                continue

            method = getattr(source, method_name, None)
            if method is None:
                continue

            try:
                result = method(**kwargs)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    logger.debug(f"[{source.name}].{method_name} 成功，{len(result)}行")
                    return result
                logger.debug(f"[{source.name}].{method_name} 返回空数据")
            except NotImplementedError:
                continue
            except Exception as e:
                source.mark_failed(e)
                errors.append(f"{source.name}: {e}")

        raise AllSourcesFailedError(
            f"所有数据源 {method_name} 失败: {'; '.join(errors)}"
        )

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取日线行情（auto-fallback）"""
        return self._call_with_fallback(
            "fetch_daily_quotes",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        """获取股票列表"""
        return self._call_with_fallback("fetch_stock_list", market=market)

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        """获取财务数据"""
        return self._call_with_fallback("fetch_financial_data", symbol=symbol)

    def fetch_money_flow(self, symbol: str, days: int = 20) -> pd.DataFrame:
        """获取资金流向"""
        return self._call_with_fallback(
            "fetch_money_flow", symbol=symbol, days=days
        )

    def fetch_stock_news(self, symbol: str, limit: int = 20) -> list[dict]:
        """获取个股新闻（auto-fallback）"""
        for source in self._sources:
            if not source.is_available():
                continue
            try:
                results = source.fetch_stock_news(symbol=symbol, limit=limit)
                if results:
                    logger.debug(f"[{source.name}].fetch_stock_news 成功，{len(results)}条")
                    return results
            except NotImplementedError:
                continue
            except Exception as e:
                source.mark_failed(e)
        return []

    def fetch_valuation(self, symbol: str) -> pd.DataFrame:
        """获取估值数据"""
        return self._call_with_fallback("fetch_valuation", symbol=symbol)

    def fetch_batch_financial(self, report_date: str = "") -> pd.DataFrame:
        """获取全市场批量财务数据"""
        return self._call_with_fallback("fetch_batch_financial", report_date=report_date)

    def fetch_spot_data(self, market: str = "A") -> pd.DataFrame:
        """获取全市场实时行情快照"""
        return self._call_with_fallback("fetch_spot_data", market=market)

    @property
    def available_sources(self) -> list[str]:
        return [s.name for s in self._sources if s.is_available()]
