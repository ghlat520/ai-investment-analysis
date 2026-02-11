"""
行情数据采集器

每日收盘后采集全A股日线行情，入库PostgreSQL。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from ..sources import DataSourceManager


class QuoteCollector:
    """日线行情采集器"""

    def __init__(self, source_manager: DataSourceManager) -> None:
        self._source = source_manager

    def collect_single(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """采集单只股票行情"""
        if end_date is None:
            end_date = date.today().isoformat()
        if start_date is None:
            start_date = (date.today() - timedelta(days=365)).isoformat()

        logger.info(f"采集行情: {symbol} [{start_date} ~ {end_date}]")
        return self._source.fetch_daily_quotes(symbol, start_date, end_date)

    def collect_batch(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """批量采集多只股票行情"""
        results = {}
        for symbol in symbols:
            try:
                df = self.collect_single(symbol, start_date, end_date)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"采集 {symbol} 行情失败: {e}")
        logger.info(f"批量采集完成: {len(results)}/{len(symbols)}")
        return results
