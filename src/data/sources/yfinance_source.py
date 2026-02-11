"""
yfinance 数据源

覆盖全球市场（A股/港股/美股），作为最终fallback。
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from .base import BaseDataSource


class YFinanceSource(BaseDataSource):
    name = "yfinance"
    priority = 5  # 最低优先级

    @staticmethod
    def _to_yf_symbol(symbol: str) -> str:
        """转换为yfinance格式"""
        if symbol.endswith(".SH"):
            return symbol.replace(".SH", ".SS")
        if symbol.endswith(".SZ"):
            return symbol  # yfinance支持.SZ
        return symbol

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import yfinance as yf

        yf_symbol = self._to_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)
        if df.empty:
            return df

        df = df.reset_index()
        rename_map = {
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns=rename_map)
        return df[["date", "open", "high", "low", "close", "volume"]]

    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        # yfinance不支持股票列表，返回空
        return pd.DataFrame()
