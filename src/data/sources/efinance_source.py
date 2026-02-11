"""
efinance 数据源

东方财富数据，最高优先级。
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from .base import BaseDataSource


class EFinanceSource(BaseDataSource):
    name = "efinance"
    priority = 0  # 最高优先级

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import efinance as ef

        code = symbol.split(".")[0]
        df = ef.stock.get_quote_history(code, beg=start_date.replace("-", ""), end=end_date.replace("-", ""))
        if df.empty:
            return df

        rename_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
            "振幅": "amplitude",
            "涨跌幅": "change_pct",
            "涨跌额": "change_amount",
        }
        df = df.rename(columns=rename_map)
        return df

    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        import efinance as ef

        if market == "A":
            df = ef.stock.get_realtime_quotes()
            if df.empty:
                return df
            rename_map = {
                "股票代码": "symbol",
                "股票名称": "name",
            }
            df = df.rename(columns=rename_map)
            df["symbol"] = df["symbol"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            df["market"] = "A"
            return df[["symbol", "name", "market"]]
        return pd.DataFrame()
