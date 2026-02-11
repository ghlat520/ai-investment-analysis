"""
AKShare 数据源

A股数据：日线行情、财务数据、资金流向、估值。
优先级1（efinance之后）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from .base import BaseDataSource


class AKShareSource(BaseDataSource):
    name = "akshare"
    priority = 1

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        import akshare as ak

        # AKShare用纯数字代码
        code = symbol.split(".")[0]
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",
        )
        if df.empty:
            return df

        # 标准化列名
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
        import akshare as ak

        if market == "A":
            df = ak.stock_zh_a_spot_em()
            rename_map = {
                "代码": "symbol",
                "名称": "name",
            }
            df = df.rename(columns=rename_map)
            # 添加后缀
            df["symbol"] = df["symbol"].apply(
                lambda x: f"{x}.SZ" if x.startswith(("0", "3")) else f"{x}.SH"
            )
            df["market"] = "A"
            return df[["symbol", "name", "market"]]
        return pd.DataFrame()

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            # 主要财务指标
            df = ak.stock_financial_analysis_indicator(symbol=code)
            if df.empty:
                return df
            # 标准化（AKShare财务数据列名可能变化，做兼容处理）
            return df
        except Exception as e:
            logger.debug(f"[akshare] 财务数据获取失败: {e}")
            return pd.DataFrame()

    def fetch_money_flow(self, symbol: str, days: int = 20) -> pd.DataFrame:
        import akshare as ak

        code = symbol.split(".")[0]
        try:
            df = ak.stock_individual_fund_flow(stock=code, market="sh" if symbol.endswith(".SH") else "sz")
            if df.empty:
                return df
            # 取最近N天
            return df.tail(days)
        except Exception as e:
            logger.debug(f"[akshare] 资金流向获取失败: {e}")
            return pd.DataFrame()
