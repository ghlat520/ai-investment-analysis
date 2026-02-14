"""
baostock 数据源

独立 TCP 协议，不受 HTTP 代理影响。适合 Clash 代理环境下获取 A 股行情/财务数据。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import pandas as pd
from loguru import logger

from .base import BaseDataSource


@contextmanager
def _bs_session() -> Generator:
    """baostock 登录/登出上下文管理器"""
    import baostock as bs

    bs.login()
    try:
        yield bs
    finally:
        bs.logout()


def _to_bs_code(symbol: str) -> str:
    """将 300054 或 300054.SZ 转为 baostock 格式 sz.300054"""
    code = symbol.split(".")[0]
    if code.startswith(("6",)):
        return f"sh.{code}"
    return f"sz.{code}"


class BaoStockSource(BaseDataSource):
    name = "baostock"
    priority = -1  # 最高优先级（比 efinance=0 更高）

    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        bs_code = _to_bs_code(symbol)

        with _bs_session() as bs:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
            )
            rows = []
            while (rs.error_code == "0") & rs.next():
                rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)

        # 类型转换
        for col in ["open", "high", "low", "close", "volume", "amount", "turn", "pctChg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 统一列名
        df = df.rename(columns={
            "turn": "turnover_rate",
            "pctChg": "change_pct",
        })

        logger.debug(f"[baostock] {symbol} 行情 {len(df)} 条")
        return df

    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        if market != "A":
            raise NotImplementedError("baostock 仅支持 A 股")

        with _bs_session() as bs:
            rs = bs.query_stock_basic()
            rows = []
            while (rs.error_code == "0") & rs.next():
                rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        # 只保留正常上市的股票
        df = df[df["type"] == "1"]  # 1=股票

        df["symbol"] = df["code"].apply(
            lambda x: x.replace("sh.", "").replace("sz.", "")
        )
        df["symbol"] = df["symbol"].apply(
            lambda x: f"{x}.SH" if x.startswith("6") else f"{x}.SZ"
        )
        df = df.rename(columns={"code_name": "name"})
        df["market"] = "A"
        return df[["symbol", "name", "market"]]

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        # BaoStock 财务数据不完整（缺 revenue_yoy/profit_yoy/debt_ratio/current_ratio/operating_cashflow），
        # 交由 AKShare 处理，数据更全面。
        raise NotImplementedError("baostock 财务数据不完整，使用 akshare 替代")
