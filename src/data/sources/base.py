"""
数据源基类

复用 daily_stock_analysis 的 auto-fallback 模式：
- 多数据源按优先级注册
- 失败自动切换
- Circuit breaker：失败源冷却
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from loguru import logger


class DataSourceError(Exception):
    """数据源错误"""


class BaseDataSource(ABC):
    """数据源基类"""

    name: str = "base"
    priority: int = 99  # 越小优先级越高

    def __init__(self, cooldown_seconds: int = 300):
        self._cooldown_seconds = cooldown_seconds
        self._last_failure_time: Optional[float] = None
        self._failure_count: int = 0

    def is_available(self) -> bool:
        """检查数据源是否可用（circuit breaker）"""
        if self._last_failure_time is None:
            return True
        elapsed = time.time() - self._last_failure_time
        if elapsed > self._cooldown_seconds:
            # 冷却期结束，重置
            self._last_failure_time = None
            self._failure_count = 0
            logger.info(f"[{self.name}] 冷却期结束，恢复可用")
            return True
        return False

    def mark_failed(self, error: Optional[Exception] = None) -> None:
        """标记数据源失败"""
        self._last_failure_time = time.time()
        self._failure_count += 1
        logger.warning(
            f"[{self.name}] 标记失败(第{self._failure_count}次)，"
            f"冷却{self._cooldown_seconds}秒。错误: {error}"
        )

    @abstractmethod
    def fetch_daily_quotes(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """获取日线行情"""

    @abstractmethod
    def fetch_stock_list(self, market: str = "A") -> pd.DataFrame:
        """获取股票列表"""

    def fetch_financial_data(self, symbol: str) -> pd.DataFrame:
        """获取财务数据（部分数据源可能不支持）"""
        raise NotImplementedError(f"{self.name} 不支持财务数据")

    def fetch_money_flow(self, symbol: str, days: int = 20) -> pd.DataFrame:
        """获取资金流向（部分数据源可能不支持）"""
        raise NotImplementedError(f"{self.name} 不支持资金流向数据")

    def fetch_valuation(self, symbol: str) -> pd.DataFrame:
        """获取估值数据"""
        raise NotImplementedError(f"{self.name} 不支持估值数据")

    def fetch_stock_news(self, symbol: str, limit: int = 20) -> list[dict]:
        """获取个股新闻（部分数据源可能不支持）

        返回: [{title, content, datetime, source}, ...]
        """
        raise NotImplementedError(f"{self.name} 不支持个股新闻")

    def fetch_batch_financial(self, report_date: str = "") -> pd.DataFrame:
        """获取全市场批量财务数据（用于筛选引擎）

        返回标准化列：symbol, name, eps, bps, roe, gross_margin,
        revenue_yoy, profit_yoy, cashflow_per_share, industry
        """
        raise NotImplementedError(f"{self.name} 不支持批量财务数据")

    def fetch_spot_data(self, market: str = "A") -> pd.DataFrame:
        """获取全市场实时行情快照（含PE/PB/市值/涨跌幅等）

        返回标准化列：symbol, name, price, change_pct, volume, amount,
        turnover_rate, pe, pb, total_market_cap, float_market_cap,
        change_pct_60d, change_pct_ytd, volume_ratio
        """
        raise NotImplementedError(f"{self.name} 不支持实时行情快照")

    def fetch_business_composition(self, symbol: str) -> pd.DataFrame:
        """获取分业务/分产品营收构成

        返回：product, revenue, revenue_pct, gross_margin, cost, report_date
        """
        raise NotImplementedError(f"{self.name} 不支持分业务营收构成")

    def fetch_profit_forecast(self, symbol: str) -> pd.DataFrame:
        """获取券商盈利预测/一致预期

        返回：year, institution, eps, profit, revenue 等
        """
        raise NotImplementedError(f"{self.name} 不支持券商盈利预测")

    def fetch_shareholder_count(self, symbol: str) -> pd.DataFrame:
        """获取股东人数变化历史

        返回：date, count, change_pct, avg_market_value, total_market_cap
        """
        raise NotImplementedError(f"{self.name} 不支持股东人数数据")

    def fetch_northbound_holdings(self, symbol: str) -> pd.DataFrame:
        """获取北向资金持股（个股）

        返回：date, hold_shares, hold_market_value, hold_ratio_float, change_shares
        """
        raise NotImplementedError(f"{self.name} 不支持北向资金数据")

    def fetch_dividend_history(self, symbol: str) -> pd.DataFrame:
        """获取分红送配历史

        返回：report_date, div_per_share, dividend_yield, eps, payout_ratio, ex_date
        """
        raise NotImplementedError(f"{self.name} 不支持分红历史数据")

    def fetch_dragon_tiger(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """获取龙虎榜+大宗交易（合并）

        返回：date, source, net_buy_amount, buy_amount, sell_amount, premium_rate, reason
        """
        raise NotImplementedError(f"{self.name} 不支持龙虎榜数据")

    def fetch_research_reports(self, symbol: str) -> dict:
        """获取个股研报数据（评级+EPS预测+机构覆盖+机构参与度）

        返回：dict，包含 rating_distribution, coverage_count, eps_forecasts,
        recent_titles, institutional_participation 等字段
        """
        raise NotImplementedError(f"{self.name} 不支持个股研报数据")

    def fetch_margin_data(self, symbol: str) -> pd.DataFrame:
        """获取个股融资融券数据

        返回：date, margin_buy, margin_balance, short_sell, short_balance, total_balance
        """
        raise NotImplementedError(f"{self.name} 不支持融资融券数据")

    def fetch_performance_forecast(self, report_date: str = "") -> pd.DataFrame:
        """获取全市场业绩预告（批量）

        返回：symbol, name, forecast_type, change_pct, forecast_content, announce_date
        """
        raise NotImplementedError(f"{self.name} 不支持业绩预告")

    def fetch_hot_rank(self) -> pd.DataFrame:
        """获取人气排名Top100

        返回：rank, symbol, name, price, change_pct
        """
        raise NotImplementedError(f"{self.name} 不支持人气排名")

    def fetch_zt_pool(self, date: str = "") -> pd.DataFrame:
        """获取涨停池

        返回：symbol, name, change_pct, price, amount, seal_amount, first_time, last_time,
        zt_count, continuous, industry
        """
        raise NotImplementedError(f"{self.name} 不支持涨停池")

    def fetch_industry_peers(self, symbol: str) -> pd.DataFrame:
        """获取同行业个股（行业成分股）

        返回：symbol, name, price, change_pct, pe, pb, turnover_rate
        """
        raise NotImplementedError(f"{self.name} 不支持行业成分股")
