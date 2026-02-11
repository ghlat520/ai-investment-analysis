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
