"""
市场适配器

每个市场声明支持哪些Agent、有哪些特有数据。
Phase 1: 仅A股
Phase 2: +港股
Phase 3: +美股
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.data.sources import DataSourceManager
from src.data.sources.akshare_source import AKShareSource
from src.data.sources.efinance_source import EFinanceSource
from src.data.sources.yfinance_source import YFinanceSource


class MarketAdapter(ABC):
    """市场适配器基类"""

    market_code: str = ""
    market_name: str = ""

    @abstractmethod
    def supported_agents(self) -> list[str]:
        """返回该市场支持的Agent列表"""

    @abstractmethod
    def create_source_manager(self) -> DataSourceManager:
        """创建该市场的数据源管理器"""

    @abstractmethod
    def get_trading_hours(self) -> dict[str, str]:
        """返回交易时间"""


class AShareAdapter(MarketAdapter):
    """A股市场适配器"""

    market_code = "A"
    market_name = "A股"

    def supported_agents(self) -> list[str]:
        return [
            "technical",
            "fundamental",
            "valuation",
            "money_flow",
            "sentiment",
            "industry",
        ]

    def create_source_manager(self) -> DataSourceManager:
        manager = DataSourceManager()
        manager.register(EFinanceSource())
        manager.register(AKShareSource())
        manager.register(YFinanceSource())
        return manager

    def get_trading_hours(self) -> dict[str, str]:
        return {
            "timezone": "Asia/Shanghai",
            "morning_open": "09:30",
            "morning_close": "11:30",
            "afternoon_open": "13:00",
            "afternoon_close": "15:00",
            "data_collection_time": "16:30",
        }


class HKStockAdapter(MarketAdapter):
    """港股市场适配器（Phase 2）"""

    market_code = "HK"
    market_name = "港股"

    def supported_agents(self) -> list[str]:
        return ["technical", "fundamental", "valuation", "sentiment"]

    def create_source_manager(self) -> DataSourceManager:
        manager = DataSourceManager()
        manager.register(YFinanceSource())
        return manager

    def get_trading_hours(self) -> dict[str, str]:
        return {
            "timezone": "Asia/Hong_Kong",
            "morning_open": "09:30",
            "morning_close": "12:00",
            "afternoon_open": "13:00",
            "afternoon_close": "16:00",
        }


class USStockAdapter(MarketAdapter):
    """美股市场适配器（Phase 3）"""

    market_code = "US"
    market_name = "美股"

    def supported_agents(self) -> list[str]:
        return ["technical", "fundamental", "valuation", "sentiment"]

    def create_source_manager(self) -> DataSourceManager:
        manager = DataSourceManager()
        manager.register(YFinanceSource())
        return manager

    def get_trading_hours(self) -> dict[str, str]:
        return {
            "timezone": "America/New_York",
            "open": "09:30",
            "close": "16:00",
        }
