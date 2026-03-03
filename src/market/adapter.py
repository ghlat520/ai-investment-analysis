"""
市场适配器

每个市场声明支持哪些Agent、有哪些特有数据。
Phase 1: 仅A股
Phase 2: +港股
Phase 3: +美股

R4: 市场环境检测 + 恐贪指数
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from loguru import logger

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


# ============================================================
# R4: 市场环境检测
# ============================================================


def detect_market_regime(index_quotes: list[dict] | None = None) -> dict:
    """检测市场环境

    基于指数数据判断牛市/熊市/震荡，并计算恐贪指数。
    如果无指数数据，返回neutral默认值。

    Args:
        index_quotes: 指数日线数据 [{date, open, high, low, close, volume}, ...]

    Returns:
        {
            "regime": "bull" | "bear" | "neutral" | "value_investing",
            "fear_greed_index": 0-100 (0=极度恐惧, 100=极度贪婪),
            "valuation_discount": 0.0-0.3 (安全边际折扣建议),
            "details": {...},
        }
    """
    result = {
        "regime": "value_investing",  # 默认使用价值投资权重
        "fear_greed_index": 50,
        "valuation_discount": 0.0,
        "details": {},
    }

    if not index_quotes or len(index_quotes) < 60:
        logger.debug("[MarketRegime] 指数数据不足60天，使用默认value_investing")
        return result

    try:
        import pandas as pd

        df = pd.DataFrame(index_quotes)
        if "close" not in df.columns:
            return result

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 60:
            return result

        # 1. 趋势判断: 60日均线 vs 200日均线
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else close.mean()
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else close.mean()
        current = close.iloc[-1]

        trend_score = 0  # -100 ~ +100
        if current > ma60 and ma60 > ma200:
            trend_score = 60  # 多头排列
        elif current > ma60:
            trend_score = 30
        elif current < ma60 and ma60 < ma200:
            trend_score = -60  # 空头排列
        elif current < ma60:
            trend_score = -30

        # 2. 市场宽度: 距60日高低点的位置
        high_60 = close.tail(60).max()
        low_60 = close.tail(60).min()
        if high_60 > low_60:
            breadth = (current - low_60) / (high_60 - low_60) * 100
        else:
            breadth = 50

        # 3. 波动率: 20日滚动波动率
        returns = close.pct_change().dropna()
        vol_20 = returns.tail(20).std() * (252 ** 0.5) * 100 if len(returns) >= 20 else 20
        # 波动率分位 (越高越恐惧)
        vol_score = min(100, max(0, vol_20 * 3))  # 粗略映射

        # 4. 动量: 20日收益率
        momentum = (current / close.iloc[-min(20, len(close))] - 1) * 100

        # 恐贪指数: 综合以上指标 (0=极度恐惧, 100=极度贪婪)
        fear_greed = (
            trend_score * 0.3  # 趋势
            + breadth * 0.3  # 市场宽度
            + (100 - vol_score) * 0.2  # 波动率(反转)
            + min(100, max(0, momentum * 5 + 50)) * 0.2  # 动量
        )
        fear_greed = max(0, min(100, fear_greed))

        result["fear_greed_index"] = round(fear_greed)
        result["details"] = {
            "trend_score": round(trend_score),
            "breadth": round(breadth),
            "volatility_20d": round(vol_20, 1),
            "momentum_20d": round(momentum, 1),
            "ma60": round(ma60, 2),
            "ma200": round(ma200, 2),
            "current": round(current, 2),
        }

        # 判断市场环境
        if fear_greed >= 75:
            result["regime"] = "bull"
            # 极度贪婪时提高安全边际要求
            result["valuation_discount"] = 0.15
        elif fear_greed >= 60:
            result["regime"] = "bull"
            result["valuation_discount"] = 0.05
        elif fear_greed <= 20:
            result["regime"] = "bear"
            # 极度恐惧 = 价值投资者的"打折季"
            result["valuation_discount"] = -0.10  # 负值=降低安全边际要求
        elif fear_greed <= 35:
            result["regime"] = "bear"
            result["valuation_discount"] = -0.05
        else:
            result["regime"] = "value_investing"  # 默认价值投资权重
            result["valuation_discount"] = 0.0

        logger.info(
            f"[MarketRegime] regime={result['regime']} "
            f"fear_greed={result['fear_greed_index']} "
            f"discount={result['valuation_discount']:.0%} "
            f"trend={trend_score} breadth={breadth:.0f} vol={vol_20:.1f}%"
        )

    except Exception as e:
        logger.warning(f"[MarketRegime] 检测失败: {e}，使用默认value_investing")

    return result
