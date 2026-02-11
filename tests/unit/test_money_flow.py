"""资金面Agent单元测试"""

from src.agents.analysts.money_flow import (
    _score_flow_intensity,
    _score_flow_price_correlation,
    _score_main_force_trend,
    _score_order_divergence,
    _score_xl_orders,
    _to_dataframe,
    analyze_money_flow,
)
from src.agents.state import StockData

import pandas as pd


# ─── 辅助 ──────────────────────────────────────────────

def _make_flow_data(days: int = 10, main_net: float = 1e6) -> list[dict]:
    """生成测试资金流向数据"""
    import datetime
    base = datetime.date(2025, 1, 1)
    return [
        {
            "日期": (base + datetime.timedelta(days=i)).isoformat(),
            "收盘价": 10.0 + i * 0.1,
            "涨跌幅": 1.0,
            "主力净流入-净额": main_net * (1 if i % 2 == 0 else -0.5),
            "主力净流入-净占比": 5.0 if main_net > 0 else -5.0,
            "超大单净流入-净额": main_net * 0.5,
            "超大单净流入-净占比": 2.0,
            "大单净流入-净额": main_net * 0.3,
            "大单净流入-净占比": 1.5,
            "中单净流入-净额": -main_net * 0.1,
            "中单净流入-净占比": -0.5,
            "小单净流入-净额": -main_net * 0.2,
            "小单净流入-净占比": -1.0,
        }
        for i in range(days)
    ]


def _make_continuous_inflow(days: int = 5) -> list[dict]:
    """生成连续流入数据"""
    import datetime
    base = datetime.date(2025, 1, 1)
    return [
        {
            "日期": (base + datetime.timedelta(days=i)).isoformat(),
            "收盘价": 10.0 + i * 0.2,
            "涨跌幅": 2.0,
            "主力净流入-净额": 5e6 + i * 1e6,  # 递增流入
            "主力净流入-净占比": 8.0 + i,
            "超大单净流入-净额": 3e6,
            "超大单净流入-净占比": 4.0,
            "大单净流入-净额": 2e6,
            "大单净流入-净占比": 3.0,
            "中单净流入-净额": -1e6,
            "中单净流入-净占比": -1.0,
            "小单净流入-净额": -2e6,
            "小单净流入-净占比": -2.0,
        }
        for i in range(days)
    ]


# ─── 测试 ──────────────────────────────────────────────


class TestToDataframe:
    def test_empty(self):
        assert _to_dataframe([]).empty

    def test_column_rename(self):
        data = _make_flow_data(3)
        df = _to_dataframe(data)
        assert "main_net" in df.columns
        assert "日期" not in df.columns


class TestMainForceTrend:
    def test_continuous_inflow(self):
        df = _to_dataframe(_make_continuous_inflow(5))
        score, desc = _score_main_force_trend(df)
        assert score >= 20
        assert "连续" in desc

    def test_insufficient_data(self):
        df = _to_dataframe(_make_flow_data(2))
        score, _ = _score_main_force_trend(df)
        assert score == 0


class TestOrderDivergence:
    def test_main_absorption(self):
        """大单流入+小单流出 = 主力吸筹"""
        data = _make_continuous_inflow(5)
        df = _to_dataframe(data)
        score, desc = _score_order_divergence(df)
        assert score == 20
        assert "吸筹" in desc


class TestXLOrders:
    def test_continuous_xl_inflow(self):
        df = _to_dataframe(_make_continuous_inflow(5))
        score, desc = _score_xl_orders(df)
        assert score >= 10


class TestFlowPriceCorrelation:
    def test_bullish_correlation(self):
        """量价齐升"""
        df = _to_dataframe(_make_continuous_inflow(5))
        score, desc = _score_flow_price_correlation(df)
        assert score > 0
        assert "齐升" in desc


class TestFlowIntensity:
    def test_high_intensity(self):
        df = _to_dataframe(_make_continuous_inflow(5))
        score, desc = _score_flow_intensity(df)
        assert score >= 10


class TestAnalyzeMoneyFlow:
    def test_no_data(self):
        stock = StockData(symbol="000001.SZ", name="测试", market="A")
        signal = analyze_money_flow(stock)
        assert signal.signal_score == 0
        assert signal.confidence == 0.0

    def test_with_data(self):
        stock = StockData(
            symbol="000001.SZ", name="测试", market="A",
            money_flow=_make_continuous_inflow(10),
        )
        signal = analyze_money_flow(stock)
        assert -100 <= signal.signal_score <= 100
        assert signal.confidence > 0
        assert signal.agent_name == "money_flow"
        assert len(signal.key_factors) > 0

    def test_score_range(self):
        stock = StockData(
            symbol="000001.SZ", name="测试", market="A",
            money_flow=_make_flow_data(10),
        )
        signal = analyze_money_flow(stock)
        assert -100 <= signal.signal_score <= 100
