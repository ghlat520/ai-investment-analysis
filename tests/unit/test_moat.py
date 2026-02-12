"""护城河Agent单元测试"""

from src.agents.analysts.moat import analyze_moat, _build_moat_context
from src.agents.state import StockData


def _make_stock(with_financial: bool = True) -> StockData:
    """创建测试用StockData"""
    financial = []
    if with_financial:
        financial = [
            {
                "report_date": "2024-03-31",
                "roe": 18.5,
                "gross_margin": 42.3,
                "net_margin": 12.1,
                "revenue_yoy": 15.2,
            },
            {
                "report_date": "2024-06-30",
                "roe": 19.2,
                "gross_margin": 43.1,
                "net_margin": 12.8,
                "revenue_yoy": 16.5,
            },
            {
                "report_date": "2024-09-30",
                "roe": 17.8,
                "gross_margin": 41.5,
                "net_margin": 11.9,
                "revenue_yoy": 14.0,
            },
        ]

    return StockData(
        symbol="000001.SZ",
        name="平安银行",
        market="A",
        financial_data=financial,
        info={"industry": "银行", "sector": "金融", "market_cap": 2500e8},
    )


def test_build_moat_context_with_data():
    stock = _make_stock(with_financial=True)
    context = _build_moat_context(stock)
    assert "银行" in context
    assert "毛利率趋势" in context
    assert "ROE趋势" in context


def test_build_moat_context_empty():
    stock = StockData(symbol="000001.SZ", name="平安银行", market="A")
    context = _build_moat_context(stock)
    assert "基础数据有限" in context


def test_analyze_moat_no_llm():
    """无LLM时应返回中性评分"""
    stock = _make_stock()
    signal = analyze_moat(stock)
    assert signal.agent_name == "moat"
    assert -100 <= signal.signal_score <= 100
    assert 0 <= signal.confidence <= 1
    assert signal.reasoning != ""


def test_analyze_moat_empty_data():
    stock = StockData(symbol="000001.SZ", name="平安银行", market="A")
    signal = analyze_moat(stock)
    assert signal.agent_name == "moat"
    assert -100 <= signal.signal_score <= 100
