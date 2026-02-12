"""商业模式Agent单元测试"""

from src.agents.analysts.business_model import analyze_business_model, _build_business_context
from src.agents.state import StockData


def _make_stock(with_financial: bool = True) -> StockData:
    """创建测试用StockData"""
    financial = []
    if with_financial:
        financial = [
            {
                "report_date": "2024-03-31",
                "revenue": 50e8,
                "net_profit": 8e8,
                "roe": 15.0,
                "gross_margin": 35.0,
                "net_margin": 16.0,
                "revenue_yoy": 12.0,
            },
            {
                "report_date": "2024-06-30",
                "revenue": 55e8,
                "net_profit": 9e8,
                "roe": 16.0,
                "gross_margin": 36.5,
                "net_margin": 16.4,
                "revenue_yoy": 14.0,
            },
        ]

    return StockData(
        symbol="600519.SH",
        name="贵州茅台",
        market="A",
        financial_data=financial,
        info={"industry": "白酒", "main_business": "贵州茅台酒系列产品的生产和销售"},
    )


def test_build_business_context_with_data():
    stock = _make_stock(with_financial=True)
    context = _build_business_context(stock)
    assert "白酒" in context
    assert "营收" in context


def test_build_business_context_empty():
    stock = StockData(symbol="600519.SH", name="贵州茅台", market="A")
    context = _build_business_context(stock)
    assert "基础数据有限" in context


def test_analyze_business_model_no_llm():
    stock = _make_stock()
    signal = analyze_business_model(stock)
    assert signal.agent_name == "business_model"
    assert -100 <= signal.signal_score <= 100
    assert 0 <= signal.confidence <= 1
    assert signal.reasoning != ""


def test_analyze_business_model_empty_data():
    stock = StockData(symbol="600519.SH", name="贵州茅台", market="A")
    signal = analyze_business_model(stock)
    assert signal.agent_name == "business_model"
