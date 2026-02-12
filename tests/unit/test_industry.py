"""产业链Agent单元测试"""

from src.agents.analysts.industry import analyze_industry, _build_industry_context
from src.agents.state import StockData


def _make_stock(with_financial: bool = True) -> StockData:
    """创建测试用StockData"""
    financial = []
    if with_financial:
        financial = [
            {
                "report_date": "2024-03-31",
                "revenue_yoy": 25.0,
                "profit_yoy": 30.0,
                "gross_margin": 55.0,
            },
            {
                "report_date": "2024-06-30",
                "revenue_yoy": 28.0,
                "profit_yoy": 35.0,
                "gross_margin": 56.2,
            },
            {
                "report_date": "2024-09-30",
                "revenue_yoy": 22.0,
                "profit_yoy": 25.0,
                "gross_margin": 54.8,
            },
        ]

    return StockData(
        symbol="300750.SZ",
        name="宁德时代",
        market="A",
        financial_data=financial,
        info={"industry": "电池", "sector": "新能源", "main_business": "锂离子电池研发、生产和销售"},
    )


def test_build_industry_context_with_data():
    stock = _make_stock(with_financial=True)
    context = _build_industry_context(stock)
    assert "电池" in context
    assert "营收增速趋势" in context


def test_build_industry_context_empty():
    stock = StockData(symbol="300750.SZ", name="宁德时代", market="A")
    context = _build_industry_context(stock)
    assert "基础数据有限" in context


def test_analyze_industry_no_llm():
    stock = _make_stock()
    signal = analyze_industry(stock)
    assert signal.agent_name == "industry"
    assert -100 <= signal.signal_score <= 100
    assert 0 <= signal.confidence <= 1
    assert signal.reasoning != ""


def test_analyze_industry_empty_data():
    stock = StockData(symbol="300750.SZ", name="宁德时代", market="A")
    signal = analyze_industry(stock)
    assert signal.agent_name == "industry"
