"""技术面Agent单元测试"""

from src.agents.analysts.technical import analyze_technical
from src.agents.state import StockData


def _make_quotes(n: int = 100, base_price: float = 20.0) -> list[dict]:
    """生成模拟行情数据"""
    import random
    random.seed(42)
    quotes = []
    price = base_price
    for i in range(n):
        change = random.uniform(-0.03, 0.03)
        price *= (1 + change)
        quotes.append({
            "date": f"2025-{(i // 30) + 1:02d}-{(i % 28) + 1:02d}",
            "open": round(price * 0.99, 2),
            "high": round(price * 1.02, 2),
            "low": round(price * 0.98, 2),
            "close": round(price, 2),
            "volume": random.randint(1000000, 10000000),
        })
    return quotes


def test_analyze_technical_basic():
    stock = StockData(
        symbol="000001.SZ",
        name="平安银行",
        market="A",
        daily_quotes=_make_quotes(100),
    )
    signal = analyze_technical(stock)
    assert signal.agent_name == "technical"
    assert -100 <= signal.signal_score <= 100
    assert 0 <= signal.confidence <= 1
    assert signal.reasoning != ""


def test_analyze_technical_empty_data():
    stock = StockData(symbol="000001.SZ", name="平安银行", market="A")
    signal = analyze_technical(stock)
    assert signal.signal_score == 0
    assert signal.confidence == 0.0


def test_analyze_technical_short_data():
    stock = StockData(
        symbol="000001.SZ",
        name="平安银行",
        market="A",
        daily_quotes=_make_quotes(10),
    )
    signal = analyze_technical(stock)
    assert signal.agent_name == "technical"
    # 数据不足时置信度应较低
    assert signal.confidence < 0.5
