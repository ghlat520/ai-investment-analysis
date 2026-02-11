"""情绪面Agent单元测试"""

from datetime import date, timedelta

from src.agents.analysts.sentiment import (
    _detect_major_events,
    _score_keyword_sentiment,
    _score_news_recency,
    _score_news_volume,
    analyze_sentiment,
)
from src.agents.state import StockData


# ─── 辅助 ──────────────────────────────────────────────

def _make_news(titles: list[str], days_ago: int = 0) -> list[dict]:
    """生成测试新闻数据"""
    dt = (date.today() - timedelta(days=days_ago)).isoformat()
    return [
        {"title": t, "content": "", "datetime": dt, "source": "测试来源"}
        for t in titles
    ]


# ─── 关键词情感测试 ──────────────────────────────────────


class TestKeywordSentiment:
    def test_positive(self):
        news = _make_news(["公司业绩增长超预期", "净利润增翻倍"])
        score, desc, kws = _score_keyword_sentiment(news)
        assert score > 0
        assert len(kws) > 0

    def test_negative(self):
        news = _make_news(["公司业绩下滑亏损", "大股东减持质押"])
        score, desc, kws = _score_keyword_sentiment(news)
        assert score < 0

    def test_empty(self):
        score, desc, kws = _score_keyword_sentiment([])
        assert score == 0

    def test_neutral(self):
        news = _make_news(["公司召开年度会议"])
        score, _, kws = _score_keyword_sentiment(news)
        assert score == 0
        assert kws == []


class TestNewsVolume:
    def test_dense(self):
        news = _make_news([f"新闻{i}" for i in range(25)])
        score, desc = _score_news_volume(news)
        assert score > 0
        assert "密集" in desc

    def test_sparse(self):
        news = _make_news(["唯一新闻"])
        score, desc = _score_news_volume(news)
        assert score < 0

    def test_empty(self):
        score, _ = _score_news_volume([])
        assert score == 0


class TestNewsRecency:
    def test_recent(self):
        news = _make_news([f"今日新闻{i}" for i in range(6)], days_ago=0)
        score, desc = _score_news_recency(news)
        assert score > 0

    def test_stale(self):
        news = _make_news(["旧新闻"], days_ago=10)
        score, desc = _score_news_recency(news)
        assert score < 0


class TestMajorEvents:
    def test_positive_event(self):
        news = _make_news(["公司业绩预增超100%"])
        score, desc, events = _detect_major_events(news)
        assert score > 0
        assert "业绩大增" in events

    def test_negative_event(self):
        news = _make_news(["公司被立案调查"])
        score, desc, events = _detect_major_events(news)
        assert score < 0
        assert "监管风险" in events

    def test_delisting_risk(self):
        news = _make_news(["公司面临退市风险"])
        score, _, events = _detect_major_events(news)
        assert score <= -20
        assert "退市风险" in events

    def test_no_events(self):
        news = _make_news(["公司召开股东大会"])
        score, _, events = _detect_major_events(news)
        assert score == 0
        assert events == []


class TestAnalyzeSentiment:
    def test_no_news(self):
        stock = StockData(symbol="000001.SZ", name="测试", market="A")
        signal = analyze_sentiment(stock)
        assert signal.signal_score == 0
        assert signal.confidence == 0.0

    def test_positive_news(self):
        stock = StockData(
            symbol="000001.SZ", name="测试", market="A",
            news=_make_news([
                "公司业绩增长超预期",
                "获得重大合同签约10亿",
                "机构买入评级",
                "公司回购股票",
            ]),
        )
        signal = analyze_sentiment(stock)
        assert signal.signal_score > 0
        assert signal.agent_name == "sentiment"
        assert signal.confidence > 0

    def test_negative_news(self):
        stock = StockData(
            symbol="000001.SZ", name="测试", market="A",
            news=_make_news([
                "公司业绩预亏",
                "大股东大量减持套现",
                "被监管处罚立案",
            ]),
        )
        signal = analyze_sentiment(stock)
        assert signal.signal_score < 0

    def test_score_range(self):
        stock = StockData(
            symbol="000001.SZ", name="测试", market="A",
            news=_make_news([f"新闻{i}" for i in range(10)]),
        )
        signal = analyze_sentiment(stock)
        assert -100 <= signal.signal_score <= 100
        assert 0 <= signal.confidence <= 1.0
