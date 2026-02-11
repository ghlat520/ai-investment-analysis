"""决策融合单元测试"""

from src.agents.fusion.engine import fuse_signals, _detect_conflicts
from src.agents.state import AgentSignal, StockData


def test_fuse_signals_basic():
    signals = [
        AgentSignal(agent_name="technical", signal_score=50, confidence=0.8, reasoning="多头排列"),
        AgentSignal(agent_name="fundamental", signal_score=30, confidence=0.7, reasoning="ROE优秀"),
        AgentSignal(agent_name="valuation", signal_score=-20, confidence=0.6, reasoning="PE偏高"),
    ]
    decision = fuse_signals(signals)
    assert -100 <= decision.final_score <= 100
    assert decision.final_action in ["建仓", "逐步建仓", "观察", "观望", "减仓", "清仓", "强烈清仓"]
    assert decision.confidence > 0
    assert len(decision.signal_summary) == 3


def test_fuse_signals_empty():
    decision = fuse_signals([])
    assert decision.final_score == 0
    assert decision.final_action == "观望"


def test_detect_conflicts():
    signals = [
        AgentSignal(agent_name="technical", signal_score=60, confidence=0.8),
        AgentSignal(agent_name="fundamental", signal_score=-50, confidence=0.7),
    ]
    conflicts = _detect_conflicts(signals)
    assert len(conflicts) == 1
    assert "technical" in conflicts[0]
    assert "fundamental" in conflicts[0]


def test_fuse_signals_st_stock():
    signals = [
        AgentSignal(agent_name="technical", signal_score=80, confidence=0.9),
    ]
    stock = StockData(symbol="000001.SZ", name="ST测试", market="A", info={"is_st": True})
    decision = fuse_signals(signals, stock=stock)
    assert decision.final_score <= -50  # ST股票被强制降分
