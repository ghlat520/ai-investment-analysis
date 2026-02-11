"""决策融合单元测试"""

from src.agents.fusion.engine import (
    _calc_risk_params,
    _detect_conflicts,
    _normalize_weights,
    _suggest_position,
    fuse_signals,
)
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


def test_normalize_weights_active_agents_only():
    """权重应自动归一化到活跃Agent"""
    raw = {"technical": 0.20, "fundamental": 0.20, "valuation": 0.20, "money_flow": 0.15, "sentiment": 0.15, "industry": 0.10}
    active = ["technical", "fundamental", "valuation"]
    normalized = _normalize_weights(raw, active)

    # 只包含活跃Agent
    assert set(normalized.keys()) == set(active)
    # 总和约等于1.0
    assert abs(sum(normalized.values()) - 1.0) < 0.001
    # 权重比例保持
    assert normalized["technical"] == normalized["fundamental"]


def test_normalize_weights_single_agent():
    """单Agent时权重为1.0"""
    normalized = _normalize_weights({"technical": 0.35}, ["technical"])
    assert normalized["technical"] == 1.0


def test_normalize_weights_unknown_agent():
    """未在配置中的Agent获得默认权重0.1"""
    normalized = _normalize_weights({"technical": 0.5}, ["technical", "new_agent"])
    assert "new_agent" in normalized
    assert sum(normalized.values()) - 1.0 < 0.001


def test_conflicts_reduce_confidence():
    """矛盾信号应降低置信度"""
    no_conflict = [
        AgentSignal(agent_name="technical", signal_score=50, confidence=0.9),
        AgentSignal(agent_name="fundamental", signal_score=40, confidence=0.9),
    ]
    with_conflict = [
        AgentSignal(agent_name="technical", signal_score=60, confidence=0.9),
        AgentSignal(agent_name="fundamental", signal_score=-50, confidence=0.9),
    ]
    d1 = fuse_signals(no_conflict)
    d2 = fuse_signals(with_conflict)
    assert d2.confidence < d1.confidence


def test_dynamic_risk_params():
    """止损止盈应随信号强度变化"""
    # 强看多 → 宽止损 + 高目标
    sl_strong, tp_strong = _calc_risk_params(80, 0.9, [])
    # 弱看多 → 紧止损 + 低目标
    sl_weak, tp_weak = _calc_risk_params(20, 0.9, [])

    assert sl_strong < sl_weak  # 更宽的止损（更负的值）
    assert tp_strong > tp_weak  # 更高的目标

    # 看空 → 更紧的止损
    sl_short, tp_short = _calc_risk_params(-50, 0.8, [])
    assert sl_short > sl_strong  # 看空止损更紧（更接近0）


def test_position_cap_at_80():
    """仓位不超过80%"""
    pos = _suggest_position(100, 1.0)
    assert pos <= 80


def test_position_zero_for_negative():
    """负分数仓位为0"""
    pos = _suggest_position(-30, 0.9)
    assert pos == 0


def test_fusion_weights_in_output():
    """输出应包含归一化后的权重"""
    signals = [
        AgentSignal(agent_name="technical", signal_score=50, confidence=0.8),
        AgentSignal(agent_name="fundamental", signal_score=30, confidence=0.7),
    ]
    decision = fuse_signals(signals)
    # 权重应归一化到这2个Agent
    assert "technical" in decision.weights_used
    assert "fundamental" in decision.weights_used
    assert abs(sum(decision.weights_used.values()) - 1.0) < 0.001


def test_fusion_conflict_resolution_text():
    """矛盾信号时应有解决方案文本"""
    signals = [
        AgentSignal(agent_name="technical", signal_score=60, confidence=0.9),
        AgentSignal(agent_name="fundamental", signal_score=-50, confidence=0.9),
    ]
    decision = fuse_signals(signals)
    assert len(decision.conflicts) > 0
    assert decision.conflict_resolution != ""
