"""决策融合单元测试"""

from src.agents.fusion.engine import (
    _calc_risk_params,
    _classify_action,
    _detect_conflicts,
    _group_signals,
    _normalize_weights,
    _suggest_position,
    fuse_signals,
)
from src.agents.state import AgentSignal, StockData


def _make_signal(name: str, score: int, confidence: float = 0.8) -> AgentSignal:
    return AgentSignal(
        agent_name=name,
        signal_score=score,
        confidence=confidence,
        reasoning=f"{name}分析结果",
        key_factors=(f"{name}因素1", f"{name}因素2"),
        risks=(f"{name}风险1",),
    )


def test_fuse_signals_basic():
    signals = [
        _make_signal("technical", 50),
        _make_signal("fundamental", 30, 0.7),
        _make_signal("valuation", -20, 0.6),
    ]
    decision = fuse_signals(signals)
    assert -100 <= decision.final_score <= 100
    assert decision.final_action in ["建仓", "逐步建仓", "观察", "观望", "减仓", "清仓", "强烈清仓"]
    assert decision.confidence > 0
    assert len(decision.signal_summary) == 3


def test_fuse_signals_8_agents():
    """8个Agent完整融合"""
    signals = [
        _make_signal("technical", 30),
        _make_signal("fundamental", 40),
        _make_signal("valuation", 20),
        _make_signal("money_flow", 10),
        _make_signal("sentiment", -5),
        _make_signal("moat", 50),
        _make_signal("business_model", 35),
        _make_signal("industry", 25),
    ]
    decision = fuse_signals(signals)
    assert -100 <= decision.final_score <= 100
    assert decision.confidence > 0
    assert len(decision.signal_summary) == 8


def test_fuse_signals_empty():
    decision = fuse_signals([])
    assert decision.final_score == 0
    assert decision.final_action == "观望"


def test_detect_conflicts():
    signals = [
        _make_signal("technical", 60),
        _make_signal("fundamental", -50, 0.7),
    ]
    conflicts = _detect_conflicts(signals)
    assert len(conflicts) == 1
    assert "technical" in conflicts[0]
    assert "fundamental" in conflicts[0]


def test_group_signals():
    signals = [
        _make_signal("technical", 50),
        _make_signal("fundamental", -30),
        _make_signal("valuation", 5),
        _make_signal("moat", 25),
    ]
    bullish, bearish, neutral = _group_signals(signals)
    assert len(bullish) == 2  # technical(50), moat(25)
    assert len(bearish) == 1  # fundamental(-30)
    assert len(neutral) == 1  # valuation(5)


def test_fuse_signals_st_stock():
    signals = [_make_signal("technical", 80, 0.9)]
    stock = StockData(symbol="000001.SZ", name="ST测试", market="A", info={"is_st": True})
    decision = fuse_signals(signals, stock=stock)
    assert decision.final_score <= -50


def test_normalize_weights_active_agents_only():
    raw = {"technical": 0.20, "fundamental": 0.20, "valuation": 0.20, "money_flow": 0.15, "sentiment": 0.15, "industry": 0.10}
    active = ["technical", "fundamental", "valuation"]
    normalized = _normalize_weights(raw, active)
    assert set(normalized.keys()) == set(active)
    assert abs(sum(normalized.values()) - 1.0) < 0.001
    assert normalized["technical"] == normalized["fundamental"]


def test_normalize_weights_single_agent():
    normalized = _normalize_weights({"technical": 0.35}, ["technical"])
    assert normalized["technical"] == 1.0


def test_normalize_weights_unknown_agent():
    normalized = _normalize_weights({"technical": 0.5}, ["technical", "new_agent"])
    assert "new_agent" in normalized
    assert sum(normalized.values()) - 1.0 < 0.001


def test_classify_action():
    assert _classify_action(70) == "建仓"
    assert _classify_action(40) == "逐步建仓"
    assert _classify_action(15) == "观察"
    assert _classify_action(0) == "观望"
    assert _classify_action(-20) == "减仓"
    assert _classify_action(-50) == "清仓"
    assert _classify_action(-80) == "强烈清仓"


def test_conflicts_reduce_confidence():
    no_conflict = [
        _make_signal("technical", 50, 0.9),
        _make_signal("fundamental", 40, 0.9),
    ]
    with_conflict = [
        _make_signal("technical", 60, 0.9),
        _make_signal("fundamental", -50, 0.9),
    ]
    d1 = fuse_signals(no_conflict)
    d2 = fuse_signals(with_conflict)
    assert d2.confidence < d1.confidence


def test_dynamic_risk_params():
    sl_strong, tp_strong = _calc_risk_params(80, 0.9, [])
    sl_weak, tp_weak = _calc_risk_params(20, 0.9, [])
    assert sl_strong < sl_weak
    assert tp_strong > tp_weak

    sl_short, tp_short = _calc_risk_params(-50, 0.8, [])
    assert sl_short > sl_strong


def test_position_cap_at_80():
    pos = _suggest_position(100, 1.0)
    assert pos <= 80


def test_position_zero_for_negative():
    pos = _suggest_position(-30, 0.9)
    assert pos == 0


def test_fusion_weights_in_output():
    signals = [
        _make_signal("technical", 50),
        _make_signal("fundamental", 30, 0.7),
    ]
    decision = fuse_signals(signals)
    assert "technical" in decision.weights_used
    assert "fundamental" in decision.weights_used
    assert abs(sum(decision.weights_used.values()) - 1.0) < 0.001


def test_fusion_conflict_resolution_text():
    signals = [
        _make_signal("technical", 60, 0.9),
        _make_signal("fundamental", -50, 0.9),
    ]
    decision = fuse_signals(signals)
    assert len(decision.conflicts) > 0
    assert decision.conflict_resolution != ""


def test_fusion_decision_new_fields():
    """验证FusionDecision新增字段（多空论据、分歧点、目标价）"""
    signals = [
        _make_signal("technical", 50),
        _make_signal("fundamental", -30),
        _make_signal("moat", 40),
    ]
    decision = fuse_signals(signals)
    assert isinstance(decision.bull_arguments, tuple)
    assert isinstance(decision.bear_arguments, tuple)
    assert isinstance(decision.divergence_points, tuple)
    assert isinstance(decision.target_prices, dict)
    # 有多空信号，应该提取到论据
    assert len(decision.bull_arguments) > 0
    assert len(decision.bear_arguments) > 0
