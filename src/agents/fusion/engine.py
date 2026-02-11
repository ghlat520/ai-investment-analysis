"""
决策融合引擎

三层仲裁：
1. 量化层：动态权重加权
2. 规则层：硬性否决
3. LLM层：矛盾仲裁（Phase 2）

Phase 1 仅实现量化层 + 规则层（固定权重）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from ..state import AgentSignal, FusionDecision, StockData

# 默认权重（neutral市场环境）
DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.35,
    "fundamental": 0.35,
    "valuation": 0.30,
}


def _load_weights(regime: str = "neutral") -> dict[str, float]:
    """从配置文件加载权重"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "weights.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        regimes = config.get("market_regimes", {})
        if regime in regimes:
            return regimes[regime]
    return DEFAULT_WEIGHTS


def _classify_action(score: int) -> str:
    """根据分数确定操作建议"""
    if score >= 60:
        return "建仓"
    if score >= 30:
        return "逐步建仓"
    if score >= 10:
        return "观察"
    if score >= -10:
        return "观望"
    if score >= -30:
        return "减仓"
    if score >= -60:
        return "清仓"
    return "强烈清仓"


def _suggest_position(score: int, confidence: float) -> int:
    """根据分数和置信度建议仓位比例"""
    if score < 0:
        return 0
    base = min(100, max(0, score))
    return int(base * confidence)


def _detect_conflicts(signals: list[AgentSignal]) -> list[str]:
    """检测信号矛盾"""
    conflicts = []
    for i, s1 in enumerate(signals):
        for s2 in signals[i + 1 :]:
            # 两个Agent方向相反且信号强度都较高
            if s1.signal_score * s2.signal_score < 0 and abs(s1.signal_score) > 30 and abs(s2.signal_score) > 30:
                conflicts.append(
                    f"{s1.agent_name}({s1.signal_score:+d}) vs "
                    f"{s2.agent_name}({s2.signal_score:+d})"
                )
    return conflicts


def fuse_signals(
    signals: list[AgentSignal],
    stock: Optional[StockData] = None,
    market_regime: str = "neutral",
) -> FusionDecision:
    """融合多Agent信号，产出最终决策

    Phase 1: 固定权重加权 + 规则否决
    Phase 2: 动态权重 + LLM仲裁
    """
    if not signals:
        return FusionDecision(
            final_score=0,
            final_action="观望",
            confidence=0.0,
            reasoning="无分析信号输入",
        )

    weights = _load_weights(market_regime)

    # Step 1: 量化加权
    weighted_sum = 0.0
    total_weight = 0.0
    signal_summary = {}

    for signal in signals:
        w = weights.get(signal.agent_name, 0.1)
        weighted_sum += signal.signal_score * signal.confidence * w
        total_weight += w * signal.confidence
        signal_summary[signal.agent_name] = signal.signal_score

    final_score = int(weighted_sum / total_weight) if total_weight > 0 else 0
    final_score = max(-100, min(100, final_score))

    # 综合置信度
    avg_confidence = sum(s.confidence for s in signals) / len(signals) if signals else 0
    confidence = round(avg_confidence, 3)

    # Step 2: 规则否决
    if stock and stock.info:
        if stock.info.get("is_st"):
            final_score = min(final_score, -50)
            logger.warning(f"[Fusion] ST股票 {stock.symbol}，强制降分至{final_score}")
        if stock.info.get("is_delisting"):
            final_score = -100
            logger.warning(f"[Fusion] 退市风险 {stock.symbol}，强制-100")

    # 矛盾检测
    conflicts = _detect_conflicts(signals)

    # 操作建议
    final_action = _classify_action(final_score)
    position_pct = _suggest_position(final_score, confidence)

    # 推理汇总
    reasoning_parts = [
        f"{s.agent_name}: {s.signal_score:+d}（{s.reasoning[:50]}...）"
        for s in signals
    ]
    reasoning = f"融合{len(signals)}个Agent信号，加权得分{final_score}。\n" + "\n".join(reasoning_parts)

    if conflicts:
        reasoning += f"\n矛盾信号: {', '.join(conflicts)}"

    return FusionDecision(
        final_score=final_score,
        final_action=final_action,
        confidence=confidence,
        position_pct=position_pct,
        stop_loss_pct=-8.0,
        take_profit_pct=15.0,
        reasoning=reasoning,
        signal_summary=signal_summary,
        conflicts=tuple(conflicts),
        conflict_resolution="Phase 1: 按固定权重加权，矛盾信号保留供参考" if conflicts else "",
        market_regime=market_regime,
        weights_used=weights,
    )
