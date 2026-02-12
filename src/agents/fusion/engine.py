"""
决策融合引擎

多空辩论式融合：
1. 量化层：加权融合（权重自动归一化到活跃Agent）→ code_score
2. 规则层：硬性否决（ST/退市）
3. 信号分组：多方/空方/中性
4. LLM层：bull vs bear辩论 → final_score
5. 代码保留加权平均作为baseline，LLM辩论为主（0.7/0.3权重）

LLM不可用时优雅降级为纯代码融合。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from ..state import AgentSignal, FusionDecision, StockData

# 默认权重（neutral市场环境，9个Agent）
DEFAULT_WEIGHTS: dict[str, float] = {
    "technical": 0.10,
    "fundamental": 0.14,
    "valuation": 0.14,
    "money_flow": 0.08,
    "sentiment": 0.07,
    "moat": 0.13,
    "business_model": 0.12,
    "industry": 0.11,
    "supply_chain": 0.11,
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


def _normalize_weights(
    raw_weights: dict[str, float], active_agents: list[str],
) -> dict[str, float]:
    """将权重归一化到实际活跃的Agent"""
    active_w = {a: raw_weights.get(a, 0.1) for a in active_agents}
    total = sum(active_w.values())
    if total <= 0:
        n = len(active_agents)
        return {a: 1.0 / n for a in active_agents}
    return {a: w / total for a, w in active_w.items()}


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
    base = min(80, max(0, score))
    return int(base * confidence)


def _calc_risk_params(
    score: int, confidence: float, signals: list[AgentSignal],
) -> tuple[float, float]:
    """动态止损止盈"""
    strength = abs(score) / 100.0

    if score >= 0:
        stop_loss = -(5.0 + strength * 5.0)
        take_profit = 8.0 + strength * 12.0
    else:
        stop_loss = -(3.0 + strength * 5.0)
        take_profit = 5.0 + strength * 5.0

    if confidence < 0.5:
        stop_loss *= 0.7
        take_profit *= 0.7

    return round(stop_loss, 1), round(take_profit, 1)


def _detect_conflicts(signals: list[AgentSignal]) -> list[str]:
    """检测信号矛盾"""
    conflicts = []
    for i, s1 in enumerate(signals):
        for s2 in signals[i + 1:]:
            if (
                s1.signal_score * s2.signal_score < 0
                and abs(s1.signal_score) > 30
                and abs(s2.signal_score) > 30
            ):
                conflicts.append(
                    f"{s1.agent_name}({s1.signal_score:+d}) vs "
                    f"{s2.agent_name}({s2.signal_score:+d})"
                )
    return conflicts


def _group_signals(signals: list[AgentSignal]) -> tuple[list[AgentSignal], list[AgentSignal], list[AgentSignal]]:
    """将信号按方向分为多方/空方/中性"""
    bullish = [s for s in signals if s.signal_score >= 20]
    bearish = [s for s in signals if s.signal_score <= -20]
    neutral = [s for s in signals if -20 < s.signal_score < 20]
    return bullish, bearish, neutral


def _build_bull_bear_arguments(
    bullish: list[AgentSignal], bearish: list[AgentSignal],
) -> tuple[list[str], list[str]]:
    """从多空信号中提取核心论据"""
    bull_args = []
    for s in sorted(bullish, key=lambda x: -x.signal_score):
        reasoning_short = s.reasoning[:100] if s.reasoning else "无详细分析"
        bull_args.append(f"[{s.agent_name}({s.signal_score:+d})] {reasoning_short}")
        for f in s.key_factors[:2]:
            bull_args.append(f"  - {f}")

    bear_args = []
    for s in sorted(bearish, key=lambda x: x.signal_score):
        reasoning_short = s.reasoning[:100] if s.reasoning else "无详细分析"
        bear_args.append(f"[{s.agent_name}({s.signal_score:+d})] {reasoning_short}")
        for r in s.risks[:2]:
            bear_args.append(f"  - {r}")

    return bull_args, bear_args


def _build_signals_text(signals: list[AgentSignal]) -> str:
    """构建信号摘要文本供LLM阅读"""
    lines = []
    for s in signals:
        lines.append(f"### {s.agent_name} (score={s.signal_score:+d}, confidence={s.confidence:.0%})")
        lines.append(f"分析: {s.reasoning}")
        if s.key_factors:
            lines.append(f"关键因素: {'; '.join(s.key_factors[:5])}")
        if s.risks:
            lines.append(f"风险: {'; '.join(s.risks[:3])}")
        lines.append("")
    return "\n".join(lines)


def _build_code_reasoning(
    signals: list[AgentSignal],
    final_score: int,
    weights_used: dict[str, float],
    conflicts: list[str],
    bull_args: list[str],
    bear_args: list[str],
) -> str:
    """构建代码版reasoning"""
    parts = []
    for s in signals:
        w = weights_used.get(s.agent_name, 0)
        parts.append(
            f"- {s.agent_name}({s.signal_score:+d}, 置信{s.confidence:.0%}, 权重{w:.0%}): "
            f"{s.reasoning[:80]}"
        )
    reasoning = f"融合{len(signals)}个Agent信号，加权得分{final_score:+d}。\n" + "\n".join(parts)

    if bull_args:
        reasoning += f"\n\n多方论据({len(bull_args)}条)"
    if bear_args:
        reasoning += f"\n空方论据({len(bear_args)}条)"
    if conflicts:
        reasoning += f"\n矛盾信号: {', '.join(conflicts)}"

    return reasoning


def _llm_debate_fusion(
    signals: list[AgentSignal],
    stock: Optional[StockData],
    code_score: int,
    code_reasoning: str,
    conflicts: list[str],
) -> dict[str, Any]:
    """LLM多空辩论式融合

    LLM拥有完全裁量权，可输出-100~+100的score_adjustment。
    返回辩论结果，包含多空论据、分歧点、目标价。
    """
    result = {
        "enhanced": False,
        "score_adjustment": 0,
        "reasoning": code_reasoning,
        "conflict_resolution": "",
        "bull_arguments": [],
        "bear_arguments": [],
        "divergence_points": [],
        "target_prices": {},
        "llm_model": "",
        "llm_tokens": 0,
        "llm_cost": 0.0,
    }

    from ..llm_enhance import llm_enhance

    symbol = stock.symbol if stock else "N/A"
    name = stock.name if stock else "N/A"

    from datetime import date

    llm_result = llm_enhance(
        agent_name="fusion",
        template_name="fusion.md",
        template_vars={
            "symbol": symbol,
            "name": name,
            "analysis_date": date.today().isoformat(),
            "code_score": str(code_score),
            "signals_text": _build_signals_text(signals),
            "conflicts_text": ", ".join(conflicts) if conflicts else "无矛盾",
        },
        code_score=code_score,
        code_reasoning=code_reasoning,
        code_factors=[],
        code_risks=[],
        max_adjustment=100,  # 融合层有完全裁量权
    )

    result["llm_model"] = llm_result["llm_model"]
    result["llm_tokens"] = llm_result["llm_tokens"]
    result["llm_cost"] = llm_result["llm_cost"]

    if llm_result["enhanced"]:
        result["enhanced"] = True
        result["score_adjustment"] = llm_result["score_adjustment"]
        if llm_result["reasoning"]:
            result["reasoning"] = llm_result["reasoning"]

        # 提取冲突解决方案
        for f in llm_result["extra_factors"]:
            if "矛盾" in f or "冲突" in f or "conflict" in f.lower():
                result["conflict_resolution"] = f
                break

    return result


def fuse_signals(
    signals: list[AgentSignal],
    stock: Optional[StockData] = None,
    market_regime: str = "neutral",
) -> FusionDecision:
    """融合多Agent信号，产出最终决策

    1. 量化层：权重自动归一化到活跃Agent → 加权得分
    2. 规则层：ST/退市硬性否决
    3. 信号分组：多方/空方/中性
    4. LLM层：bull vs bear辩论式融合（可选）
    5. 最终得分：LLM可用时以LLM为主(0.7) + code为辅(0.3)
    """
    start = time.time()

    if not signals:
        return FusionDecision(
            final_score=0,
            final_action="观望",
            confidence=0.0,
            reasoning="无分析信号输入",
        )

    # Step 1: 权重归一化到活跃Agent
    raw_weights = _load_weights(market_regime)
    active_agents = [s.agent_name for s in signals]
    weights = _normalize_weights(raw_weights, active_agents)

    # Step 2: 量化加权
    weighted_sum = 0.0
    total_weight = 0.0
    signal_summary = {}

    for signal in signals:
        w = weights.get(signal.agent_name, 0.1)
        weighted_sum += signal.signal_score * signal.confidence * w
        total_weight += w * signal.confidence
        signal_summary[signal.agent_name] = signal.signal_score

    code_score = int(weighted_sum / total_weight) if total_weight > 0 else 0
    code_score = max(-100, min(100, code_score))

    # 综合置信度
    avg_confidence = sum(s.confidence for s in signals) / len(signals)

    # Step 3: 规则否决
    if stock and stock.info:
        if stock.info.get("is_st"):
            code_score = min(code_score, -50)
            logger.warning(f"[Fusion] ST股票 {stock.symbol}，强制降分至{code_score}")
        if stock.info.get("is_delisting"):
            code_score = -100
            logger.warning(f"[Fusion] 退市风险 {stock.symbol}，强制-100")

    # Step 4: 信号分组 + 矛盾检测
    bullish, bearish, neutral_signals = _group_signals(signals)
    bull_args, bear_args = _build_bull_bear_arguments(bullish, bearish)
    conflicts = _detect_conflicts(signals)

    if conflicts:
        penalty = min(0.2, 0.1 * len(conflicts))
        avg_confidence = max(0.1, avg_confidence - penalty)
        logger.info(f"[Fusion] 检测到{len(conflicts)}个矛盾信号，置信度-{penalty:.0%}")

    confidence = round(avg_confidence, 3)

    # 代码版reasoning
    code_reasoning = _build_code_reasoning(
        signals, code_score, weights, conflicts, bull_args, bear_args,
    )

    # Step 5: LLM辩论式融合（可选）
    llm_result = _llm_debate_fusion(signals, stock, code_score, code_reasoning, conflicts)

    # 最终得分计算
    if llm_result["enhanced"]:
        # LLM可用：以LLM辩论结论为主(0.7) + code为辅(0.3)
        llm_adjusted_score = code_score + llm_result["score_adjustment"]
        llm_adjusted_score = max(-100, min(100, llm_adjusted_score))
        final_score = int(0.3 * code_score + 0.7 * llm_adjusted_score)
        final_score = max(-100, min(100, final_score))
        confidence = min(1.0, confidence + 0.05)
    else:
        # LLM不可用：纯code_score
        final_score = code_score

    reasoning = llm_result["reasoning"]

    # 操作建议
    final_action = _classify_action(final_score)
    position_pct = _suggest_position(final_score, confidence)
    stop_loss, take_profit = _calc_risk_params(final_score, confidence, signals)

    # 矛盾解决方案
    conflict_resolution = llm_result.get("conflict_resolution", "")
    if conflicts and not conflict_resolution:
        conflict_resolution = "按加权权重融合，矛盾信号已通过置信度折扣反映在最终评分中"

    # 目标价（从LLM或valuation agent提取）
    target_prices = llm_result.get("target_prices", {})
    if not target_prices:
        for s in signals:
            if s.agent_name == "valuation" and isinstance(s.metadata, dict):
                raw = s.metadata.get("raw_response", {})
                if isinstance(raw, dict) and "target_prices" in raw:
                    target_prices = raw["target_prices"]
                    break

    # 多空论据（合并LLM和代码提取的）
    final_bull_args = llm_result.get("bull_arguments", []) or bull_args
    final_bear_args = llm_result.get("bear_arguments", []) or bear_args
    divergence_points = llm_result.get("divergence_points", [])

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[Fusion] 完成: score={final_score:+d} action={final_action} "
        f"confidence={confidence:.0%} position={position_pct}% "
        f"bull={len(bullish)} bear={len(bearish)} neutral={len(neutral_signals)} "
        f"llm={'Y' if llm_result['enhanced'] else 'N'} {elapsed_ms}ms"
    )

    return FusionDecision(
        final_score=final_score,
        final_action=final_action,
        confidence=confidence,
        position_pct=position_pct,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
        reasoning=reasoning,
        signal_summary=signal_summary,
        conflicts=tuple(conflicts),
        conflict_resolution=conflict_resolution,
        market_regime=market_regime,
        weights_used=weights,
        bull_arguments=tuple(final_bull_args),
        bear_arguments=tuple(final_bear_args),
        divergence_points=tuple(divergence_points),
        target_prices=target_prices if isinstance(target_prices, dict) else {},
    )
