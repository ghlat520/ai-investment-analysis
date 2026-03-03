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
        # 默认使用value_investing权重
        default_regime = config.get("default", "neutral")
        regimes = config.get("market_regimes", {})
        if regime in regimes:
            return regimes[regime]
        if default_regime in regimes:
            return regimes[default_regime]
    return DEFAULT_WEIGHTS


def _load_fusion_config() -> dict:
    """加载融合配置（融合比例、择时因子列表等）"""
    config_path = Path(__file__).parent.parent.parent.parent / "config" / "weights.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _get_fusion_ratio(agent_name: str, fusion_config: dict) -> tuple[float, float]:
    """获取某Agent的code/LLM融合比例

    Returns:
        (code_weight, llm_weight)
    """
    ratios = fusion_config.get("fusion_ratios", {})

    # 检查code_dominant列表
    code_dom = ratios.get("code_dominant", {})
    if agent_name in code_dom.get("agents", []):
        return code_dom.get("code_weight", 0.7), code_dom.get("llm_weight", 0.3)

    # 检查llm_dominant列表
    llm_dom = ratios.get("llm_dominant", {})
    if agent_name in llm_dom.get("agents", []):
        return llm_dom.get("code_weight", 0.3), llm_dom.get("llm_weight", 0.7)

    # 默认平衡
    return 0.5, 0.5


def _calc_timing_signal(signals: list[AgentSignal], timing_agents: list[str]) -> str:
    """从择时因子计算择时信号

    择时因子（technical, money_flow, sentiment）不影响"是否值得买"，
    仅影响"何时买"。
    """
    timing_scores = []
    for s in signals:
        if s.agent_name in timing_agents:
            timing_scores.append(s.signal_score)

    if not timing_scores:
        return "中性"

    avg = sum(timing_scores) / len(timing_scores)
    if avg >= 30:
        return "积极买入"
    elif avg >= 10:
        return "偏多"
    elif avg >= -10:
        return "中性"
    elif avg >= -30:
        return "偏空(等待)"
    else:
        return "极度悲观(逆向买入机会)"  # 价值投资逆向逻辑


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

    # 数据质量警告
    data_warnings = stock.info.get("data_warnings", []) if stock and stock.info else []
    warnings_text = ""
    if data_warnings:
        warnings_text = "\n\n### ⚠️ 数据质量风险提示\n"
        for w in data_warnings:
            warnings_text += f"- {w}\n"
        warnings_text += "请在最终建议中明确提及以上数据局限性。\n"

    llm_result = llm_enhance(
        agent_name="fusion",
        template_name="fusion.md",
        template_vars={
            "symbol": symbol,
            "name": name,
            "analysis_date": date.today().isoformat(),
            "code_score": str(code_score),
            "signals_text": _build_signals_text(signals) + warnings_text,
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

        # V2: 提取丰富字段（风险交叉验证、操作策略、多空论据等）
        raw = llm_result.get("raw_response") if isinstance(llm_result.get("raw_response"), dict) else {}
        if raw:
            # 优先从LLM raw_response提取个性化冲突解决方案
            if raw.get("conflict_resolution"):
                result["conflict_resolution"] = raw["conflict_resolution"]
            if raw.get("bull_arguments"):
                result["bull_arguments"] = raw["bull_arguments"]
            if raw.get("bear_arguments"):
                result["bear_arguments"] = raw["bear_arguments"]
            if raw.get("divergence_points"):
                result["divergence_points"] = raw["divergence_points"]
            if raw.get("target_prices"):
                result["target_prices"] = raw["target_prices"]
            if raw.get("risk_cross_validation"):
                result["risk_cross_validation"] = raw["risk_cross_validation"]
            if raw.get("operation_strategy"):
                result["operation_strategy"] = raw["operation_strategy"]
            if raw.get("sub_scores"):
                result["sub_scores"] = raw["sub_scores"]

    return result


def _calc_score_range(
    signals: list[AgentSignal],
    final_score: int,
    confidence: float,
    conflicts: list[str],
) -> tuple[int, int]:
    """计算95%置信区间

    基于以下因素：
    1. Agent评分方差（越大区间越宽）
    2. 综合置信度（越低区间越宽）
    3. 矛盾信号数量（越多区间越宽）
    """
    import statistics

    if len(signals) < 2:
        # 单Agent时，基于置信度给出默认区间
        half_width = int(30 * (1 - confidence) + 10)
        return final_score - half_width, final_score + half_width

    # 计算Agent评分的标准差
    scores = [s.signal_score for s in signals]
    try:
        std_dev = statistics.stdev(scores)
    except statistics.StatisticsError:
        std_dev = 20  # 默认标准差

    # 基础区间宽度 = 1.96 * 标准差 / sqrt(n)（95%置信区间公式）
    n = len(signals)
    base_half_width = int(1.96 * std_dev / (n ** 0.5))

    # 根据置信度调整
    confidence_adjustment = (1 - confidence) * 20

    # 根据矛盾数量调整
    conflict_adjustment = len(conflicts) * 5

    # 最终半宽
    half_width = int(base_half_width + confidence_adjustment + conflict_adjustment)
    half_width = max(10, min(40, half_width))  # 限制在10-40分

    low = max(-100, final_score - half_width)
    high = min(100, final_score + half_width)

    return low, high


def _extract_key_uncertainties(
    signals: list[AgentSignal],
    conflicts: list[str],
    divergence_points: list[str],
) -> list[str]:
    """从Agent信号中提取关键不确定性因素"""
    uncertainties = []

    # 从矛盾信号中提取
    for conflict in conflicts[:3]:
        uncertainties.append(f"信号分歧: {conflict}")

    # 从分歧点中提取
    uncertainties.extend(divergence_points[:3])

    # 从低置信度Agent的风险中提取
    for s in signals:
        if s.confidence < 0.6 and s.risks:
            for risk in s.risks[:1]:
                if risk not in uncertainties:
                    uncertainties.append(f"[{s.agent_name}风险] {risk}")

    # 去重并限制数量
    seen = set()
    unique = []
    for u in uncertainties:
        if u not in seen:
            seen.add(u)
            unique.append(u)
            if len(unique) >= 5:
                break

    return unique


def _generate_verification_points(
    signals: list[AgentSignal],
    stock: Optional[StockData],
    divergence_points: list[str],
) -> list[dict[str, Any]]:
    """生成验证时点及监控指标"""
    from datetime import datetime

    points = []

    # 基于分歧点生成验证指标
    for div in divergence_points[:3]:
        points.append({
            "type": "分歧验证",
            "description": div,
            "timing": "下季度财报",
            "indicator": "关注相关业务线数据变化",
        })

    # 基于Agent风险生成验证点
    for s in signals:
        if s.risks and s.confidence < 0.7:
            for risk in s.risks[:1]:
                points.append({
                    "type": "风险监控",
                    "agent": s.agent_name,
                    "description": risk,
                    "timing": "持续监控",
                    "indicator": f"关注{s.agent_name}相关指标变化",
                })

    # 去重
    seen = set()
    unique = []
    for p in points:
        key = p.get("description", "")
        if key not in seen:
            seen.add(key)
            unique.append(p)
            if len(unique) >= 4:
                break

    # 如果没有提取到，添加默认验证点
    if not unique:
        unique.append({
            "type": "综合验证",
            "description": "整体分析结论验证",
            "timing": "30天后",
            "indicator": "股价走势 vs 预期方向",
        })

    return unique


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
    6. P2: 计算置信区间和关键不确定性
    """
    start = time.time()

    if not signals:
        return FusionDecision(
            final_score=0,
            final_action="观望",
            confidence=0.0,
            reasoning="无分析信号输入",
            score_range_low=-20,
            score_range_high=20,
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
        # P2: 管理层风险否决
        if stock.info.get("management_red_flag"):
            code_score = min(code_score, -30)
            logger.warning(f"[Fusion] 管理层风险 {stock.symbol}，强制降分至{code_score}")

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

    # R3: 使用分场景融合比例
    fusion_config = _load_fusion_config()
    fusion_ratio = fusion_config.get("fusion_ratios", {}).get("fusion", {})
    fusion_code_w = fusion_ratio.get("code_weight", 0.5)
    fusion_llm_w = fusion_ratio.get("llm_weight", 0.5)

    # 最终得分计算
    if llm_result["enhanced"]:
        llm_adjusted_score = code_score + llm_result["score_adjustment"]
        llm_adjusted_score = max(-100, min(100, llm_adjusted_score))
        # R3: 融合层使用可配置比例（默认0.5/0.5）
        final_score = int(fusion_code_w * code_score + fusion_llm_w * llm_adjusted_score)
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
        # 构建有针对性的冲突说明（而非千篇一律）
        bull_names = [f"{s.agent_name}({s.signal_score:+d})" for s in bullish[:3]]
        bear_names = [f"{s.agent_name}({s.signal_score:+d})" for s in bearish[:3]]
        conflict_resolution = (
            f"多方（{', '.join(bull_names)}）与空方（{', '.join(bear_names)}）"
            f"存在{len(conflicts)}处矛盾。"
            f"按权重加权后偏向{'多方' if final_score > 0 else '空方' if final_score < 0 else '中性'}，"
            f"置信度因矛盾折扣至{confidence:.0%}。建议关注矛盾焦点后续数据验证。"
        )

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
    risk_cross_validation = llm_result.get("risk_cross_validation", [])
    operation_strategy = llm_result.get("operation_strategy", {})
    sub_scores = llm_result.get("sub_scores", {})

    # P2: 计算置信区间
    score_range_low, score_range_high = _calc_score_range(
        signals, final_score, confidence, conflicts
    )

    # P2: 提取关键不确定性
    key_uncertainties = _extract_key_uncertainties(
        signals, conflicts, list(divergence_points)
    )

    # P2: 生成验证时点
    verification_points = _generate_verification_points(
        signals, stock, list(divergence_points)
    )

    # 从LLM结果中提取不确定性信息（如果有）
    if llm_result.get("enhanced"):
        raw = llm_result.get("raw_response") if isinstance(llm_result.get("raw_response"), dict) else {}
        if raw:
            if raw.get("key_uncertainties"):
                key_uncertainties = list(raw["key_uncertainties"])[:5]
            if raw.get("verification_points"):
                verification_points = list(raw["verification_points"])[:4]

    # === R3: 价值投资三层输出 ===
    timing_agents = fusion_config.get("timing_agents", ["technical", "money_flow", "sentiment"])

    # Layer 1: 内在价值判断
    intrinsic_value_range = {}
    margin_of_safety_val = 0.0
    value_grade = "N/A"

    valuation_signal = next((s for s in signals if s.agent_name == "valuation"), None)
    if valuation_signal and isinstance(valuation_signal.metadata, dict):
        vmeta = valuation_signal.metadata

        # --- 内在价值区间智能构建（三级优先级） ---

        # 优先级1: YAML目标价折现（最准确，分业务线精算）
        _tp = vmeta.get("target_prices", {})
        _tp_conservative = _tp.get("conservative", 0) if isinstance(_tp, dict) else 0
        _tp_base = _tp.get("base", 0) if isinstance(_tp, dict) else 0
        _tp_optimistic = _tp.get("optimistic", 0) if isinstance(_tp, dict) else 0

        if _tp_conservative and _tp_conservative > 0:
            # 目标价按年折现到当前（折现率10%，对应1/2/3年）
            intrinsic_value_range = {
                "low": round(_tp_conservative / 1.10, 2),
                "base": round(_tp_base / (1.10 ** 2), 2) if _tp_base > 0 else round(_tp_conservative / 1.10, 2),
                "high": round(_tp_optimistic / (1.10 ** 3), 2) if _tp_optimistic > 0 else round(_tp_base / (1.10 ** 2), 2) if _tp_base > 0 else round(_tp_conservative / 1.10, 2),
            }

        # 优先级2: 过滤不适用方法后构建区间
        if not intrinsic_value_range:
            iv = vmeta.get("intrinsic_value", {})
            if iv:
                # 获取profit_yoy判断是否成长股
                _profit_yoy = vmeta.get("profit_yoy")
                _is_growth_stock = _profit_yoy is not None and _profit_yoy > 20

                iv_filtered = {}
                for method, val in iv.items():
                    if not isinstance(val, (int, float)) or val <= 0:
                        continue
                    # 成长股排除Graham Number（对高增长股严重低估）
                    if _is_growth_stock and method == "graham_number":
                        continue
                    iv_filtered[method] = val

                if iv_filtered:
                    iv_values = list(iv_filtered.values())
                    intrinsic_value_range = {
                        "low": min(iv_values),
                        "base": sum(iv_values) / len(iv_values),
                        "high": max(iv_values),
                    }

        # 优先级3: 原逻辑兜底（所有方法）
        if not intrinsic_value_range:
            iv = vmeta.get("intrinsic_value", {})
            if iv:
                iv_values = [v for v in iv.values() if isinstance(v, (int, float)) and v > 0]
                if iv_values:
                    intrinsic_value_range = {
                        "low": min(iv_values),
                        "base": sum(iv_values) / len(iv_values),
                        "high": max(iv_values),
                    }
        # 安全边际
        mos_data = vmeta.get("margin_of_safety", {})
        if mos_data:
            mos_values = [
                d.get("margin_of_safety", 0) for d in mos_data.values()
                if isinstance(d, dict) and "margin_of_safety" in d
            ]
            if mos_values:
                margin_of_safety_val = round(sum(mos_values) / len(mos_values), 4)

        value_grade = vmeta.get("valuation_grade", "N/A")

    # Layer 2: 企业质量评级
    quality_score = 0
    quality_grade = "N/A"
    piotroski_val = 0
    moat_grade_val = ""

    fundamental_signal = next((s for s in signals if s.agent_name == "fundamental"), None)
    if fundamental_signal and isinstance(fundamental_signal.metadata, dict):
        fmeta = fundamental_signal.metadata
        vm = fmeta.get("value_metrics", {})
        piotroski_val = vm.get("piotroski_f_score", 0)

        # 质量评分: 基于fundamental score + piotroski + moat
        quality_score = max(0, min(100, int((fundamental_signal.signal_score + 100) / 2)))

    moat_signal = next((s for s in signals if s.agent_name == "moat"), None)
    if moat_signal:
        if moat_signal.signal_score >= 40:
            moat_grade_val = "宽"
        elif moat_signal.signal_score >= 10:
            moat_grade_val = "窄"
        else:
            moat_grade_val = "无"
        # 护城河影响质量评分
        quality_score = max(0, min(100, quality_score + moat_signal.signal_score // 5))

    if quality_score >= 80:
        quality_grade = "优秀"
    elif quality_score >= 60:
        quality_grade = "良好"
    elif quality_score >= 40:
        quality_grade = "一般"
    elif quality_score >= 20:
        quality_grade = "较差"
    else:
        quality_grade = "危险"

    # Layer 3: 择时信号
    timing_signal = _calc_timing_signal(signals, timing_agents)

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[Fusion] 完成: score={final_score:+d}[{score_range_low:+d}~{score_range_high:+d}] "
        f"action={final_action} confidence={confidence:.0%} position={position_pct}% "
        f"value_grade={value_grade} quality={quality_score}({quality_grade}) moat={moat_grade_val} "
        f"mos={margin_of_safety_val:.0%} timing={timing_signal} "
        f"bull={len(bullish)} bear={len(bearish)} neutral={len(neutral_signals)} "
        f"uncertainties={len(key_uncertainties)} llm={'Y' if llm_result['enhanced'] else 'N'} {elapsed_ms}ms"
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
        risk_cross_validation=tuple(risk_cross_validation) if isinstance(risk_cross_validation, list) else (),
        operation_strategy=operation_strategy if isinstance(operation_strategy, dict) else {},
        sub_scores=sub_scores if isinstance(sub_scores, dict) else {},
        # P2: 不确定性字段
        score_range_low=score_range_low,
        score_range_high=score_range_high,
        key_uncertainties=tuple(key_uncertainties),
        verification_points=tuple(verification_points),
        # R3: 价值投资三层输出
        intrinsic_value_range=intrinsic_value_range,
        margin_of_safety=margin_of_safety_val,
        value_grade=value_grade,
        quality_score=quality_score,
        quality_grade=quality_grade,
        piotroski_f_score=piotroski_val,
        moat_grade=moat_grade_val,
        timing_signal=timing_signal,
    )
