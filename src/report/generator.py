"""
研报生成器

将融合决策和各Agent分析结果生成结构化投研报告。
Phase 1: Markdown格式 + 可选LLM增强核心逻辑
Phase 3: HTML/PDF
"""

from __future__ import annotations

from datetime import date
from typing import Any

from loguru import logger


def _action_to_stars(score: int) -> str:
    """分数转星级评级"""
    if score >= 60:
        return "★★★★★（强烈看多）"
    if score >= 30:
        return "★★★★（看多）"
    if score >= 10:
        return "★★★☆（偏多）"
    if score >= -10:
        return "★★★（中性）"
    if score >= -30:
        return "★★☆（偏空）"
    if score >= -60:
        return "★★（看空）"
    return "★（强烈看空）"


_AGENT_DISPLAY = {
    "technical": "技术面",
    "fundamental": "基本面",
    "valuation": "估值",
    "money_flow": "资金面",
    "sentiment": "情绪面",
    "industry": "产业链",
}


def _build_report_context(state: dict[str, Any]) -> str:
    """构建给LLM的报告上下文"""
    fusion = state.get("fusion")
    signals = state.get("signals", [])

    parts = []
    if fusion:
        parts.append(f"综合评分: {fusion.final_score:+d}, 建议: {fusion.final_action}")
    for s in signals:
        name = _AGENT_DISPLAY.get(s.agent_name, s.agent_name)
        parts.append(f"{name}({s.signal_score:+d}): {s.reasoning[:100]}")
    return "\n".join(parts)


def _llm_enhance_report(state: dict[str, Any]) -> dict[str, Any]:
    """LLM生成更自然的核心逻辑摘要

    返回: {"enhanced": bool, "core_logic": str, "llm_model": str, ...}
    """
    result = {"enhanced": False, "core_logic": "", "llm_model": "", "llm_tokens": 0, "llm_cost": 0.0}

    stock = state.get("stock")
    fusion = state.get("fusion")
    if stock is None or fusion is None:
        return result

    try:
        from src.agents.llm_enhance import llm_enhance

        llm_result = llm_enhance(
            agent_name="report",
            template_name="report.md",
            template_vars={
                "symbol": stock.symbol,
                "name": stock.name,
                "analysis_date": state.get("analysis_date", date.today().isoformat()),
                "report_context": _build_report_context(state),
            },
            code_score=fusion.final_score,
            code_reasoning=fusion.reasoning,
            code_factors=[],
            code_risks=[],
        )

        result["llm_model"] = llm_result["llm_model"]
        result["llm_tokens"] = llm_result["llm_tokens"]
        result["llm_cost"] = llm_result["llm_cost"]

        if llm_result["enhanced"] and llm_result["reasoning"]:
            result["enhanced"] = True
            result["core_logic"] = llm_result["reasoning"]
    except Exception as e:
        logger.debug(f"[Report] LLM增强跳过: {e}")

    return result


def generate_report(state: dict[str, Any]) -> str:
    """生成Markdown研报

    state 是 LangGraph 的 TypedDict，通过 dict 方式访问。
    """
    stock = state.get("stock")
    fusion = state.get("fusion")
    signals = state.get("signals", [])
    analysis_date = state.get("analysis_date", date.today().isoformat())

    if stock is None or fusion is None:
        return "# 分析失败\n\n无法生成报告：缺少股票数据或融合决策。"

    symbol = stock.symbol
    name = stock.name
    rating = _action_to_stars(fusion.final_score)

    # LLM增强核心逻辑
    llm_report = _llm_enhance_report(state)

    # === 构建报告 ===
    lines = [
        f"# {name}({symbol}) 投研分析报告",
        "",
        f"**分析日期**: {analysis_date}",
        "",
        f"## 综合评级: {rating}",
        "",
        f"- **综合评分**: {fusion.final_score:+d}/100",
        f"- **置信度**: {fusion.confidence:.0%}",
        f"- **建议操作**: {fusion.final_action}",
        f"- **建议仓位**: {fusion.position_pct}%",
        f"- **止损位**: {fusion.stop_loss_pct:+.1f}%",
        f"- **目标位**: {fusion.take_profit_pct:+.1f}%",
        "",
    ]

    # 核心逻辑
    lines.append("## 核心逻辑")
    lines.append("")
    if llm_report["enhanced"]:
        lines.append(llm_report["core_logic"])
    else:
        lines.append(fusion.reasoning)
    lines.append("")

    # 融合权重
    if fusion.weights_used:
        lines.append("## 信号权重")
        lines.append("")
        for agent_name, weight in sorted(fusion.weights_used.items(), key=lambda x: -x[1]):
            display = _AGENT_DISPLAY.get(agent_name, agent_name)
            score = fusion.signal_summary.get(agent_name, 0)
            lines.append(f"- {display}: 权重{weight:.0%} | 评分{score:+d}")
        lines.append("")

    # 各维度分析
    if signals:
        lines.append("## 各维度分析")
        lines.append("")
        for signal in signals:
            display_name = _AGENT_DISPLAY.get(signal.agent_name, signal.agent_name)
            llm_tag = f" [{signal.llm_model}]" if signal.llm_model else ""

            lines.append(f"### {display_name}（{signal.signal_score:+d}）{llm_tag}")
            lines.append("")
            lines.append(f"- **置信度**: {signal.confidence:.0%}")
            lines.append(f"- **分析**: {signal.reasoning}")
            if signal.key_factors:
                lines.append(f"- **关键因素**: {', '.join(signal.key_factors[:5])}")
            if signal.risks:
                lines.append(f"- **风险**: {', '.join(signal.risks)}")
            lines.append("")

    # 矛盾信号
    if fusion.conflicts:
        lines.append("## 信号矛盾")
        lines.append("")
        for conflict in fusion.conflicts:
            lines.append(f"- {conflict}")
        if fusion.conflict_resolution:
            lines.append(f"- **解决方案**: {fusion.conflict_resolution}")
        lines.append("")

    # 风险提示
    all_risks = []
    for signal in signals:
        all_risks.extend(signal.risks)
    if all_risks:
        lines.append("## 风险提示")
        lines.append("")
        for risk in dict.fromkeys(all_risks):  # 去重保持顺序
            lines.append(f"- {risk}")
        lines.append("")

    # LLM成本统计
    total_tokens = sum(s.llm_tokens_used for s in signals)
    total_cost = sum(s.llm_cost_usd for s in signals)
    if llm_report["llm_tokens"]:
        total_tokens += llm_report["llm_tokens"]
        total_cost += llm_report["llm_cost"]

    # 数据来源
    lines.extend([
        "## 数据来源与时效性",
        "",
        f"- 分析日期: {analysis_date}",
        f"- 分析Agent数: {len(signals)}",
    ])
    if total_tokens > 0:
        lines.append(f"- LLM消耗: {total_tokens} tokens (${total_cost:.4f})")
    else:
        lines.append("- LLM: 未使用（纯代码分析）")
    lines.extend([
        "- 本报告由AI投研助手系统自动生成，仅供参考",
        "",
        "---",
        "*免责声明：本报告由AI系统自动生成，不构成投资建议。投资有风险，入市需谨慎。*",
    ])

    report = "\n".join(lines)
    logger.info(f"研报生成完成: {symbol} ({len(report)} chars)")
    return report
