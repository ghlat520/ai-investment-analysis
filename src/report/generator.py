"""
研报生成器

将融合决策和各Agent分析结果生成结构化投研报告。
Phase 1: Markdown格式
Phase 3: HTML/PDF
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.agents.state import AnalysisState


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


def generate_report(state: AnalysisState) -> str:
    """生成Markdown研报"""
    stock = state.stock
    fusion = state.fusion
    signals = state.signals

    if stock is None or fusion is None:
        return "# 分析失败\n\n无法生成报告：缺少股票数据或融合决策。"

    symbol = stock.symbol
    name = stock.name
    rating = _action_to_stars(fusion.final_score)

    # 构建报告
    lines = [
        f"# {name}({symbol}) 投研分析报告",
        "",
        f"**分析日期**: {state.analysis_date or date.today().isoformat()}",
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
        "## 核心逻辑",
        "",
        fusion.reasoning,
        "",
    ]

    # 各维度分析
    if signals:
        lines.append("## 各维度分析")
        lines.append("")
        for signal in signals:
            display_name = {
                "technical": "技术面",
                "fundamental": "基本面",
                "valuation": "估值",
                "money_flow": "资金面",
                "sentiment": "情绪面",
                "industry": "产业链",
            }.get(signal.agent_name, signal.agent_name)

            lines.append(f"### {display_name}（{signal.signal_score:+d}）")
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
        for risk in set(all_risks):
            lines.append(f"- {risk}")
        lines.append("")

    # 数据来源
    lines.extend([
        "## 数据来源与时效性",
        "",
        f"- 分析日期: {state.analysis_date or date.today().isoformat()}",
        f"- 分析Agent数: {len(signals)}",
        "- 本报告由AI投研助手系统自动生成，仅供参考",
        "",
        "---",
        "*免责声明：本报告由AI系统自动生成，不构成投资建议。投资有风险，入市需谨慎。*",
    ])

    report = "\n".join(lines)
    logger.info(f"研报生成完成: {symbol} ({len(report)} chars)")
    return report
