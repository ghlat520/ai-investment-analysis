"""
研报生成器

将融合决策和各Agent分析结果生成结构化投研报告。
支持8-Agent深度分析 + 多空辩论 + 目标价 + 操作建议。
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
    "moat": "护城河",
    "business_model": "商业模式",
    "industry": "行业",
    "supply_chain": "产业链",
}


def _build_report_context(state: dict[str, Any]) -> str:
    """构建给LLM的报告上下文"""
    fusion = state.get("fusion")
    signals = state.get("signals", [])

    parts = []
    if fusion:
        parts.append(f"综合评分: {fusion.final_score:+d}, 建议: {fusion.final_action}")
        if fusion.bull_arguments:
            parts.append(f"\n多方论据: {'; '.join(fusion.bull_arguments[:5])}")
        if fusion.bear_arguments:
            parts.append(f"\n空方论据: {'; '.join(fusion.bear_arguments[:5])}")
        if fusion.target_prices:
            tp = fusion.target_prices
            parts.append(f"\n目标价: 保守{tp.get('conservative', 0):.2f} / 中性{tp.get('base', 0):.2f} / 乐观{tp.get('optimistic', 0):.2f}")
    for s in signals:
        name = _AGENT_DISPLAY.get(s.agent_name, s.agent_name)
        parts.append(f"{name}({s.signal_score:+d}): {s.reasoning[:120]}")
    return "\n".join(parts)


def _llm_enhance_report(state: dict[str, Any]) -> dict[str, Any]:
    """LLM生成更自然的核心逻辑摘要"""
    result = {
        "enhanced": False, "core_logic": "",
        "operation_advice": {}, "key_tracking": [],
        "llm_model": "", "llm_tokens": 0, "llm_cost": 0.0,
    }

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
    """生成Markdown研报"""
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

    # 目标价
    if fusion.target_prices:
        tp = fusion.target_prices
        conservative = tp.get("conservative", 0)
        base = tp.get("base", 0)
        optimistic = tp.get("optimistic", 0)
        if any(v > 0 for v in [conservative, base, optimistic]):
            lines.append("## 目标价")
            lines.append("")
            lines.append(f"| 情景 | 目标价 |")
            lines.append(f"|------|--------|")
            if conservative > 0:
                lines.append(f"| 保守 | {conservative:.2f}元 |")
            if base > 0:
                lines.append(f"| 中性 | {base:.2f}元 |")
            if optimistic > 0:
                lines.append(f"| 乐观 | {optimistic:.2f}元 |")
            pw = tp.get("probability_weighted", 0)
            if pw > 0:
                lines.append(f"| 概率加权 | {pw:.2f}元 |")
            lines.append("")

    # 核心逻辑
    lines.append("## 核心逻辑")
    lines.append("")
    if llm_report["enhanced"]:
        lines.append(llm_report["core_logic"])
    else:
        lines.append(fusion.reasoning)
    lines.append("")

    # 多空博弈
    if fusion.bull_arguments or fusion.bear_arguments:
        lines.append("## 多空博弈")
        lines.append("")
        if fusion.bull_arguments:
            lines.append("### 多方论据")
            lines.append("")
            for arg in fusion.bull_arguments[:8]:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        display = _AGENT_DISPLAY.get(k, k)
                        lines.append(f"- [{display}] {v}")
                else:
                    lines.append(f"- {arg}")
            lines.append("")
        if fusion.bear_arguments:
            lines.append("### 空方论据")
            lines.append("")
            for arg in fusion.bear_arguments[:8]:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        display = _AGENT_DISPLAY.get(k, k)
                        lines.append(f"- [{display}] {v}")
                else:
                    lines.append(f"- {arg}")
            lines.append("")

    # 可验证分歧点
    if fusion.divergence_points:
        lines.append("## 可验证分歧点")
        lines.append("")
        for dp in fusion.divergence_points:
            if isinstance(dp, dict):
                for k, v in dp.items():
                    lines.append(f"- **{k}**: {v}")
            else:
                lines.append(f"- {dp}")
        lines.append("")

    # 融合权重
    if fusion.weights_used:
        lines.append("## 信号权重")
        lines.append("")
        lines.append("| 维度 | 权重 | 评分 |")
        lines.append("|------|------|------|")
        for agent_name, weight in sorted(fusion.weights_used.items(), key=lambda x: -x[1]):
            display = _AGENT_DISPLAY.get(agent_name, agent_name)
            score = fusion.signal_summary.get(agent_name, 0)
            lines.append(f"| {display} | {weight:.0%} | {score:+d} |")
        lines.append("")

    # === V2: 基本面深度分析（收入构成、红旗/绿旗）===
    fundamental_signal = next((s for s in signals if s.agent_name == "fundamental"), None)
    if fundamental_signal and isinstance(fundamental_signal.metadata, dict):
        meta = fundamental_signal.metadata
        # 收入构成表
        rev_breakdown = meta.get("revenue_breakdown")
        if rev_breakdown and isinstance(rev_breakdown, list) and len(rev_breakdown) > 0:
            lines.append("## 基本面深度分析")
            lines.append("")
            lines.append("### 收入构成")
            lines.append("")
            lines.append("| 业务板块 | 营收(亿) | 占比 | 毛利率 | 增速 | 角色 | 未来预测 |")
            lines.append("|---------|---------|------|--------|------|------|---------|")
            for seg in rev_breakdown:
                if isinstance(seg, dict):
                    lines.append(
                        f"| {seg.get('segment', seg.get('name', '-'))} "
                        f"| {seg.get('revenue_billion', seg.get('revenue', '-'))} "
                        f"| {seg.get('revenue_pct', '-')}% "
                        f"| {seg.get('gross_margin', '-')}% "
                        f"| {seg.get('growth_rate', '-')}% "
                        f"| {seg.get('role', '-')} "
                        f"| {seg.get('future_estimate', seg.get('future_growth_estimate', '-'))} |"
                    )
            lines.append("")

        # 质量评级
        quality = meta.get("quality_rating")
        if quality:
            lines.append(f"### 财务质量评级: {quality}")
            lines.append("")

        # 红旗/绿旗检测
        red_flags = meta.get("red_flags_detail")
        green_flags = meta.get("green_flags_detail")
        if red_flags or green_flags:
            lines.append("### 红旗/绿旗检测")
            lines.append("")
            lines.append("| 检测项 | 状态 | 说明 | 严重等级 |")
            lines.append("|--------|------|------|---------|")
            if red_flags and isinstance(red_flags, list):
                for rf in red_flags:
                    if isinstance(rf, dict) and rf.get("triggered"):
                        lines.append(
                            f"| {rf.get('code', '')}: {rf.get('name', '')} "
                            f"| {'命中' if rf.get('triggered') else '未命中'} "
                            f"| {rf.get('detail', '-')} "
                            f"| {rf.get('severity', '-')} |"
                        )
            if green_flags and isinstance(green_flags, list):
                for gf in green_flags:
                    if isinstance(gf, dict) and gf.get("triggered"):
                        lines.append(
                            f"| {gf.get('code', '')}: {gf.get('name', '')} "
                            f"| {'命中' if gf.get('triggered') else '未命中'} "
                            f"| {gf.get('detail', '-')} "
                            f"| - |"
                        )
            lines.append("")

    # === V2: 估值深度分析（模型选择、计算过程、情景分析）===
    valuation_signal = next((s for s in signals if s.agent_name == "valuation"), None)
    if valuation_signal and isinstance(valuation_signal.metadata, dict):
        vmeta = valuation_signal.metadata

        model_sel = vmeta.get("model_selection")
        if model_sel and isinstance(model_sel, dict):
            lines.append("## 估值深度分析")
            lines.append("")
            lines.append("### 估值模型选择")
            lines.append("")
            lines.append(f"- **公司特征**: {model_sel.get('company_profile', '-')}")
            lines.append(f"- **主模型**: {model_sel.get('primary_model', '-')} — {model_sel.get('primary_reason', model_sel.get('primary_model_reason', '-'))}")
            lines.append(f"- **辅助模型**: {model_sel.get('secondary_model', '-')} — {model_sel.get('secondary_reason', model_sel.get('secondary_model_reason', '-'))}")
            lines.append("")

        # 因子权重表
        factor_table = vmeta.get("factor_table")
        if factor_table and isinstance(factor_table, list) and len(factor_table) > 0:
            lines.append("### 估值因子权重表")
            lines.append("")
            lines.append("| 因子 | 权重 | 当前值 | 历史分位 | 行业均值 | 评分 | 来源 |")
            lines.append("|------|------|--------|---------|---------|------|------|")
            for f in factor_table:
                if isinstance(f, dict):
                    lines.append(
                        f"| {f.get('factor', '-')} "
                        f"| {f.get('weight', '-')} "
                        f"| {f.get('current', f.get('current_value', '-'))} "
                        f"| {f.get('percentile', f.get('historical_percentile', '-'))} "
                        f"| {f.get('industry_avg', '-')} "
                        f"| {f.get('score', '-')} "
                        f"| {f.get('source', f.get('data_source', '-'))} |"
                    )
            lines.append("")

        # 主模型计算过程
        primary_val = vmeta.get("primary_valuation")
        if primary_val and isinstance(primary_val, dict):
            lines.append("### 目标价计算过程")
            lines.append("")
            lines.append(f"**{primary_val.get('model', '主模型')}**: {primary_val.get('formula', '')}")
            lines.append("")
            calc = primary_val.get("calculation", primary_val.get("calculation_steps", ""))
            if calc:
                lines.append(calc)
                lines.append("")
            tp = primary_val.get("target_price", 0)
            if tp:
                lines.append(f"**主模型目标价: {tp:.2f}元**")
                lines.append("")

        # 情景分析表
        scenario = vmeta.get("scenario_analysis")
        if scenario and isinstance(scenario, dict):
            lines.append("### 情景分析")
            lines.append("")
            lines.append("| 情景 | 概率 | 核心假设 | 触发条件 | PE | EPS | 目标价 | 较现价空间 |")
            lines.append("|------|------|---------|---------|----|----|--------|----------|")
            for label, key in [("保守", "conservative"), ("中性", "neutral"), ("乐观", "optimistic")]:
                s = scenario.get(key, {})
                if isinstance(s, dict):
                    lines.append(
                        f"| {label} "
                        f"| {s.get('probability', '-')}% "
                        f"| {s.get('assumptions', '-')[:40]} "
                        f"| {s.get('trigger', s.get('trigger_conditions', '-'))[:30]} "
                        f"| {s.get('pe', s.get('pe_assumption', '-'))} "
                        f"| {s.get('eps', s.get('eps_forecast', '-'))} "
                        f"| {s.get('target_price', '-')} "
                        f"| {s.get('upside_pct', s.get('upside', '-'))}% |"
                    )
            lines.append("")

    # 各维度分析
    if signals:
        lines.append("## 各维度分析")
        lines.append("")
        for signal in signals:
            display_name = _AGENT_DISPLAY.get(signal.agent_name, signal.agent_name)
            llm_tag = f" [{signal.llm_model}]" if signal.llm_model else ""
            mode_tag = ""
            if isinstance(signal.metadata, dict):
                llm_mode = signal.metadata.get("llm_mode", "")
                if llm_mode == "primary":
                    mode_tag = " (LLM主导)"

            lines.append(f"### {display_name}（{signal.signal_score:+d}）{llm_tag}{mode_tag}")
            lines.append("")
            lines.append(f"- **置信度**: {signal.confidence:.0%}")
            lines.append(f"- **分析**: {signal.reasoning}")
            if signal.key_factors:
                lines.append(f"- **关键因素**: {', '.join(signal.key_factors[:5])}")
            if signal.risks:
                lines.append(f"- **风险**: {', '.join(signal.risks[:5])}")
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
        for risk in dict.fromkeys(all_risks):
            lines.append(f"- {risk}")
        lines.append("")

    # === V2: 风险交叉验证矩阵 ===
    risk_cv = fusion.risk_cross_validation if fusion.risk_cross_validation else None
    op_strategy = fusion.operation_strategy if fusion.operation_strategy else None

    if risk_cv and isinstance(risk_cv, list) and len(risk_cv) > 0:
        lines.append("## 风险交叉验证")
        lines.append("")
        lines.append("| 风险 | 识别Agent | 确认状态 | 影响评估 | 缓解措施 |")
        lines.append("|------|----------|---------|---------|---------|")
        for r in risk_cv:
            if isinstance(r, dict):
                agents = ", ".join(r.get("identified_by", [])) if isinstance(r.get("identified_by"), list) else str(r.get("identified_by", "-"))
                lines.append(
                    f"| {r.get('risk', r.get('risk_description', '-'))} "
                    f"| {agents} "
                    f"| {r.get('status', r.get('confirmation_status', '-'))} "
                    f"| {r.get('impact', r.get('impact_assessment', '-'))} "
                    f"| {r.get('mitigation', '-')} |"
                )
        lines.append("")

    # === V2: 操作策略 ===
    if op_strategy and isinstance(op_strategy, dict):
        lines.append("## 操作策略")
        lines.append("")

        # 建仓计划
        entry = op_strategy.get("entry")
        if entry and isinstance(entry, dict):
            lines.append("### 建仓计划")
            lines.append("")
            lines.append("| 批次 | 价格 | 仓位 | 条件 |")
            lines.append("|------|------|------|------|")
            ip = entry.get("initial_price", 0)
            ipp = entry.get("initial_position_pct", 0)
            ic = entry.get("condition", "-")
            if ip:
                lines.append(f"| 首次建仓 | {ip:.2f} | {ipp}% | {ic} |")
            scaling = entry.get("scaling", [])
            if isinstance(scaling, list):
                for i, s in enumerate(scaling, 1):
                    if isinstance(s, dict):
                        lines.append(
                            f"| 第{i}次加仓 | {s.get('price', '-')} | {s.get('add_pct', '-')}% | {s.get('trigger', s.get('condition', '-'))} |"
                        )
            lines.append("")

        # 止盈计划
        tp_plan = op_strategy.get("take_profit")
        if tp_plan and isinstance(tp_plan, list) and len(tp_plan) > 0:
            lines.append("### 止盈计划")
            lines.append("")
            lines.append("| 目标 | 价格 | 减仓比例 | 依据 |")
            lines.append("|------|------|---------|------|")
            for i, t in enumerate(tp_plan, 1):
                if isinstance(t, dict):
                    lines.append(
                        f"| 第{i}目标 | {t.get('target_price', '-')} | {t.get('reduce_pct', '-')}% | {t.get('basis', '-')} |"
                    )
            lines.append("")

        # 止损规则
        sl = op_strategy.get("stop_loss")
        if sl and isinstance(sl, dict):
            lines.append("### 止损规则")
            lines.append("")
            hsp = sl.get("hard_stop_price", 0)
            hspc = sl.get("hard_stop_pct", 0)
            if hsp:
                lines.append(f"- **硬性止损**: {hsp:.2f}元（{hspc:+.1f}%），无条件执行")
            cstops = sl.get("conditional_stops", [])
            if isinstance(cstops, list):
                for cs in cstops:
                    if isinstance(cs, dict):
                        lines.append(f"- **条件止损**: {cs.get('condition', '-')} → {cs.get('action', '-')}")
            lines.append("")

        # 短中期建议
        st = op_strategy.get("short_term")
        mt = op_strategy.get("mid_term")
        if st or mt:
            lines.append("### 操作建议")
            lines.append("")
            if st:
                lines.append(f"- **短期（1-3月）**: {st}")
            if mt:
                lines.append(f"- **中期（6-12月）**: {mt}")
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
