"""
ESG分析Agent (环境、社会、治理)

评估公司的ESG表现，识别可持续发展和非财务风险。
LLM-first模式: LLM做主角，代码提供量化锚点。

P3新增：覆盖非财务维度的风险评估。

ESG维度：
- E (环境): 碳排放、环保合规、资源效率、绿色转型
- S (社会): 员工关系、供应链责任、产品安全、社区贡献
- G (治理): 董事会结构、信息披露、股东权益、商业伦理
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from ..state import AgentSignal, StockData


def _calc_esg_anchor(stock: StockData) -> tuple[int, str, dict]:
    """计算ESG量化锚点分数和说明

    基于可量化的ESG指标评估公司ESG表现。
    锚点分数范围 -60 ~ +60，LLM 最终评分应在锚点 ±30 以内。

    评估维度：
    1. 环境风险：行业污染属性、环保处罚记录
    2. 社会责任：员工稳定性、薪酬竞争力
    3. 治理质量：股权结构、信息披露质量

    Returns:
        (anchor_score, anchor_explanation, anchor_details)
    """
    score = 0
    details = []
    anchor_details = {}

    info = stock.info or {}

    # ========== 环境维度 (E) ==========

    # 1. 行业环境风险属性
    industry = info.get("industry", "")
    high_pollution_industries = [
        "化工", "石化", "钢铁", "有色金属", "煤炭", "造纸", "水泥", "电力",
        "纺织", "印染", "电镀", "制药", "农药", "化肥"
    ]
    medium_pollution_industries = [
        "汽车", "机械", "电子制造", "食品加工", "酿酒", "皮革"
    ]

    env_risk_level = "low"
    if any(p in industry for p in high_pollution_industries):
        env_risk_level = "high"
        score -= 10
        details.append(f"行业[{industry}]属高污染行业，环境风险高→-10")
    elif any(p in industry for p in medium_pollution_industries):
        env_risk_level = "medium"
        score -= 3
        details.append(f"行业[{industry}]有中等环境风险→-3")
    else:
        details.append(f"行业[{industry}]环境风险较低→+0")

    anchor_details["env_risk_level"] = env_risk_level

    # 2. 环保处罚记录（如有）
    env_penalties = info.get("environmental_penalties", [])
    if env_penalties:
        penalty_count = len(env_penalties)
        total_penalty = sum(p.get("amount", 0) for p in env_penalties if isinstance(p, dict))
        if penalty_count >= 3 or total_penalty > 1000000:  # 100万以上
            score -= 15
            details.append(f"环保处罚{penalty_count}次/累计{total_penalty/10000:.0f}万→-15")
        elif penalty_count >= 1:
            score -= 8
            details.append(f"环保处罚{penalty_count}次→-8")
        anchor_details["env_penalties"] = {"count": penalty_count, "total": total_penalty}

    # 3. 绿色转型/碳中和承诺（如有）
    green_initiatives = info.get("green_initiatives", [])
    if green_initiatives:
        score += 5
        details.append(f"有绿色转型举措({len(green_initiatives)}项)→+5")
        anchor_details["green_initiatives"] = green_initiatives

    # ========== 社会维度 (S) ==========

    # 1. 员工稳定性（从财务数据推断）
    if stock.financial_data:
        df = pd.DataFrame(stock.financial_data)
        if "report_date" in df.columns:
            df["report_date"] = pd.to_datetime(df["report_date"])
            df = df.sort_values("report_date")

        # 员工人数变化
        if "employee_count" in df.columns:
            emp_vals = pd.to_numeric(df["employee_count"], errors="coerce").dropna()
            if len(emp_vals) >= 2:
                emp_change = (emp_vals.iloc[-1] - emp_vals.iloc[-2]) / emp_vals.iloc[-2] * 100
                if emp_change < -20:
                    score -= 8
                    details.append(f"员工数减少{emp_change:.1f}%（可能裁员）→-8")
                elif emp_change < -10:
                    score -= 4
                    details.append(f"员工数减少{emp_change:.1f}%→-4")
                elif emp_change > 10:
                    score += 3
                    details.append(f"员工数增长{emp_change:.1f}%→+3")
                anchor_details["employee_change_pct"] = round(emp_change, 1)

        # 人均薪酬（管理费用/员工数，粗略估计）
        if all(c in df.columns for c in ["admin_expense", "employee_count"]):
            admin = pd.to_numeric(df["admin_expense"], errors="coerce")
            emp = pd.to_numeric(df["employee_count"], errors="coerce")
            if len(admin) >= 1 and len(emp) >= 1 and emp.iloc[-1] > 0:
                per_capita = admin.iloc[-1] / emp.iloc[-1]
                # 人均薪酬10万以下偏低，20万以上较高
                if per_capita < 100000:
                    score -= 3
                    details.append(f"人均管理费{per_capita/10000:.1f}万（偏低）→-3")
                elif per_capita > 200000:
                    score += 3
                    details.append(f"人均管理费{per_capita/10000:.1f}万（较高）→+3")
                anchor_details["per_capita_admin"] = round(per_capita / 10000, 1)

    # 2. 安全生产事故（如有）
    safety_incidents = info.get("safety_incidents", [])
    if safety_incidents:
        incident_count = len(safety_incidents)
        if incident_count >= 2:
            score -= 10
            details.append(f"安全生产事故{incident_count}次→-10")
        elif incident_count >= 1:
            score -= 5
            details.append(f"安全生产事故1次→-5")
        anchor_details["safety_incidents"] = incident_count

    # 3. 供应链责任（如有信息）
    supply_chain_compliance = info.get("supply_chain_compliance", {})
    if supply_chain_compliance:
        audit_rate = supply_chain_compliance.get("audit_rate", 0)
        if audit_rate >= 0.8:
            score += 5
            details.append(f"供应链审计覆盖率{audit_rate:.0%}→+5")
        elif audit_rate > 0:
            score += 2
            details.append(f"供应链审计覆盖率{audit_rate:.0%}→+2")
        anchor_details["supply_chain_audit_rate"] = audit_rate

    # ========== 治理维度 (G) ==========

    # 1. 股权结构（是否一股独大）
    top_shareholder_pct = info.get("top_shareholder_pct")
    if top_shareholder_pct is not None:
        if top_shareholder_pct > 50:
            score -= 8
            details.append(f"第一大股东持股{top_shareholder_pct:.1f}%（一股独大风险）→-8")
        elif top_shareholder_pct > 40:
            score -= 4
            details.append(f"第一大股东持股{top_shareholder_pct:.1f}%（较高）→-4")
        elif top_shareholder_pct < 20:
            score += 3
            details.append(f"股权分散（第一大股东{top_shareholder_pct:.1f}%）→+3")
        anchor_details["top_shareholder_pct"] = top_shareholder_pct

    # 2. 独立董事比例
    independent_director_ratio = info.get("independent_director_ratio")
    if independent_director_ratio is not None:
        if independent_director_ratio >= 0.4:  # 证监会要求1/3
            score += 3
            details.append(f"独立董事占比{independent_director_ratio:.0%}→+3")
        elif independent_director_ratio < 0.33:
            score -= 5
            details.append(f"独立董事占比{independent_director_ratio:.0%}（低于要求）→-5")
        anchor_details["independent_director_ratio"] = independent_director_ratio

    # 3. 信息披露质量（从财报审计意见推断）
    audit_opinion = info.get("audit_opinion", "")
    if "无保留" in audit_opinion or "标准" in audit_opinion:
        score += 5
        details.append(f"审计意见：{audit_opinion}→+5")
    elif "保留" in audit_opinion or "无法表示" in audit_opinion:
        score -= 15
        details.append(f"审计意见：{audit_opinion}（重大风险）→-15")
    elif "强调" in audit_opinion:
        score -= 5
        details.append(f"审计意见：{audit_opinion}（需关注）→-5")
    anchor_details["audit_opinion"] = audit_opinion

    # 4. 关联交易占比
    related_party_ratio = info.get("related_party_transaction_ratio")
    if related_party_ratio is not None:
        if related_party_ratio > 0.3:
            score -= 8
            details.append(f"关联交易占比{related_party_ratio:.0%}（较高）→-8")
        elif related_party_ratio > 0.1:
            score -= 3
            details.append(f"关联交易占比{related_party_ratio:.0%}→-3")
        anchor_details["related_party_ratio"] = related_party_ratio

    # 5. 监管处罚记录
    regulatory_penalties = info.get("regulatory_penalties", [])
    if regulatory_penalties:
        penalty_count = len(regulatory_penalties)
        if penalty_count >= 3:
            score -= 15
            details.append(f"监管处罚{penalty_count}次（合规风险高）→-15")
        elif penalty_count >= 1:
            score -= 8
            details.append(f"监管处罚{penalty_count}次→-8")
        anchor_details["regulatory_penalties"] = penalty_count

    # ========== 综合评分 ==========

    # 限制锚点范围
    anchor_score = max(-60, min(60, score))

    # 生成锚点说明
    if details:
        explanation = "ESG锚点：" + "；".join(details)
    else:
        explanation = "ESG数据有限，锚点基于行业属性估算"

    # 添加锚点范围说明
    if anchor_score >= 30:
        explanation += f"。综合锚点{anchor_score:+d}分（ESG表现良好）"
    elif anchor_score >= 0:
        explanation += f"。综合锚点{anchor_score:+d}分（ESG表现一般）"
    elif anchor_score >= -30:
        explanation += f"。综合锚点{anchor_score:+d}分（ESG存在风险）"
    else:
        explanation += f"。综合锚点{anchor_score:+d}分（ESG风险较高）"

    anchor_details["anchor_score"] = anchor_score

    return anchor_score, explanation, anchor_details


def _build_esg_context(stock: StockData) -> str:
    """构建ESG分析上下文"""
    lines = []
    info = stock.info or {}

    # 基本信息
    lines.append(f"行业: {info.get('industry', '未知')}")
    lines.append(f"市值: {info.get('market_cap', 0)/1e8:.1f}亿")

    # 环境相关
    env_penalties = info.get("environmental_penalties", [])
    if env_penalties:
        lines.append(f"\n环保处罚记录: {len(env_penalties)}次")
        for p in env_penalties[:3]:
            if isinstance(p, dict):
                lines.append(f"  - {p.get('date', '')}: {p.get('reason', '')}, 罚款{p.get('amount', 0)/10000:.0f}万")

    green_initiatives = info.get("green_initiatives", [])
    if green_initiatives:
        lines.append(f"\n绿色转型举措: {len(green_initiatives)}项")
        for g in green_initiatives[:3]:
            lines.append(f"  - {g}")

    # 社会相关
    safety_incidents = info.get("safety_incidents", [])
    if safety_incidents:
        lines.append(f"\n安全生产事故: {len(safety_incidents)}次")

    supply_chain = info.get("supply_chain_compliance", {})
    if supply_chain:
        lines.append(f"\n供应链责任:")
        lines.append(f"  - 审计覆盖率: {supply_chain.get('audit_rate', 0):.0%}")

    # 治理相关
    lines.append(f"\n股权结构:")
    lines.append(f"  - 第一大股东: {info.get('top_shareholder_pct', '未知')}%")
    lines.append(f"  - 独立董事占比: {info.get('independent_director_ratio', '未知')}")

    lines.append(f"\n审计意见: {info.get('audit_opinion', '未知')}")

    related_party = info.get("related_party_transaction_ratio")
    if related_party is not None:
        lines.append(f"关联交易占比: {related_party:.0%}")

    regulatory = info.get("regulatory_penalties", [])
    if regulatory:
        lines.append(f"\n监管处罚: {len(regulatory)}次")
        for r in regulatory[:3]:
            if isinstance(r, dict):
                lines.append(f"  - {r.get('date', '')}: {r.get('reason', '')}")

    return "\n".join(lines)


def analyze_esg(stock: StockData) -> AgentSignal:
    """ESG分析主函数

    分析流程：
    1. 计算量化锚点分数
    2. 构建ESG上下文
    3. 调用LLM进行定性分析
    4. 生成最终信号
    """
    start = time.time()
    symbol = stock.symbol
    name = stock.name

    # Step 1: 计算量化锚点
    anchor_score, anchor_explanation, anchor_details = _calc_esg_anchor(stock)

    # 锚点范围
    anchor_low = anchor_score - 30
    anchor_high = anchor_score + 30

    logger.info(f"[ESG] {symbol} anchor={anchor_score:+d} range=[{anchor_low:+d}, {anchor_high:+d}]")

    # Step 2: 构建ESG上下文
    esg_context = _build_esg_context(stock)

    # Step 3: 准备LLM调用
    from ..llm_enhance import llm_enhance

    llm_result = llm_enhance(
        agent_name="esg",
        template_name="esg.md",
        template_vars={
            "symbol": symbol,
            "name": name,
            "analysis_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "esg_context": esg_context,
            "anchor_score": anchor_score,
            "anchor_score_low": anchor_low,
            "anchor_score_high": anchor_high,
            "anchor_explanation": anchor_explanation,
        },
        code_score=anchor_score,
        code_reasoning=anchor_explanation,
        code_factors=[],
        code_risks=[],
        max_adjustment=30,  # ESG评分调整限制在±30
    )

    # Step 4: 处理LLM结果
    if llm_result["enhanced"]:
        score = anchor_score + llm_result["score_adjustment"]
        reasoning = llm_result["reasoning"]
    else:
        score = anchor_score
        reasoning = f"ESG量化评估：{anchor_explanation}"

    # 强制评分在锚点范围内
    if abs(score - anchor_score) > 30:
        original_score = score
        score = max(anchor_low, min(anchor_high, score))
        logger.warning(
            f"[ESG] {symbol} score {original_score:+d} clamped to {score:+d} "
            f"(anchor range [{anchor_low:+d}, {anchor_high:+d}])"
        )

    # 限制最终分数范围
    score = max(-100, min(100, int(score)))

    # 提取关键因素和风险
    raw_response = llm_result.get("raw_response", {})
    if isinstance(raw_response, dict):
        key_factors = tuple(raw_response.get("key_strengths", [])[:5])
        risks = tuple(raw_response.get("key_risks", [])[:5])
        e_score = raw_response.get("e_score", 0)
        s_score = raw_response.get("s_score", 0)
        g_score = raw_response.get("g_score", 0)
    else:
        key_factors = ()
        risks = ()
        e_score, s_score, g_score = 0, 0, 0

    # 置信度：ESG数据通常较难获取，置信度适中
    if llm_result["enhanced"]:
        confidence = 0.65
    else:
        confidence = 0.5

    elapsed_ms = int((time.time() - start) * 1000)

    logger.info(
        f"[ESG] {symbol} final_score={score:+d} E={e_score} S={s_score} G={g_score} "
        f"confidence={confidence:.0%} llm={'Y' if llm_result['enhanced'] else 'N'} {elapsed_ms}ms"
    )

    return AgentSignal(
        agent_name="esg",
        signal_score=score,
        confidence=confidence,
        reasoning=reasoning,
        key_factors=key_factors,
        risks=risks,
        data_quality=1.0 if esg_context else 0.5,
        metadata={
            "anchor_score": anchor_score,
            "anchor_range": [anchor_low, anchor_high],
            "e_score": e_score,
            "s_score": s_score,
            "g_score": g_score,
            "raw_response": raw_response,
            **anchor_details,
        },
        llm_model=llm_result.get("llm_model", ""),
        llm_tokens_used=llm_result.get("llm_tokens", 0),
        llm_cost_usd=llm_result.get("llm_cost", 0.0),
        execution_time_ms=elapsed_ms,
    )
