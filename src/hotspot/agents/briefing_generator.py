"""
研报生成Agent

输入：所有 AnalyzedTheme
输出：完整的markdown盘前纪要
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from ..state import AnalyzedTheme

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def generate_briefing(
    analyzed_themes: list[AnalyzedTheme],
    analysis_date: str,
    config: dict[str, Any] | None = None,
) -> str:
    """生成盘前热点研报

    Args:
        analyzed_themes: 完成分析的主题列表
        analysis_date: 分析日期 YYYY-MM-DD
        config: hotspot.yaml 配置

    Returns:
        Markdown格式研报
    """
    config = config or {}

    if not analyzed_themes:
        return f"# 盘前热点纪要 | {analysis_date}\n\n暂无热点主题。"

    # 构建主题文本
    themes_text = _build_themes_text(analyzed_themes)

    # 加载并填充prompt
    template = (_PROMPTS_DIR / "briefing.md").read_text(encoding="utf-8")
    system_section, user_section = _parse_prompt_sections(template)
    user_prompt = user_section.format(
        analysis_date=analysis_date,
        themes_text=themes_text,
    )

    # 调用LLM
    try:
        from src.llm.router import get_llm_router

        router = get_llm_router()
        response = router.invoke(
            system_prompt=system_section,
            user_prompt=user_prompt,
            provider=config.get("llm_provider", "ollama"),
            model_name=config.get("llm_model", "qwen2.5:14b"),
            agent_name="briefing_generator",
            temperature=0.4,
            max_tokens=6000,
            timeout=300,
        )

        logger.info(
            f"[BriefingGenerator] LLM响应: {response.total_tokens}tokens, "
            f"{response.latency_ms}ms"
        )

        return response.content.strip()

    except Exception as e:
        logger.error(f"[BriefingGenerator] 研报生成失败: {e}")
        # 降级：直接用数据生成简单研报
        return _fallback_briefing(analyzed_themes, analysis_date)


def _build_themes_text(themes: list[AnalyzedTheme]) -> str:
    """将AnalyzedTheme列表格式化为LLM输入文本"""
    parts = []
    for i, at in enumerate(themes, 1):
        t = at.theme
        chain = at.industry_chain

        lines = [
            f"### 主题{i}：{t.title}",
            f"- 摘要：{t.summary}",
            f"- 催化剂：{t.catalyst}",
            f"- 时间维度：{t.timeline}",
            f"- 关联概念：{', '.join(t.source_concepts)}",
            f"- 可操作性：{at.actionability}",
            f"- 投资逻辑：{at.investment_logic}",
            f"- 风险：{t.risk}",
            "",
            "产业链：",
        ]

        # 上游
        if chain.upstream:
            lines.append("**上游**：")
            for s in chain.upstream:
                lines.append(f"  - {s.name}({s.symbol}) — {s.role} | {s.reason}")

        # 中游
        if chain.midstream:
            lines.append("**中游**：")
            for s in chain.midstream:
                lines.append(f"  - {s.name}({s.symbol}) — {s.role} | {s.reason}")

        # 下游
        if chain.downstream:
            lines.append("**下游**：")
            for s in chain.downstream:
                lines.append(f"  - {s.name}({s.symbol}) — {s.role} | {s.reason}")

        if chain.value_flow:
            lines.append(f"\n价值传导：{chain.value_flow}")

        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts)


def _fallback_briefing(themes: list[AnalyzedTheme], analysis_date: str) -> str:
    """降级：不用LLM直接生成简单研报"""
    lines = [f"# 盘前热点纪要 | {analysis_date}", ""]

    for i, at in enumerate(themes, 1):
        t = at.theme
        chain = at.industry_chain

        lines.append(f"## 主题{i}：{t.title}")
        lines.append(f"\n**核心逻辑**：{t.summary}")
        lines.append(f"\n**催化剂**：{t.catalyst}")
        lines.append(f"\n**产业链**：")

        if chain.upstream:
            names = " → ".join(f"{s.name}" for s in chain.upstream)
            lines.append(f"- 上游：{names}")
        if chain.midstream:
            names = " → ".join(f"{s.name}" for s in chain.midstream)
            lines.append(f"- 中游：{names}")
        if chain.downstream:
            names = " → ".join(f"{s.name}" for s in chain.downstream)
            lines.append(f"- 下游：{names}")

        lines.append(f"\n**核心标的**：")
        lines.append("| 代码 | 名称 | 位置 | 角色 | 受益逻辑 |")
        lines.append("|------|------|------|------|---------|")
        all_stocks = list(chain.upstream) + list(chain.midstream) + list(chain.downstream)
        for s in all_stocks:
            pos = {"upstream": "上游", "midstream": "中游", "downstream": "下游"}.get(s.chain_position, "")
            lines.append(f"| {s.symbol} | {s.name} | {pos} | {s.role} | {s.reason} |")

        lines.append(f"\n**风险提示**：{t.risk}")
        lines.append("\n---\n")

    lines.append("\n*本报告由AI自动生成，仅供参考，不构成投资建议。*")
    return "\n".join(lines)


def _parse_prompt_sections(template_text: str) -> tuple[str, str]:
    """解析prompt模板的系统/用户分段"""
    system_marker = "## 系统提示词"
    user_marker = "## 用户提示词"

    if system_marker not in template_text or user_marker not in template_text:
        return "", template_text

    sys_start = template_text.index(system_marker) + len(system_marker)
    user_start_marker = template_text.index(user_marker)
    user_start = user_start_marker + len(user_marker)

    system_prompt = template_text[sys_start:user_start_marker].strip()
    user_prompt = template_text[user_start:].strip()
    return system_prompt, user_prompt
