"""
产业链分析Agent

输入：单个 HotTheme + 该主题关联的概念板块成分股
输出：IndustryChain + AnalyzedTheme
作用：LLM分类股票到上中下游，解释每只股票的角色
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from ..state import AnalyzedTheme, ChainStock, HotTheme, IndustryChain

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def analyze_theme(
    theme: HotTheme,
    enriched_concepts: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> AnalyzedTheme | None:
    """分析单个主题的产业链

    Args:
        theme: 投资主题
        enriched_concepts: 带成分股的概念板块列表
        config: hotspot.yaml 配置

    Returns:
        AnalyzedTheme 或 None（分析失败时）
    """
    config = config or {}

    # 收集该主题关联的所有成分股（去重）
    stocks_text = _collect_theme_stocks(theme, enriched_concepts)

    # 加载并填充prompt
    template = (_PROMPTS_DIR / "theme_analyze.md").read_text(encoding="utf-8")
    system_section, user_section = _parse_prompt_sections(template)
    user_prompt = user_section.format(
        theme_title=theme.title,
        theme_summary=theme.summary,
        theme_catalyst=theme.catalyst,
        stocks_text=stocks_text,
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
            agent_name="theme_analyzer",
            temperature=0.3,
            max_tokens=3000,
            timeout=300,
        )

        logger.info(
            f"[ThemeAnalyzer] 「{theme.title}」LLM响应: "
            f"{response.total_tokens}tokens, {response.latency_ms}ms"
        )

        # 解析JSON
        return _parse_analysis(theme, response.content)

    except Exception as e:
        logger.error(f"[ThemeAnalyzer] 「{theme.title}」分析失败: {e}")
        return None


def _collect_theme_stocks(theme: HotTheme, enriched_concepts: list[dict[str, Any]]) -> str:
    """收集主题关联的成分股并格式化为文本"""
    seen = set()
    lines = []

    # 找到与主题关联的概念板块
    source_names = set(theme.source_concepts)
    related = [c for c in enriched_concepts if c.get("name", "") in source_names]

    # 如果没有精确匹配，取所有概念的成分股
    if not related:
        related = enriched_concepts

    for concept in related:
        concept_name = concept.get("name", "")
        stocks = concept.get("stocks", [])
        for stock in stocks:
            key = stock.get("symbol", stock.get("code", ""))
            if key in seen:
                continue
            seen.add(key)
            name = stock.get("name", "?")
            symbol = stock.get("symbol", "?")
            change = stock.get("change_pct", 0)
            cap = stock.get("market_cap", 0)
            cap_str = f"{cap / 1e8:.0f}亿" if cap > 0 else "N/A"
            lines.append(
                f"- {name}({symbol}) | 涨跌{change:+.2f}% | 市值{cap_str} | 来自概念「{concept_name}」"
            )

    return "\n".join(lines) if lines else "暂无成分股数据"


def _parse_analysis(theme: HotTheme, content: str) -> AnalyzedTheme | None:
    """解析LLM返回的产业链分析JSON"""
    from src.agents.llm_enhance import _parse_json_response

    parsed = _parse_json_response(content)
    if parsed is None or not isinstance(parsed, dict):
        logger.warning(f"[ThemeAnalyzer] 「{theme.title}」无法解析JSON")
        return None

    try:
        upstream = _parse_chain_stocks(parsed.get("upstream", []), "upstream")
        midstream = _parse_chain_stocks(parsed.get("midstream", []), "midstream")
        downstream = _parse_chain_stocks(parsed.get("downstream", []), "downstream")

        chain = IndustryChain(
            upstream=tuple(upstream),
            midstream=tuple(midstream),
            downstream=tuple(downstream),
            value_flow=str(parsed.get("value_flow", "")),
        )

        return AnalyzedTheme(
            theme=theme,
            industry_chain=chain,
            investment_logic=str(parsed.get("investment_logic", "")),
            actionability=str(parsed.get("actionability", "medium")),
        )

    except Exception as e:
        logger.warning(f"[ThemeAnalyzer] 「{theme.title}」构建AnalyzedTheme失败: {e}")
        return None


def _parse_chain_stocks(items: list[dict[str, Any]], position: str) -> list[ChainStock]:
    """解析产业链某个位置的股票列表"""
    stocks = []
    if not isinstance(items, list):
        return stocks
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            stocks.append(ChainStock(
                symbol=str(item.get("symbol", "")),
                name=str(item.get("name", "")),
                chain_position=position,
                role=str(item.get("role", "")),
                reason=str(item.get("reason", "")),
            ))
        except Exception:
            pass
    return stocks


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
