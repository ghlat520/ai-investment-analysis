"""
主题提取Agent

输入：ranked_concepts + hot_stocks + policy_news
输出：3-5个 HotTheme（JSON）
作用：将相关概念板块归并为投资主题，过滤纯炒作热点
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from ..state import HotTheme

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def extract_themes(
    ranked_concepts: list[dict[str, Any]],
    hot_stocks: list[dict[str, Any]],
    cctv_news: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> list[HotTheme]:
    """从市场数据中提取投资主题

    Args:
        ranked_concepts: 排序后的热门概念（含成分股）
        hot_stocks: 人气榜股票
        cctv_news: 近期政策新闻
        config: hotspot.yaml 配置

    Returns:
        3-5个 HotTheme
    """
    config = config or {}
    min_score = config.get("min_relevance_score", 50)

    # 构建prompt变量
    concepts_text = _build_concepts_text(ranked_concepts)
    hot_stocks_text = _build_hot_stocks_text(hot_stocks[:20])
    news_text = _build_news_text(cctv_news[:30])

    # 加载并填充prompt
    template = (_PROMPTS_DIR / "theme_extract.md").read_text(encoding="utf-8")
    system_section, user_section = _parse_prompt_sections(template)
    user_prompt = user_section.format(
        concepts_text=concepts_text,
        hot_stocks_text=hot_stocks_text,
        news_text=news_text,
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
            agent_name="theme_extractor",
            temperature=0.3,
            max_tokens=3000,
            timeout=300,
        )

        logger.info(
            f"[ThemeExtractor] LLM响应: {response.total_tokens}tokens, "
            f"{response.latency_ms}ms"
        )

        # 解析JSON
        themes = _parse_themes(response.content, min_score)
        logger.info(f"[ThemeExtractor] 提取主题: {len(themes)}个")
        return themes

    except Exception as e:
        logger.error(f"[ThemeExtractor] LLM调用失败: {e}")
        return []


def _build_concepts_text(concepts: list[dict[str, Any]]) -> str:
    """构建概念板块文本"""
    lines = []
    for i, c in enumerate(concepts, 1):
        stocks = c.get("stocks", [])
        stock_names = ", ".join(s["name"] for s in stocks[:5]) if stocks else "无成分股数据"
        lines.append(
            f"{i}. **{c['name']}** (涨幅{c.get('change_pct', 0):+.2f}%, "
            f"龙头{c.get('leader_name', '?')}{c.get('leader_change', 0):+.2f}%)\n"
            f"   成分股: {stock_names}"
        )
    return "\n".join(lines)


def _build_hot_stocks_text(stocks: list[dict[str, Any]]) -> str:
    """构建人气股票文本"""
    lines = []
    for i, s in enumerate(stocks[:20], 1):
        name = s.get("股票名称", s.get("名称", "?"))
        code = s.get("股票代码", s.get("代码", "?"))
        lines.append(f"{i}. {name}({code})")
    return "\n".join(lines) if lines else "暂无数据"


def _build_news_text(news: list[dict[str, Any]]) -> str:
    """构建新闻文本"""
    lines = []
    for n in news[:30]:
        title = n.get("title", n.get("标题", ""))
        if title:
            lines.append(f"- {title}")
    return "\n".join(lines) if lines else "暂无政策新闻"


def _parse_themes(content: str, min_score: int) -> list[HotTheme]:
    """解析LLM返回的主题JSON"""
    from src.agents.llm_enhance import _parse_json_response

    # 尝试解析为数组
    parsed = _parse_json_response(content)
    if parsed is None:
        logger.warning("[ThemeExtractor] 无法解析JSON")
        return []

    # 如果是dict包裹的数组
    if isinstance(parsed, dict):
        for key in ("themes", "data", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    if not isinstance(parsed, list):
        return []

    themes = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        score = int(item.get("relevance_score", 0))
        if score < min_score:
            continue
        try:
            theme = HotTheme(
                title=str(item.get("title", "")),
                summary=str(item.get("summary", "")),
                relevance_score=score,
                source_concepts=tuple(item.get("source_concepts", [])),
                catalyst=str(item.get("catalyst", "")),
                timeline=str(item.get("timeline", "短期")),
                risk=str(item.get("risk", "")),
            )
            themes.append(theme)
        except Exception as e:
            logger.warning(f"[ThemeExtractor] 解析单个主题失败: {e}")

    # 按relevance_score降序
    themes.sort(key=lambda t: t.relevance_score, reverse=True)
    return themes


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
