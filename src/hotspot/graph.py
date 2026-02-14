"""
LangGraph DAG — 热点分析流水线

流程:
    collect_and_rank (代码, ~30s)
        ↓
    extract_themes (LLM, ~6min)
        ↓
    analyze_themes_batch (LLM × N, ~4min/主题, 顺序执行)
        ↓
    generate_briefing (LLM, ~7min)
        ↓
      END

顺序执行原因：Ollama单实例无法高效并行LLM调用。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import yaml
from langgraph.graph import END, StateGraph
from loguru import logger

from .state import AnalyzedTheme, HotspotState


def _load_hotspot_config() -> dict[str, Any]:
    """加载 config/hotspot.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "hotspot.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f).get("hotspot", {})
    return {}


# ---------- Node functions ----------


def collect_and_rank(state: dict[str, Any]) -> dict[str, Any]:
    """采集市场数据并排序热门概念（纯代码，无LLM）"""
    from .collectors.market_data import (
        collect_market_snapshot,
        enrich_concepts_with_stocks,
        rank_hot_concepts,
    )

    config = _load_hotspot_config()

    logger.info("[Hotspot] 开始采集市场数据...")
    snapshot = collect_market_snapshot()

    logger.info("[Hotspot] 排序热门概念...")
    top_n = config.get("top_concepts", 8)
    ranked = rank_hot_concepts(snapshot, top_n=top_n)

    logger.info("[Hotspot] 拉取成分股...")
    stocks_per = config.get("stocks_per_concept", 10)
    enriched = enrich_concepts_with_stocks(ranked, stocks_per_concept=stocks_per)

    return {
        "market_snapshot": snapshot,
        "ranked_concepts": enriched,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
    }


def extract_themes(state: dict[str, Any]) -> dict[str, Any]:
    """从市场数据中提取投资主题（LLM）"""
    from .agents.theme_extractor import extract_themes as do_extract

    config = _load_hotspot_config()
    snapshot = state.get("market_snapshot", {})
    ranked_concepts = state.get("ranked_concepts", [])

    logger.info("[Hotspot] 提取投资主题...")
    themes = do_extract(
        ranked_concepts=ranked_concepts,
        hot_stocks=snapshot.get("hot_stocks", []),
        cctv_news=snapshot.get("cctv_news", []),
        config=config,
    )

    # 限制主题数量
    max_themes = config.get("max_themes", 5)
    themes = themes[:max_themes]
    logger.info(f"[Hotspot] 提取到 {len(themes)} 个主题")

    return {"themes": themes}


def analyze_themes_batch(state: dict[str, Any]) -> dict[str, Any]:
    """逐主题进行产业链分析（LLM × N，顺序执行）"""
    from .agents.theme_analyzer import analyze_theme

    config = _load_hotspot_config()
    themes = state.get("themes", [])
    enriched_concepts = state.get("ranked_concepts", [])

    analyzed = []
    errors = []

    for i, theme in enumerate(themes, 1):
        logger.info(f"[Hotspot] 分析主题 {i}/{len(themes)}：{theme.title}")
        try:
            result = analyze_theme(theme, enriched_concepts, config)
            if result:
                analyzed.append(result)
                logger.info(f"[Hotspot] 主题「{theme.title}」分析完成")
            else:
                errors.append(f"主题「{theme.title}」分析返回空结果")
        except Exception as e:
            errors.append(f"主题「{theme.title}」分析失败: {e}")
            logger.error(f"[Hotspot] 主题「{theme.title}」分析失败: {e}")

    return {"analyzed_themes": analyzed, "errors": errors}


def generate_briefing_node(state: dict[str, Any]) -> dict[str, Any]:
    """生成盘前研报（LLM）"""
    from .agents.briefing_generator import generate_briefing

    config = _load_hotspot_config()
    analyzed_themes = state.get("analyzed_themes", [])
    analysis_date = state.get("analysis_date", datetime.now().strftime("%Y-%m-%d"))

    logger.info("[Hotspot] 生成盘前研报...")
    briefing = generate_briefing(analyzed_themes, analysis_date, config)
    logger.info(f"[Hotspot] 研报生成完成，长度: {len(briefing)} 字符")

    return {"briefing": briefing}


# ---------- Graph construction ----------


def build_hotspot_graph() -> StateGraph:
    """构建热点分析流程图

    结构:
        collect_and_rank
            ↓
        extract_themes
            ↓
        analyze_themes_batch
            ↓
        generate_briefing
            ↓
          END
    """
    graph = StateGraph(HotspotState)

    graph.add_node("collect_and_rank", collect_and_rank)
    graph.add_node("extract_themes", extract_themes)
    graph.add_node("analyze_themes_batch", analyze_themes_batch)
    graph.add_node("generate_briefing", generate_briefing_node)

    graph.set_entry_point("collect_and_rank")
    graph.add_edge("collect_and_rank", "extract_themes")
    graph.add_edge("extract_themes", "analyze_themes_batch")
    graph.add_edge("analyze_themes_batch", "generate_briefing")
    graph.add_edge("generate_briefing", END)

    return graph


def compile_hotspot_graph():
    """编译并返回可执行的热点分析图"""
    graph = build_hotspot_graph()
    return graph.compile()
