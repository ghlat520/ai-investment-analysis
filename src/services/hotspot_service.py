"""
热点分析服务层

提供阻塞式和流式两种执行方式，供 CLI 和 API 复用。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger


def run_hotspot(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """运行完整热点分析流水线（阻塞式）

    Returns:
        LangGraph 最终 state dict
    """
    from src.hotspot.graph import compile_hotspot_graph

    graph = compile_hotspot_graph()
    return graph.invoke({})


def run_hotspot_streaming(
    on_stage_done: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """运行热点分析流水线（流式，逐阶段回调）

    Args:
        on_stage_done: 回调函数 (node_name, node_output)

    Returns:
        最终累积 state dict
    """
    from src.hotspot.graph import compile_hotspot_graph

    graph = compile_hotspot_graph()
    final_state: dict[str, Any] = {}

    for event in graph.stream({}):
        for node_name, node_output in event.items():
            # 累积到 final_state
            if "market_snapshot" in node_output:
                final_state["market_snapshot"] = node_output["market_snapshot"]
            if "ranked_concepts" in node_output:
                final_state["ranked_concepts"] = node_output["ranked_concepts"]
            if "themes" in node_output:
                final_state.setdefault("themes", [])
                final_state["themes"].extend(node_output["themes"])
            if "analyzed_themes" in node_output:
                final_state.setdefault("analyzed_themes", [])
                final_state["analyzed_themes"].extend(node_output["analyzed_themes"])
            if "briefing" in node_output:
                final_state["briefing"] = node_output["briefing"]
            if "analysis_date" in node_output:
                final_state["analysis_date"] = node_output["analysis_date"]
            if "errors" in node_output:
                final_state.setdefault("errors", [])
                final_state["errors"].extend(node_output["errors"])

            if on_stage_done:
                on_stage_done(node_name, node_output)

    return final_state
