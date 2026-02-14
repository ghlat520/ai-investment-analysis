"""
热点分析共享状态定义

数据结构：
- HotTheme: 提炼后的投资主题
- ChainStock: 产业链中的个股
- IndustryChain: 上中下游产业链
- AnalyzedTheme: 完成分析的主题（含产业链+投资逻辑）
- HotspotState: LangGraph 共享状态
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict


@dataclass(frozen=True)
class HotTheme:
    """提炼后的投资主题"""

    title: str              # "小米YU7大定火爆，汽车零部件链受益"
    summary: str            # 2-3句核心逻辑
    relevance_score: int    # 1-100
    source_concepts: tuple[str, ...]  # 关联的概念板块
    catalyst: str           # 催化剂事件
    timeline: str           # 短期/中期/长期
    risk: str               # 主要风险


@dataclass(frozen=True)
class ChainStock:
    """产业链中的个股"""

    symbol: str             # "002460.SZ"
    name: str               # "赣锋锂业"
    chain_position: str     # upstream / midstream / downstream
    role: str               # "锂矿开采"
    reason: str             # "固态电池需求拉动锂盐价格"


@dataclass(frozen=True)
class IndustryChain:
    """上中下游产业链"""

    upstream: tuple[ChainStock, ...]
    midstream: tuple[ChainStock, ...]
    downstream: tuple[ChainStock, ...]
    value_flow: str         # 价值传导逻辑


@dataclass(frozen=True)
class AnalyzedTheme:
    """完成分析的主题"""

    theme: HotTheme
    industry_chain: IndustryChain
    investment_logic: str   # 投资逻辑总结
    actionability: str      # high/medium/low


def _merge_errors(existing: list[str], new: list[str]) -> list[str]:
    return existing + new


def _merge_themes(existing: list[HotTheme], new: list[HotTheme]) -> list[HotTheme]:
    return existing + new


def _merge_analyzed(existing: list[AnalyzedTheme], new: list[AnalyzedTheme]) -> list[AnalyzedTheme]:
    return existing + new


class HotspotState(TypedDict, total=False):
    """LangGraph 热点分析流程的共享状态"""

    # 输入：市场快照数据（AKShare采集）
    market_snapshot: dict[str, Any]

    # 排序后的热门概念（代码排序，非LLM）
    ranked_concepts: list[dict[str, Any]]

    # LLM提取的投资主题
    themes: Annotated[list[HotTheme], _merge_themes]

    # 完成产业链分析的主题
    analyzed_themes: Annotated[list[AnalyzedTheme], _merge_analyzed]

    # 最终markdown研报
    briefing: str

    # 运行标识
    analysis_date: str

    # 错误收集
    errors: Annotated[list[str], _merge_errors]
