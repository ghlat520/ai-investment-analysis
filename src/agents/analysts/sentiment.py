"""
情绪面分析Agent

分析新闻舆情、市场情绪、利好/利空事件。
数据来源：akshare stock_news_em + 可选搜索引擎。

代码/LLM = 30/70: 代码做基础统计，LLM做核心情感分析和事件影响评估。
"""

from __future__ import annotations

import re
import time
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any

from loguru import logger

from ..state import AgentSignal, StockData


# ─── 关键词情绪字典 ──────────────────────────────────────


_POSITIVE_KEYWORDS = [
    "业绩增长", "净利润增", "营收增", "超预期", "创新高", "突破",
    "获批", "中标", "签约", "战略合作", "回购", "增持", "分红",
    "利好", "涨停", "大涨", "新高", "上调", "评级",
    "景气", "龙头", "高增长", "翻倍", "盈利",
    "机构看好", "买入评级", "目标价上调",
]

_NEGATIVE_KEYWORDS = [
    "业绩下滑", "净利润降", "营收降", "亏损", "不及预期",
    "减持", "质押", "违规", "处罚", "诉讼", "立案",
    "利空", "跌停", "大跌", "下调", "退市", "ST",
    "暴雷", "风险", "预亏", "计提", "商誉减值",
    "被罚", "限售", "解禁", "停牌",
    "做空", "卖出评级", "目标价下调",
]


# ─── 评分函数 ──────────────────────────────────────────────


def _score_keyword_sentiment(news: list[dict[str, Any]]) -> tuple[int, str, list[str]]:
    """关键词情感评分

    统计新闻标题/内容中利好/利空关键词出现频率。
    """
    if not news:
        return 0, "无新闻数据", []

    pos_count = 0
    neg_count = 0
    found_keywords = []

    for item in news:
        text = str(item.get("title", "")) + str(item.get("content", ""))
        for kw in _POSITIVE_KEYWORDS:
            if kw in text:
                pos_count += 1
                found_keywords.append(f"利好:{kw}")
        for kw in _NEGATIVE_KEYWORDS:
            if kw in text:
                neg_count += 1
                found_keywords.append(f"利空:{kw}")

    total = pos_count + neg_count
    if total == 0:
        return 0, "未检测到明显情绪关键词", found_keywords

    # 情绪比例 → 评分
    sentiment_ratio = (pos_count - neg_count) / total  # -1 ~ +1
    score = int(sentiment_ratio * 25)  # -25 ~ +25

    desc = f"利好关键词{pos_count}个, 利空{neg_count}个"
    return score, desc, found_keywords


def _score_news_volume(news: list[dict[str, Any]]) -> tuple[int, str]:
    """新闻数量评分

    异常多的新闻通常意味着市场关注度高（短期可能有大波动）。
    过少的新闻可能意味着被市场忽视。
    """
    count = len(news)
    if count == 0:
        return 0, "无新闻"
    if count >= 20:
        return 5, f"新闻密集({count}条), 市场高度关注"
    if count >= 10:
        return 3, f"新闻较多({count}条), 市场关注"
    if count >= 3:
        return 0, f"新闻正常({count}条)"
    return -3, f"新闻稀少({count}条), 市场关注度低"


def _score_news_recency(news: list[dict[str, Any]]) -> tuple[int, str]:
    """新闻时效性评分

    有今日/昨日新闻 → 信息及时
    最近新闻超过7天前 → 信息陈旧
    """
    if not news:
        return 0, "无新闻"

    today = date.today()
    recent_count = 0

    for item in news:
        dt_str = item.get("datetime", item.get("date", ""))
        if not dt_str:
            continue
        try:
            if isinstance(dt_str, str):
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).date()
            elif hasattr(dt_str, "date"):
                dt = dt_str.date() if hasattr(dt_str, "date") else dt_str
            else:
                continue
            if (today - dt).days <= 3:
                recent_count += 1
        except (ValueError, TypeError):
            continue

    if recent_count >= 5:
        return 5, f"近3日{recent_count}条新闻（信息充足）"
    if recent_count >= 2:
        return 3, f"近3日{recent_count}条新闻"
    if recent_count >= 1:
        return 0, f"近3日{recent_count}条新闻（信息有限）"
    return -5, "近3日无新闻（信息滞后）"


def _detect_major_events(news: list[dict[str, Any]]) -> tuple[int, str, list[str]]:
    """重大事件检测

    检测：业绩预告、重大合同、股权变动、监管处罚等
    """
    if not news:
        return 0, "无重大事件", []

    events = []
    event_score = 0

    # 重大正面事件
    positive_events = {
        r"业绩(预增|预盈|大增|翻倍)": ("业绩大增", 15),
        r"(中标|签约|合同).*(亿|百亿|千万)": ("重大合同", 10),
        r"(获批|批准|通过).*(新药|专利|资质)": ("重大获批", 10),
        r"(回购|增持).*股": ("股票回购/增持", 8),
        r"(战略|重大).*合作": ("战略合作", 5),
    }

    # 重大负面事件
    negative_events = {
        r"业绩(预亏|预减|下滑|暴雷)": ("业绩暴雷", -20),
        r"(减持|套现).*(亿|大量|清仓)": ("大额减持", -15),
        r"(立案|调查|违规|处罚)": ("监管风险", -15),
        r"(退市|摘牌|暂停上市)": ("退市风险", -25),
        r"商誉(减值|计提)": ("商誉减值", -10),
    }

    for item in news:
        text = str(item.get("title", "")) + str(item.get("content", ""))
        for pattern, (event_name, score) in positive_events.items():
            if re.search(pattern, text):
                events.append(event_name)
                event_score += score
        for pattern, (event_name, score) in negative_events.items():
            if re.search(pattern, text):
                events.append(event_name)
                event_score += score

    # 去重
    events = list(dict.fromkeys(events))
    event_score = max(-30, min(30, event_score))

    if events:
        desc = "重大事件: " + ", ".join(events[:5])
    else:
        desc = "无重大事件"
    return event_score, desc, events


# ─── 摘要构建 ──────────────────────────────────────────────


def _build_news_summary(news: list[dict[str, Any]], component_scores: dict) -> str:
    """构建新闻摘要供LLM阅读"""
    lines = [f"新闻总数: {len(news)}条"]

    for name, score in component_scores.items():
        lines.append(f"{name}评分: {score:+d}")

    lines.append("")
    lines.append("--- 最新新闻标题 ---")
    for item in news[:15]:
        title = item.get("title", "无标题")
        dt = item.get("datetime", item.get("date", ""))
        source = item.get("source", "")
        if dt and isinstance(dt, str) and len(dt) > 10:
            dt = dt[:10]
        lines.append(f"- [{source}] {title} ({dt})")

    return "\n".join(lines)


def _build_news_content(news: list[dict[str, Any]], n: int = 8) -> str:
    """构建新闻内容供LLM深度分析"""
    lines = []
    for i, item in enumerate(news[:n], 1):
        title = item.get("title", "无标题")
        content = item.get("content", "")
        dt = item.get("datetime", item.get("date", ""))
        source = item.get("source", "")

        # 截断过长内容
        if len(content) > 300:
            content = content[:300] + "..."

        lines.append(f"### 新闻{i}")
        lines.append(f"**标题**: {title}")
        if source:
            lines.append(f"**来源**: {source}")
        if dt:
            lines.append(f"**日期**: {dt}")
        if content:
            lines.append(f"**内容**: {content}")
        lines.append("")

    return "\n".join(lines) if lines else "暂无新闻内容"


# ─── 主函数 ────────────────────────────────────────────────


def analyze_sentiment(stock: StockData) -> AgentSignal:
    """情绪面分析主函数

    30%代码基础评分（关键词+数量+时效+事件） + 70%LLM深度情感分析。
    LLM不可用时自动降级为纯代码分析。
    """
    start = time.time()

    news = stock.news
    if not news:
        return AgentSignal(
            agent_name="sentiment",
            signal_score=0,
            confidence=0.0,
            reasoning="无新闻数据，无法进行情绪面分析",
            data_quality=0.0,
        )

    # 各维度评分
    scores = []
    factors = []

    kw_score, kw_desc, found_kws = _score_keyword_sentiment(news)
    scores.append(kw_score)
    factors.append(f"关键词情感: {kw_desc}")

    vol_score, vol_desc = _score_news_volume(news)
    scores.append(vol_score)
    factors.append(f"新闻数量: {vol_desc}")

    recency_score, recency_desc = _score_news_recency(news)
    scores.append(recency_score)
    factors.append(f"时效性: {recency_desc}")

    event_score, event_desc, events = _detect_major_events(news)
    scores.append(event_score)
    factors.append(f"重大事件: {event_desc}")

    # 代码评分（-100 ~ +100）
    total = sum(scores)
    # 满分约65 (25+5+5+30)，归一化到100
    code_score = max(-100, min(100, int(total * 100 / 65)))

    component_scores = {
        "keyword_sentiment": kw_score,
        "news_volume": vol_score,
        "news_recency": recency_score,
        "major_events": event_score,
    }

    # 风险提示
    risks = []
    if event_score < -10:
        risks.append("检测到重大负面事件")
    if kw_score < -15:
        risks.append("利空关键词密集")

    # --- LLM增强（核心 - 情绪面高度依赖LLM）---
    from ..llm_enhance import llm_enhance

    llm_result = llm_enhance(
        agent_name="sentiment",
        template_name="sentiment.md",
        template_vars={
            "symbol": stock.symbol,
            "name": stock.name,
            "analysis_date": date.today().isoformat(),
            "news_summary": _build_news_summary(news, component_scores),
            "news_content": _build_news_content(news),
        },
        code_score=code_score,
        code_reasoning=f"情绪面代码评分{code_score}。" + "；".join(factors),
        code_factors=factors,
        code_risks=risks,
    )

    # 合并LLM结果
    signal_score = max(-100, min(100, code_score + llm_result["score_adjustment"]))
    reasoning = llm_result["reasoning"]
    all_factors = factors + llm_result["extra_factors"]
    all_risks = list(risks) + llm_result["extra_risks"]

    # 置信度
    news_count = len(news)
    confidence = min(0.8, news_count / 15)  # 新闻越多越可靠，但上限0.8（新闻不如硬数据可靠）
    if llm_result["enhanced"]:
        confidence = min(1.0, confidence + 0.15)  # LLM对情绪面贡献大

    elapsed_ms = int((time.time() - start) * 1000)

    return AgentSignal(
        agent_name="sentiment",
        signal_score=signal_score,
        confidence=round(confidence, 3),
        reasoning=reasoning,
        key_factors=tuple(all_factors),
        risks=tuple(all_risks),
        data_quality=round(min(1.0, news_count / 10), 2),
        llm_model=llm_result["llm_model"],
        llm_tokens_used=llm_result["llm_tokens"],
        llm_cost_usd=llm_result["llm_cost"],
        metadata={
            "total_raw_score": total,
            "code_score": code_score,
            "llm_adjustment": llm_result["score_adjustment"],
            "llm_enhanced": llm_result["enhanced"],
            "component_scores": component_scores,
            "news_count": news_count,
            "found_keywords": found_kws[:20],
            "major_events": events,
        },
        execution_time_ms=elapsed_ms,
    )
