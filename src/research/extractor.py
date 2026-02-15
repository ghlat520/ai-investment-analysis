"""
研报PDF解析与摘要提取

核心流程：
1. parse_research_pdfs()   — pdfplumber 提取全文，截断至 MAX_PDF_CHARS
2. extract_research_summaries() — 一次 LLM 调用，提取 7 个维度的结构化摘要
3. get_research_context()  — 各 Agent 调用入口，返回自己维度的摘要文本

设计决策：
- 一次 LLM 提取所有维度（vs 逐 Agent 提取）→ 节省 ~90s
- 每维度摘要 ≤3000 字，PDF 输入 ≤12000 字
- 无研报时返回空字符串，不影响现有流程
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger


MAX_PDF_CHARS = 12000

# Agent 名称到研报摘要维度的映射
_AGENT_TO_DIMENSION = {
    "fundamental": "fundamental",
    "valuation": "valuation",
    "sentiment": "sentiment",
    "moat": "moat",
    "business_model": "business_model",
    "industry": "industry",
    "supply_chain": "supply_chain",
}

_EXTRACTION_PROMPT = """你是专业的金融研报分析师。请从以下券商研报中提取关键信息，按7个维度整理摘要。

目标股票：{symbol}（{name}）

### 研报原文
{pdf_text}

### 提取要求
请为以下每个维度提取相关信息摘要（每个维度不超过500字）。如果研报中没有涉及某个维度，该维度留空字符串。

输出严格JSON格式：
```json
{{
  "fundamental": "基本面相关：营收、利润、ROE、毛利率等财务指标和趋势",
  "valuation": "估值相关：PE、PB、目标价、DCF估值、可比估值等",
  "sentiment": "市场情绪相关：机构评级、投资评级变化、市场关注度等",
  "moat": "护城河相关：竞争壁垒、品牌、专利、转换成本、网络效应等",
  "business_model": "商业模式相关：盈利模式、客户结构、收入来源、定价能力等",
  "industry": "行业分析相关：行业趋势、市场空间、竞争格局、政策影响等",
  "supply_chain": "产业链相关：上下游关系、供应商/客户集中度、产业链地位等"
}}
```"""


def parse_research_pdfs(dir_path: str | Path) -> str:
    """解析目录下所有PDF研报，合并提取全文

    Args:
        dir_path: PDF 文件所在目录

    Returns:
        合并后的纯文本（截断至 MAX_PDF_CHARS）
    """
    import pdfplumber

    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning(f"[研报] 目录不存在: {dir_path}")
        return ""

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"[研报] 目录为空: {dir_path}")
        return ""

    all_text: list[str] = []
    total_chars = 0

    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        all_text.append(text)
                        total_chars += len(text)
                        if total_chars >= MAX_PDF_CHARS:
                            break
            logger.info(f"[研报] 已解析: {pdf_file.name} ({total_chars}字)")
        except Exception as e:
            logger.warning(f"[研报] PDF解析失败 {pdf_file.name}: {e}")

        if total_chars >= MAX_PDF_CHARS:
            break

    merged = "\n\n".join(all_text)
    if len(merged) > MAX_PDF_CHARS:
        merged = merged[:MAX_PDF_CHARS] + "\n...(已截断)"
        logger.info(f"[研报] 文本截断至 {MAX_PDF_CHARS} 字")

    return merged


def extract_research_summaries(
    pdf_text: str,
    symbol: str,
    name: str,
) -> dict[str, str]:
    """一次 LLM 调用，从研报全文提取 7 个维度的结构化摘要

    Args:
        pdf_text: PDF 全文文本
        symbol: 股票代码
        name: 股票名称

    Returns:
        {dimension: summary_text} 字典，失败时返回空 dict
    """
    if not pdf_text.strip():
        return {}

    from src.agents.llm_enhance import _load_agent_config, _parse_json_response
    from src.llm.router import get_llm_router

    cfg = _load_agent_config("research_extractor")
    provider = cfg.get("llm_provider", "ollama")
    model = cfg.get("llm_model", "qwen2.5:14b")
    temperature = cfg.get("temperature", 0.2)
    max_tokens = cfg.get("max_tokens", 4000)
    timeout = cfg.get("timeout", 180)

    prompt = _EXTRACTION_PROMPT.format(
        symbol=symbol,
        name=name,
        pdf_text=pdf_text,
    )

    try:
        router = get_llm_router()
        response = router.invoke(
            system_prompt="你是专业金融研报分析师，擅长从研报中提取结构化信息。必须输出严格JSON。",
            user_prompt=prompt,
            provider=provider,
            model_name=model,
            agent_name="research_extractor",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        parsed = _parse_json_response(response.content)
        if parsed is None:
            logger.warning("[研报] LLM返回非JSON，研报摘要提取失败")
            return {}

        # 确保所有维度都是字符串
        result = {}
        for dim in _AGENT_TO_DIMENSION.values():
            val = parsed.get(dim, "")
            result[dim] = str(val) if val else ""

        non_empty = sum(1 for v in result.values() if v)
        logger.info(
            f"[研报] 摘要提取完成: {non_empty}/7 个维度有内容, "
            f"{response.total_tokens}tokens, ${response.cost_usd:.4f}"
        )
        return result

    except Exception as e:
        logger.warning(f"[研报] 摘要提取失败: {e}")
        return {}


def _build_online_research_context(stock: Any, agent_name: str) -> str:
    """从在线采集的研报数据构建上下文文本"""
    reports = getattr(stock, "info", {}).get("research_reports", {})
    if not reports:
        return ""

    parts = []

    if agent_name in ("sentiment", "valuation", "fundamental", "moat"):
        # 机构一致评级
        rating_dist = reports.get("rating_distribution", {})
        if rating_dist:
            total = reports.get("total_reports", sum(rating_dist.values()))
            coverage = reports.get("coverage_count", 0)
            dist_str = ", ".join(f"{k}{v}条" for k, v in rating_dist.items())
            parts.append(f"**机构覆盖**: {coverage}家券商, 近期{total}篇研报")
            parts.append(f"**评级分布**: {dist_str}")

        # 评级变动
        changes = reports.get("rating_changes", {})
        if changes:
            change_str = ", ".join(f"{k}{v}条" for k, v in changes.items())
            parts.append(f"**评级变动**: {change_str}")

        # 机构参与度
        participation = reports.get("latest_participation")
        if participation is not None:
            parts.append(f"**机构参与度**: {participation:.1f}%")

    if agent_name == "sentiment":
        # 研报标题（机构观点信号）
        titles = reports.get("recent_titles", [])
        if titles:
            parts.append("\n**近期研报标题**:")
            for t in titles[:8]:
                inst = t.get("institution", "")
                rating = t.get("rating", "")
                date = t.get("date", "")
                title = t.get("title", "")
                parts.append(f"- [{date}] {inst}({rating}): {title}")

    if agent_name == "valuation":
        # EPS预测（交叉验证）
        eps_forecasts = reports.get("eps_forecasts", [])
        if eps_forecasts:
            parts.append("\n**各券商EPS预测（交叉验证用）**:")
            parts.append("| 机构 | 日期 | " + " | ".join(
                k for k in eps_forecasts[0].keys() if k not in ("institution", "date")
            ) + " |")
            parts.append("|" + "---|" * (len(eps_forecasts[0])) )
            for f in eps_forecasts[:5]:
                eps_vals = " | ".join(
                    str(round(v, 3)) if isinstance(v, float) else str(v)
                    for k, v in f.items() if k not in ("institution", "date")
                )
                parts.append(f"| {f.get('institution', '')} | {f.get('date', '')} | {eps_vals} |")

    return "\n".join(parts) if parts else ""


def get_research_context(stock: Any, agent_name: str) -> str:
    """各 Agent 调用入口，获取自己维度的研报摘要

    Args:
        stock: StockData 实例
        agent_name: Agent 名称（如 "fundamental"）

    Returns:
        该维度的研报摘要文本，无研报时返回空字符串
    """
    parts = []

    # 1. PDF 研报摘要（手动上传）
    dimension = _AGENT_TO_DIMENSION.get(agent_name, "")
    if dimension:
        summaries = getattr(stock, "info", {}).get("research_summaries", {})
        if summaries:
            text = summaries.get(dimension, "")
            if text:
                parts.append(text)

    # 2. 在线研报数据（自动采集）
    online_ctx = _build_online_research_context(stock, agent_name)
    if online_ctx:
        parts.append(online_ctx)

    return "\n\n".join(parts) if parts else ""
