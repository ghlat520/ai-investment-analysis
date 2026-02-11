"""
LLM增强层

为Agent提供统一的LLM调用接口：
- 加载prompt模板
- 调用LLM获取结构化JSON响应
- 解析并验证响应
- 无API Key或调用失败时优雅降级
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(template_name: str) -> str:
    """加载prompt模板"""
    path = _PROMPTS_DIR / template_name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_json_response(text: str) -> Optional[dict]:
    """从LLM响应中提取JSON

    支持：
    1. 纯JSON
    2. ```json ... ``` 包裹的JSON
    3. 文本中嵌入的 { ... } JSON
    """
    text = text.strip()

    # 尝试1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试2: 提取 ```json ... ``` 块
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 尝试3: 提取第一个 { ... } 块
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    # 尝试4: json_repair
    try:
        import json_repair
        return json_repair.loads(text)
    except Exception:
        pass

    return None


def _has_llm_key(provider: str) -> bool:
    """检查是否配置了LLM API Key"""
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        return bool(key) and key != "sk-xxx"
    elif provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        return bool(key) and key != "sk-ant-xxx"
    return False


def _load_agent_config(agent_name: str) -> dict[str, Any]:
    """加载Agent的LLM配置"""
    config_path = Path(__file__).parent.parent.parent / "config" / "agents.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            agents_cfg = yaml.safe_load(f).get("agents", {})
        return agents_cfg.get(agent_name, {})
    return {}


def llm_enhance(
    agent_name: str,
    template_name: str,
    template_vars: dict[str, str],
    code_score: int,
    code_reasoning: str,
    code_factors: list[str],
    code_risks: list[str],
) -> dict[str, Any]:
    """调用LLM增强Agent分析

    Args:
        agent_name: Agent名称（如 "technical"）
        template_name: prompt模板文件名（如 "technical.md"）
        template_vars: 模板变量
        code_score: 代码计算的基础分数
        code_reasoning: 代码生成的分析描述
        code_factors: 代码识别的关键因素
        code_risks: 代码识别的风险

    Returns:
        dict with keys:
        - enhanced: bool (是否使用了LLM增强)
        - score_adjustment: int (LLM建议的分数调整)
        - reasoning: str (LLM生成的深度分析，或code_reasoning)
        - extra_factors: list[str] (LLM发现的额外因素)
        - extra_risks: list[str] (LLM发现的额外风险)
        - llm_model: str
        - llm_tokens: int
        - llm_cost: float
        - llm_latency_ms: int
    """
    result = {
        "enhanced": False,
        "score_adjustment": 0,
        "reasoning": code_reasoning,
        "extra_factors": [],
        "extra_risks": [],
        "llm_model": "",
        "llm_tokens": 0,
        "llm_cost": 0.0,
        "llm_latency_ms": 0,
    }

    # 加载Agent配置
    cfg = _load_agent_config(agent_name)
    provider = cfg.get("llm_provider", "openai")
    model = cfg.get("llm_model", "gpt-4o-mini")
    temperature = cfg.get("temperature", 0.3)
    max_tokens = cfg.get("max_tokens", 2000)
    timeout = cfg.get("timeout", 30)

    # 检查API Key
    if not _has_llm_key(provider):
        logger.debug(f"[{agent_name}] LLM未配置({provider})，使用纯代码分析")
        return result

    # 加载并填充prompt
    try:
        template = _load_prompt(template_name)
        prompt = template.format(**template_vars)
    except Exception as e:
        logger.warning(f"[{agent_name}] Prompt加载失败: {e}")
        return result

    # 调用LLM
    try:
        from src.llm.router import get_llm_router

        router = get_llm_router()
        system_prompt = (
            f"你是{agent_name}分析Agent。"
            f"当前代码评分为{code_score}，你的任务是补充代码无法识别的定性分析。"
            f"必须输出严格的JSON格式。"
        )

        response = router.invoke(
            system_prompt=system_prompt,
            user_prompt=prompt,
            provider=provider,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        result["llm_model"] = response.model
        result["llm_tokens"] = response.total_tokens
        result["llm_cost"] = response.cost_usd
        result["llm_latency_ms"] = response.latency_ms

        # 解析JSON
        parsed = _parse_json_response(response.content)
        if parsed is None:
            logger.warning(f"[{agent_name}] LLM返回非JSON，使用原始文本作为reasoning")
            result["enhanced"] = True
            result["reasoning"] = response.content[:500]
            return result

        # 提取调整分数（限制范围）
        adj = parsed.get("score_adjustment", 0)
        if isinstance(adj, (int, float)):
            result["score_adjustment"] = max(-15, min(15, int(adj)))

        # 提取reasoning
        llm_reasoning = parsed.get("reasoning", "")
        if llm_reasoning:
            result["reasoning"] = llm_reasoning

        # 提取额外因素
        for key in ["key_patterns", "key_strengths", "key_levels"]:
            items = parsed.get(key, [])
            if isinstance(items, list):
                result["extra_factors"].extend([str(x) for x in items])

        # 提取额外风险
        for key in ["risks", "key_risks", "watch_items"]:
            items = parsed.get(key, [])
            if isinstance(items, list):
                result["extra_risks"].extend([str(x) for x in items])

        result["enhanced"] = True
        logger.info(
            f"[{agent_name}] LLM增强完成: adj={result['score_adjustment']:+d}, "
            f"{response.total_tokens}tokens, ${response.cost_usd:.4f}"
        )

    except Exception as e:
        logger.warning(f"[{agent_name}] LLM调用失败，回退到纯代码: {e}")

    return result
