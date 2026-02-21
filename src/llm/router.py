"""
LLM路由器

功能：
1. 统一接口适配多 LLM provider（OpenAI/Anthropic）
2. 成本追踪
3. 响应缓存（Phase 2 Redis）
4. 按Agent配置路由到不同模型
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger


@dataclass
class LLMResponse:
    """LLM响应"""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False


# 价格表（USD per 1K tokens）
PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-20250414": {"input": 0.0008, "output": 0.004},
    # 智谱 GLM（参考 https://open.bigmodel.cn/pricing）
    "glm-5": {"input": 0.001, "output": 0.001},  # 约 1元/百万tokens
    "glm-4-plus": {"input": 0.05, "output": 0.05},
    "glm-4-flash": {"input": 0.0001, "output": 0.0001},
    # Ollama 本地模型（免费）
    "qwen2.5:14b": {"input": 0.0, "output": 0.0},
    "qwen2.5:7b": {"input": 0.0, "output": 0.0},
    "qwen2.5:32b": {"input": 0.0, "output": 0.0},
}


def _calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """计算LLM调用成本"""
    pricing = PRICING.get(model, {"input": 0.001, "output": 0.003})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000


class LLMRouter:
    """LLM路由器"""

    def __init__(self) -> None:
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._call_count: int = 0
        self._models: dict[str, Any] = {}
        # Ollama 并发控制
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._queue_depth: dict[str, int] = {}
        self._queue_lock = threading.Lock()

    def _get_semaphore(self, provider: str) -> Optional[threading.Semaphore]:
        """获取provider对应的并发信号量，非Ollama返回None"""
        if provider != "ollama":
            return None
        if provider not in self._semaphores:
            from config.settings import get_settings
            n = get_settings().llm.ollama_max_concurrent
            self._semaphores[provider] = threading.Semaphore(n)
            self._queue_depth[provider] = 0
            logger.info(f"[LLM] Ollama 并发信号量初始化: max_concurrent={n}")
        return self._semaphores[provider]

    def _get_model(self, provider: str, model_name: str, **kwargs: Any) -> Any:
        """获取或创建LLM模型实例"""
        key = f"{provider}:{model_name}"
        if key not in self._models:
            if provider == "ollama":
                from langchain_openai import ChatOpenAI

                from config.settings import get_settings
                settings = get_settings()
                self._models[key] = ChatOpenAI(
                    model=model_name,
                    base_url=settings.llm.ollama_base_url,
                    api_key="ollama",  # Ollama 不需要真实 key
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    request_timeout=settings.llm.ollama_request_timeout,
                )
            elif provider == "openai":
                from langchain_openai import ChatOpenAI
                self._models[key] = ChatOpenAI(
                    model=model_name,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    request_timeout=kwargs.get("timeout", 60),
                )
            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                self._models[key] = ChatAnthropic(
                    model=model_name,
                    temperature=kwargs.get("temperature", 0.3),
                    max_tokens=kwargs.get("max_tokens", 2000),
                    timeout=kwargs.get("timeout", 60),
                )
            else:
                raise ValueError(f"不支持的provider: {provider}")
        return self._models[key]

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        agent_name: str = "",
        **kwargs: Any,
    ) -> LLMResponse:
        """调用LLM"""
        model = self._get_model(provider, model_name, **kwargs)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        sem = self._get_semaphore(provider)
        label = agent_name or model_name

        if sem is not None:
            with self._queue_lock:
                self._queue_depth[provider] += 1
                depth = self._queue_depth[provider]
            logger.info(f"[LLM] [{label}] 等待 Ollama 推理槽位 (队列深度: {depth})")
            sem.acquire()
            with self._queue_lock:
                self._queue_depth[provider] -= 1

        start = time.time()
        try:
            response = model.invoke(messages)
            latency_ms = int((time.time() - start) * 1000)

            # 提取token使用量
            usage = getattr(response, "usage_metadata", {}) or {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            cost = _calc_cost(model_name, input_tokens, output_tokens)

            # 累计统计
            self._total_cost += cost
            self._total_tokens += total_tokens
            self._call_count += 1

            logger.debug(
                f"[LLM] [{label}] {model_name} | {total_tokens} tokens | "
                f"${cost:.4f} | {latency_ms}ms"
            )

            return LLMResponse(
                content=response.content,
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"[LLM] [{label}] {model_name} 调用失败: {e}")
            raise
        finally:
            if sem is not None:
                sem.release()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self._total_cost, 4),
            "total_tokens": self._total_tokens,
            "call_count": self._call_count,
        }


# 全局实例
_router: Optional[LLMRouter] = None


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
