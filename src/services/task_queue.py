"""
异步任务队列

职责：
1. ThreadPoolExecutor 执行分析任务
2. SSE 事件广播（agent_completed 粒度）
3. 防止重复提交
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

if TYPE_CHECKING:
    from asyncio import Queue as AsyncQueue


class TaskStatus(str, Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskInfo:
    task_id: str
    symbol: str
    market: str = "A"
    stock_name: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: Optional[str] = None
    completed_agents: list[str] = field(default_factory=list)
    total_agents: int = 9
    result: Optional[dict[str, Any]] = None
    run_id: Optional[str] = None
    error: Optional[str] = None
    research_dir: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "symbol": self.symbol,
            "market": self.market,
            "stock_name": self.stock_name,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "completed_agents": self.completed_agents,
            "total_agents": self.total_agents,
            "run_id": self.run_id,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DuplicateTaskError(Exception):
    def __init__(self, symbol: str, existing_task_id: str):
        self.symbol = symbol
        self.existing_task_id = existing_task_id
        super().__init__(f"股票 {symbol} 正在分析中 (task_id: {existing_task_id})")


class AnalysisTaskQueue:
    """单例任务队列"""

    _instance: Optional[AnalysisTaskQueue] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = 2):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._tasks: dict[str, TaskInfo] = {}
        self._analyzing_stocks: dict[str, str] = {}  # symbol -> task_id
        self._futures: dict[str, Future] = {}
        self._subscribers: list[AsyncQueue] = []
        self._subscribers_lock = threading.Lock()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._data_lock = threading.RLock()
        self._max_history = 100
        self._initialized = True
        logger.info(f"[TaskQueue] 初始化完成，最大并发: {max_workers}")

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="analysis_task_",
            )
        return self._executor

    # ========== 任务提交与查询 ==========

    def submit_task(
        self, symbol: str, market: str = "A", research_dir: str | None = None,
    ) -> TaskInfo:
        with self._data_lock:
            if symbol in self._analyzing_stocks:
                raise DuplicateTaskError(symbol, self._analyzing_stocks[symbol])

            task_id = uuid.uuid4().hex[:12]
            task_info = TaskInfo(
                task_id=task_id,
                symbol=symbol,
                market=market,
                research_dir=research_dir,
                message="任务已加入队列",
            )
            self._tasks[task_id] = task_info
            self._analyzing_stocks[symbol] = task_id

            future = self.executor.submit(
                self._execute_task, task_id, symbol, market, research_dir,
            )
            self._futures[task_id] = future

            logger.info(f"[TaskQueue] 任务已提交: {symbol} -> {task_id}")

        self._broadcast_event("task_created", task_info.to_dict())
        return task_info

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._data_lock:
            return self._tasks.get(task_id)

    def list_pending_tasks(self) -> list[TaskInfo]:
        with self._data_lock:
            return [
                t
                for t in self._tasks.values()
                if t.status in (TaskStatus.PENDING, TaskStatus.COLLECTING, TaskStatus.ANALYZING)
            ]

    def list_all_tasks(self, limit: int = 50) -> list[TaskInfo]:
        with self._data_lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            return tasks[:limit]

    # ========== 任务执行 ==========

    def _execute_task(
        self, task_id: str, symbol: str, market: str, research_dir: str | None = None,
    ) -> None:
        # 更新状态: collecting
        with self._data_lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = TaskStatus.COLLECTING
            task.started_at = datetime.now()
            task.message = "正在采集数据..."
            task.progress = 5
        self._broadcast_event("task_started", task.to_dict())

        try:
            from src.services.analysis_service import collect_stock_data, run_analysis_streaming
            from src.data.storage.persist import persist_analysis

            # 1. 数据采集
            stock_data = collect_stock_data(symbol, market, research_dir=research_dir)

            with self._data_lock:
                task.stock_name = stock_data.name
                task.status = TaskStatus.ANALYZING
                task.message = "正在分析..."
                task.progress = 10
            self._broadcast_event("task_collecting_done", task.to_dict())

            # 2. 流式分析
            def on_agent_done(node_name: str, node_output: dict):
                # 跳过非 analyst 节点的内部更新
                skip_nodes = {"data_loader", "fusion", "report"}
                if node_name in skip_nodes:
                    if node_name == "fusion":
                        with self._data_lock:
                            task.message = "融合决策中..."
                            task.progress = 85
                        self._broadcast_event("task_fusing", task.to_dict())
                    elif node_name == "report":
                        with self._data_lock:
                            task.message = "生成研报..."
                            task.progress = 95
                        self._broadcast_event("task_reporting", task.to_dict())
                    return

                with self._data_lock:
                    if node_name not in task.completed_agents:
                        task.completed_agents.append(node_name)
                    done_count = len(task.completed_agents)
                    task.progress = 10 + int(done_count / task.total_agents * 70)
                    task.message = f"已完成 {done_count}/{task.total_agents} 个 Agent"

                # 提取 signal 信息
                agent_signal_data = None
                if "signals" in node_output and node_output["signals"]:
                    sig = node_output["signals"][0]
                    agent_signal_data = {
                        "agent_name": sig.agent_name,
                        "signal_score": sig.signal_score,
                        "confidence": sig.confidence,
                        "execution_time_ms": sig.execution_time_ms,
                    }

                self._broadcast_event(
                    "agent_completed",
                    {
                        "task_id": task_id,
                        "agent_name": node_name,
                        "completed_agents": list(task.completed_agents),
                        "progress": task.progress,
                        "signal": agent_signal_data,
                    },
                )

            result = run_analysis_streaming(stock_data, on_agent_done=on_agent_done)

            # 3. 持久化
            run_id = None
            try:
                run_id = persist_analysis(result)
                logger.info(f"[TaskQueue] 持久化成功: run_id={run_id}")
            except Exception as e:
                logger.error(f"[TaskQueue] 持久化失败: {type(e).__name__}: {e}", exc_info=True)

            # 4. 完成
            with self._data_lock:
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                task.completed_at = datetime.now()
                task.message = "分析完成"
                task.run_id = run_id
                task.result = self._extract_result_summary(result)
                if symbol in self._analyzing_stocks:
                    del self._analyzing_stocks[symbol]

            self._broadcast_event("task_completed", task.to_dict())
            logger.info(f"[TaskQueue] 任务完成: {task_id} ({symbol})")
            self._cleanup_old_tasks()

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[TaskQueue] 任务失败: {task_id} ({symbol}): {error_msg}")
            with self._data_lock:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error_msg[:500]
                task.message = f"分析失败: {error_msg[:100]}"
                if symbol in self._analyzing_stocks:
                    del self._analyzing_stocks[symbol]
            self._broadcast_event("task_failed", task.to_dict())
            self._cleanup_old_tasks()

    @staticmethod
    def _extract_result_summary(result: dict[str, Any]) -> dict[str, Any]:
        """提取结果摘要（不包含大量原始数据）"""
        fusion = result.get("fusion")
        signals = result.get("signals", [])
        summary: dict[str, Any] = {}

        if fusion:
            summary["fusion"] = {
                "final_score": fusion.final_score,
                "final_action": fusion.final_action,
                "confidence": fusion.confidence,
                "position_pct": fusion.position_pct,
                "stop_loss_pct": fusion.stop_loss_pct,
                "take_profit_pct": fusion.take_profit_pct,
                "reasoning": fusion.reasoning,
                "signal_summary": fusion.signal_summary,
                "conflicts": list(fusion.conflicts) if fusion.conflicts else [],
                "conflict_resolution": fusion.conflict_resolution,
                "market_regime": fusion.market_regime,
                "weights_used": fusion.weights_used,
                "bull_arguments": list(fusion.bull_arguments) if fusion.bull_arguments else [],
                "bear_arguments": list(fusion.bear_arguments) if fusion.bear_arguments else [],
                "divergence_points": list(fusion.divergence_points) if fusion.divergence_points else [],
                "target_prices": fusion.target_prices,
            }

        summary["signals"] = [
            {
                "agent_name": s.agent_name,
                "signal_score": s.signal_score,
                "confidence": s.confidence,
                "reasoning": s.reasoning,
                "key_factors": list(s.key_factors) if s.key_factors else [],
                "risks": list(s.risks) if s.risks else [],
                "llm_model": s.llm_model,
                "execution_time_ms": s.execution_time_ms,
            }
            for s in signals
        ]

        summary["report"] = result.get("report", "")
        summary["errors"] = result.get("errors", [])
        return summary

    def _cleanup_old_tasks(self) -> None:
        with self._data_lock:
            if len(self._tasks) <= self._max_history:
                return
            completed = sorted(
                [t for t in self._tasks.values() if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)],
                key=lambda t: t.created_at,
            )
            to_remove = len(self._tasks) - self._max_history
            for task in completed[:to_remove]:
                del self._tasks[task.task_id]
                self._futures.pop(task.task_id, None)

    # ========== SSE ==========

    def subscribe(self, queue: AsyncQueue) -> None:
        with self._subscribers_lock:
            self._subscribers.append(queue)
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            logger.debug(f"[TaskQueue] 新订阅者，当前: {len(self._subscribers)}")

    def unsubscribe(self, queue: AsyncQueue) -> None:
        with self._subscribers_lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)
                logger.debug(f"[TaskQueue] 订阅者离开，当前: {len(self._subscribers)}")

    def _broadcast_event(self, event_type: str, data: dict[str, Any]) -> None:
        event = {"type": event_type, "data": data}
        with self._subscribers_lock:
            subscribers = self._subscribers.copy()
            loop = self._main_loop
        if not subscribers or loop is None:
            return
        for queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception:
                pass

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


def get_task_queue() -> AnalysisTaskQueue:
    return AnalysisTaskQueue()
