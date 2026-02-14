"""
热点分析任务队列

职责：
1. ThreadPoolExecutor 执行热点分析任务
2. SSE 事件广播（阶段粒度）
3. 同一天去重（只允许一个热点任务）
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

if TYPE_CHECKING:
    from asyncio import Queue as AsyncQueue


class HotspotTaskStatus(str, Enum):
    PENDING = "pending"
    COLLECTING = "collecting"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class HotspotTaskInfo:
    task_id: str
    status: HotspotTaskStatus = HotspotTaskStatus.PENDING
    progress: int = 0
    message: Optional[str] = None
    current_theme: int = 0
    total_themes: int = 0
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "current_theme": self.current_theme,
            "total_themes": self.total_themes,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DuplicateHotspotError(Exception):
    def __init__(self, existing_task_id: str):
        self.existing_task_id = existing_task_id
        super().__init__(f"今日热点分析已在运行 (task_id: {existing_task_id})")


class HotspotTaskQueue:
    """热点分析任务队列（单例）"""

    _instance: Optional[HotspotTaskQueue] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._executor: Optional[ThreadPoolExecutor] = None
        self._tasks: dict[str, HotspotTaskInfo] = {}
        self._today_task_id: Optional[str] = None
        self._today_date: Optional[date] = None
        self._futures: dict[str, Future] = {}
        self._subscribers: list[AsyncQueue] = []
        self._subscribers_lock = threading.Lock()
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._data_lock = threading.RLock()
        self._initialized = True
        logger.info("[HotspotQueue] 初始化完成")

    @property
    def executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="hotspot_task_",
            )
        return self._executor

    def submit_task(self) -> HotspotTaskInfo:
        """提交热点分析任务"""
        with self._data_lock:
            today = date.today()
            # 同一天去重
            if self._today_date == today and self._today_task_id:
                existing = self._tasks.get(self._today_task_id)
                if existing and existing.status in (
                    HotspotTaskStatus.PENDING,
                    HotspotTaskStatus.COLLECTING,
                    HotspotTaskStatus.EXTRACTING,
                    HotspotTaskStatus.ANALYZING,
                    HotspotTaskStatus.GENERATING,
                ):
                    raise DuplicateHotspotError(self._today_task_id)

            task_id = uuid.uuid4().hex[:12]
            task_info = HotspotTaskInfo(
                task_id=task_id,
                message="热点分析任务已加入队列",
            )
            self._tasks[task_id] = task_info
            self._today_task_id = task_id
            self._today_date = today

            future = self.executor.submit(self._execute_task, task_id)
            self._futures[task_id] = future

            logger.info(f"[HotspotQueue] 任务已提交: {task_id}")

        self._broadcast_event("hotspot_task_created", task_info.to_dict())
        return task_info

    def get_task(self, task_id: str) -> Optional[HotspotTaskInfo]:
        with self._data_lock:
            return self._tasks.get(task_id)

    def get_latest_task(self) -> Optional[HotspotTaskInfo]:
        """获取最新一次热点分析任务"""
        with self._data_lock:
            if not self._tasks:
                return None
            return max(self._tasks.values(), key=lambda t: t.created_at)

    def get_latest_completed(self) -> Optional[HotspotTaskInfo]:
        """获取最新已完成的热点分析"""
        with self._data_lock:
            completed = [
                t for t in self._tasks.values()
                if t.status == HotspotTaskStatus.COMPLETED
            ]
            if not completed:
                return None
            return max(completed, key=lambda t: t.completed_at or t.created_at)

    def list_tasks(self, limit: int = 20) -> list[HotspotTaskInfo]:
        with self._data_lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            return tasks[:limit]

    def _execute_task(self, task_id: str) -> None:
        """执行热点分析任务"""
        with self._data_lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = HotspotTaskStatus.COLLECTING
            task.started_at = datetime.now()
            task.message = "正在采集市场数据..."
            task.progress = 5
        self._broadcast_event("hotspot_task_started", task.to_dict())

        try:
            from src.services.hotspot_service import run_hotspot_streaming

            def on_stage_done(node_name: str, node_output: dict):
                with self._data_lock:
                    if node_name == "collect_and_rank":
                        task.status = HotspotTaskStatus.EXTRACTING
                        task.message = "正在提取投资主题..."
                        task.progress = 20
                        self._broadcast_event("hotspot_collecting_done", task.to_dict())

                    elif node_name == "extract_themes":
                        themes = node_output.get("themes", [])
                        task.total_themes = len(themes)
                        task.status = HotspotTaskStatus.ANALYZING
                        task.message = f"正在分析产业链 (0/{task.total_themes})..."
                        task.progress = 35
                        theme_data = [
                            {"title": t.title, "relevance_score": t.relevance_score}
                            for t in themes
                        ]
                        self._broadcast_event("hotspot_themes_extracted", {
                            **task.to_dict(),
                            "themes": theme_data,
                        })

                    elif node_name == "analyze_themes_batch":
                        analyzed = node_output.get("analyzed_themes", [])
                        task.current_theme = len(analyzed)
                        task.status = HotspotTaskStatus.GENERATING
                        task.message = "正在生成研报..."
                        task.progress = 75
                        self._broadcast_event("hotspot_themes_analyzed", task.to_dict())

                    elif node_name == "generate_briefing":
                        task.progress = 95
                        task.message = "研报生成完成"

            result = run_hotspot_streaming(on_stage_done=on_stage_done)

            # 提取结果摘要
            summary = self._extract_result_summary(result)

            with self._data_lock:
                task.status = HotspotTaskStatus.COMPLETED
                task.progress = 100
                task.completed_at = datetime.now()
                task.message = "热点分析完成"
                task.result = summary

            self._broadcast_event("hotspot_completed", {
                **task.to_dict(),
                "briefing_preview": (result.get("briefing", ""))[:500],
            })
            logger.info(f"[HotspotQueue] 任务完成: {task_id}")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[HotspotQueue] 任务失败: {task_id}: {error_msg}")
            with self._data_lock:
                task.status = HotspotTaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error_msg[:500]
                task.message = f"分析失败: {error_msg[:100]}"
            self._broadcast_event("hotspot_failed", task.to_dict())

    @staticmethod
    def _extract_result_summary(result: dict[str, Any]) -> dict[str, Any]:
        """提取结果摘要"""
        summary: dict[str, Any] = {}

        themes = result.get("themes", [])
        summary["themes"] = [
            {
                "title": t.title,
                "summary": t.summary,
                "relevance_score": t.relevance_score,
                "catalyst": t.catalyst,
                "timeline": t.timeline,
                "risk": t.risk,
            }
            for t in themes
        ]

        analyzed = result.get("analyzed_themes", [])
        summary["analyzed_themes"] = []
        for at in analyzed:
            chain = at.industry_chain
            theme_summary = {
                "title": at.theme.title,
                "investment_logic": at.investment_logic,
                "actionability": at.actionability,
                "upstream": [
                    {"symbol": s.symbol, "name": s.name, "role": s.role, "reason": s.reason}
                    for s in chain.upstream
                ],
                "midstream": [
                    {"symbol": s.symbol, "name": s.name, "role": s.role, "reason": s.reason}
                    for s in chain.midstream
                ],
                "downstream": [
                    {"symbol": s.symbol, "name": s.name, "role": s.role, "reason": s.reason}
                    for s in chain.downstream
                ],
                "value_flow": chain.value_flow,
            }
            summary["analyzed_themes"].append(theme_summary)

        summary["briefing"] = result.get("briefing", "")
        summary["errors"] = result.get("errors", [])
        summary["analysis_date"] = result.get("analysis_date", "")

        return summary

    # ========== SSE ==========

    def subscribe(self, queue: AsyncQueue) -> None:
        with self._subscribers_lock:
            self._subscribers.append(queue)
            try:
                self._main_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

    def unsubscribe(self, queue: AsyncQueue) -> None:
        with self._subscribers_lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

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


def get_hotspot_queue() -> HotspotTaskQueue:
    return HotspotTaskQueue()
