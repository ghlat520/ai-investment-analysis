"""
热点分析端点

POST /analyze       - 触发热点分析
GET  /status/{id}   - 查询任务状态
GET  /stream        - SSE 实时推送
GET  /latest        - 最新一期结果
GET  /history       - 历史列表
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

from src.services.hotspot_task_queue import (
    DuplicateHotspotError,
    get_hotspot_queue,
)

from ..schemas.hotspot import (
    BriefingResponse,
    DuplicateHotspotErrorResponse,
    HotspotHistoryResponse,
    HotspotTaskAccepted,
    HotspotTaskInfoResponse,
)

router = APIRouter()


@router.post(
    "/analyze",
    responses={
        202: {"model": HotspotTaskAccepted},
        409: {"model": DuplicateHotspotErrorResponse},
    },
    summary="触发热点分析",
)
def trigger_hotspot_analysis() -> JSONResponse:
    queue = get_hotspot_queue()
    try:
        task_info = queue.submit_task()
        return JSONResponse(
            status_code=202,
            content=HotspotTaskAccepted(
                task_id=task_info.task_id,
                status="pending",
                message="热点分析任务已加入队列",
            ).model_dump(),
        )
    except DuplicateHotspotError as e:
        return JSONResponse(
            status_code=409,
            content=DuplicateHotspotErrorResponse(
                message=str(e),
                existing_task_id=e.existing_task_id,
            ).model_dump(),
        )


@router.get("/status/{task_id}", summary="查询热点分析任务状态")
def get_hotspot_status(task_id: str):
    queue = get_hotspot_queue()
    task = queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    resp = task.to_dict()
    if task.result:
        resp["result"] = task.result
    return resp


@router.get("/latest", summary="获取最新热点分析结果")
def get_latest_hotspot():
    queue = get_hotspot_queue()
    task = queue.get_latest_completed()
    if not task:
        raise HTTPException(status_code=404, detail="暂无热点分析结果")
    resp = task.to_dict()
    if task.result:
        resp["result"] = task.result
    return resp


@router.get("/history", response_model=HotspotHistoryResponse, summary="热点分析历史")
def get_hotspot_history(limit: int = Query(20, ge=1, le=100)):
    queue = get_hotspot_queue()
    tasks = queue.list_tasks(limit=limit)
    return HotspotHistoryResponse(
        total=len(tasks),
        tasks=[
            HotspotTaskInfoResponse(
                task_id=t.task_id,
                status=t.status.value,
                progress=t.progress,
                message=t.message,
                current_theme=t.current_theme,
                total_themes=t.total_themes,
                error=t.error,
                created_at=t.created_at.isoformat(),
                started_at=t.started_at.isoformat() if t.started_at else None,
                completed_at=t.completed_at.isoformat() if t.completed_at else None,
            )
            for t in tasks
        ],
    )


@router.get(
    "/stream",
    summary="SSE 实时推送热点分析进度",
    description="Server-Sent Events 流式推送热点分析状态",
)
async def hotspot_stream():
    async def event_generator():
        queue = get_hotspot_queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        yield _format_sse("connected", {"message": "Connected to hotspot stream"})

        # 发送当前进行中的任务
        latest = queue.get_latest_task()
        if latest and latest.status.value not in ("completed", "failed"):
            yield _format_sse("hotspot_task_created", latest.to_dict())

        queue.subscribe(event_queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=30)
                    yield _format_sse(event["type"], event["data"])
                except asyncio.TimeoutError:
                    yield _format_sse("heartbeat", {"ts": datetime.now().isoformat()})
        except asyncio.CancelledError:
            pass
        finally:
            queue.unsubscribe(event_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _format_sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
