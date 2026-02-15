"""
分析端点

POST /analyze       - 提交分析任务
GET  /status/{id}   - 查询任务状态
GET  /tasks         - 任务列表
GET  /tasks/stream  - SSE 实时推送
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.services.task_queue import DuplicateTaskError, get_task_queue

from ..schemas.analysis import (
    AnalyzeRequest,
    DuplicateTaskErrorResponse,
    TaskAccepted,
    TaskInfoResponse,
    TaskListResponse,
)

router = APIRouter()


@router.get("/search", summary="搜索股票（支持模糊输入）")
def search_stocks(q: str = Query("", min_length=1, description="搜索关键词，如 '鼎龙' '港股智谱' '300054' 'BABA'")):
    from src.services.stock_search import search_stocks as do_search

    results = do_search(q, limit=10)
    return {
        "query": q,
        "results": [
            {"symbol": r.symbol, "name": r.name, "market": r.market}
            for r in results
        ],
    }


@router.post(
    "/analyze",
    responses={
        202: {"model": TaskAccepted},
        409: {"model": DuplicateTaskErrorResponse},
    },
    summary="提交股票分析任务",
)
def trigger_analysis(request: AnalyzeRequest) -> JSONResponse:
    task_queue = get_task_queue()
    research_dir = _get_research_dir(request.symbol) if request.use_research else None
    try:
        task_info = task_queue.submit_task(
            symbol=request.symbol,
            market=request.market,
            research_dir=research_dir,
            auto_research=request.auto_research,
        )
        return JSONResponse(
            status_code=202,
            content=TaskAccepted(
                task_id=task_info.task_id,
                status="pending",
                message=f"分析任务已加入队列: {request.symbol}",
            ).model_dump(),
        )
    except DuplicateTaskError as e:
        return JSONResponse(
            status_code=409,
            content=DuplicateTaskErrorResponse(
                message=str(e),
                symbol=e.symbol,
                existing_task_id=e.existing_task_id,
            ).model_dump(),
        )


@router.get("/tasks", response_model=TaskListResponse, summary="获取任务列表")
def get_task_list(
    limit: int = Query(20, ge=1, le=100),
):
    task_queue = get_task_queue()
    all_tasks = task_queue.list_all_tasks(limit=limit)
    return TaskListResponse(
        total=len(all_tasks),
        tasks=[
            TaskInfoResponse(
                task_id=t.task_id,
                symbol=t.symbol,
                market=t.market,
                stock_name=t.stock_name,
                status=t.status.value,
                progress=t.progress,
                message=t.message,
                completed_agents=t.completed_agents,
                total_agents=t.total_agents,
                run_id=t.run_id,
                error=t.error,
                created_at=t.created_at.isoformat(),
                started_at=t.started_at.isoformat() if t.started_at else None,
                completed_at=t.completed_at.isoformat() if t.completed_at else None,
            )
            for t in all_tasks
        ],
    )


@router.get("/status/{task_id}", summary="查询任务状态")
def get_task_status(task_id: str):
    task_queue = get_task_queue()
    task = task_queue.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    resp = task.to_dict()
    # 完成的任务附带结果摘要
    if task.result:
        resp["result"] = task.result
    return resp


@router.get(
    "/tasks/stream",
    summary="SSE 实时推送",
    description="Server-Sent Events 流式推送任务状态",
)
async def task_stream():
    async def event_generator():
        task_queue = get_task_queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        yield _format_sse("connected", {"message": "Connected to task stream"})

        # 发送当前进行中的任务
        for task in task_queue.list_pending_tasks():
            yield _format_sse("task_created", task.to_dict())

        task_queue.subscribe(event_queue)
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
            task_queue.unsubscribe(event_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_RESEARCH_BASE_DIR = "/tmp/ai-invest-research"


def _get_research_dir(symbol: str) -> str | None:
    """获取股票的研报目录路径（如果存在）"""
    import os

    path = os.path.join(_RESEARCH_BASE_DIR, symbol.replace(".", "_"))
    return path if os.path.isdir(path) else None


@router.post("/research/upload", summary="上传研报PDF")
async def upload_research(
    symbol: str = Query(..., description="股票代码"),
    files: list[UploadFile] = File(..., description="研报PDF文件"),
) -> dict:
    """上传研报PDF到临时目录，供后续分析使用"""
    import os

    upload_dir = os.path.join(_RESEARCH_BASE_DIR, symbol.replace(".", "_"))
    os.makedirs(upload_dir, exist_ok=True)

    saved = []
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            continue
        filepath = os.path.join(upload_dir, f.filename)
        content = await f.read()
        with open(filepath, "wb") as fout:
            fout.write(content)
        saved.append(f.filename)

    return {
        "symbol": symbol,
        "uploaded": saved,
        "research_dir": upload_dir,
    }


def _format_sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
