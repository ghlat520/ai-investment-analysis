#!/usr/bin/env bash
#
# AI 投研助手 - 一键启动脚本
#
# 用法：
#   ./scripts/start.sh          # 生产模式（前端用 dist 静态文件）
#   ./scripts/start.sh --dev    # 开发模式（前后端分开运行，支持热更新）
#   ./scripts/start.sh --stop   # 停止所有服务
#

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv/bin/python"
WEB_DIR="$PROJECT_DIR/web"
PID_DIR="$PROJECT_DIR/.pids"
BACKEND_PORT=8000
FRONTEND_PORT=5173

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

mkdir -p "$PID_DIR"

log() { echo -e "${GREEN}[AI投研]${NC} $1"; }
warn() { echo -e "${YELLOW}[AI投研]${NC} $1"; }
err() { echo -e "${RED}[AI投研]${NC} $1"; }

stop_services() {
    log "停止服务..."
    if [ -f "$PID_DIR/backend.pid" ]; then
        kill "$(cat "$PID_DIR/backend.pid")" 2>/dev/null && log "后端已停止" || true
        rm -f "$PID_DIR/backend.pid"
    fi
    if [ -f "$PID_DIR/frontend.pid" ]; then
        kill "$(cat "$PID_DIR/frontend.pid")" 2>/dev/null && log "前端已停止" || true
        rm -f "$PID_DIR/frontend.pid"
    fi
    # 兜底清理
    lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    log "所有服务已停止"
}

check_deps() {
    if [ ! -f "$VENV" ]; then
        err "未找到 Python 虚拟环境: $VENV"
        err "请先运行: cd $PROJECT_DIR && uv venv && uv pip install -e '.[web]'"
        exit 1
    fi
}

start_backend() {
    log "启动后端 API (port $BACKEND_PORT)..."
    cd "$PROJECT_DIR"
    "$VENV" -m uvicorn src.server:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        $BACKEND_EXTRA_ARGS \
        > "$PID_DIR/backend.log" 2>&1 &
    echo $! > "$PID_DIR/backend.pid"
    sleep 2

    if curl -s "http://localhost:$BACKEND_PORT/api/health" > /dev/null 2>&1; then
        log "后端启动成功: http://localhost:$BACKEND_PORT"
        log "API 文档: http://localhost:$BACKEND_PORT/docs"
    else
        err "后端启动失败，查看日志: $PID_DIR/backend.log"
        exit 1
    fi
}

start_frontend_dev() {
    log "启动前端开发服务器 (port $FRONTEND_PORT)..."
    cd "$WEB_DIR"

    if [ ! -d "node_modules" ]; then
        warn "安装前端依赖..."
        npm install
    fi

    npx vite --port $FRONTEND_PORT > "$PID_DIR/frontend.log" 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    sleep 2
    log "前端启动成功: http://localhost:$FRONTEND_PORT"
}

build_frontend() {
    log "构建前端..."
    cd "$WEB_DIR"

    if [ ! -d "node_modules" ]; then
        warn "安装前端依赖..."
        npm install
    fi

    npx vite build
    log "前端构建完成: $WEB_DIR/dist/"
}

# ========== Main ==========

case "${1:-}" in
    --stop)
        stop_services
        exit 0
        ;;
    --dev)
        stop_services
        check_deps
        BACKEND_EXTRA_ARGS="--reload"
        start_backend
        start_frontend_dev
        echo ""
        log "========================================="
        log "  开发模式已启动"
        log "  前端: http://localhost:$FRONTEND_PORT"
        log "  后端: http://localhost:$BACKEND_PORT"
        log "  API:  http://localhost:$BACKEND_PORT/docs"
        log "  停止: ./scripts/start.sh --stop"
        log "========================================="
        echo ""
        log "按 Ctrl+C 停止..."
        trap stop_services EXIT
        wait
        ;;
    *)
        stop_services
        check_deps
        # 生产模式：构建前端 → 由 FastAPI 静态文件服务
        build_frontend
        BACKEND_EXTRA_ARGS=""
        start_backend
        echo ""
        log "========================================="
        log "  生产模式已启动"
        log "  访问: http://localhost:$BACKEND_PORT"
        log "  API:  http://localhost:$BACKEND_PORT/docs"
        log "  停止: ./scripts/start.sh --stop"
        log "========================================="
        echo ""
        log "按 Ctrl+C 停止..."
        trap stop_services EXIT
        wait
        ;;
esac
