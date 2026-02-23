#!/usr/bin/env bash
#
# AI 投研助手 - 一键部署脚本
#
# 用法：
#   ./deploy.sh                  # 交互式部署（提示输入密码）
#   ./deploy.sh --setup-remote   # 仅初始化远程服务器环境
#   ./deploy.sh --deploy-only    # 仅同步代码和重启（跳过环境检查）
#
set -e

# ==================== 配置区域 ====================
REMOTE_HOST=""           # 运行时填入，如 192.168.1.100
REMOTE_USER=""           # 运行时填入
REMOTE_PASS=""           # 运行时填入
REMOTE_DIR=""            # 运行时填入
SERVICE_NAME="ai-investment"
BACKEND_PORT=8000

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[部署]${NC} $1"; }
info() { echo -e "${BLUE}[信息]${NC} $1"; }
warn() { echo -e "${YELLOW}[警告]${NC} $1"; }
err() { echo -e "${RED}[错误]${NC} $1"; }

# ==================== 工具函数 ====================

# 使用 expect 执行远程命令
remote_exec() {
    local cmd="$1"
    expect -c "
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    \"$REMOTE_USER@$REMOTE_HOST\" \"$cmd\"
expect {
    \"password:\" { send \"$REMOTE_PASS\r\"; exp_continue }
    eof
}
"
}

# 使用 expect 同步文件
remote_sync() {
    local src="$1"
    local dst="$2"
    local exclude_file="$3"

    if [ -n "$exclude_file" ] && [ -f "$exclude_file" ]; then
        expect -c "
spawn rsync -avz --delete \
    --exclude-from=\"$exclude_file\" \
    -e \"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null\" \
    \"$src\" \"$dst\"
expect {
    \"password:\" { send \"$REMOTE_PASS\r\"; exp_continue }
    eof
}
"
    else
        expect -c "
spawn rsync -avz --delete \
    -e \"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null\" \
    \"$src\" \"$dst\"
expect {
    \"password:\" { send \"$REMOTE_PASS\r\"; exp_continue }
    eof
}
"
    fi
}

# 上传单个文件
remote_upload() {
    local src="$1"
    local dst="$2"
    expect -c "
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    \"$src\" \"$dst\"
expect {
    \"password:\" { send \"$REMOTE_PASS\r\"; exp_continue }
    eof
}
"
}

# ==================== 用户输入 ====================

collect_credentials() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}   AI 投研助手 - 一键部署${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""

    # 用户名
    if [ -z "$REMOTE_USER" ]; then
        read -p "请输入远程服务器用户名: " REMOTE_USER
    fi

    # 密码（隐藏输入）
    if [ -z "$REMOTE_PASS" ]; then
        read -s -p "请输入远程服务器密码: " REMOTE_PASS
        echo ""
    fi

    # 部署目录
    if [ -z "$REMOTE_DIR" ]; then
        REMOTE_DIR="/home/$REMOTE_USER/ai-investment"
        read -p "请输入部署目录 [$REMOTE_DIR]: " custom_dir
        REMOTE_DIR="${custom_dir:-$REMOTE_DIR}"
    fi

    echo ""
    log "部署配置:"
    echo "  服务器: $REMOTE_USER@$REMOTE_HOST"
    echo "  目录:   $REMOTE_DIR"
    echo ""
}

# ==================== 环境初始化 ====================

setup_remote_env() {
    log "检查远程服务器环境..."

    # 检测系统类型
    local os_type=$(remote_exec "cat /etc/os-release 2>/dev/null | grep '^ID=' | cut -d'=' -f2 | tr -d '\"' || echo 'unknown'" | tail -1)

    log "检测到系统: $os_type"

    # 检查并安装 Python 3.11+
    log "检查 Python 版本..."
    remote_exec "
if command -v python3 &>/dev/null; then
    ver=\$(python3 -c 'import sys; print(sys.version_info.major*100 + sys.version_info.minor)' 2>/dev/null || echo 0)
    if [ \"\$ver\" -ge 311 ]; then
        echo 'Python OK:' && python3 --version
    else
        echo 'Python version too old'
    fi
else
    echo 'Python not found'
fi
"

    # 检查并安装 Node.js
    log "检查 Node.js 版本..."
    remote_exec "
if command -v node &>/dev/null; then
    ver=\$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ \"\$ver\" -ge 18 ]; then
        echo 'Node.js OK:' && node -v
    else
        echo 'Node.js version too old'
    fi
else
    echo 'Node.js not found'
fi
"

    # 安装 uv
    log "确保 uv 包管理器已安装..."
    remote_exec "export PATH=\"\$HOME/.local/bin:\$PATH\"; command -v uv &>/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)"

    # 创建部署目录
    log "创建部署目录: $REMOTE_DIR"
    remote_exec "mkdir -p $REMOTE_DIR"

    # 创建 systemd 服务文件
    log "创建 systemd 服务..."
    local service_content="[Unit]
Description=AI Investment Analysis System
After=network.target

[Service]
Type=simple
User=$REMOTE_USER
WorkingDirectory=$REMOTE_DIR
Environment=PATH=$REMOTE_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin:/home/$REMOTE_USER/.local/bin
ExecStart=$REMOTE_DIR/.venv/bin/uvicorn src.server:app --host 0.0.0.0 --port $BACKEND_PORT
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

# 注意：数据库默认使用 SQLite，无需安装 PostgreSQL
# Redis 是 Phase 2 功能，当前未使用"

    echo "$service_content" > /tmp/$SERVICE_NAME.service
    remote_upload "/tmp/$SERVICE_NAME.service" "$REMOTE_USER@$REMOTE_HOST:/tmp/$SERVICE_NAME.service"
    remote_exec "sudo mv /tmp/$SERVICE_NAME.service /etc/systemd/system/ && sudo systemctl daemon-reload"
    rm -f /tmp/$SERVICE_NAME.service

    log "远程环境初始化完成"
}

# ==================== 代码同步 ====================

sync_code() {
    log "同步代码到远程服务器..."

    # 创建临时 exclude 文件
    local exclude_file=$(mktemp)
    cat > "$exclude_file" << 'EOF'
.git
.venv
node_modules
.pytest_cache
.idea
.vscode
.vite
__pycache__
*.pyc
*.pyo
*.egg-info
.eggs
dist
build
*.log
*.pid
.pids
.env
.env.local
*.bak
*.swp
.DS_Store
Thumbs.db
data/
output/
test_pdfs/
EOF

    remote_sync "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" "$exclude_file"

    rm -f "$exclude_file"
    log "代码同步完成"
}

# ==================== 安装依赖 ====================

install_deps() {
    log "安装远程依赖..."

    # 安装 Python 依赖
    log "安装 Python 依赖..."
    remote_exec "
cd $REMOTE_DIR && \
export PATH=\"\$HOME/.local/bin:\$PATH\" && \
uv venv && \
uv pip install -e '.[web]'
"

    # 安装前端依赖并构建
    log "构建前端..."
    remote_exec "
cd $REMOTE_DIR/web && \
npm install && \
npm run build
"

    log "依赖安装完成"
}

# ==================== 配置 ====================

setup_env_file() {
    log "检查 .env 配置..."

    # 简化处理：直接上传本地 .env（如果存在且用户同意）
    if [ -f "$LOCAL_DIR/.env" ]; then
        read -p "是否上传本地 .env 到远程？[y/N]: " upload_env
        if [ "$upload_env" = "y" ] || [ "$upload_env" = "Y" ]; then
            remote_upload "$LOCAL_DIR/.env" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/.env"
            log ".env 已上传"
        fi
    else
        warn "本地没有 .env 文件，将复制 .env.example"
        remote_exec "cp $REMOTE_DIR/.env.example $REMOTE_DIR/.env 2>/dev/null || echo '.env.example not found'"
    fi
}

# ==================== 服务管理 ====================

start_service() {
    log "启动服务..."

    remote_exec "
cd $REMOTE_DIR && \
sudo systemctl enable $SERVICE_NAME && \
sudo systemctl restart $SERVICE_NAME
"

    log "等待服务启动..."
    sleep 3

    check_service
}

check_service() {
    log "检查服务状态..."

    remote_exec "
if systemctl is-active --quiet $SERVICE_NAME; then
    echo '✓ 服务运行正常'
    systemctl status $SERVICE_NAME --no-pager | head -10
else
    echo '✗ 服务未运行'
    sudo journalctl -u $SERVICE_NAME -n 20 --no-pager
fi
"
}

# ==================== 主流程 ====================

check_tools() {
    if ! command -v expect &>/dev/null; then
        err "未找到 expect 命令"
        echo "macOS 安装: brew install expect"
        exit 1
    fi

    if ! command -v rsync &>/dev/null; then
        err "未找到 rsync 命令"
        exit 1
    fi
}

test_connection() {
    log "测试 SSH 连接..."
    remote_exec "echo 'SSH 连接成功'" | tail -1
}

main() {
    # 检查工具
    check_tools

    # 收集凭据
    collect_credentials

    # 测试连接
    test_connection

    # 根据参数执行不同操作
    case "${1:-}" in
        --setup-remote)
            setup_remote_env
            ;;
        --deploy-only)
            sync_code
            install_deps
            setup_env_file
            start_service
            ;;
        --status)
            check_service
            ;;
        *)
            # 完整部署流程
            setup_remote_env
            sync_code
            install_deps
            setup_env_file
            start_service
            ;;
    esac

    echo ""
    log "========================================="
    log "  部署完成！"
    log "========================================="
    echo ""
    echo -e "  访问地址: ${GREEN}http://$REMOTE_HOST:$BACKEND_PORT${NC}"
    echo -e "  API 文档: ${GREEN}http://$REMOTE_HOST:$BACKEND_PORT/docs${NC}"
    echo ""
    echo "常用命令:"
    echo "  查看状态: ./deploy.sh --status"
    echo "  重新部署: ./deploy.sh --deploy-only"
    echo "  查看日志: ssh $REMOTE_USER@$REMOTE_HOST 'sudo journalctl -u $SERVICE_NAME -f'"
    echo ""
}

# 执行
main "$@"
