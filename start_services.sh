#!/bin/bash

# 文本信息检索系统 - 前后端同时启动脚本
# 作者: 系统管理员
# 版本: 1.0.0

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
BACKEND_DIR="server"
FRONTEND_DIR="web"
BACKEND_PORT=5000
FRONTEND_PORT=8080
LOG_DIR="logs"

# 创建日志目录
mkdir -p $LOG_DIR

# 信号处理函数 - 优雅关闭服务
cleanup() {
    echo -e "\n${YELLOW}正在关闭服务...${NC}"
    
    # 关闭后端服务
    if [ ! -z "$BACKEND_PID" ]; then
        echo -e "${BLUE}关闭后端服务 (PID: $BACKEND_PID)${NC}"
        kill -TERM $BACKEND_PID 2>/dev/null
        wait $BACKEND_PID 2>/dev/null
    fi
    
    # 关闭前端服务
    if [ ! -z "$FRONTEND_PID" ]; then
        echo -e "${BLUE}关闭前端服务 (PID: $FRONTEND_PID)${NC}"
        kill -TERM $FRONTEND_PID 2>/dev/null
        wait $FRONTEND_PID 2>/dev/null
    fi
    
    echo -e "${GREEN}所有服务已关闭${NC}"
    exit 0
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 检查端口是否被占用
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1
    else
        return 0
    fi
}

# 等待服务启动
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${BLUE}等待 $service_name 服务启动...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s $url >/dev/null 2>&1; then
            echo -e "${GREEN}$service_name 服务已启动${NC}"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "\n${RED}$service_name 服务启动超时${NC}"
    return 1
}

# 检查Python环境和依赖
check_python_env() {
    echo -e "${BLUE}检查Python环境...${NC}"
    
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到Python3${NC}"
        return 1
    fi
    
    # 检查必要的Python包
    local required_packages=("flask" "flask-cors" "pandas" "numpy" "jieba" "scikit-learn")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            missing_packages+=($package)
        fi
    done
    
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo -e "${YELLOW}缺少以下Python包: ${missing_packages[*]}${NC}"
        echo -e "${BLUE}正在安装缺少的包...${NC}"
        pip3 install ${missing_packages[*]}
    fi
    
    echo -e "${GREEN}Python环境检查完成${NC}"
    return 0
}

# 检查Node.js环境
check_node_env() {
    echo -e "${BLUE}检查Node.js环境...${NC}"
    
    if ! command -v node &> /dev/null; then
        echo -e "${RED}错误: 未找到Node.js${NC}"
        return 1
    fi
    
    if ! command -v npm &> /dev/null; then
        echo -e "${RED}错误: 未找到npm${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Node.js环境检查完成${NC}"
    return 0
}

# 启动后端服务
start_backend() {
    echo -e "${BLUE}启动后端服务...${NC}"
    
    # 检查端口
    if ! check_port $BACKEND_PORT; then
        echo -e "${RED}端口 $BACKEND_PORT 已被占用${NC}"
        return 1
    fi
    
    # 进入后端目录
    cd $BACKEND_DIR
    
    # 启动Flask应用
    python3 sever.py --host localhost --port $BACKEND_PORT > ../$LOG_DIR/backend.log 2>&1 &
    BACKEND_PID=$!
    
    cd ..
    
    # 等待后端服务启动
    if wait_for_service "http://localhost:$BACKEND_PORT/health" "后端"; then
        echo -e "${GREEN}后端服务启动成功 (PID: $BACKEND_PID, Port: $BACKEND_PORT)${NC}"
        return 0
    else
        echo -e "${RED}后端服务启动失败${NC}"
        return 1
    fi
}

# 启动前端服务
start_frontend() {
    echo -e "${BLUE}启动前端服务...${NC}"
    
    # 检查端口
    if ! check_port $FRONTEND_PORT; then
        echo -e "${RED}端口 $FRONTEND_PORT 已被占用${NC}"
        return 1
    fi
    
    # 进入前端目录
    cd $FRONTEND_DIR
    
    # 检查是否已安装依赖
    if [ ! -d "node_modules" ]; then
        echo -e "${BLUE}安装前端依赖...${NC}"
        npm install
    fi
    
    # 启动Vue开发服务器
    npm run serve > ../$LOG_DIR/frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    cd ..
    
    # 等待前端服务启动
    if wait_for_service "http://localhost:$FRONTEND_PORT" "前端"; then
        echo -e "${GREEN}前端服务启动成功 (PID: $FRONTEND_PID, Port: $FRONTEND_PORT)${NC}"
        return 0
    else
        echo -e "${RED}前端服务启动失败${NC}"
        return 1
    fi
}

# 主函数
main() {
    echo -e "${GREEN}=== 文本信息检索系统启动脚本 ===${NC}"
    echo -e "${BLUE}开始启动前后端服务...${NC}"
    
    # 环境检查
    if ! check_python_env; then
        echo -e "${RED}Python环境检查失败${NC}"
        exit 1
    fi
    
    if ! check_node_env; then
        echo -e "${RED}Node.js环境检查失败${NC}"
        exit 1
    fi
    
    # 启动后端服务
    if ! start_backend; then
        echo -e "${RED}后端服务启动失败${NC}"
        exit 1
    fi
    
    # 启动前端服务
    if ! start_frontend; then
        echo -e "${RED}前端服务启动失败${NC}"
        cleanup
        exit 1
    fi
    
    # 显示服务信息
    echo -e "\n${GREEN}=== 服务启动完成 ===${NC}"
    echo -e "${BLUE}后端服务: http://localhost:$BACKEND_PORT${NC}"
    echo -e "${BLUE}前端服务: http://localhost:$FRONTEND_PORT${NC}"
    echo -e "${BLUE}API健康检查: http://localhost:$BACKEND_PORT/health${NC}"
    echo -e "\n${YELLOW}按 Ctrl+C 停止所有服务${NC}\n"
    
    # 保持脚本运行，等待用户中断
    while true; do
        sleep 5
        
        # 检查进程是否还在运行
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "${RED}后端服务意外停止${NC}"
            cleanup
            exit 1
        fi
        
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            echo -e "${RED}前端服务意外停止${NC}"
            cleanup
            exit 1
        fi
    done
}

# 执行主函数
main "$@"

