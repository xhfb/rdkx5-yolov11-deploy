#!/bin/bash
set -e # 遇到错误立即停止

# ==========================================
# RDK X5 YOLOv11 Deploy Tool (Linux Organized)
# ==========================================

# 配置
CONTAINER_NAME="rdk_converter_v1"
IMAGE_NAME="openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------------------
# 1. 参数解析 (CLI Mode)
# ------------------------------------------
MODEL=""
CALIB=""

show_help() {
    echo ""
    echo "========================================================"
    echo "       RDK X5 YOLOv11 Deploy Tool - Help"
    echo "========================================================"
    echo ""
    echo "Usage: deploy.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help                 Show this help message and exit"
    echo "  --model <path>             Path to the YOLOv11 .pt model file"
    echo "  --calibrate_images <path>  Path to the calibration images folder"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh"
    echo "      Run in interactive mode (prompts for input)"
    echo ""
    echo "  ./deploy.sh --model /path/to/yolov11n.pt --calibrate_images /path/to/calib"
    echo "      Run with specified model and calibration images"
    echo ""
    echo "  ./deploy.sh -h"
    echo "      Show this help message"
    echo ""
    echo "Description:"
    echo "  This tool converts YOLOv11 PyTorch models (.pt) to RDK X5 compatible"
    echo "  format using Docker-based AI toolchain. The converted model files"
    echo "  will be saved in the runs/ directory with a timestamp."
    echo ""
    echo "Requirements:"
    echo "  - Docker installed and running"
    echo "  - YOLOv11 .pt model file"
    echo "  - Calibration images folder (for quantization)"
    echo ""
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        --model) MODEL="$2"; shift ;;
        --calibrate_images) CALIB="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ------------------------------------------
# 2. 交互模式 (Interactive Mode)
# ------------------------------------------
if [ -z "$MODEL" ] || [ -z "$CALIB" ]; then
    clear
    echo -e "${BLUE}========================================================${NC}"
    echo -e "${BLUE}       RDK X5 YOLOv11 Deploy Tool (Linux)       ${NC}"
    echo -e "${BLUE}========================================================${NC}"
    echo ""
    echo -e "${YELLOW}[INFO] No parameters detected. Entering Input Mode.${NC}"
    echo ""
fi

# 获取模型路径
while [ -z "$MODEL" ]; do
    echo -e "[1/2] Please input .pt MODEL path:"
    read -p "> " MODEL
    # 去除可能的引号 (Linux终端拖入文件通常会自动加引号)
    MODEL=$(echo "$MODEL" | tr -d "'\"")
    
    if [ ! -f "$MODEL" ]; then
        echo -e "${RED}[ERROR] File not found: $MODEL${NC}"
        MODEL=""
    fi
done

echo ""

# 获取图片路径
while [ -z "$CALIB" ]; do
    echo -e "[2/2] Please input IMAGES folder path:"
    read -p "> " CALIB
    # 去除可能的引号
    CALIB=$(echo "$CALIB" | tr -d "'\"")
    
    if [ ! -d "$CALIB" ]; then
        echo -e "${RED}[ERROR] Folder not found: $CALIB${NC}"
        CALIB=""
    fi
done

# ------------------------------------------
# 3. 准备工作区 (Workspace Prep)
# ------------------------------------------
echo ""
echo -e "${BLUE}[Host] Preparing workspace...${NC}"

MODEL_NAME=$(basename "$MODEL")
# CALIB_NAME=$(basename "$CALIB")

# --- 生成时间戳 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR_NAME="run_${TIMESTAMP}"
RUN_DIR_PATH="$(pwd)/runs/${RUN_DIR_NAME}"

# 创建目录
mkdir -p "$RUN_DIR_PATH"
echo -e "${GREEN}[Host] Output directory created: runs/${RUN_DIR_NAME}${NC}"

# --- 复制模型 ---
echo "[Host] Copying model to run directory..."
cp "$MODEL" "$RUN_DIR_PATH/$MODEL_NAME"
MODEL_PATH_FOR_DOCKER="/data/runs/${RUN_DIR_NAME}/${MODEL_NAME}"

# --- 复制图片 ---
echo "[Host] Copying images to run directory..."
IMAGES_DEST_DIR="$RUN_DIR_PATH/source_images"
mkdir -p "$IMAGES_DEST_DIR"
# 递归复制图片内容
cp -r "$CALIB"/. "$IMAGES_DEST_DIR/"
CALIB_PATH_FOR_DOCKER="/data/runs/${RUN_DIR_NAME}/source_images"

# ------------------------------------------
# 4. Docker 执行
# ------------------------------------------
echo -e "${BLUE}[Host] Checking Docker environment...${NC}"

# 检查容器状态
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo -e "${GREEN}[Host] Found running container. Reusing...${NC}"
else
    echo -e "${YELLOW}[Host] Container not running. Recreating...${NC}"
    # 强制删除旧容器（如果有）
    docker rm -f ${CONTAINER_NAME} >/dev/null 2>&1 || true
    
    # 启动新容器 (使用 sleep 挂起)
    docker run -d \
        --name ${CONTAINER_NAME} \
        --entrypoint /bin/bash \
        -v "$(pwd)":/data \
        ${IMAGE_NAME} \
        -c "while true; do sleep 3600; done"
        
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FATAL ERROR] Failed to start Docker container.${NC}"
        exit 1
    fi
fi

# ------------------------------------------
# 5. 调用内部脚本
# ------------------------------------------
echo ""
echo -e "${BLUE}[Host] Running internal script...${NC}"
echo "--------------------------------------------------------"

# 确保脚本有执行权限 (以防万一)
docker exec ${CONTAINER_NAME} chmod +x /data/scripts/internal_runner.sh

# 执行转换
# 参数: 1.模型路径 2.图片路径 3.运行目录名
docker exec ${CONTAINER_NAME} /bin/bash /data/scripts/internal_runner.sh \
    "$MODEL_PATH_FOR_DOCKER" \
    "$CALIB_PATH_FOR_DOCKER" \
    "$RUN_DIR_NAME"

RET_CODE=$?

echo ""
if [ $RET_CODE -eq 0 ]; then
    echo -e "${GREEN}[Success] All Done!${NC}"
    echo -e "${GREEN}[Result] Files are located in: runs/${RUN_DIR_NAME}${NC}"
    
    # 尝试打开文件夹 (兼容 Linux/Mac)
    if command -v xdg-open &> /dev/null; then
        xdg-open "$RUN_DIR_PATH"
    elif command -v open &> /dev/null; then
        open "$RUN_DIR_PATH"
    fi
else
    echo -e "${RED}[Error] Conversion failed.${NC}"
    echo "Last 20 lines of logs:"
    docker logs ${CONTAINER_NAME} --tail 20
fi
