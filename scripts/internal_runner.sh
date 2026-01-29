#!/bin/bash
set -e

# 接收参数
MODEL_FILE="$1"
CALIB_DIR="$2"
RUN_DIR_NAME="$3"

WORK_ROOT="/data"
VENV_DIR="${WORK_ROOT}/venv_rdk"
CURRENT_OUTPUT_DIR="${WORK_ROOT}/runs/${RUN_DIR_NAME}"
PROCESSED_CALIB="${CURRENT_OUTPUT_DIR}/calib_processed"
PY_EXEC="${VENV_DIR}/bin/python"

echo "=========================================="
echo "📦 [Docker] 开始转换流程 (CPU轻量版)"
echo "   任务ID: ${RUN_DIR_NAME}"
echo "=========================================="

cd ${WORK_ROOT}

# 1. 环境检查与安装
if [ ! -d "${VENV_DIR}" ]; then
    echo ">>> [1/5] 初始化虚拟环境..."
    python3 -m venv ${VENV_DIR}

    # 【核心优化】先安装 CPU 版 PyTorch
    # 避免下载数 GB 的 NVIDIA GPU 库，解决 Hash Mismatch 和下载慢的问题
    echo ">>> [1/5-A] 安装 CPU 版 PyTorch (极速模式)..."
    "${VENV_DIR}/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu --retries 5

    # 生成其他依赖
    cat > ${WORK_ROOT}/requirements.txt << EOF
ultralytics>=8.3.0
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy<2.0.0
onnx>=1.15.0
onnxruntime>=1.16.0
tqdm>=4.66.0
PyYAML>=6.0.0
EOF
    echo ">>> [1/5-B] 安装其余依赖..."
    "${VENV_DIR}/bin/pip" install -r ${WORK_ROOT}/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --retries 5
else
    echo ">>> [1/5] 虚拟环境已就绪"
fi

# 2. 修改源码
echo ">>> [2/5] 注入 RDK X5 适配代码..."
"${PY_EXEC}" scripts/modify_code.py

# 3. 导出 ONNX
echo ">>> [3/5] 导出 ONNX..."
cat > scripts/temp_export.py << EOF
from ultralytics import YOLO
import sys
import os

try:
    print("Loading model...")
    model = YOLO('${MODEL_FILE}')
    
    save_dir = '${CURRENT_OUTPUT_DIR}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_split.onnx')
    
    print(f"Exporting ONNX to {save_path}...")
    model.export(
        format='onnx', 
        imgsz=640, 
        opset=11, 
        simplify=False, 
        dynamic=False, 
        half=False
    )
    
    exported_file = '${MODEL_FILE}'.replace('.pt', '.onnx')
    if os.path.exists(exported_file):
        os.rename(exported_file, save_path)
        print("Moved ONNX to output dir.")
        
except Exception as e:
    print(f"导出出错: {e}")
    sys.exit(1)
EOF
"${PY_EXEC}" scripts/temp_export.py

# 4. 准备校准数据
echo ">>> [4/5] 预处理校准图片..."
mkdir -p ${PROCESSED_CALIB}
"${PY_EXEC}" scripts/prepare_data.py "${CALIB_DIR}" "${PROCESSED_CALIB}"

# 5. 量化
echo ">>> [5/5] 执行 BPU 量化..."
cat > ${CURRENT_OUTPUT_DIR}/config.yaml << EOF
model_parameters:
  onnx_model: '${CURRENT_OUTPUT_DIR}/model_split.onnx'
  march: 'bayes-e'
  working_dir: '${CURRENT_OUTPUT_DIR}/build_cache'
  output_model_file_prefix: 'yolov11_final'
  layer_out_dump: False
  node_info: {
    "/model.10/m/m.0/attn/Softmax": {
      'ON': 'BPU', 
      'InputType': 'int16', 
      'OutputType': 'int16'
    }
  }
input_parameters:
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
  input_shape: '1x3x640x640'
calibration_parameters:
  cal_data_dir: '${PROCESSED_CALIB}'
  cal_data_type: 'float32'
  calibration_type: 'default'
compiler_parameters:
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O3'
EOF

cd ${CURRENT_OUTPUT_DIR}
hb_mapper makertbin --model-type onnx --config config.yaml

echo "=========================================="
if [ -f "${CURRENT_OUTPUT_DIR}/build_cache/yolov11_final.bin" ]; then
    cp "${CURRENT_OUTPUT_DIR}/build_cache/yolov11_final.bin" "${CURRENT_OUTPUT_DIR}/"
    echo "🏆 转换成功！"
    echo "📂 结果路径: runs/${RUN_DIR_NAME}"
    
    if command -v hrt_model_exec &> /dev/null; then
        hrt_model_exec model_info --model_file "${CURRENT_OUTPUT_DIR}/yolov11_final.bin" | grep "subgraph"
    fi
    exit 0
else
    echo "❌ 转换失败，未生成 bin 文件。"
    exit 1
fi