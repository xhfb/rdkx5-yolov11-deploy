# 🚀 RDK X5 YOLOv11 高性能部署工具箱 (Auto-Deploy Toolkit)

[![RDK X5](https://img.shields.io/badge/Platform-RDK__X5-blue)](https://developer.d-robotics.cc/)
[![YOLOv11](https://img.shields.io/badge/Model-YOLOv11-green)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Environment-Docker-blue)](https://www.docker.com/)

> 参考文档：
> - [YOLO11n-INT8 教程](https://github.com/1760hwy/YOLO11n-INT8/blob/master/docs/tutorial_zh.md)
> - [YOLOv11 高帧率部署到 RDK X5 教程](https://blog.csdn.net/zhangqian4622/article/details/151119019)

---

## ✨ 核心特性

- **🐳 零环境依赖**：基于 Docker 容器化编译，无需在宿主机安装复杂的 Python 环境或地平线工具链，保持系统整洁。
- **🖱️ 跨平台支持**：
  - **Windows**: 支持拖拽式交互，双击脚本即可运行。
  - **Linux**: 提供标准 Shell 脚本支持。
- **📂 自动归档**：转换结果自动按时间戳保存至 `runs/` 目录，包含模型、日志、中间 ONNX 和配置文件。
- **🛠️ 板端 Demo**：提供适配 RDK X5 的 Python 推理代码，开箱即用。

---

## 🏗️ 项目结构

```text
RDKX5_YOLOV11_CONVERT/
├── deploy.bat                 # [Windows] 启动脚本 (支持拖拽)
├── deploy.sh                  # [Linux] 启动脚本 (命令行)
├── requirements.txt           # Docker 内环境依赖列表
├── scripts/                   # 核心工具脚本 (Docker 内部调用)
│   ├── config.yaml            # 自动生成的量化配置模板
│   ├── internal_runner.sh     # 容器内总控脚本
│   ├── modify_code.py         # 自动修改 Ultralytics Head 源码
│   ├── prepare_data.py        # 校准数据预处理
│   ├── temp_export.py         # ONNX 导出脚本
│   └── test_model_output.py   # 模型结构验证
├── RDKx5_demo/                # [板端] RDK X5 上运行的推理代码
│   └── camera_infer.py        # 摄像头实时检测 Demo
├── runs/                      # [自动生成] 存放所有运行结果
│   └── run_2026xxxx_xxxx/     # 每次运行的独立文件夹
│       ├── yolov11_final.bin  # 最终可部署模型
│       ├── model_split.onnx   # 中间态 ONNX
│       └── ...
└── venv_rdk/                  # [自动生成] 脚本运行时的临时环境 (建议加入 .gitignore)
```

---

## 🛠️ 环境准备

### 1. 安装 Docker

| 平台 | 安装方式 |
|------|----------|
| Windows | 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Linux | 安装 Docker Engine |

### 2. 拉取工具链镜像

```bash
docker pull openexplorer/ai_toolchain_ubuntu_20_x5_gpu:v1.2.8
```

### 3. 准备模型与数据

| 资源 | 要求 |
|------|------|
| 模型 | 训练好的 `yolov11n.pt`（推荐使用 n 版本以获得最佳性能） |
| 校准图片 | 约 100 张来自训练集或 COCO 数据集的图片（.jpg 格式），放入一个文件夹中（例如 `coco_calib/`） |

---

## 🚀 使用指南 (转换模型)

### 🪟 Windows 用户

**方式 A：拖拽式交互**

1. 直接双击运行 `deploy.bat`
2. 将 `.pt` 文件拖入终端窗口，按回车
3. 将校准图片文件夹拖入窗口，按回车
4. 等待脚本执行完成

**方式 B：命令行参数**

```bash
deploy.bat --model best.pt --calibrate_images ./images
```

> **注意**：脚本会自动打开生成的 `runs/run_xxx` 文件夹，获取 `yolov11_final.bin`

### 🐧 Linux 用户

1. 赋予脚本执行权限：

```bash
chmod +x deploy.sh
```

2. 运行脚本：

```bash
./deploy.sh --model best.pt --calibrate_images ./path/to/images
```

> 结果将保存在 `runs/` 目录下

---

## 🤖 板端部署 (RDK X5)

### 1. 传输模型

转换完成后，将生成的 `.bin` 文件传输到 RDK X5 开发板。

### 2. 安装依赖

在 RDK X5 上安装必要的库：

```bash
pip install scipy
```

### 3. 运行 Demo

```bash
# 在 RDK X5 终端执行
# 注意：如果使用 SSH 连接且需要显示画面，请确保配置了 X11 转发或连接了 HDMI
python3 camera_infer.py
```

> **提示**：请在 `camera_infer.py` 中修改模型路径指向你的 `.bin` 文件

---

## 📝 常见问题

### Q: 转换失败怎么办？

1. 检查模型文件是否有效（使用 `yolov11n.pt` 验证）
2. 确保校准图片数量充足（建议 100 张以上）
3. 查看 `runs/` 目录下的日志文件定位问题

### Q: 如何验证模型转换成功？

转换完成后，检查以下文件：
- `yolov11_final.bin` - 最终可部署模型
- `model_split.onnx` - 中间态 ONNX 文件
- `config.yaml` - 配置文件

---

## 📄 许可证

本项目遵循 MIT 许可证开源。
