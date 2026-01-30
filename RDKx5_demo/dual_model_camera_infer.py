#!/usr/bin/env python3
"""
双模型摄像头实时检测 - 多线程版本
同时加载两个bin模型，对摄像头画面进行推理，结果叠加显示
"""
import cv2
import numpy as np
import time
import threading
from queue import Queue
from hobot_dnn import pyeasy_dnn as dnn


class YOLOv11Detector:
    """YOLOv11检测器类"""
    
    def __init__(self, model_path, conf_thresh=0.3, nms_thresh=0.5, cls_num=80, 
                 class_names=None, color_offset=0, name="Model"):
        """
        初始化检测器
        
        Args:
            model_path: bin模型路径
            conf_thresh: 置信度阈值 (0.0-1.0)
            nms_thresh: NMS阈值 (0.0-1.0)
            cls_num: 类别数量
            class_names: 类别名称列表（可选）
            color_offset: 颜色偏移量，用于区分不同模型的检测结果
            name: 模型名称，用于日志显示
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = 640
        self.reg_max = 16  # DFL的最大回归距离
        self.strides = [8, 16, 32]  # 三个检测头的stride
        self.cls_num = cls_num
        self.name = name
        self.color_offset = color_offset
        
        # 加载模型
        models = dnn.load(model_path)
        self.model = models[0]
        print(f"✅ [{self.name}] 模型加载成功: {model_path}")
        
        # 预计算anchor grid（加速后处理）
        self._init_anchors()
        
        # 类别名称
        if class_names is not None:
            self.class_names = class_names
        else:
            # 默认COCO 80类类别名称
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
        
        # 为每个类别生成随机颜色（带偏移以区分不同模型）
        np.random.seed(42 + color_offset)
        self.colors = np.random.randint(0, 255, size=(max(len(self.class_names), cls_num), 3), dtype=int)
    
    def _init_anchors(self):
        """
        预计算anchor grid
        对于640x640输入，三个检测头的grid大小为：
        - stride=8:  80x80
        - stride=16: 40x40
        - stride=32: 20x20
        """
        self.grids = []
        for stride in self.strides:
            h = w = self.input_size // stride
            # 生成网格坐标 (h, w, 2)
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
            self.grids.append(grid)
    
    def bgr_to_nv12(self, img):
        """
        BGR图片转NV12格式 + Letterbox缩放
        
        NV12格式说明：
        - Y平面: 640x640 (亮度)
        - UV平面: 320x640 (色度，U和V交错存储)
        - 总大小: 640x960
        
        Args:
            img: BGR图片 (H, W, 3)
        
        Returns:
            nv12: NV12数据 (960, 640)
            scale: 缩放比例
            pad_left: 左边padding
            pad_top: 上边padding
        """
        h, w = img.shape[:2]
        
        # 计算缩放比例（保持宽高比）
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Letterbox resize
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top = (self.input_size - new_h) // 2
        left = (self.input_size - new_w) // 2
        canvas[top:top+new_h, left:left+new_w] = resized
        
        # BGR to YUV (I420格式)
        yuv = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)
        
        # 提取Y、U、V平面
        y = yuv[:self.input_size, :]
        u = yuv[self.input_size:self.input_size+self.input_size//4, :].reshape(
            self.input_size//2, self.input_size//2)
        v = yuv[self.input_size+self.input_size//4:, :].reshape(
            self.input_size//2, self.input_size//2)
        
        # 组装NV12 (UV交错存储)
        uv = np.empty((self.input_size//2, self.input_size), dtype=np.uint8)
        uv[:, 0::2] = u
        uv[:, 1::2] = v
        
        nv12 = np.concatenate([y, uv], axis=0)
        
        return nv12, scale, left, top
    
    def dfl_decode(self, bbox_raw):
        """
        DFL (Distribution Focal Loss) 解码
        
        将64维的分布特征解码为4维的bbox坐标(ltrb)
        
        原理：
        1. 将64维reshape为(4, 16)，每个方向16个bin
        2. 对每个方向做Softmax，得到概率分布
        3. 计算期望值（加权求和）作为最终距离
        
        Args:
            bbox_raw: (N, 64) DFL特征
        
        Returns:
            ltrb: (N, 4) 边界框距离(left, top, right, bottom)
        """
        # Reshape: (N, 64) -> (N, 4, 16)
        bbox = bbox_raw.reshape(-1, 4, self.reg_max)
        
        # Softmax归一化
        bbox_exp = np.exp(bbox - np.max(bbox, axis=-1, keepdims=True))
        bbox_softmax = bbox_exp / np.sum(bbox_exp, axis=-1, keepdims=True)
        
        # 计算期望值 (加权求和)
        weights = np.arange(self.reg_max).reshape(1, 1, -1)
        ltrb = np.sum(bbox_softmax * weights, axis=-1)
        
        return ltrb
    
    def detect(self, img, nv12_data=None, preprocess_info=None):
        """
        执行目标检测
        
        流程：
        1. 预处理：BGR -> NV12（如果未提供）
        2. BPU推理：forward
        3. 后处理：解码 + NMS
        
        Args:
            img: 输入图片 (BGR格式)
            nv12_data: 预处理后的NV12数据（可选，用于共享预处理结果）
            preprocess_info: 预处理信息 (scale, pad_left, pad_top)（可选）
        
        Returns:
            boxes: 检测框 (N, 4) xyxy格式
            scores: 置信度 (N,)
            classes: 类别ID (N,)
        """
        orig_h, orig_w = img.shape[:2]
        
        # 1. 预处理（如果未提供预处理数据）
        if nv12_data is None:
            nv12, scale, pad_left, pad_top = self.bgr_to_nv12(img)
        else:
            nv12 = nv12_data
            scale, pad_left, pad_top = preprocess_info
        
        # 2. BPU推理
        outputs = self.model.forward(nv12)
        
        # 3. 后处理
        boxes, scores, classes = self._postprocess(
            outputs, scale, pad_left, pad_top, orig_w, orig_h
        )
        
        return boxes, scores, classes
    
    def _postprocess(self, outputs, scale, pad_left, pad_top, orig_w, orig_h):
        """
        后处理：解码 + 筛选 + NMS
        
        输出格式：
        - outputs[0-2]: bbox特征 (stride=8/16/32)
        - outputs[3-5]: class分数 (stride=8/16/32)
        
        优化策略：
        - 利用Sigmoid单调性，先筛选再计算
        - 减少不必要的DFL解码
        """
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # 分离bbox和cls输出
        bbox_outputs = outputs[:3]
        cls_outputs = outputs[3:]
        
        # 遍历三个检测头
        for i, (bbox_out, cls_out, grid, stride) in enumerate(
            zip(bbox_outputs, cls_outputs, self.grids, self.strides)):
            
            # 获取原始输出 (量化后的int16数据会自动转为float32)
            bbox_feat = bbox_out.buffer.reshape(-1, 64)   # (H*W, 64)
            cls_feat = cls_out.buffer.reshape(-1, self.cls_num)  # (H*W, cls_num)
            
            # ====== 优化：先筛选再计算 ======
            # Sigmoid是单调函数，可以在logit空间直接比较
            cls_max = np.max(cls_feat, axis=1)
            
            # 计算阈值对应的logit值
            # sigmoid(x) > thresh  <==>  x > log(thresh / (1-thresh))
            thresh_logit = np.log(self.conf_thresh / (1 - self.conf_thresh))
            
            # 筛选高置信度候选框
            valid_mask = cls_max > thresh_logit
            
            if not np.any(valid_mask):
                continue
            
            # 只对有效候选框进行后续计算
            valid_bbox = bbox_feat[valid_mask]
            valid_cls = cls_feat[valid_mask]
            valid_grid = grid[valid_mask]
            
            # ====== 类别分数计算 ======
            # Sigmoid激活
            scores = 1 / (1 + np.exp(-valid_cls))
            max_scores = np.max(scores, axis=1)
            max_classes = np.argmax(scores, axis=1)
            
            # ====== 边界框解码 ======
            # DFL解码得到ltrb距离
            ltrb = self.dfl_decode(valid_bbox)
            
            # 计算anchor中心点坐标
            x_center = (valid_grid[:, 0] + 0.5) * stride
            y_center = (valid_grid[:, 1] + 0.5) * stride
            
            # ltrb转xyxy（去除padding，还原到原图尺度）
            x1 = (x_center - ltrb[:, 0] * stride - pad_left) / scale
            y1 = (y_center - ltrb[:, 1] * stride - pad_top) / scale
            x2 = (x_center + ltrb[:, 2] * stride - pad_left) / scale
            y2 = (y_center + ltrb[:, 3] * stride - pad_top) / scale
            
            # 裁剪到图像边界
            x1 = np.clip(x1, 0, orig_w)
            y1 = np.clip(y1, 0, orig_h)
            x2 = np.clip(x2, 0, orig_w)
            y2 = np.clip(y2, 0, orig_h)
            
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            
            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_classes.append(max_classes)
        
        if not all_boxes:
            return np.array([]), np.array([]), np.array([])
        
        # ====== 合并所有尺度的检测结果 ======
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        classes = np.concatenate(all_classes, axis=0)
        
        # ====== NMS去重 ======
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_thresh,
            self.nms_thresh
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices], scores[indices], classes[indices]
        
        return np.array([]), np.array([]), np.array([])
    
    def draw(self, img, boxes, scores, classes, prefix=""):
        """
        在图片上绘制检测结果
        
        Args:
            img: 输入图片
            boxes: 检测框
            scores: 置信度
            classes: 类别ID
            prefix: 标签前缀（用于区分不同模型）
        
        Returns:
            img: 绘制后的图片
        """
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            cls_idx = int(cls)
            color = tuple(map(int, self.colors[cls_idx % len(self.colors)]))
            
            # 获取类别名称
            if cls_idx < len(self.class_names):
                class_name = self.class_names[cls_idx]
            else:
                class_name = f"class_{cls_idx}"
            
            label = f"{prefix}{class_name}: {score:.2f}"
            
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签（带背景）
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(img, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return img


class InferenceThread(threading.Thread):
    """推理线程类"""
    
    def __init__(self, detector, input_queue, output_queue, name="InferThread"):
        """
        初始化推理线程
        
        Args:
            detector: YOLOv11Detector实例
            input_queue: 输入队列（帧数据）
            output_queue: 输出队列（检测结果）
            name: 线程名称
        """
        super().__init__(name=name)
        self.detector = detector
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.daemon = True
    
    def run(self):
        """线程主循环"""
        while self.running:
            try:
                # 从输入队列获取帧数据（带超时）
                data = self.input_queue.get(timeout=0.1)
                if data is None:
                    continue
                
                frame, frame_id, nv12_data, preprocess_info = data
                
                # 执行推理
                boxes, scores, classes = self.detector.detect(
                    frame, nv12_data, preprocess_info
                )
                
                # 将结果放入输出队列
                self.output_queue.put((frame_id, boxes, scores, classes))
                
            except Exception as e:
                if self.running:
                    # 队列超时是正常的，其他异常需要记录
                    if "Empty" not in str(type(e).__name__):
                        print(f"⚠️ [{self.name}] 推理异常: {e}")
    
    def stop(self):
        """停止线程"""
        self.running = False


class DualModelInference:
    """双模型推理管理器"""
    
    def __init__(self, model1_config, model2_config):
        """
        初始化双模型推理
        
        Args:
            model1_config: 模型1配置字典
            model2_config: 模型2配置字典
        
        配置字典格式：
        {
            'model_path': str,      # 模型路径
            'conf_thresh': float,   # 置信度阈值
            'nms_thresh': float,    # NMS阈值
            'cls_num': int,         # 类别数量
            'class_names': list,    # 类别名称（可选）
            'name': str,            # 模型名称
            'label_prefix': str,    # 标签前缀
        }
        """
        print("=" * 70)
        print("🚀 初始化双模型推理系统")
        print("=" * 70)
        
        # 创建检测器
        self.detector1 = YOLOv11Detector(
            model_path=model1_config['model_path'],
            conf_thresh=model1_config.get('conf_thresh', 0.3),
            nms_thresh=model1_config.get('nms_thresh', 0.5),
            cls_num=model1_config.get('cls_num', 80),
            class_names=model1_config.get('class_names'),
            color_offset=0,
            name=model1_config.get('name', 'Model1')
        )
        
        self.detector2 = YOLOv11Detector(
            model_path=model2_config['model_path'],
            conf_thresh=model2_config.get('conf_thresh', 0.3),
            nms_thresh=model2_config.get('nms_thresh', 0.5),
            cls_num=model2_config.get('cls_num', 80),
            class_names=model2_config.get('class_names'),
            color_offset=100,  # 颜色偏移，区分两个模型
            name=model2_config.get('name', 'Model2')
        )
        
        self.label_prefix1 = model1_config.get('label_prefix', '[M1]')
        self.label_prefix2 = model2_config.get('label_prefix', '[M2]')
        
        # 创建队列
        self.input_queue1 = Queue(maxsize=2)
        self.input_queue2 = Queue(maxsize=2)
        self.output_queue1 = Queue(maxsize=2)
        self.output_queue2 = Queue(maxsize=2)
        
        # 创建推理线程
        self.thread1 = InferenceThread(
            self.detector1, self.input_queue1, self.output_queue1,
            name=f"Thread-{model1_config.get('name', 'Model1')}"
        )
        self.thread2 = InferenceThread(
            self.detector2, self.input_queue2, self.output_queue2,
            name=f"Thread-{model2_config.get('name', 'Model2')}"
        )
        
        # 结果缓存
        self.results_cache = {}
        self.cache_lock = threading.Lock()
        
        print("✅ 双模型推理系统初始化完成")
    
    def start(self):
        """启动推理线程"""
        self.thread1.start()
        self.thread2.start()
        print("🏃 推理线程已启动")
    
    def stop(self):
        """停止推理线程"""
        self.thread1.stop()
        self.thread2.stop()
        self.thread1.join(timeout=1.0)
        self.thread2.join(timeout=1.0)
        print("⏹️ 推理线程已停止")
    
    def preprocess(self, frame):
        """
        共享预处理
        
        Args:
            frame: 输入帧
        
        Returns:
            nv12_data: NV12数据
            preprocess_info: (scale, pad_left, pad_top)
        """
        nv12, scale, pad_left, pad_top = self.detector1.bgr_to_nv12(frame)
        return nv12, (scale, pad_left, pad_top)
    
    def submit_frame(self, frame, frame_id):
        """
        提交帧进行推理
        
        Args:
            frame: 输入帧
            frame_id: 帧ID
        """
        # 共享预处理结果
        nv12_data, preprocess_info = self.preprocess(frame)
        
        # 提交到两个推理队列
        try:
            self.input_queue1.put_nowait((frame, frame_id, nv12_data, preprocess_info))
        except:
            pass  # 队列满则跳过
        
        try:
            self.input_queue2.put_nowait((frame, frame_id, nv12_data, preprocess_info))
        except:
            pass  # 队列满则跳过
    
    def get_results(self, frame_id, timeout=0.05):
        """
        获取推理结果
        
        Args:
            frame_id: 帧ID
            timeout: 超时时间
        
        Returns:
            result1: (boxes, scores, classes) 或 None
            result2: (boxes, scores, classes) 或 None
        """
        result1 = None
        result2 = None
        
        # 尝试从输出队列获取结果
        try:
            while not self.output_queue1.empty():
                fid, boxes, scores, classes = self.output_queue1.get_nowait()
                with self.cache_lock:
                    self.results_cache[('m1', fid)] = (boxes, scores, classes)
        except:
            pass
        
        try:
            while not self.output_queue2.empty():
                fid, boxes, scores, classes = self.output_queue2.get_nowait()
                with self.cache_lock:
                    self.results_cache[('m2', fid)] = (boxes, scores, classes)
        except:
            pass
        
        # 从缓存获取结果
        with self.cache_lock:
            if ('m1', frame_id) in self.results_cache:
                result1 = self.results_cache.pop(('m1', frame_id))
            if ('m2', frame_id) in self.results_cache:
                result2 = self.results_cache.pop(('m2', frame_id))
            
            # 清理旧缓存
            old_keys = [k for k in self.results_cache.keys() if k[1] < frame_id - 10]
            for k in old_keys:
                del self.results_cache[k]
        
        return result1, result2
    
    def draw_results(self, frame, result1, result2):
        """
        绘制两个模型的检测结果
        
        Args:
            frame: 输入帧
            result1: 模型1结果
            result2: 模型2结果
        
        Returns:
            frame: 绘制后的帧
        """
        if result1 is not None:
            boxes, scores, classes = result1
            frame = self.detector1.draw(frame, boxes, scores, classes, self.label_prefix1)
        
        if result2 is not None:
            boxes, scores, classes = result2
            frame = self.detector2.draw(frame, boxes, scores, classes, self.label_prefix2)
        
        return frame


def main():
    """主函数：双模型摄像头实时检测"""
    
    print("=" * 70)
    print("🎥 双模型摄像头实时检测 - 多线程版本")
    print("=" * 70)
    
    # ========== 配置两个模型 ==========
    # 模型1配置
    model1_config = {
        'model_path': '/home/sunrise/RDK_infer/yolov11/yolov11_model1.bin',
        'conf_thresh': 0.3,
        'nms_thresh': 0.5,
        'cls_num': 80,
        'class_names': None,  # 使用默认COCO类别
        'name': 'YOLO-Model1',
        'label_prefix': '[M1]',
    }
    
    # 模型2配置
    model2_config = {
        'model_path': '/home/sunrise/RDK_infer/yolov11/yolov11_model2.bin',
        'conf_thresh': 0.3,
        'nms_thresh': 0.5,
        'cls_num': 80,
        'class_names': None,  # 使用默认COCO类别
        'name': 'YOLO-Model2',
        'label_prefix': '[M2]',
    }
    
    # 初始化双模型推理系统
    dual_infer = DualModelInference(model1_config, model2_config)
    
    # 打开摄像头
    # USB摄像头使用0，MIPI摄像头使用8
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n📹 摄像头已打开 (640x480)")
    print("🎬 开始双模型实时检测 (按 'q' 退出)")
    print("-" * 70)
    
    # 设置显示权限（通过SSH运行时需要）
    import os
    os.environ['DISPLAY'] = ':0'
    
    # 启动推理线程
    dual_infer.start()
    
    # FPS统计
    fps_list = []
    frame_count = 0
    
    # 上一帧的结果（用于平滑显示）
    last_result1 = None
    last_result2 = None
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("⚠️  无法读取摄像头帧")
                break
            
            # 计时开始
            start = time.time()
            
            # 提交帧进行推理
            dual_infer.submit_frame(frame, frame_count)
            
            # 获取推理结果
            result1, result2 = dual_infer.get_results(frame_count)
            
            # 使用最新结果或上一帧结果
            if result1 is not None:
                last_result1 = result1
            if result2 is not None:
                last_result2 = result2
            
            # 绘制结果
            result_frame = frame.copy()
            result_frame = dual_infer.draw_results(result_frame, last_result1, last_result2)
            
            # 计算FPS
            elapsed = time.time() - start
            fps = 1.0 / max(elapsed, 0.001)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
            
            # 统计检测数量
            count1 = len(last_result1[0]) if last_result1 is not None else 0
            count2 = len(last_result2[0]) if last_result2 is not None else 0
            
            # 在图片上显示FPS和检测数量
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Model1: {count1} objs", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(result_frame, f"Model2: {count2} objs", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            
            # 显示画面（会显示在HDMI显示器上）
            cv2.imshow('Dual Model Detection', result_frame)
            
            # 终端日志
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"帧: {frame_count:4d} | FPS: {avg_fps:5.1f} | "
                      f"M1检测: {count1:2d} | M2检测: {count2:2d}")
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断 (Ctrl+C)")
    
    finally:
        # 停止推理线程
        dual_infer.stop()
        
        # 清理资源
        cap.release()
        cv2.destroyAllWindows()
        
        if len(fps_list) > 0:
            print("\n" + "=" * 70)
            print("📊 最终统计")
            print("=" * 70)
            print(f"总帧数: {frame_count}")
            print(f"平均FPS: {np.mean(fps_list):.1f}")
            print("=" * 70)
        
        print("\n✅ 程序结束")


def main_sync():
    """
    同步版本主函数（备选方案）
    不使用多线程，顺序执行两个模型的推理
    适用于调试或资源受限的情况
    """
    
    print("=" * 70)
    print("🎥 双模型摄像头实时检测 - 同步版本")
    print("=" * 70)
    
    # ========== 配置两个模型 ==========
    # 模型1配置
    detector1 = YOLOv11Detector(
        model_path='/home/sunrise/RDK_infer/yolov11/yolov11_model1.bin',
        conf_thresh=0.3,
        nms_thresh=0.5,
        cls_num=80,
        color_offset=0,
        name='YOLO-Model1'
    )
    
    # 模型2配置
    detector2 = YOLOv11Detector(
        model_path='/home/sunrise/RDK_infer/yolov11/yolov11_model2.bin',
        conf_thresh=0.3,
        nms_thresh=0.5,
        cls_num=80,
        color_offset=100,
        name='YOLO-Model2'
    )
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n📹 摄像头已打开 (640x480)")
    print("🎬 开始双模型实时检测 (按 'q' 退出)")
    print("-" * 70)
    
    # 设置显示权限
    import os
    os.environ['DISPLAY'] = ':0'
    
    # FPS统计
    fps_list = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  无法读取摄像头帧")
                break
            
            start = time.time()
            
            # 共享预处理
            nv12, scale, pad_left, pad_top = detector1.bgr_to_nv12(frame)
            preprocess_info = (scale, pad_left, pad_top)
            
            # 顺序执行两个模型推理
            boxes1, scores1, classes1 = detector1.detect(frame, nv12, preprocess_info)
            boxes2, scores2, classes2 = detector2.detect(frame, nv12, preprocess_info)
            
            # 绘制结果
            result_frame = frame.copy()
            result_frame = detector1.draw(result_frame, boxes1, scores1, classes1, "[M1]")
            result_frame = detector2.draw(result_frame, boxes2, scores2, classes2, "[M2]")
            
            # 计算FPS
            elapsed = time.time() - start
            fps = 1.0 / max(elapsed, 0.001)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
            
            # 显示信息
            cv2.putText(result_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Model1: {len(boxes1)} objs", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(result_frame, f"Model2: {len(boxes2)} objs", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            
            cv2.imshow('Dual Model Detection (Sync)', result_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"帧: {frame_count:4d} | FPS: {avg_fps:5.1f} | "
                      f"M1检测: {len(boxes1):2d} | M2检测: {len(boxes2):2d}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断 (Ctrl+C)")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if len(fps_list) > 0:
            print("\n" + "=" * 70)
            print("📊 最终统计")
            print("=" * 70)
            print(f"总帧数: {frame_count}")
            print(f"平均FPS: {np.mean(fps_list):.1f}")
            print("=" * 70)
        
        print("\n✅ 程序结束")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='双模型摄像头实时检测')
    parser.add_argument('--sync', action='store_true',
                        help='使用同步模式（不使用多线程）')
    parser.add_argument('--model1', type=str,
                        default='/home/sunrise/RDK_infer/yolov11/yolov11_model1.bin',
                        help='模型1路径')
    parser.add_argument('--model2', type=str,
                        default='/home/sunrise/RDK_infer/yolov11/yolov11_model2.bin',
                        help='模型2路径')
    parser.add_argument('--conf1', type=float, default=0.3,
                        help='模型1置信度阈值')
    parser.add_argument('--conf2', type=float, default=0.3,
                        help='模型2置信度阈值')
    parser.add_argument('--cls1', type=int, default=80,
                        help='模型1类别数量')
    parser.add_argument('--cls2', type=int, default=80,
                        help='模型2类别数量')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头ID (USB=0, MIPI=8)')
    
    args = parser.parse_args()
    
    if args.sync:
        # 同步模式
        main_sync()
    else:
        # 多线程模式（需要修改main函数以支持命令行参数）
        # 这里简化处理，直接调用main()
        # 如需使用命令行参数，可以修改main()函数接收参数
        main()