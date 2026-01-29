# camera_detect_final.py - 完整的实时检测脚本
#!/usr/bin/env python3
"""
YOLOv11n 摄像头实时检测 - 完整版
包含：预处理、BPU推理、后处理、NMS、绘制、显示
"""
import cv2
import numpy as np
import time
from hobot_dnn import pyeasy_dnn as dnn

class YOLOv11Detector:
    """YOLOv11检测器类"""
    
    def __init__(self, model_path, conf_thresh=0.3, nms_thresh=0.5,cls_num=80):
        """
        初始化检测器
        
        Args:
            model_path: bin模型路径
            conf_thresh: 置信度阈值 (0.0-1.0)
            nms_thresh: NMS阈值 (0.0-1.0)
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = 640
        self.reg_max = 16  # DFL的最大回归距离
        self.strides = [8, 16, 32]  # 三个检测头的stride
        self.cls_num=cls_num
        # 加载模型
        models = dnn.load(model_path)
        self.model = models[0]
        print(f"✅ 模型加载成功: {model_path}")
        
        # 预计算anchor grid（加速后处理）
        self._init_anchors()
        
        # COCO 80类类别名称
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
        
        # 为每个类别生成随机颜色
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=int)
    
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
    
    def detect(self, img):
        """
        执行目标检测
        
        流程：
        1. 预处理：BGR -> NV12
        2. BPU推理：forward
        3. 后处理：解码 + NMS
        
        Args:
            img: 输入图片 (BGR格式)
        
        Returns:
            boxes: 检测框 (N, 4) xyxy格式
            scores: 置信度 (N,)
            classes: 类别ID (N,)
        """
        orig_h, orig_w = img.shape[:2]
        
        # 1. 预处理
        nv12, scale, pad_left, pad_top = self.bgr_to_nv12(img)
        
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
            cls_feat = cls_out.buffer.reshape(-1, self.cls_num) #(-1, 80)         # (H*W, 80)
            
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
    
    def draw(self, img, boxes, scores, classes):
        """
        在图片上绘制检测结果
        
        Args:
            img: 输入图片
            boxes: 检测框
            scores: 置信度
            classes: 类别ID
        
        Returns:
            img: 绘制后的图片
        """
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            color = tuple(map(int, self.colors[int(cls)]))
            label = f"{self.class_names[int(cls)]}: {score:.2f}"
            
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


def main():
    """主函数：摄像头实时检测"""
    
    print("=" * 70)
    print("🎥 YOLOv11n 摄像头实时检测")
    print("=" * 70)
    
    # 初始化检测器
    detector = YOLOv11Detector(
        model_path='/home/sunrise/RDK_infer/yolov11/yolov11_final.bin',
        conf_thresh=0.3,   # 置信度阈值（可调整）
        nms_thresh=0.5,     # NMS阈值（可调整）
        cls_num=80,  # 类别数量
    )
    
    # 打开摄像头
    # USB摄像头使用0，MIPI摄像头使用8
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    print("\n📹 摄像头已打开 (640x480)")
    print("🎬 开始实时检测 (按 'q' 退出)")
    print("-" * 70)
    
    # 设置显示权限（通过SSH运行时需要）
    import os
    os.environ['DISPLAY'] = ':0'
    
    # FPS统计
    fps_list = []
    frame_count = 0
    
    try:
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                print("⚠️  无法读取摄像头帧")
                break
            
            # 计时开始
            start = time.time()
            
            # 执行检测
            boxes, scores, classes = detector.detect(frame)
            
            # 绘制结果
            result = detector.draw(frame.copy(), boxes, scores, classes)
            
            # 计算FPS
            elapsed = time.time() - start
            fps = 1.0 / elapsed
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)
            
            # 在图片上显示FPS和检测数量
            cv2.putText(result, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result, f"Objects: {len(boxes)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示画面（会显示在HDMI显示器上）
            cv2.imshow('YOLOv11n Detection', result)
            
            # 终端日志
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"帧: {frame_count:4d} | FPS: {avg_fps:5.1f} | 检测: {len(boxes):2d} 个物体")
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断 (Ctrl+C)")
    
    finally:
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


if __name__ == '__main__':
    main()