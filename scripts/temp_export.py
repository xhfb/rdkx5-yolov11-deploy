from ultralytics import YOLO
import sys
import os

try:
    print("Loading model...")
    model = YOLO('/data/runs/run_20260130_055153/yolov11n_20250629.pt')
    
    save_dir = '/data/runs/run_20260130_055153'
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
    
    exported_file = '/data/runs/run_20260130_055153/yolov11n_20250629.pt'.replace('.pt', '.onnx')
    if os.path.exists(exported_file):
        os.rename(exported_file, save_path)
        print("Moved ONNX to output dir.")
        
except Exception as e:
    print(f"导出出错: {e}")
    sys.exit(1)
