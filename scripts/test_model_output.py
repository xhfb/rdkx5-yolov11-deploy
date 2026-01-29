# test_model_output.py - 验证修改是否生效
import torch
from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11n.pt')

# 创建测试输入
dummy_input = torch.randn(1, 3, 640, 640)

# 测试forward
model.model.eval()
with torch.no_grad():
    outputs = model.model(dummy_input)

# 检查输出
print(f"输出类型: {type(outputs)}")
print(f"输出数量: {len(outputs)}")

if isinstance(outputs, tuple) and len(outputs) == 6:
    print("✅ 修改成功！输出已分离为6个tensor")
    for i, out in enumerate(outputs):
        print(f"  Output {i}: {out.shape}")
else:
    print("❌ 修改失败！请检查head.py")