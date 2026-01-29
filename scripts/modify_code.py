import os
import sys
import ultralytics

def modify_head():
    # 1. 定位文件
    try:
        head_path = os.path.join(os.path.dirname(ultralytics.__file__), 'nn/modules/head.py')
        print(f">>> [Auto] 定位源码: {head_path}")
    except Exception as e:
        print(f"❌ 错误: 无法找到 ultralytics 库: {e}")
        sys.exit(1)

    # 2. 读取所有行
    with open(head_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3. 定义新代码 (RDK X5 专用)
    new_forward_code = [
        "    def forward(self, x):\n",
        "        \"\"\"Modified for RDK X5 BPU: Split bbox and cls outputs\"\"\"\n",
        "        if self.end2end:\n",
        "            return self.forward_end2end(x)\n",
        "        \n",
        "        # 适配地平线BPU，分离输出为6个Tensor\n",
        "        bboxes = [self.cv2[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]\n",
        "        clses = [self.cv3[i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl)]\n",
        "        return (*bboxes, *clses)\n",
        "\n"
    ]

    new_lines = []
    in_detect_class = False
    skipping_forward = False
    inserted = False

    # 4. 状态机扫描
    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        # 状态 A: 进入 Detect 类
        if stripped.startswith("class Detect") and "nn.Module" in line:
            in_detect_class = True
            new_lines.append(line)
            continue

        # 状态 B: 在 Detect 类中
        if in_detect_class:
            # 检测是否退出了 Detect 类
            if indent == 0 and stripped.startswith("class ") and not stripped.startswith("class Detect"):
                in_detect_class = False
                skipping_forward = False
            
            # --- 【关键修改点】精确匹配 def forward( ---
            # 只有当行内容是 "def forward(" 开头时才匹配
            # 这样 "def forward_head(" 就不会被误伤了
            if indent == 4 and stripped.startswith("def forward("):
                skipping_forward = True
                print(f"    🗑️  发现目标 forward 函数 (行 {i+1})，开始移除...")
                
                if not inserted:
                    print("    ✨ 插入 RDK X5 专用 forward 代码...")
                    new_lines.extend(new_forward_code)
                    inserted = True
                continue

            # 状态 C: 正在删除旧 forward
            if skipping_forward:
                # 遇到下一个方法定义 (缩进4的 def) 或者 类结束，停止删除
                # 注意：这里会保留 forward_head (如果它在 forward 下面的话)
                if (indent == 4 and stripped.startswith("def ")) or (indent == 0 and stripped):
                    skipping_forward = False
                    print(f"    ⏹️  旧 forward 移除结束 (行 {i+1})。")
                    # 下一个函数（可能是 forward_head 或其他）会被正常保留
                else:
                    # 还在旧 forward 块里，跳过
                    continue

        # 其他所有行照常保留
        new_lines.append(line)

    # 5. 写回文件
    with open(head_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("✅ 源码重构完成！(已避开 forward_head，精准替换 forward)")
    return True

if __name__ == "__main__":
    if not modify_head():
        sys.exit(1)