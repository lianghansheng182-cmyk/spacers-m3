import json
import os
import sys
from pathlib import Path

# 配置
DATA_DIR = Path("datasets/lora_train_v1")
TRAIN_FILE = DATA_DIR / "train.jsonl"
DEV_FILE = DATA_DIR / "train_dev.jsonl"

errors = []

def check_bbox(bbox):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "bbox 不是 4 个元素的列表"
    x1, y1, x2, y2 = bbox
    if not all(isinstance(v, int) for v in bbox):
        return False, "bbox 元素不是整数"
    if not (0 <= x1 <= 100 and 0 <= y1 <= 100 and 0 <= x2 <= 100 and 0 <= y2 <= 100):
        return False, f"bbox 值超出 0-100 范围: {bbox}"
    if x1 >= x2:
        return False, f"x1 >= x2: {bbox}"
    if y1 >= y2:
        return False, f"y1 >= y2: {bbox}"
    return True, ""

def validate_file(filepath, name):
    print(f"\n校验 {name}: {filepath}")
    if not filepath.exists():
        errors.append(f"{name} 文件不存在")
        return
    
    count = 0
    required_fields = {
        'caption': ['image', 'question', 'answer', 'task'],
        'vqa': ['image', 'question', 'answer', 'task'],
        'grounding': ['image', 'question', 'target', 'bbox', 'task'],
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{name} 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            
            count += 1
            task = data.get('task')
            if task not in required_fields:
                errors.append(f"{name} 第 {line_num} 行 task 字段无效: {task}")
                continue
            
            for field in required_fields[task]:
                if field not in data:
                    errors.append(f"{name} 第 {line_num} 行缺少字段: {field}")
            
            if task == 'grounding':
                bbox = data.get('bbox')
                if bbox is None:
                    errors.append(f"{name} 第 {line_num} 行缺少 bbox")
                else:
                    valid, msg = check_bbox(bbox)
                    if not valid:
                        errors.append(f"{name} 第 {line_num} 行 bbox 无效: {msg}")
    
    print(f"  共 {count} 条记录")

print("=" * 60)
print("校验 LoRA 训练数据")
print("=" * 60)

validate_file(TRAIN_FILE, "train.jsonl")
validate_file(DEV_FILE, "train_dev.jsonl")

print("\n" + "=" * 60)
if errors:
    print(f"❌ 发现 {len(errors)} 个错误:")
    for e in errors[:20]:  # 只显示前20个
        print(f"  - {e}")
    if len(errors) > 20:
        print(f"  ... 还有 {len(errors) - 20} 个错误")
    sys.exit(1)
else:
    print("✅ 所有校验通过！")
    sys.exit(0)