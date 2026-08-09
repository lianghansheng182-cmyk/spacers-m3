import json
import os
import sys

print("=" * 50)
print("VRSBench 扩大验证集校验工具")
print("=" * 50)

# 路径
jsonl_path = "datasets/vrsbench_eval_v1/candidate_manifest.jsonl"
errors = []

# 1. 检查文件是否存在
if not os.path.exists(jsonl_path):
    print("❌ 错误: 找不到 candidate_manifest.jsonl")
    sys.exit(1)

# 2. 读取并解析 JSONL
print(f"\n📂 读取: {jsonl_path}")
samples = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            samples.append(sample)
        except json.JSONDecodeError as e:
            errors.append(f"第 {line_num} 行 JSON 格式错误: {e}")

print(f"   读取了 {len(samples)} 条样本")

# 3. 检查总数是否为 90
if len(samples) != 90:
    errors.append(f"样本数量应为 90 条，实际为 {len(samples)} 条")

# 4. 检查三类任务是否各 30 条
task_counts = {}
for s in samples:
    task = s.get('task')
    task_counts[task] = task_counts.get(task, 0) + 1

for task, count in task_counts.items():
    print(f"   - {task}: {count} 条")

if task_counts.get('caption', 0) != 30:
    errors.append(f"Caption 应为 30 条，实际为 {task_counts.get('caption', 0)} 条")
if task_counts.get('vqa', 0) != 30:
    errors.append(f"VQA 应为 30 条，实际为 {task_counts.get('vqa', 0)} 条")
if task_counts.get('grounding', 0) != 30:
    errors.append(f"Grounding 应为 30 条，实际为 {task_counts.get('grounding', 0)} 条")

# 5. 检查 ID 是否重复
ids = [s.get('id') for s in samples]
if len(set(ids)) != len(ids):
    errors.append(f"存在重复 ID，总样本数 {len(ids)}，唯一 ID 数 {len(set(ids))}")

# 6. 必需字段检查
required_fields = {
    'caption': ['id', 'task', 'image', 'question', 'answer', 'source_split'],
    'vqa': ['id', 'task', 'image', 'question', 'answer', 'question_type', 'source_split'],
    'grounding': ['id', 'task', 'image', 'question', 'target', 'bbox', 'bbox_format', 'bbox_space', 'source_split']
}

for sample in samples:
    task = sample.get('task')
    if task in required_fields:
        for field in required_fields[task]:
            if field not in sample:
                errors.append(f"样本 {sample.get('id', 'unknown')} 缺少字段: {field}")

# 7. Grounding bbox 格式检查
for sample in samples:
    if sample.get('task') == 'grounding':
        bbox = sample.get('bbox')
        if bbox is None:
            errors.append(f"样本 {sample.get('id', 'unknown')} 缺少 bbox")
        elif not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"样本 {sample.get('id', 'unknown')} 的 bbox 格式错误，应为 4 个数字的列表")
        else:
            for val in bbox:
                if not isinstance(val, (int, float)) or val < 0 or val > 100:
                    errors.append(f"样本 {sample.get('id', 'unknown')} 的 bbox 值 {val} 超出 0-100 范围")

# 8. source_split 检查
for sample in samples:
    if 'source_split' not in sample:
        errors.append(f"样本 {sample.get('id', 'unknown')} 缺少 source_split")
    else:
        if sample['source_split'] not in ['train', 'val', 'test']:
            errors.append(f"样本 {sample.get('id', 'unknown')} 的 source_split 无效: {sample['source_split']}")

# 9. 输出结果
print("\n" + "=" * 50)
if errors:
    print(f"❌ 发现 {len(errors)} 个错误:")
    for e in errors:
        print(f"   - {e}")
    print("\n校验失败 ❌")
    sys.exit(1)
else:
    print("✅ 所有校验通过！")
    print("   - 总数: 90 条")
    print("   - 三类任务各 30 条")
    print("   - 无 ID 重复")
    print("   - 所有必需字段齐全")
    print("   - Grounding bbox 格式正确")
    print("   - source_split 有效")
    sys.exit(0)