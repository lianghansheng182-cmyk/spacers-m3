import json
import os
import shutil
from pathlib import Path

# ============================================================
# 配置区域（组长使用前需要修改这里）
# ============================================================
# VRSBench 原始图片所在目录（组长需要改成自己的路径）
VRSBENCH_IMAGES_DIR = Path(r"E:\path\to\VRSBench\images")  # ← 修改这里

# 输出目录（相对于项目根目录）
OUTPUT_DIR = Path("datasets/vrsbench_eval_v1/images")
MANIFEST_PATH = Path("datasets/vrsbench_eval_v1/candidate_manifest.jsonl")
# ============================================================

def main():
    # 检查清单文件是否存在
    if not MANIFEST_PATH.exists():
        print(f"❌ 错误: 找不到 {MANIFEST_PATH}")
        print("   请先运行 extract_90_candidates.py 生成候选清单")
        return

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取候选清单
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    print(f"📂 读取到 {len(samples)} 条候选样本")
    print(f"📁 图片源目录: {VRSBENCH_IMAGES_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print()

    # 提取图片
    copied = 0
    missing = 0
    for sample in samples:
        # 从 image 路径中提取文件名
        image_path = sample['image']
        filename = os.path.basename(image_path)
        src = VRSBENCH_IMAGES_DIR / filename
        dst = OUTPUT_DIR / filename

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
            print(f"✅ 已复制: {filename}")
        else:
            missing += 1
            print(f"❌ 找不到: {filename} (尝试路径: {src})")

    print()
    print(f"✅ 完成！成功复制 {copied} 张图片，{missing} 张缺失。")
    if missing > 0:
        print("⚠️ 请检查 VRSBENCH_IMAGES_DIR 路径是否正确，或手动补齐缺失图片。")

if __name__ == "__main__":
    main()