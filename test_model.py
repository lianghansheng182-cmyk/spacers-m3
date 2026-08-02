import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import json
import time
import os

print("=" * 50)
print("开始测试 Qwen2-VL-2B 模型（本地加载）")
print("=" * 50)

# 检查设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型（使用本地下载的模型）
print("\n⏳ 正在加载本地模型...")
model_path = "E:/Qwen2-VL-2B-Instruct/models/Qwen--Qwen2-VL-2B-Instruct/snapshots/master"

try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        ignore_mismatched_sizes=True
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 读取测试数据
jsonl_path = "datasets/mini_eval/mini_eval.jsonl"
if not os.path.exists(jsonl_path):
    print(f"❌ 找不到文件: {jsonl_path}")
    exit()

with open(jsonl_path, 'r', encoding='utf-8') as f:
    samples = [json.loads(line) for line in f.readlines()]

print(f"\n✅ 加载了 {len(samples)} 条测试样本")

# 只测试前3条（节省时间）
test_samples = samples[:3]

for idx, sample in enumerate(test_samples, 1):
    print(f"\n{'='*30}")
    print(f"测试样本 {idx}/{len(test_samples)}")
    print(f"任务类型: {sample['task']}")
    print(f"问题: {sample['question']}")
    print(f"标准答案: {sample['answer']}")
    
    # 加载图片
    image_path = sample['image']
    if not os.path.exists(image_path):
        print(f"⚠️ 图片不存在: {image_path}")
        continue
    
    try:
        image = Image.open(image_path)
        print(f"✅ 图片加载成功: {image_path}")
        print(f"   图片尺寸: {image.size}")
        
        # 构造对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample['question']}
                ]
            }
        ]
        
        # 处理输入
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # 推理计时
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        end_time = time.time()
        
        # 解码输出
        response = processor.decode(outputs[0], skip_special_tokens=True)
        elapsed_time = end_time - start_time
        
        print(f"\n🤖 模型回答: {response}")
        print(f"⏱️ 推理耗时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")

print("\n" + "=" * 50)
print("测试完成！")