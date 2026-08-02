import json

# 读取原始数据
with open('VRSBench_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 分别存储两种任务的样本
captions = []
vqas = []

# 遍历所有数据，按任务分类
for item in data:
    # 提取图片名
    image_name = item['image']
    
    # 从conversations中提取问题和答案
    question = None
    answer = None
    task_type = None
    
    for conv in item['conversations']:
        if conv['from'] == 'human':
            raw_question = conv['value']
            # 移除 <image> 标签
            clean_question = raw_question.replace('<image>', '').strip()
            
            if '[caption]' in clean_question:
                task_type = 'caption'
                clean_question = clean_question.replace('[caption]', '').strip()
                question = clean_question
            elif '[vqa]' in clean_question:
                task_type = 'vqa'
                clean_question = clean_question.replace('[vqa]', '').strip()
                question = clean_question
        elif conv['from'] == 'gpt':
            answer = conv['value']
    
    # 根据问题类型存入不同列表
    if question and answer:
        if task_type == 'caption':
            captions.append({
                'image': image_name,
                'question': question,
                'answer': answer
            })
        elif task_type == 'vqa':
            vqas.append({
                'image': image_name,
                'question': question,
                'answer': answer
            })

print(f"找到 Caption 样本: {len(captions)} 条")
print(f"找到 VQA 样本: {len(vqas)} 条")

# 各取前10条
selected_captions = captions[:10]
selected_vqas = vqas[:10]

# 生成 mini_eval.jsonl
output_file = 'mini_eval.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    # 写入 Caption 样本
    for idx, item in enumerate(selected_captions, 1):
        sample = {
            "id": f"cap_{idx:03d}",
            "task": "caption",
            "image": f"datasets/mini_eval/images/{item['image']}",
            "question": item['question'],
            "answer": item['answer'],
            "source_split": "train"
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 写入 VQA 样本
    for idx, item in enumerate(selected_vqas, 1):
        # 简单判断问题类型
        q_text = item['question'].lower()
        if '多少' in q_text or '几个' in q_text or 'count' in q_text:
            q_type = 'counting'
        elif '是' in q_text or '存在' in q_text or '有' in q_text:
            q_type = 'existence'
        else:
            q_type = 'general'
        
        sample = {
            "id": f"vqa_{idx:03d}",
            "task": "vqa",
            "image": f"datasets/mini_eval/images/{item['image']}",
            "question": item['question'],
            "answer": item['answer'],
            "question_type": q_type,
            "source_split": "train"
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\n✅ 已生成 {output_file}")
print(f"包含 {len(selected_captions)} 条 Caption + {len(selected_vqas)} 条 VQA = {len(selected_captions) + len(selected_vqas)} 条样本")