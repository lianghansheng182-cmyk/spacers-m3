# Qwen3-VL-2B Zero-Shot Error Analysis

分析对象：`results/vrsbench_mini/qwen3_vl_2b_zero_shot/`

数据来源：现有 30 条 baseline 输出，未重新跑模型推理。验收命令已通过；`generate_benchmark_report.py` 读取现有 `predictions.jsonl` 聚合 30 条样本。

## Overall

- VQA：10 条中 5 条 wrong，accuracy = 0.5。
- Grounding：10 条中 3 条 IoU < 0.5，mean IoU = 0.565188，Acc@0.5 = 0.7。
- Caption：本轮没有自动错误标签，未作为 8 个错误样本统计。
- Grounding 输出格式不统一，例如 `box`、`box_3d`、`box_1`，但 10 条均成功解析；本轮 3 个 Grounding 错误不是解析失败。

## Error Cases

| ID | Task | Image | GT | Prediction | Main Category | Prompt-solvable? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `vqa_001` | VQA | `datasets/mini_eval/images/00002_0000.png` | `expressway-toll-station` | `blue structure` | 视觉理解错误 | Partial | 图像中心是蓝顶收费站/卡口。模型只描述颜色和外观，没有识别出功能类别。需要更强的遥感设施类别识别；prompt 可要求回答功能类别，但不一定稳定解决。 |
| `vqa_005` | VQA | `datasets/mini_eval/images/00003_0000.png` | `urban` | `residential` | 输出格式或答案表达问题 | Yes | 图中是密集建成区，`residential` 是合理子类，但 exact match 只接受 `urban`。更像标签体系/答案粒度不一致，可通过限定候选标签或要求使用数据集标签风格改善。 |
| `vqa_008` | VQA | `datasets/mini_eval/images/00003_0000.png` | `middle` | `bottom` | 计数/属性判断错误 | Partial | 车站位于图像下半部但主体/轨道区域跨越中下位置，`middle` 与 `bottom` 边界较模糊。prompt 可定义 top/middle/bottom 三等分和按目标中心点判断。 |
| `vqa_009` | VQA | `datasets/mini_eval/images/00003_0000.png` | `Yes` | `no` | 视觉理解错误 | Yes | 轨道沿图像右侧方向延伸；模型没有正确处理 eastward。若默认北向上，应明确 east = image right，并要求只答 yes/no。 |
| `vqa_010` | VQA | `datasets/mini_eval/images/00004_0000.png` | `Chimney` | `cooling tower` | 输出格式或答案表达问题 | Partial | 中心大型圆形工业塔视觉上很像 cooling tower，GT 使用 `Chimney`。这是类别词汇/粒度差异，prompt 可要求用简短通用标签，但稳定对齐可能需要训练或标签映射。 |
| `grounding_002` | Grounding | `datasets/mini_eval/images/P0003_0002.png` | `[11, 96, 24, 100]` | `[0.0, 89.9, 11.3, 100.0]`, IoU 0.007276 | Grounding目标理解错误 | Partial | 目标是底部偏左的白色大车，预测落到更靠左的边缘车辆/局部目标。多车辆场景中描述“bottom-left large vehicle”消歧不足，且框明显左偏。 |
| `grounding_009` | Grounding | `datasets/mini_eval/images/P0007_0004.png` | `[5, 24, 12, 57]` | `[5.6, 37.2, 15.4, 55.6]`, IoU 0.401145 | Grounding框偏大/偏小/偏移 | Partial | 模型找到了左侧道路边的大车，但框整体向下偏移，漏掉上半部分，右边界也偏宽。属于定位精度问题。 |
| `grounding_010` | Grounding | `datasets/mini_eval/images/P0007_0005.png` | `[78, 59, 84, 94]` | `[77.9, 59.9, 96.1, 94.8]`, IoU 0.319398 | Grounding框偏大/偏小/偏移 | Partial | 模型定位到右侧车辆区域，但横向框过宽，把相邻右侧车辆/拖挂也包含进来。目标理解大体正确，主要是单目标 tight box 控制不足。 |

## Root Cause Summary

1. `vqa_001`：视觉理解错误，遥感设施类别未识别，只回答外观。
2. `vqa_005`：答案表达/标签粒度问题，`residential` 与 `urban` 语义接近但 exact match 失败。
3. `vqa_008`：空间属性判断边界模糊，需定义 top/middle/bottom 判定规则。
4. `vqa_009`：方向理解错误，eastward 需要和图像坐标明确绑定。
5. `vqa_010`：类别词汇/答案表达问题，`cooling tower` 与 `Chimney` 存在近义或标注粒度差异。
6. `grounding_002`：Grounding 目标理解错误，多车场景消歧失败并左偏。
7. `grounding_009`：Grounding 框偏移，目标基本正确但 y1 明显偏下。
8. `grounding_010`：Grounding 框偏大，目标区域正确但包含相邻对象。

## Prompt vs Training

更可能通过 Prompt 解决：

- `vqa_005`：限定答案标签集合或要求使用上位类。
- `vqa_008`：明确 top/middle/bottom 按目标中心点和三等分区域判断。
- `vqa_009`：明确北向上、east = image right，并强制 yes/no。
- Grounding 输出格式不统一：虽然本轮解析成功，但 prompt 可统一为单行 `<box>x1 y1 x2 y2</box>` 或 JSON `{"box":[x1,y1,x2,y2]}`。

更可能需要 LoRA/训练或数据增强解决：

- `vqa_001`：收费站/卡口这类遥感设施功能识别。
- `vqa_010`：工业设施细粒度类别与数据集标签对齐。
- `grounding_002`：多实例车辆中的目标消歧。
- `grounding_009`、`grounding_010`：归一化框的 tight box 精度，尤其是长条车辆和相邻实例边界。

## Prompt v2 Suggestions

Caption v2：

```text
Describe the remote-sensing image in 2-4 concise sentences.
Mention the main scene type, key objects, their approximate positions, and visible counts only when clear.
Use remote-sensing category names when possible, such as toll station, train station, chimney, vehicle, road, track, building, farmland, urban area, rural area.
Do not invent fine details that are not visible.
```

VQA v2：

```text
Answer the question using the shortest dataset-style answer.
If the question asks for a scene type, prefer broad labels such as urban, rural, industrial, residential, farmland, water, road, railway.
If the question asks top/middle/bottom or left/center/right, judge by the target object's center point in the image thirds.
Assume north is up in the image: east is right, west is left, north is top, south is bottom.
For yes/no questions, answer only "yes" or "no".
For object/category questions, answer the functional object category, not only color or appearance.
```

Grounding v2：

```text
Locate exactly one target object described in the question.
Return only one bounding box in normalized 0-100 xyxy coordinates: [x1, y1, x2, y2].
The box must tightly cover the full visible extent of the target object and exclude adjacent objects.
If several similar objects appear, use all location words in the target description, such as left edge, right side, bottom-left, top, or center, to disambiguate.
Do not output 3D boxes, multiple boxes, markdown fences, or extra explanation.
```
