# VRSBench 扩大验证集（v1）

## 用途

该目录用于项目内部扩大验证，不代表官方 VRSBench 榜单成绩。

当前共 90 条样本：

| 任务 | 数量 | 来源 |
|---|---:|---|
| Caption | 30 | VRSBench train |
| VQA | 30 | VRSBench train |
| Grounding | 30 | VRSBench val |
| 合计 | 90 | 官方 VRSBench 图片与标注 |

90 张本次实际使用的图片已整理到：

```text
datasets/vrsbench_eval_v1/images/
```

因此仅使用本仓库即可复现这 90 条内部验证，不需要另外下载完整 VRSBench 才能运行本子集。

## 文件

```text
datasets/vrsbench_eval_v1/
├── candidate_manifest.jsonl
├── bbox_repairs.json
├── images/
└── README.md
```

- `candidate_manifest.jsonl`：保留从官方数据抽取的候选样本和原始归一化标注。
- `bbox_repairs.json`：仅记录已确认的退化 Grounding 框及内部评测修复规则。
- `images/`：90 条样本对应图片。

## Grounding 坐标

Grounding 使用：

```text
bbox_format = xyxy
bbox_space  = normalized_0_100
```

有效框必须满足：

```text
0 <= x1 < x2 <= 100
0 <= y1 < y2 <= 100
```

### 已知退化标注

`grounding_ext_004` 在官方 0-100 整数归一化结果中为：

```text
[14, 46, 20, 46]
```

该框高度为 0，不适合直接计算 IoU。

为避免静默篡改官方来源，`candidate_manifest.jsonl` 保留原始值，并在
`bbox_repairs.json` 中显式记录内部评测修复：

```text
[14, 46, 20, 47]
```

修复规则为最小确定性扩展：只把 `y2` 增加 1 个归一化单位。正式报告中如使用该样本，应说明这一内部修复。

评测前可生成已应用修复的 manifest：

```bash
python scripts/data/materialize_vrsbench_eval_v1.py
```

输出：

```text
datasets/vrsbench_eval_v1/resolved_manifest.jsonl
```

后续模型评测应优先使用 `resolved_manifest.jsonl`。

## 数据校验

运行：

```bash
python scripts/data/validate_vrsbench_eval_v1.py
```

校验内容包括：

- 总数是否为 90；
- Caption / VQA / Grounding 是否各 30 条；
- ID 是否重复；
- 必需字段是否齐全；
- `source_split` 是否有效；
- 90 条图片路径是否真实存在；
- Grounding bbox 是否为 `xyxy`；
- bbox 是否位于 0-100；
- 是否满足 `x1 < x2`、`y1 < y2`；
- 退化 bbox 是否存在显式、可追溯的修复记录。

## 数据说明

本子集仅包含项目当前内部验证实际需要的 90 张图片，不包含完整 VRSBench 原始数据。

Caption / VQA 使用 train 数据，因此相关结果应称为“内部验证结果”，不能当作 VRSBench 官方测试成绩。Grounding 样本来自 val。

不得自行编造问题、答案或目标框。若后续发现官方归一化标注存在异常，应保留原始值，并通过显式修复记录或替换为另一条官方样本处理。
