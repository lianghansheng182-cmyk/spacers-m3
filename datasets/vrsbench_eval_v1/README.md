\# VRSBench 扩大验证集 (v1)



\## 数据来源

本验证集从 VRSBench 官方数据集中抽取，包括：

\- 训练集：`VRSBench\_train.json`

\- 验证集：`VRSBench\_EVAL\_referring.json`



\## 样本数量

| 任务类型 | 数量 |

|----------|------|

| Caption（图像描述） | 30 条 |

| VQA（视觉问答） | 30 条 |

| Grounding（目标定位） | 30 条 |

| \*\*合计\*\* | \*\*90 条\*\* |



\## 数据划分

\- Caption 和 VQA：来自 `train` 集

\- Grounding：来自 `val` 集



\## 字段格式

\### Caption

```json

{

&#x20; "id": "cap\_ext\_001",

&#x20; "task": "caption",

&#x20; "image": "datasets/vrsbench\_eval\_v1/images/xxx.png",

&#x20; "question": "请描述这张图片",

&#x20; "answer": "图片描述内容",

&#x20; "source\_split": "train"

}

