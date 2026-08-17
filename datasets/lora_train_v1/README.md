# LoRA Training Data for VRSBench

## 数据来源
- VRSBench_train.json
- VRSBench 官方训练集图片

## 数据量
| 任务 | 训练集 | 验证集 |
|------|--------|--------|
| Caption | 3000 | 0 |
| VQA | 3000 | 525 |
| Grounding | 3000 | 75 |
| **合计** | **9000** | **600** |

## 泄漏检查
- 是否包含评测集图片: True
- 泄漏图片数: 0

## 排除的图片
共排除 104 张图片

## 文件说明
- train.jsonl: 训练集
- train_dev.jsonl: 验证集
- leakage_audit_report.json: 泄漏审计报告
