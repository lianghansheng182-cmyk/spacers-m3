# VRSBench mini_eval Grounding 人工审核总结

## 审核范围

本次人工审核对应现有 `mini_eval` 中 10 条 Grounding 样本，并生成了 10 张带标准框的预览图：

```text
results/data_audit/grounding_previews/
```

## 审核结果

- 审核总数：10
- keep：10
- fix：0
- uncertain：0
- invalid：0

现有 10 条 mini_eval Grounding 标准框均保留。

## 说明

本文件只总结原 30 条 `mini_eval` 中的 Grounding 审核结果。

扩大验证集 `datasets/vrsbench_eval_v1/` 另发现一条官方 0-100 整数归一化后的退化框：`grounding_ext_004`。该问题不属于原 10 条 mini_eval 审核样本，已通过 `datasets/vrsbench_eval_v1/bbox_repairs.json` 单独记录和处理。
