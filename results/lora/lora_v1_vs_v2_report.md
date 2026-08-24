# LoRA v1 vs LoRA v2 Report

Generated: 2026-08-24T21:26:00

## V2 Data Change

Caption and VQA source samples are kept from v1. Grounding train samples add one Prompt-v1-style Chinese wrapper per source sample, with unchanged GT bbox and canonical assistant output.

```json
{
  "created": "2026-08-24T18:52:24",
  "source_dir": "datasets/vrsbench_lora_v1",
  "target_dir": "datasets/vrsbench_lora_v2",
  "train_count": 12000,
  "dev_count": 600,
  "train_task_counts": {
    "caption": 3000,
    "grounding": 6000,
    "vqa": 3000
  },
  "dev_task_counts": {
    "caption": 200,
    "grounding": 200,
    "vqa": 200
  },
  "augmented_grounding_train_samples": 3000,
  "unique_referenced_images": 3199,
  "augmentation": {
    "grounding_prompt_v1_zh_per_source": 1,
    "assistant_format": "{<x1><y1><x2><y2>}",
    "bbox_format": "xyxy",
    "bbox_space": "normalized_0_100"
  }
}
```

## Smoke Test

```json
{
  "status": "PASS",
  "mode": "smoke",
  "output_dir": "/home/lhs/spacers_m3/results/lora/qwen3_vl_2b_lora_v2_smoke",
  "optimizer_steps": 50,
  "micro_steps": 400,
  "loss_first_window": 3.317212608456612,
  "loss_last_window": 1.0194842470809817,
  "loss_min": 0.012480583041906357,
  "loss_max": 7.010941028594971,
  "loss_nan": false,
  "oom_count": 0,
  "elapsed_seconds": 209.196,
  "peak_vram_gib": 4.987,
  "total_parameters": 2136248320,
  "trainable_parameters": 8716288,
  "trainable_ratio_percent": 0.408,
  "task_weight_config": {
    "caption": 1.0,
    "vqa": 1.0,
    "grounding": 1.75
  },
  "optimizer_step_task_counts": {
    "caption": 111,
    "vqa": 94,
    "grounding": 195
  },
  "checkpoints": [
    "checkpoint-step-25",
    "checkpoint-step-50"
  ],
  "final_adapter": "/home/lhs/spacers_m3/results/lora/qwen3_vl_2b_lora_v2_smoke/adapter_final"
}
```

## Grounding Target Audit

```json
{
  "status": "PASS",
  "grounding_rows": 6000,
  "sampled_rows": 30,
  "target_kind_counts": {
    "bbox_brace_angle": 30
  },
  "example_target": "{<37><62><71><94>}<|im_end|>\n"
}
```

## Full Training

```json
{
  "status": "PASS",
  "mode": "full",
  "output_dir": "/home/lhs/spacers_m3/results/lora/qwen3_vl_2b_lora_v2",
  "optimizer_steps": 1125,
  "micro_steps": 9000,
  "loss_first_window": 3.321793419122696,
  "loss_last_window": 0.7909562805667519,
  "loss_min": 0.0014690958196297288,
  "loss_max": 7.131561756134033,
  "loss_nan": false,
  "oom_count": 0,
  "elapsed_seconds": 7231.196,
  "peak_vram_gib": 5.077,
  "total_parameters": 2136248320,
  "trainable_parameters": 8716288,
  "trainable_ratio_percent": 0.408,
  "task_weight_config": {
    "caption": 1.0,
    "vqa": 1.0,
    "grounding": 1.75
  },
  "optimizer_step_task_counts": {
    "caption": 2361,
    "vqa": 2396,
    "grounding": 4243
  },
  "checkpoints": [
    "checkpoint-step-100",
    "checkpoint-step-1000",
    "checkpoint-step-1025",
    "checkpoint-step-1050",
    "checkpoint-step-1075",
    "checkpoint-step-1100",
    "checkpoint-step-1125",
    "checkpoint-step-125",
    "checkpoint-step-150",
    "checkpoint-step-175",
    "checkpoint-step-200",
    "checkpoint-step-225",
    "checkpoint-step-25",
    "checkpoint-step-250",
    "checkpoint-step-275",
    "checkpoint-step-300",
    "checkpoint-step-325",
    "checkpoint-step-350",
    "checkpoint-step-375",
    "checkpoint-step-400",
    "checkpoint-step-425",
    "checkpoint-step-450",
    "checkpoint-step-475",
    "checkpoint-step-50",
    "checkpoint-step-500",
    "checkpoint-step-525",
    "checkpoint-step-550",
    "checkpoint-step-575",
    "checkpoint-step-600",
    "checkpoint-step-625",
    "checkpoint-step-650",
    "checkpoint-step-675",
    "checkpoint-step-700",
    "checkpoint-step-725",
    "checkpoint-step-75",
    "checkpoint-step-750",
    "checkpoint-step-775",
    "checkpoint-step-800",
    "checkpoint-step-825",
    "checkpoint-step-850",
    "checkpoint-step-875",
    "checkpoint-step-900",
    "checkpoint-step-925",
    "checkpoint-step-950",
    "checkpoint-step-975"
  ],
  "final_adapter": "/home/lhs/spacers_m3/results/lora/qwen3_vl_2b_lora_v2/adapter_final"
}
```

## Mini Eval 30

| Metric | Base | LoRA v1 | LoRA v2 | v2-v1 |
|---|---:|---:|---:|---:|
| VQA accuracy | 0.5 | 0.7 | 0.7 | +0 |
| Grounding mean IoU | 0.565188 | 0.617452 | 0.639133 | +0.021681 |
| Grounding Acc@0.5 | 0.7 | 0.8 | 0.8 | +0 |
| Grounding Acc@0.7 | 0.4 | 0.2 | 0.4 | +0.2 |
| Grounding parse failures | 0 | 0 | 0 | +0 |
| Mean latency seconds | 4.20324 | 1.7851 | 1.32359 | -0.46151 |
| P95 latency seconds | 12.1252 | 4.81884 | 3.40144 | -1.4174 |
| Peak VRAM MB | 4184.17 | 4201.32 | 4201.32 | +0 |

## VRSBench Eval v1 90

| Metric | Base | LoRA v1 | LoRA v2 | v2-v1 |
|---|---:|---:|---:|---:|
| VQA accuracy | 0.6 | 0.7 | 0.733333 | +0.033333 |
| Grounding mean IoU | 0.436447 | 0.443984 | 0.382554 | -0.06143 |
| Grounding Acc@0.5 | 0.466667 | 0.466667 | 0.4 | -0.066667 |
| Grounding Acc@0.7 | 0.3 | 0.266667 | 0.266667 | +0 |
| Grounding parse failures | 0 | 1 | 0 | -1 |
| Mean latency seconds | 3.49209 | 1.76847 | 1.51805 | -0.250419 |
| P95 latency seconds | 10.7164 | 5.11779 | 4.22675 | -0.891039 |
| Peak VRAM MB | 4184.17 | 4203.17 | 4203.17 | +0 |

## 90-Eval Grounding V1 vs V2

```json
{
  "improved": 9,
  "declined": 10,
  "unchanged": 11,
  "parse failure": 0
}
```

V2 geometry/error categories:

```json
{
  "bbox boundary inaccurate": 1,
  "bbox too large": 4,
  "bbox too small": 7,
  "mostly correct": 12,
  "target mismatch or multi-target": 6
}
```

| sample_id | v1 prediction | v2 prediction | GT | v1 IoU | v2 IoU | delta IoU | v2 parse | delta category | v2 error category |
|---|---|---|---:|---:|---:|---:|---|---|---|
| grounding_ext_001 | <code>{<45><66><63><82>}</code> | <code>{<45><67><63><83>}</code> | [46, 68, 63, 82] | 0.826389 | 0.826389 | +0.000000 | ok | unchanged | mostly correct |
| grounding_ext_002 | <code>{<57><47><74><68>}</code> | <code>{<59><48><73><65>}</code> | [56, 48, 73, 66] | 0.768000 | 0.777778 | +0.009778 | ok | unchanged | mostly correct |
| grounding_ext_003 | <code>{<0><0><67><62>}</code> | <code>{<0><0><71><63>}</code> | [0, 0, 80, 67] | 0.775000 | 0.834515 | +0.059515 | ok | improved | mostly correct |
| grounding_ext_004 | <code>{<14><47><18><50>}</code> | <code>{<14><43><18><49>}</code> | [14, 46, 20, 47] | 0.000000 | 0.153846 | +0.153846 | ok | improved | bbox too large |
| grounding_ext_005 | <code>{<2><15><13><21>}</code> | <code>{<10><15><16><20>}</code> | [41, 5, 44, 9] | 0.000000 | 0.000000 | +0.000000 | ok | unchanged | target mismatch or multi-target |
| grounding_ext_006 | <code>{<71><45><79><77>}</code> | <code>{<70><44><78><99>}</code> | [71, 44, 89, 100] | 0.253968 | 0.362183 | +0.108215 | ok | improved | bbox too small |
| grounding_ext_007 | <code>{<39><83><45><91>}</code> | <code>{<33><85><44><90>}</code> | [35, 84, 57, 91] | 0.262500 | 0.274390 | +0.011890 | ok | unchanged | bbox too small |
| grounding_ext_008 | <code>{<10><29><48><66>}</code> | <code>{<11><28><47><62>}</code> | [10, 28, 47, 65] | 0.923077 | 0.894083 | -0.028994 | ok | declined | mostly correct |
| grounding_ext_009 | <code>{<64><57><100><99>}</code> | <code>{<64><58><99><100>}</code> | [66, 58, 100, 100] | 0.901682 | 0.916667 | +0.014985 | ok | unchanged | mostly correct |
| grounding_ext_010 | <code>{<44><81><54><93>}</code> | <code>{<44><81><53><92>}</code> | [48, 83, 83, 100] | 0.091603 | 0.069337 | -0.022266 | ok | declined | bbox too small |
| grounding_ext_011 | <code>{<6><80><38><100>}</code> | <code>{<6><79><41><100>}</code> | [7, 80, 37, 100] | 0.937500 | 0.816327 | -0.121173 | ok | declined | mostly correct |
| grounding_ext_012 | <code>{52><9><60><16><10><13></code> | <code>{<53><8><59><14>}</code> | [52, 7, 60, 16] | 0.000000 | 0.500000 | +0.500000 | ok | improved | mostly correct |
| grounding_ext_013 | <code>{<37><85><43><93>}</code> | <code>{<25><88><35><92>}</code> | [0, 96, 14, 100] | 0.000000 | 0.000000 | +0.000000 | ok | unchanged | target mismatch or multi-target |
| grounding_ext_014 | <code>{<50><41><100><99>}</code> | <code>{<46><70><73><99>}</code> | [44, 41, 100, 100] | 0.877724 | 0.236985 | -0.640739 | ok | declined | bbox too small |
| grounding_ext_015 | <code>{56><0><76><70>}</code> | <code>{<57><62><76><70>}</code> | [62, 0, 75, 68] | 0.631429 | 0.081420 | -0.550009 | ok | declined | bbox too small |
| grounding_ext_016 | <code>{<11><12><74><50>}</code> | <code>{<11><12><74><49>}</code> | [17, 18, 66, 42] | 0.491228 | 0.504505 | +0.013277 | ok | unchanged | mostly correct |
| grounding_ext_017 | <code>{<68><72><88><93>}</code> | <code>{<72><73><85><85>}</code> | [70, 68, 89, 88] | 0.562500 | 0.410526 | -0.151974 | ok | declined | bbox too small |
| grounding_ext_018 | <code>{<29><17><48><57>}</code> | <code>{<29><17><47><57>}</code> | [29, 18, 47, 57] | 0.923684 | 0.975000 | +0.051316 | ok | improved | mostly correct |
| grounding_ext_019 | <code>{<85><87><96><100>}</code> | <code>{<69><79><77><92>}</code> | [89, 87, 95, 100] | 0.545455 | 0.000000 | -0.545455 | ok | declined | target mismatch or multi-target |
| grounding_ext_020 | <code>{<79><61><85><76>}</code> | <code>{<81><63><84><75>}</code> | [80, 62, 84, 75] | 0.577778 | 0.692308 | +0.114530 | ok | improved | mostly correct |
| grounding_ext_021 | <code>{<19><56><23><62>}</code> | <code>{<22><61><28><64>}</code> | [20, 59, 27, 63] | 0.209302 | 0.277778 | +0.068476 | ok | improved | bbox boundary inaccurate |
| grounding_ext_022 | <code>{<80><37><100><66>}</code> | <code>{<81><37><99><63>}</code> | [75, 23, 100, 71] | 0.483333 | 0.390000 | -0.093333 | ok | declined | bbox too small |
| grounding_ext_023 | <code>{<50><11><65><20>}</code> | <code>{<51><12><65><19>}</code> | [55, 13, 65, 18] | 0.370370 | 0.510204 | +0.139834 | ok | improved | mostly correct |
| grounding_ext_024 | <code>{<8><45><11><51>}</code> | <code>{<6><43><10><50>}</code> | [7, 48, 10, 50] | 0.200000 | 0.214286 | +0.014286 | ok | unchanged | bbox too large |
| grounding_ext_025 | <code>{<53><0><65><5>}</code> | <code>{<33><36><36><40>}</code> | [35, 0, 39, 4] | 0.000000 | 0.000000 | +0.000000 | ok | unchanged | target mismatch or multi-target |
| grounding_ext_026 | <code>{<61><0><72><6>}</code> | <code>{<16><0><24><3>}</code> | [2, 7, 5, 19] | 0.000000 | 0.000000 | +0.000000 | ok | unchanged | target mismatch or multi-target |
| grounding_ext_027 | <code>{<6><71><11><76>}</code> | <code>{<5><71><11><77>}</code> | [13, 55, 15, 60] | 0.000000 | 0.000000 | +0.000000 | ok | unchanged | bbox too large |
| grounding_ext_028 | <code>{<10><1><44><55>}</code> | <code>{<8><20><42><53>}</code> | [6, 20, 46, 57] | 0.559737 | 0.758108 | +0.198371 | ok | improved | mostly correct |
| grounding_ext_029 | <code>{<88><52><95><99>}</code> | <code>{<65><52><86><100>}</code> | [89, 54, 100, 96] | 0.467532 | 0.000000 | -0.467532 | ok | declined | bbox too large |
| grounding_ext_030 | <code>{<57><20><75><37>}</code> | <code>{<11><25><36><48>}</code> | [58, 21, 74, 34] | 0.679739 | 0.000000 | -0.679739 | ok | declined | target mismatch or multi-target |
