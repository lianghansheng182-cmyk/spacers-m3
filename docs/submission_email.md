# SPACERS-M3 提交邮件说明

您好，

本次提交包中的源代码、LoRA v1 adapter、技术报告、复现脚本和小型验证数据已随主提交 ZIP 提供。为避免主提交 ZIP 过大，正式复现需要的基座模型和公开 benchmark raw data 已单独放入百度网盘分享文件夹。

百度网盘链接：

```text
https://pan.baidu.com/s/1GrAYgwE0YkL8rRalwX4g4Q?pwd=r0k4
```

提取码：

```text
r0k4
```

分享文件夹：

```text
external_uploads
```

访问权限：公开分享链接 + 提取码，永久有效。

## 外部大文件

| 文件 | SHA256 | 解压后放置路径 |
| --- | --- | --- |
| `Qwen3-VL-2B-Instruct__SpaceRS-M3__base_model.zip` | `f97c7387001f1d371a0aa12e5232abaf42cb1c84f73fd9c7b4883bba56af70a7` | `submission_final/02_source_code/models/Qwen3-VL-2B-Instruct/` |
| `raw_benchmark_data__SpaceRS-M3.zip` | `9fe29f2dad8455de88dee725122523e50d9537cf64d1db5ccf17ea5ecd651208` | `submission_final/02_source_code/data/raw/` |

LoRA v1 adapter 已随主提交包放置于：

```text
submission_final/03_model/qwen3_vl_2b_lora_v1_adapter/
```

因此 LoRA adapter 不是 external artifact，不需要从百度网盘额外下载。External artifact 仅用于基座模型、公开 benchmark raw data 和其他不适合放入主提交 ZIP 的大容量文件。

推荐复现环境为 WSL2 Ubuntu / Linux。Windows native 仅提供静态检查或有限单图替代流程，不作为正式 full smoke 验证环境。
