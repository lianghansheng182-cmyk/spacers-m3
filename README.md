# Qwen3-VL Single Image Smoke Test

项目绝对路径：`/home/lhs/spacers_m3`

## 文件路径

- 虚拟环境：`/home/lhs/spacers_m3/.venv`
- 推理脚本：`/home/lhs/spacers_m3/src/inference/run_single.py`
- 运行脚本：`/home/lhs/spacers_m3/scripts/run_single.sh`
- 测试图片：`/home/lhs/spacers_m3/data/test/demo.jpeg`
- 本地模型目录：`/home/lhs/spacers_m3/models/Qwen3-VL-2B-Instruct`
- 结果 JSON：`/home/lhs/spacers_m3/results/smoke_test.json`
- 环境检查日志：`/home/lhs/spacers_m3/logs/environment_check.txt`
- 依赖列表：`/home/lhs/spacers_m3/requirements.txt`
- pip freeze：`/home/lhs/spacers_m3/logs/pip_freeze.txt`

## 激活环境

```bash
cd /home/lhs/spacers_m3
source /home/lhs/spacers_m3/.venv/bin/activate
```

## GPU 检查

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

## 单图推理

```bash
bash /home/lhs/spacers_m3/scripts/run_single.sh
```

运行脚本默认从 `/home/lhs/spacers_m3/models/Qwen3-VL-2B-Instruct` 加载模型；如需改用其他本地模型目录：

```bash
SPACERS_M3_MODEL_PATH=/path/to/Qwen3-VL-2B-Instruct bash /home/lhs/spacers_m3/scripts/run_single.sh
```

或直接调用：

```bash
/home/lhs/spacers_m3/.venv/bin/python /home/lhs/spacers_m3/src/inference/run_single.py \
  --image /home/lhs/spacers_m3/data/test/demo.jpeg \
  --question "请详细描述图像中的主要场景、物体及其空间位置关系。" \
  --output /home/lhs/spacers_m3/results/smoke_test.json
```

## 替换测试图片

把新的本地图片复制或移动到：

```bash
/home/lhs/spacers_m3/data/test/demo.jpeg
```

也可以保留原文件，直接在命令中把 `--image` 改成你的图片绝对路径。

## 常见报错处理

- `图片不存在或不是文件`：检查 `--image` 是否为真实文件路径。
- `CUDA 显存不足`：关闭其他 GPU 进程后重试，或使用分辨率更小的图片；本项目不使用量化。
- `torch.cuda.is_available() == False`：检查 Windows NVIDIA 驱动、WSL GPU 支持和 `nvidia-smi` 是否正常。
- `ModuleNotFoundError`：确认已激活 `/home/lhs/spacers_m3/.venv`，必要时运行 `python -m pip install -r /home/lhs/spacers_m3/requirements.txt`。
- 模型下载失败：检查 Hugging Face 网络访问，删除未完成缓存后重试运行脚本。
- 结果 JSON 缺少字段：重新运行 `bash /home/lhs/spacers_m3/scripts/run_single.sh`，并检查终端错误输出。
