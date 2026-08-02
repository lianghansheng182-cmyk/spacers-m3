from modelscope import snapshot_download

# 模型名称
model_name = "Qwen/Qwen2-VL-2B-Instruct"
# 下载到你电脑的指定目录，请替换为你的实际路径
local_dir = "E:/Qwen2-VL-2B-Instruct"

print(f"⏳ 开始从 ModelScope 下载模型: {model_name}")
print(f"📁 下载到: {local_dir}")
print("⏳ 文件较大，请耐心等待...")

model_dir = snapshot_download(
    model_name,
    cache_dir=local_dir,
    revision='master' # 推荐指定版本分支
)

print(f"✅ 模型下载完成！保存在: {model_dir}")