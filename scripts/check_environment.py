import subprocess
import torch
import os
import platform
from datetime import datetime

os.makedirs("logs", exist_ok=True)

output_lines = []
output_lines.append("=" * 40)
output_lines.append(f"Environment Information")
output_lines.append(f"Generated at: {datetime.now()}")
output_lines.append("=" * 40)

output_lines.append("\\n--- Python Version ---")
output_lines.append(platform.python_version())

output_lines.append("\\n--- PyTorch Version ---")
output_lines.append(torch.__version__)

output_lines.append("\\n--- CUDA Available ---")
output_lines.append(str(torch.cuda.is_available()))

output_lines.append("\\n--- GPU Full Name ---")
if torch.cuda.is_available():
    output_lines.append(torch.cuda.get_device_name(0))
else:
    output_lines.append("N/A")

output_lines.append("\\n--- GPU Total Memory (MB) ---")
if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
    output_lines.append(f"{total_mem:.2f}")
else:
    output_lines.append("N/A")

output_lines.append("\\n--- NVIDIA Driver Version ---")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, timeout=5)
    output_lines.append(result.stdout.strip().split('\\n')[0] if result.stdout else "N/A")
except:
    output_lines.append("N/A")

output_lines.append("\\n--- Disk Free Space (GB) ---")
try:
    import shutil
    free_gb = shutil.disk_usage('.').free / 1024**3
    output_lines.append(f"{free_gb:.2f}")
except:
    output_lines.append("N/A")

output_lines.append("\\n" + "=" * 40)
output_lines.append("Done.")

with open("logs/environment_info.txt", "w", encoding="utf-8") as f:
    f.write("\\n".join(output_lines))

print("\\n".join(output_lines))
print("\\n✅ Environment info saved to logs/environment_info.txt")