import platform
import psutil
import torch
from cpuinfo import get_cpu_info

print("=" * 50)
print("SYSTEM CONFIGURATION")
print("=" * 50)

# OS Information
print("OS:", platform.platform())

# CPU Information
cpu_info = get_cpu_info()

print("\nCPU Information:")
print("Model:", cpu_info.get("brand_raw", "Unknown"))
print("Architecture:", platform.machine())
print("Physical Cores:", psutil.cpu_count(logical=False))
print("Logical Cores:", psutil.cpu_count(logical=True))
print("Max Frequency:",
      f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "Unknown")

# RAM Information
print("\nRAM Information:")
print(f"Total RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")

# Disk Information
disk = psutil.disk_usage('/')
print("\nStorage Information:")
print(f"Total Storage: {disk.total / 1024**3:.2f} GB")
print(f"Free Storage: {disk.free / 1024**3:.2f} GB")

# GPU Information
if torch.cuda.is_available():
    print("\nGPU Information:")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )
    print("CUDA Version:", torch.version.cuda)
else:
    print("\nGPU Information:")
    print("GPU: Not Available")

# Python & PyTorch
print("\nSoftware Information:")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)

print("=" * 50)
