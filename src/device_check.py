# src/device_check.py
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config


def check_devices():
    """检查设备配置"""
    config = Config()

    print("=" * 50)
    print("设备检查")
    print("=" * 50)

    print(f"配置设备: {config.device}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name()}")

    # 测试张量创建
    cpu_tensor = torch.tensor([1, 2, 3])
    gpu_tensor = torch.tensor([1, 2, 3]).to(config.device)

    print(f"CPU张量设备: {cpu_tensor.device}")
    print(f"GPU张量设备: {gpu_tensor.device}")

    # 测试掩码创建
    from src.utils import create_look_ahead_mask
    mask = create_look_ahead_mask(10, config.device)
    print(f"掩码设备: {mask.device}")

    print("设备检查完成!")


if __name__ == "__main__":
    check_devices()