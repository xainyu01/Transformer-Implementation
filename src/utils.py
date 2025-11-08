import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os


def create_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """创建填充掩码"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]


def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建前瞻掩码（防止看到未来信息）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0  # [seq_len, seq_len]


def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0) -> tuple:
    """创建所有必要的掩码"""
    device = src.device  # 获取输入张量的设备

    # 源序列填充掩码
    src_mask = create_padding_mask(src, pad_idx)

    # 目标序列填充掩码
    tgt_padding_mask = create_padding_mask(tgt, pad_idx)

    # 目标序列前瞻掩码 - 确保在正确的设备上
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len, device)

    # 组合目标序列掩码 - 修复形状处理
    tgt_look_ahead_mask = tgt_look_ahead_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, tgt_len, tgt_len]
    tgt_mask = tgt_padding_mask & tgt_look_ahead_mask  # [batch_size, 1, tgt_len, tgt_len]

    return src_mask, tgt_mask


def save_plots(train_losses: list, val_losses: list, path: str):
    """保存训练曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(path, 'training_curve.png'))
    plt.close()