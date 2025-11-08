# src/analyze_results.py
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_training_curves(results_path):
    """绘制训练曲线"""
    # 这里可以读取训练过程中保存的损失值并绘制图表
    # 由于我们已经在训练过程中保存了图表，这里主要是提供一个分析接口

    if os.path.exists(os.path.join(results_path, 'training_curve.png')):
        print("训练曲线已保存在 results/training_curve.png")

        # 可以在这里添加更多的分析代码
        # 比如计算训练速度、分析收敛情况等

    else:
        print("未找到训练曲线文件")


def model_summary(config):
    """模型参数统计"""
    from src.transformer import Transformer
    from src.data_loader import IWSLTDataLoader

    data_loader = IWSLTDataLoader(config)
    train_loader, _ = data_loader.get_data_loaders()

    src_vocab_size = data_loader.src_tokenizer.get_vocab_size()
    tgt_vocab_size = data_loader.tgt_tokenizer.get_vocab_size()

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n模型参数统计:")
    print("=" * 40)
    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)")

    return total_params, trainable_params


if __name__ == "__main__":
    from src.config import Config

    config = Config()

    print("训练结果分析")
    print("=" * 50)

    # 模型参数统计
    total_params, trainable_params = model_summary(config)

    # 绘制训练曲线
    plot_training_curves(config.results_path)

    print("\n分析完成!")