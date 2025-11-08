#!/usr/bin/env python3


import os
import sys

# 修复OpenMP库冲突错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import Trainer
from src.config import Config


def main():
    """主函数"""
    print("=" * 60)
    print("Transformer从零实现 - 大模型期中作业")
    print("=" * 60)

    # 初始化配置
    config = Config()

    # 创建必要的目录
    os.makedirs(config.checkpoint_path, exist_ok=True)
    os.makedirs(config.results_path, exist_ok=True)

    # 检查数据是否存在
    train_en_path = os.path.join(config.data_path, 'train.en')
    if not os.path.exists(train_en_path):
        print(f"Error: Training data not found at {train_en_path}")
        print("Please ensure you have the IWSLT2017 dataset in the data/iwslt2017 directory")
        return

    # 开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()