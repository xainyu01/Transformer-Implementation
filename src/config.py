# src/config.py
import torch


class Config:
    # 模型参数 - 增大模型规模
    d_model = 512  # 增大嵌入维度
    nhead = 8  # 增加注意力头数
    num_encoder_layers = 6  # 增加编码器层数
    num_decoder_layers = 6  # 增加解码器层数
    dim_feedforward = 2048  # 增大前馈网络维度
    dropout = 0.1

    # 训练参数
    batch_size = 32  # 增大批次大小
    epochs = 30  # 增加训练轮数
    learning_rate = 0.0001
    weight_decay = 0.0001
    clip_grad = 1.0

    # 数据参数
    max_length = 100  # 增加序列最大长度
    vocab_size = 30000  # 增大词汇表大小

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径
    data_path = './data/iwslt2017'
    checkpoint_path = './checkpoints'
    results_path = './results'