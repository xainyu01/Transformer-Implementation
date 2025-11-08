# src/transformer.py
import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置编码 - 从零实现"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力 - 从零实现"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # 线性变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        batch_size = query.size(0)

        # 线性变换
        Q = self.w_q(query)  # [batch_size, query_len, d_model]
        K = self.w_k(key)  # [batch_size, key_len, d_model]
        V = self.w_v(value)  # [batch_size, value_len, d_model]

        # 获取序列长度
        query_len = query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)

        # 分头
        Q = Q.view(batch_size, query_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, value_len, self.nhead, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            # 确保掩码形状与注意力分数匹配
            if mask.dim() == 4:
                # 如果掩码是4D [batch_size, 1, seq_len, seq_len]，需要调整以匹配多头
                if mask.size(1) == 1:  # 单头掩码
                    mask = mask.repeat(1, self.nhead, 1, 1)  # 扩展到多头
                scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )

        # 输出线性变换
        return self.w_o(attn_output)


class PositionWiseFeedForward(nn.Module):
    """位置前馈网络 - 从零实现"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """编码器层 - 从零实现"""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))

        return src


class DecoderLayer(nn.Module):
    """解码器层 - 从零实现"""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None):
        # 自注意力 + 残差连接 + 层归一化
        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_output))

        # 交叉注意力 + 残差连接 + 层归一化
        cross_attn_output = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_output))

        return tgt


class Transformer(nn.Module):
    """完整的Transformer模型 - 从零实现（增大版本）"""

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, max_seq_length: int = 5000):
        super().__init__()
        self.d_model = d_model

        # 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # 编码器和解码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None):
        # 词嵌入 + 位置编码
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        src_embedded = self.dropout(src_embedded)

        # 编码器层
        memory = src_embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask)

        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None):
        # 词嵌入 + 位置编码
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        tgt_embedded = self.dropout(tgt_embedded)

        # 解码器层
        output = tgt_embedded
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        return self.output_layer(output)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        return output