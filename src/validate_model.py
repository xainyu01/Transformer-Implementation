# src/debug_transformer.py
import torch
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer, MultiHeadAttention
from src.config import Config


def debug_attention():
    """调试多头注意力"""
    print("调试多头注意力...")

    config = Config()

    # 测试多头注意力
    try:
        attention = MultiHeadAttention(d_model=64, nhead=4).to(config.device)

        # 测试自注意力
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64).to(config.device)

        output = attention(x, x, x)
        print(f"✅ 自注意力成功! 输入形状: {x.shape}, 输出形状: {output.shape}")

        # 测试交叉注意力（不同序列长度）
        query_len = 8
        key_len = 12
        query = torch.randn(batch_size, query_len, 64).to(config.device)
        key = torch.randn(batch_size, key_len, 64).to(config.device)
        value = torch.randn(batch_size, key_len, 64).to(config.device)

        output = attention(query, key, value)
        print(f"✅ 交叉注意力成功! query: {query.shape}, key: {key.shape}, value: {value.shape}, 输出: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ 注意力调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_transformer():
    """调试完整Transformer"""
    print("调试完整Transformer...")

    config = Config()

    try:
        model = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=1000,
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        ).to(config.device)

        print("✅ Transformer创建成功!")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试不同序列长度的输入
        batch_size = 2
        src_len = 10
        tgt_len = 8

        src = torch.randint(0, 1000, (batch_size, src_len)).to(config.device)
        tgt = torch.randint(0, 1000, (batch_size, tgt_len)).to(config.device)

        # 简单掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=config.device))
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :, :-1, :-1])

        print("✅ Transformer前向传播成功!")
        print(f"输入形状: src {src.shape}, tgt {tgt.shape}")
        print(f"输出形状: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Transformer调试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Transformer调试")
    print("=" * 50)

    success1 = debug_attention()
    success2 = debug_transformer()

    if success1 and success2:
        print("🎉 所有调试通过!")
    else:
        print("❌ 调试失败，请检查代码")