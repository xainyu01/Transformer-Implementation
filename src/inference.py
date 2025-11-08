# src/inference.py
import torch
import os
import sys

# 修复OpenMP库冲突错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.config import Config
from src.data_loader import IWSLTDataLoader


def load_model_for_inference(checkpoint_path, config):
    """加载训练好的模型进行推理"""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    # 从检查点恢复分词器
    src_tokenizer = checkpoint['src_tokenizer']
    tgt_tokenizer = checkpoint['tgt_tokenizer']

    # 初始化模型
    model = Transformer(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
    ).to(config.device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, src_tokenizer, tgt_tokenizer


def translate(model, src_tokenizer, tgt_tokenizer, text, config, max_length=50):
    """翻译文本"""
    # 编码源文本
    src_encoding = src_tokenizer.encode(text)
    src_ids = [src_tokenizer.token_to_id("[SOS]")] + src_encoding.ids + [src_tokenizer.token_to_id("[EOS]")]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(config.device)

    # 创建源语言掩码
    src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)

    # 编码目标语言（开始时只有[SOS]）
    tgt_ids = [tgt_tokenizer.token_to_id("[SOS]")]

    for i in range(max_length):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(config.device)

        # 创建目标语言掩码
        tgt_len = len(tgt_ids)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=config.device))
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_mask, tgt_mask)

        # 获取下一个token
        next_token_id = output[0, -1, :].argmax().item()
        tgt_ids.append(next_token_id)

        # 如果遇到[EOS]则停止
        if next_token_id == tgt_tokenizer.token_to_id("[EOS]"):
            break

    # 解码目标文本
    decoded = tgt_tokenizer.decode(tgt_ids[1:-1])  # 去掉[SOS]和[EOS]
    return decoded


if __name__ == "__main__":
    config = Config()

    # 加载模型
    checkpoint_path = "./checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print("加载训练好的模型进行推理...")
        model, src_tokenizer, tgt_tokenizer = load_model_for_inference(checkpoint_path, config)

        # 测试翻译
        test_sentences = [
            "Hello world",
            "How are you?",
            "I am a student",
            "Good morning",
            "What is your name?",
            "Thank you very much"
        ]

        print("\n翻译示例:")
        print("=" * 50)
        for sent in test_sentences:
            translation = translate(model, src_tokenizer, tgt_tokenizer, sent, config)
            print(f"EN: {sent}")
            print(f"DE: {translation}")
            print("-" * 30)
    else:
        print("未找到训练好的模型。请先运行训练。")