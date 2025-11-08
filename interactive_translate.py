#!/usr/bin/env python3

import os
import sys
import torch
import pickle

# 修复OpenMP库冲突错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.transformer import Transformer


class InteractiveTranslator:
    def __init__(self, checkpoint_path=None):
        self.config = Config()
        
        if checkpoint_path is None:
            checkpoint_path = "./checkpoints/best_model.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"错误: 未找到模型文件 {checkpoint_path}")
            sys.exit(1)
        
        # 加载模型
        self.model, self.src_tokenizer, self.tgt_tokenizer = self.load_model(checkpoint_path)
        print("模型加载成功!")
        
        # 显示模型信息
        self.display_model_info()
    
    def safe_load_checkpoint(self, checkpoint_path):
        
        try:
            # 首先尝试使用weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=False)
            return checkpoint
        except Exception as e:
            print(f"标准加载失败: {e}")
            print("尝试替代加载方法...")
            
            # 替代方法：手动加载
            try:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                return checkpoint
            except Exception as e2:
                print(f"替代加载也失败: {e2}")
                raise
    
    def load_model(self, checkpoint_path):
        """加载训练好的模型"""
        print(f"正在加载模型: {checkpoint_path}")
        checkpoint = self.safe_load_checkpoint(checkpoint_path)
        
        # 从检查点恢复分词器
        src_tokenizer = checkpoint['src_tokenizer']
        tgt_tokenizer = checkpoint['tgt_tokenizer']
        
        # 初始化模型 - 使用与训练时相同的max_seq_length
        model = Transformer(
            src_vocab_size=src_tokenizer.get_vocab_size(),
            tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=0.0,  # 推理时关闭dropout
            max_seq_length=100  # 与训练时保持一致
        ).to(self.config.device)
        
        # 加载模型权重 - 忽略位置编码的大小不匹配
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        # 过滤掉位置编码的权重，因为大小不匹配
        filtered_checkpoint_state_dict = {}
        for key, value in checkpoint_state_dict.items():
            if 'pos_encoding' not in key:
                filtered_checkpoint_state_dict[key] = value
            else:
                print(f"跳过位置编码权重: {key}")
        
        # 加载过滤后的状态字典
        model.load_state_dict(filtered_checkpoint_state_dict, strict=False)
        model.eval()
        
        return model, src_tokenizer, tgt_tokenizer
    
    def display_model_info(self):
        """显示模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n模型信息:")
        print(f"- 源语言词汇表大小: {self.src_tokenizer.get_vocab_size()}")
        print(f"- 目标语言词汇表大小: {self.tgt_tokenizer.get_vocab_size()}")
        print(f"- 模型总参数: {total_params:,}")
        print(f"- 运行设备: {self.config.device}")
        print("-" * 50)
    
    def preprocess_text(self, text):
        """预处理输入文本"""
        # 简单的文本清理
        text = text.strip()
        if not text:
            return ""
        # 确保句子以标点结尾
        if text[-1] not in ['.', '!', '?']:
            text += '.'
        return text
    
    def translate(self, text, max_length=50):
        """翻译单个句子"""
        if not text:
            return ""
        
        # 预处理文本
        text = self.preprocess_text(text)
        
        try:
            # 编码源文本
            src_encoding = self.src_tokenizer.encode(text)
            src_ids = [self.src_tokenizer.token_to_id("[SOS]")] + src_encoding.ids[:self.config.max_length-2] + [self.src_tokenizer.token_to_id("[EOS]")]
            src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(self.config.device)
            
            # 创建源语言掩码
            src_mask = (src_tensor != 0).unsqueeze(1).unsqueeze(2)
            
            # 开始翻译（开始时只有[SOS]）
            tgt_ids = [self.tgt_tokenizer.token_to_id("[SOS]")]
            
            with torch.no_grad():
                # 编码器前向传播（一次性计算）
                memory = self.model.encode(src_tensor, src_mask)
                
                # 解码器逐步生成
                for i in range(max_length):
                    tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(self.config.device)
                    
                    # 创建目标语言掩码
                    tgt_len = len(tgt_ids)
                    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.config.device))
                    tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
                    
                    # 解码器前向传播
                    output = self.model.decode(tgt_tensor, memory, tgt_mask, src_mask)
                    
                    # 获取下一个token（使用贪心搜索）
                    next_token_id = output[0, -1, :].argmax().item()
                    tgt_ids.append(next_token_id)
                    
                    # 如果遇到[EOS]则停止
                    if next_token_id == self.tgt_tokenizer.token_to_id("[EOS]"):
                        break
            
            # 解码目标文本
            decoded_tokens = []
            for token_id in tgt_ids[1:]:  # 跳过[SOS]
                if token_id == self.tgt_tokenizer.token_to_id("[EOS]"):
                    break
                if token_id not in [self.tgt_tokenizer.token_to_id("[PAD]"), 
                                  self.tgt_tokenizer.token_to_id("[SOS]"),
                                  self.tgt_tokenizer.token_to_id("[EOS]")]:
                    decoded_tokens.append(token_id)
            
            if not decoded_tokens:
                return "[翻译失败]"
            
            translation = self.tgt_tokenizer.decode(decoded_tokens)
            return translation
            
        except Exception as e:
            print(f"翻译过程中出现错误: {e}")
            return f"[翻译错误: {str(e)}]"
    
    def interactive_mode(self):
        """交互式翻译模式"""
        print("\n🎯 交互式翻译模式已启动!")
        print("输入英文句子，模型将返回德文翻译")
        print("输入 'quit' 或 'exit' 退出程序")
        print("输入 'examples' 查看示例")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n📝 请输入英文: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("感谢使用翻译工具!")
                    break
                
                elif user_input.lower() in ['examples', '示例']:
                    self.show_examples()
                    continue
                
                elif not user_input:
                    continue
                
                # 翻译
                print("🔄 翻译中...")
                translation = self.translate(user_input)
                print(f"🇩🇪 德文翻译: {translation}")
                
            except KeyboardInterrupt:
                print("\n\n感谢使用翻译工具!")
                break
            except Exception as e:
                print(f"翻译过程中出现错误: {e}")
    
    def show_examples(self):
        """显示翻译示例"""
        examples = [
            "Hello, how are you?",
            "I love machine learning.",
            "What is your name?",
            "The weather is beautiful today.",
            "Can you help me with this problem?",
            "Artificial intelligence is changing the world."
        ]
        
        print("\n📚 翻译示例:")
        print("-" * 40)
        for example in examples:
            translation = self.translate(example)
            print(f"EN: {example}")
            print(f"DE: {translation}")
            print()


def main():
    """主函数"""
    print(" Transformer交互式翻译工具 (修复位置编码问题)")
    print("=" * 50)
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer英德翻译工具')
    parser.add_argument('--model', '-m', type=str, default='./checkpoints/best_model.pth',
                       help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 初始化翻译器
    translator = InteractiveTranslator(args.model)
    
    # 运行交互式翻译模式
    translator.interactive_mode()


if __name__ == "__main__":
    main()