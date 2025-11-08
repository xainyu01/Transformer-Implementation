#!/usr/bin/env python3

#消融实验 


import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 修复OpenMP库冲突错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.transformer import Transformer, PositionalEncoding
from src.data_loader import IWSLTDataLoader


class IdentityPositionalEncoding(nn.Module):
    """恒等位置编码 - 什么都不做，用于消融实验"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x  # 直接返回输入，不做任何位置编码


class NoPositionalEncodingTransformer(Transformer):
    """无位置编码的Transformer变体"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用恒等位置编码替换原来的位置编码
        self.pos_encoding = IdentityPositionalEncoding(self.d_model, kwargs.get('max_seq_length', 5000))


class FewerHeadsTransformer(Transformer):
    """减少注意力头数的Transformer变体"""
    def __init__(self, *args, **kwargs):
        # 强制使用4个头而不是默认的8个
        kwargs['nhead'] = 4
        super().__init__(*args, **kwargs)


class FewerLayersTransformer(Transformer):
    """减少层数的Transformer变体"""
    def __init__(self, *args, **kwargs):
        # 强制使用3层而不是默认的6层
        kwargs['num_encoder_layers'] = 3
        kwargs['num_decoder_layers'] = 3
        super().__init__(*args, **kwargs)


class SmallerModelTransformer(Transformer):
    """更小模型的Transformer变体"""
    def __init__(self, *args, **kwargs):
        # 减少模型维度
        kwargs['d_model'] = 256
        kwargs['dim_feedforward'] = 1024
        super().__init__(*args, **kwargs)


class AblationStudy:
    def __init__(self, config):
        self.config = config
        self.data_loader = IWSLTDataLoader(config)
        self.results = {}
        
    def train_ablated_model(self, model_name, model, train_loader, valid_loader, epochs=5):
        """训练消融模型"""
        print(f"\n训练消融模型: {model_name}")
        print("=" * 50)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # 训练
            model.train()
            train_loss = 0
            for src, tgt in tqdm(train_loader, desc=f'Epoch {epoch}'):
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)
                
                # 创建掩码
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_len = tgt.size(1) - 1
                tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.config.device))
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
                
                # 前向传播
                output = model(src, tgt[:, :-1], src_mask, tgt_mask)
                loss = criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证
            val_loss = self.validate_model(model, valid_loader, criterion)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        return {
            'name': model_name,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'final_val_loss': val_losses[-1]
        }
    
    def validate_model(self, model, valid_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt in valid_loader:
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)
                
                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_len = tgt.size(1) - 1
                tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.config.device))
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)
                
                output = model(src, tgt[:, :-1], src_mask, tgt_mask)
                loss = criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )
                total_loss += loss.item()
        
        return total_loss / len(valid_loader)
    
    def create_ablated_models(self, src_vocab_size, tgt_vocab_size):
        """创建不同的消融模型"""
        models = {}
        
        # 1. 基准模型 (完整Transformer)
        models['baseline'] = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_seq_length=100
        ).to(self.config.device)
        
        # 2. 无位置编码
        models['no_positional_encoding'] = NoPositionalEncodingTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_seq_length=100
        ).to(self.config.device)
        
        # 3. 减少注意力头数
        models['fewer_heads'] = FewerHeadsTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=self.config.d_model,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_seq_length=100
        ).to(self.config.device)
        
        # 4. 减少层数
        models['fewer_layers'] = FewerLayersTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_seq_length=100
        ).to(self.config.device)
        
        # 5. 减少模型维度
        models['smaller_dim'] = SmallerModelTransformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            nhead=self.config.nhead,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            dropout=self.config.dropout,
            max_seq_length=100
        ).to(self.config.device)
        
        return models
    
    def run_ablation_study(self, epochs=5):
        """运行消融实验"""
        print("开始消融实验")
        print("=" * 60)
        
        # 获取数据
        train_loader, valid_loader = self.data_loader.get_data_loaders()
        src_vocab_size = self.data_loader.src_tokenizer.get_vocab_size()
        tgt_vocab_size = self.data_loader.tgt_tokenizer.get_vocab_size()
        
        # 创建消融模型
        models = self.create_ablated_models(src_vocab_size, tgt_vocab_size)
        
        # 训练并评估每个模型
        for name, model in models.items():
            result = self.train_ablated_model(name, model, train_loader, valid_loader, epochs)
            self.results[name] = result
        
        # 分析结果
        self.analyze_results()
        
        # 绘制比较图表
        self.plot_comparison()
    
    def analyze_results(self):
        """分析消融实验结果"""
        print("\n" + "=" * 60)
        print("消融实验结果分析")
        print("=" * 60)
        
        baseline_loss = self.results['baseline']['final_val_loss']
        
        print(f"{'模型变体':<25} {'最终验证损失':<15} {'相对性能下降':<15}")
        print("-" * 60)
        
        for name, result in self.results.items():
            loss = result['final_val_loss']
            performance_drop = ((loss - baseline_loss) / baseline_loss) * 100
            print(f"{name:<25} {loss:<15.4f} {performance_drop:>10.1f}%")
    
    def plot_comparison(self):
        """绘制消融实验比较图"""
        plt.figure(figsize=(12, 8))
        
        # 绘制验证损失曲线
        plt.subplot(2, 1, 1)
        for name, result in self.results.items():
            plt.plot(result['val_losses'], label=name, linewidth=2)
        
        plt.title('消融实验 - 验证损失比较')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 绘制最终性能条形图
        plt.subplot(2, 1, 2)
        names = list(self.results.keys())
        losses = [self.results[name]['final_val_loss'] for name in names]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        bars = plt.bar(names, losses, color=colors)
        plt.title('最终验证损失比较')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        
        # 在条形上添加数值
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('./results/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n消融实验图表已保存至: ./results/ablation_study.png")


def main():
    """主函数"""
    print("Transformer消融实验 (修复版)")
    print("=" * 50)
    
    # 初始化配置
    config = Config()
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    # 运行消融实验
    ablation_study = AblationStudy(config)
    ablation_study.run_ablation_study(epochs=5)  # 每个模型训练5个epoch


if __name__ == "__main__":
    main()