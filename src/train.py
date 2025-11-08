import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transformer import Transformer
from src.data_loader import IWSLTDataLoader
from src.utils import create_masks, save_plots
from src.config import Config


class Trainer:
    def __init__(self, config):
        self.config = config
        self.data_loader = IWSLTDataLoader(config)

    # 在 src/train.py 中的 initialize_model 方法后添加内存使用信息

    def initialize_model(self):
        """初始化模型"""
        # 获取数据加载器以初始化分词器
        train_loader, _ = self.data_loader.get_data_loaders()

        # 使用分词器的词汇表大小
        src_vocab_size = self.data_loader.src_tokenizer.get_vocab_size()
        tgt_vocab_size = self.data_loader.tgt_tokenizer.get_vocab_size()

        model = Transformer(
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

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model initialized with {total_params:,} parameters")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size: {total_params * 4 / (1024 ** 2):.2f} MB (FP32)")

        return model

    def get_optimizer_and_scheduler(self, model):
        """获取优化器和学习率调度器"""
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        return optimizer, scheduler

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training')

        for batch_idx, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(self.config.device), tgt.to(self.config.device)

            # 创建掩码 - 简化版本
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
            tgt_len = tgt.size(1) - 1
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.config.device))
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

            # 前向传播
            try:
                output = model(
                    src,
                    tgt[:, :-1],
                    src_mask,
                    tgt_mask
                )

                # 计算损失
                loss = criterion(
                    output.contiguous().view(-1, output.size(-1)),
                    tgt[:, 1:].contiguous().view(-1)
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad)

                optimizer.step()

                total_loss += loss.item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"src shape: {src.shape}, tgt shape: {tgt.shape}")
                raise e

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, model, valid_loader, criterion):
        """验证"""
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, tgt in tqdm(valid_loader, desc='Validating'):
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)

                # 创建掩码 - 简化版本
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

    def evaluate_model_quality(self, model, data_loader, num_examples=5):
        """评估模型质量 - 生成示例翻译"""
        model.eval()
        examples = []

        # 获取一些样本进行测试
        data_iter = iter(data_loader)
        for _ in range(min(num_examples, len(data_loader))):
            try:
                src, tgt = next(data_iter)
                src, tgt = src.to(self.config.device), tgt.to(self.config.device)

                # 使用第一个样本
                src_sample = src[0:1]
                tgt_sample = tgt[0:1]

                # 创建掩码
                src_mask = (src_sample != 0).unsqueeze(1).unsqueeze(2)
                tgt_len = tgt_sample.size(1) - 1
                tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.config.device))
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(1)

                # 生成翻译
                with torch.no_grad():
                    output = model(src_sample, tgt_sample[:, :-1], src_mask, tgt_mask)
                    predicted = output.argmax(-1)

                # 解码文本
                src_text = self.decode_tokens(src_sample[0].cpu().numpy(), data_loader.src_tokenizer)
                tgt_text = self.decode_tokens(tgt_sample[0, 1:].cpu().numpy(), data_loader.tgt_tokenizer)  # 跳过[SOS]
                pred_text = self.decode_tokens(predicted[0].cpu().numpy(), data_loader.tgt_tokenizer)

                examples.append({
                    'source': src_text,
                    'target': tgt_text,
                    'predicted': pred_text
                })

            except Exception as e:
                print(f"Error in example generation: {e}")
                continue

        return examples

    def decode_tokens(self, tokens, tokenizer):
        """将token ID解码为文本"""
        # 过滤掉特殊token和填充token
        valid_tokens = []
        for token in tokens:
            if token == tokenizer.token_to_id("[EOS]"):
                break
            if token not in [tokenizer.token_to_id("[PAD]"), tokenizer.token_to_id("[SOS]"),
                             tokenizer.token_to_id("[EOS]")]:
                valid_tokens.append(token)

        if not valid_tokens:
            return ""

        return tokenizer.decode(valid_tokens)

    def calculate_perplexity(self, loss):
        """计算困惑度"""
        return math.exp(loss)

    def plot_training_progress(self, train_losses, val_losses, examples=None):
        """绘制训练进度"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制损失曲线
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 计算并绘制困惑度
        train_perplexities = [self.calculate_perplexity(loss) for loss in train_losses]
        val_perplexities = [self.calculate_perplexity(loss) for loss in val_losses]

        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, train_perplexities, 'b--', alpha=0.5, label='Train PPL')
        ax1_twin.plot(epochs, val_perplexities, 'r--', alpha=0.5, label='Val PPL')
        ax1_twin.set_ylabel('Perplexity')
        ax1_twin.legend(loc='upper right')

        # 显示示例翻译
        if examples:
            ax2.axis('off')
            text_content = "Example Translations:\n\n"
            for i, example in enumerate(examples[:3]):  # 只显示前3个例子
                text_content += f"Example {i + 1}:\n"
                text_content += f"  Source: {example['source']}\n"
                text_content += f"  Target: {example['target']}\n"
                text_content += f"  Predicted: {example['predicted']}\n\n"

            ax2.text(0.1, 0.9, text_content, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.results_path, 'training_progress.png'))
        plt.close()

    def train(self):
        """完整训练流程"""
        # 准备数据
        train_loader, valid_loader = self.data_loader.get_data_loaders()

        # 初始化模型
        model = self.initialize_model()

        # 优化器和损失函数
        optimizer, scheduler = self.get_optimizer_and_scheduler(model)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0是pad_token_id

        # 训练记录
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        print("Starting training...")

        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            train_losses.append(train_loss)

            # 验证
            val_loss = self.validate(model, valid_loader, criterion)
            val_losses.append(val_loss)

            print(f'Epoch: {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(
                f'Train Perplexity: {self.calculate_perplexity(train_loss):.2f} | Val Perplexity: {self.calculate_perplexity(val_loss):.2f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'src_tokenizer': self.data_loader.src_tokenizer,
                    'tgt_tokenizer': self.data_loader.tgt_tokenizer
                }, f'{self.config.checkpoint_path}/best_model.pth')
                print(f"Saved best model with val_loss: {val_loss:.4f}")

            # 更新学习率
            scheduler.step()

            # 每5个epoch或最后一个epoch生成示例和可视化
            if epoch % 5 == 0 or epoch == self.config.epochs:
                examples = self.evaluate_model_quality(model, valid_loader, num_examples=3)
                self.plot_training_progress(train_losses, val_losses, examples)
                print("\nExample translations:")
                for i, example in enumerate(examples):
                    print(f"Example {i + 1}:")
                    print(f"  Source: {example['source']}")
                    print(f"  Target: {example['target']}")
                    print(f"  Predicted: {example['predicted']}")
                    print()

        # 最终评估
        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)

        # 加载最佳模型进行最终评估
        checkpoint = torch.load(f'{self.config.checkpoint_path}/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        final_val_loss = self.validate(model, valid_loader, criterion)
        final_perplexity = self.calculate_perplexity(final_val_loss)

        print(f"Final Validation Loss: {final_val_loss:.4f}")
        print(f"Final Perplexity: {final_perplexity:.2f}")

        # 生成最终示例
        final_examples = self.evaluate_model_quality(model, valid_loader, num_examples=5)
        self.plot_training_progress(train_losses, val_losses, final_examples)

        print("\nFinal Example Translations:")
        for i, example in enumerate(final_examples):
            print(f"Example {i + 1}:")
            print(f"  Source: {example['source']}")
            print(f"  Target: {example['target']}")
            print(f"  Predicted: {example['predicted']}")
            print()

        print("Training completed!")


if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()