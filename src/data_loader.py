# src/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class IWSLTDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_tokenizer, tgt_tokenizer, max_length=100):
        self.src_data = self.read_file(src_file)
        self.tgt_data = self.read_file(tgt_file)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length

        # 确保源和目标句子数相同
        assert len(self.src_data) == len(self.tgt_data)
        print(f"Loaded {len(self.src_data)} samples")

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx]
        tgt_text = self.tgt_data[idx]

        # 编码源语言和目标语言
        src_encoding = self.src_tokenizer.encode(src_text)
        tgt_encoding = self.tgt_tokenizer.encode(tgt_text)

        # 获取token IDs
        src_ids = src_encoding.ids[:self.max_length]
        tgt_ids = tgt_encoding.ids[:self.max_length]

        # 添加开始和结束标记
        src_ids = [self.src_tokenizer.token_to_id("[SOS]")] + src_ids + [self.src_tokenizer.token_to_id("[EOS]")]
        tgt_ids = [self.tgt_tokenizer.token_to_id("[SOS]")] + tgt_ids + [self.tgt_tokenizer.token_to_id("[EOS]")]

        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


class IWSLTDataLoader:
    def __init__(self, config):
        self.config = config
        self.src_tokenizer = None
        self.tgt_tokenizer = None

    def train_tokenizer(self, file_path, vocab_size=30000):
        """训练BPE分词器"""
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        )

        # 从文件训练分词器 - 使用完整数据
        with open(file_path, 'r', encoding='utf-8') as f:
            tokenizer.train_from_iterator(f, trainer)

        return tokenizer

    def initialize_tokenizers(self, train_src_file, train_tgt_file):
        """初始化分词器"""
        print("Training source tokenizer...")
        src_tokenizer = self.train_tokenizer(train_src_file, self.config.vocab_size)
        print("Training target tokenizer...")
        tgt_tokenizer = self.train_tokenizer(train_tgt_file, self.config.vocab_size)

        print(f"Source vocabulary size: {src_tokenizer.get_vocab_size()}")
        print(f"Target vocabulary size: {tgt_tokenizer.get_vocab_size()}")

        return src_tokenizer, tgt_tokenizer

    def collate_fn(self, batch):
        """批处理函数"""
        src_batch, tgt_batch = [], []

        for src, tgt in batch:
            src_batch.append(src)
            tgt_batch.append(tgt)

        # 填充序列
        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_batch, batch_first=True, padding_value=0
        )
        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            tgt_batch, batch_first=True, padding_value=0
        )

        return src_batch, tgt_batch

    def get_data_loaders(self):
        """获取数据加载器"""
        # 训练集和验证集文件
        train_src_file = os.path.join(self.config.data_path, 'train.en')
        train_tgt_file = os.path.join(self.config.data_path, 'train.de')
        valid_src_file = os.path.join(self.config.data_path, 'valid.en')
        valid_tgt_file = os.path.join(self.config.data_path, 'valid.de')
        test_src_file = os.path.join(self.config.data_path, 'test.en')
        test_tgt_file = os.path.join(self.config.data_path, 'test.de')

        # 检查文件是否存在
        if not os.path.exists(train_src_file):
            raise FileNotFoundError(f"Training source file not found: {train_src_file}")
        if not os.path.exists(train_tgt_file):
            raise FileNotFoundError(f"Training target file not found: {train_tgt_file}")

        # 初始化分词器
        self.src_tokenizer, self.tgt_tokenizer = self.initialize_tokenizers(train_src_file, train_tgt_file)

        # 创建数据集 - 使用完整数据
        train_dataset = IWSLTDataset(
            train_src_file, train_tgt_file, self.src_tokenizer, self.tgt_tokenizer,
            self.config.max_length
        )

        # 使用验证集文件
        if os.path.exists(valid_src_file) and os.path.exists(valid_tgt_file):
            valid_dataset = IWSLTDataset(
                valid_src_file, valid_tgt_file, self.src_tokenizer, self.tgt_tokenizer,
                self.config.max_length
            )
        else:
            print("Validation files not found, using 10% of training data for validation")
            # 使用训练集的10%作为验证集
            train_size = int(0.9 * len(train_dataset))
            valid_size = len(train_dataset) - train_size
            train_dataset, valid_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, valid_size]
            )

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            collate_fn=self.collate_fn, shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.config.batch_size,
            collate_fn=self.collate_fn
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}")

        return train_loader, valid_loader