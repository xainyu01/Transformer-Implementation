# Transformer从零实现

##  项目简介

基于PyTorch从零实现的完整Transformer模型，支持Encoder-Decoder架构，包含训练、推理和消融实验功能。

## 环境配置

### 系统要求
- **GPU**: NVIDIA GPU with 8GB+ VRAM (推荐)
- **内存**: 16GB+ RAM
- **Python**: 3.9

### 安装依赖
```bash
# 创建conda环境
conda create -n transformer python=3.9
conda activate transformer

# 安装依赖包
pip install -r requirements.txt
```

##  快速开始
当完成环境配置后，你可以开始训练模型了。
直接运行main可以重新开始训练.
运行ablation_study可以复现消融实验.
运行interactive_translate可以进行交互式翻译.

### 数据准备
数据集已经保存到
>data/iwslt2017/


### 训练模型
```bash
# 使用运行脚本（推荐）
chmod +x scripts/run.sh
./scripts/run.sh train

# 或直接运行Python脚本
export PYTHONHASHSEED=42
python src/main.py
```

### 运行消融实验
```bash
./scripts/run.sh ablation
```

### 交互式翻译
```bash
./scripts/run.sh translate
```

##  项目结构

```
.
├── src/                    # 源代码
│   ├── transformer.py      # Transformer模型实现
│   ├── data_loader.py      # 数据加载与预处理
│   ├── train.py           # 训练循环
│   ├── inference.py       # 推理模块
├── scripts/
│   └── run.sh            # 自动化运行脚本
├── data/                 # 数据集目录
├── checkpoints/          # 模型保存
├── results/             # 实验结果
└── requirements.txt      # 依赖列表
├── README.md
├── ablation_study.py  # 消融实验
├── interactive_translate.py # 交互翻译
└── main.py           # 程序入口
```

##  配置参数

主要模型参数（可在`src/config.py`中调整）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 512 | 模型维度 |
| `nhead` | 8 | 注意力头数 |
| `num_encoder_layers` | 6 | 编码器层数 |
| `num_decoder_layers` | 6 | 解码器层数 |
| `dim_feedforward` | 2048 | 前馈网络维度 |
| `batch_size` | 32 | 批大小 |
| `learning_rate` | 0.0001 | 学习率 |

##  运行命令详解

### 完整训练流程


# 运行完整训练
>python src/main.py \


### 消融实验
```bash
python src/ablation_study.py 
```

### 交互翻译
```bash
python src/interactive_translate.py --checkpoint checkpoints/best_model.pt
```

##  输出文件

训练完成后，在`results/`目录下生成：
- `training_curve.png` - 训练损失曲线
- `ablation_study.png` - 消融实验结果
- 模型检查点保存在`checkpoints/`目录

##  训练时间估计

- **基准模型**: 约6-8小时 (RTX 5060, 30个epoch)
- **消融实验**: 约12-16小时 (所有变体)

##  常见问题

1. **CUDA内存不足**: 减小`batch_size`或模型维度
2. **数据集下载失败**: 检查网络连接，或手动下载数据集
3. **依赖冲突**: 使用conda环境隔离

---

*最后更新: 2025年11月*