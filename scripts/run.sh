#!/bin/bash

# Transformer从零实现 - 运行脚本
# 作者: 刘裕轩


set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        log_error "Python未安装，请先安装Python 3.8+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $PYTHON_VERSION"
}

# 检查PyTorch和CUDA
check_pytorch() {
    log_step "检查PyTorch环境..."
    
    if python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" &> /dev/null; then
        log_info "PyTorch已安装"
    else
        log_error "PyTorch未安装，请运行: pip install torch torchvision torchaudio"
        exit 1
    fi
    
    # 检查CUDA
    if python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')" &> /dev/null; then
        CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            log_info "CUDA可用，使用GPU训练"
        else
            log_warning "CUDA不可用，将使用CPU训练（速度较慢）"
        fi
    fi
}

# 检查依赖
check_dependencies() {
    log_step "检查项目依赖..."
    
    REQUIRED_PACKAGES=("tokenizers" "matplotlib" "tqdm" "numpy")
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if python -c "import $package" &> /dev/null; then
            log_info "$package 已安装"
        else
            log_error "$package 未安装，请运行: pip install -r requirements.txt"
            exit 1
        fi
    done
}

# 创建目录结构
create_directories() {
    log_step "创建项目目录..."
    
    DIRS=("checkpoints" "results" "data/iwslt2017" "logs")
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
}

# 设置环境变量
setup_environment() {
    log_step "设置环境变量..."
    
    export PYTHONHASHSEED=42
    export CUDA_LAUNCH_BLOCKING=1
    export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
    
    log_info "设置随机种子: 42"
    log_info "设置PYTHONPATH"
}

# 训练模型
train_model() {
    local seed=${1:-42}
    local config=${2:-"base"}
    
    log_step "开始训练Transformer模型..."
    log_info "随机种子: $seed"
    log_info "配置: $config"
    
    python src/main.py \
        --seed "$seed" \
        --config "$config" \
        --batch_size 32 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --d_model 512 \
        --nhead 8 \
        --num_encoder_layers 6 \
        --num_decoder_layers 6 \
        --dim_feedforward 2048 \
        --dropout 0.1 \
        --log_dir "logs/training_$seed.log"
    
    if [ $? -eq 0 ]; then
        log_info "训练完成！模型保存在 checkpoints/ 目录"
    else
        log_error "训练失败！请检查日志文件: logs/training_$seed.log"
        exit 1
    fi
}

# 运行消融实验
run_ablation_study() {
    local seed=${1:-42}
    
    log_step "开始消融实验..."
    log_info "随机种子: $seed"
    
    python src/ablation_study.py \
        --seed "$seed" \
        --experiments "all" \
        --output_dir "results/ablation" \
        --log_dir "logs/ablation_$seed.log"
    
    if [ $? -eq 0 ]; then
        log_info "消融实验完成！结果保存在 results/ablation/ 目录"
    else
        log_error "消融实验失败！请检查日志文件: logs/ablation_$seed.log"
        exit 1
    fi
}

# 交互式翻译
start_interactive_translation() {
    local checkpoint=${1:-"checkpoints/best_model.pt"}
    
    log_step "启动交互式翻译..."
    
    if [ ! -f "$checkpoint" ]; then
        log_warning "模型检查点不存在: $checkpoint"
        log_info "请先训练模型或指定正确的检查点路径"
        log_info "可用检查点:"
        ls -la checkpoints/ || log_info "checkpoints目录为空"
        return 1
    fi
    
    log_info "使用模型: $checkpoint"
    python src/interactive_translate.py \
        --checkpoint "$checkpoint" \
        --max_length 50
}

# 验证模型
validate_model() {
    local checkpoint=${1:-"checkpoints/best_model.pt"}
    
    log_step "验证模型..."
    
    if [ ! -f "$checkpoint" ]; then
        log_error "模型检查点不存在: $checkpoint"
        return 1
    fi
    
    python src/validate_model.py \
        --checkpoint "$checkpoint" \
        --batch_size 32
}

# 分析结果
analyze_results() {
    log_step "分析实验结果..."
    
    if [ ! -f "src/analyze_results.py" ]; then
        log_warning "分析脚本不存在: src/analyze_results.py"
        return 1
    fi
    
    python src/analyze_results.py \
        --results_dir "results" \
        --output_dir "results/analysis"
}

# 清理临时文件
cleanup() {
    log_step "清理临时文件..."
    
    # 清理Python缓存文件
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name ".DS_Store" -delete 2>/dev/null || true
    
    log_info "临时文件清理完成"
}

# 显示系统信息
show_system_info() {
    log_step "系统信息:"
    
    # CPU信息
    if command -v nproc &> /dev/null; then
        log_info "CPU核心数: $(nproc)"
    fi
    
    # 内存信息
    if command -v free &> /dev/null; then
        log_info "内存信息:"
        free -h
    fi
    
    # GPU信息
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    else
        log_info "未检测到NVIDIA GPU"
    fi
}

# 显示帮助信息
show_help() {
    echo -e "${GREEN}Transformer从零实现 - 运行脚本${NC}"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  train [seed] [config]   训练模型 (默认: seed=42, config=base)"
    echo "  ablation [seed]         运行消融实验 (默认: seed=42)"
    echo "  translate [checkpoint]  交互式翻译 (默认: checkpoints/best_model.pt)"
    echo "  validate [checkpoint]   验证模型性能"
    echo "  analyze                 分析实验结果"
    echo "  setup                   环境检查和设置"
    echo "  clean                   清理临时文件"
    echo "  info                    显示系统信息"
    echo "  all [seed]              运行完整流程 (训练+消融+验证)"
    echo "  help                    显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 train               使用默认参数训练模型"
    echo "  $0 train 123 base      使用种子123训练模型"
    echo "  $0 ablation 456        运行消融实验"
    echo "  $0 translate           启动交互式翻译"
    echo "  $0 all                 运行完整流程"
    echo "  $0 setup               检查环境依赖"
    echo ""
    echo "环境要求:"
    echo "  - Python 3.8+"
    echo "  - PyTorch 1.9+"
    echo "  - 8GB+ GPU内存 (推荐)"
    echo "  - 5GB+ 磁盘空间"
}

# 环境设置
setup_environment() {
    log_step "开始环境设置..."
    
    check_python
    check_pytorch
    check_dependencies
    create_directories
    setup_environment_vars
    
    log_info "环境设置完成！"
}

# 完整流程
run_full_pipeline() {
    local seed=${1:-42}
    
    log_step "开始完整流程..."
    
    # 环境检查
    setup_environment
    
    # 训练模型
    train_model "$seed"
    
    # 运行消融实验
    run_ablation_study "$seed"
    
    # 验证模型
    validate_model "checkpoints/best_model.pt"
    
    # 分析结果
    analyze_results
    
    log_info "完整流程执行完毕！"
}

# 主函数
main() {
    local command=$1
    local arg1=$2
    local arg2=$3
    
    case $command in
        "train")
            setup_environment
            train_model "$arg1" "$arg2"
            ;;
        "ablation")
            setup_environment
            run_ablation_study "$arg1"
            ;;
        "translate")
            start_interactive_translation "$arg1"
            ;;
        "validate")
            validate_model "$arg1"
            ;;
        "analyze")
            analyze_results
            ;;
        "setup")
            setup_environment
            ;;
        "clean")
            cleanup
            ;;
        "info")
            show_system_info
            ;;
        "all")
            run_full_pipeline "$arg1"
            ;;
        "help"|"")
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 脚本入口
if [ "$#" -eq 0 ]; then
    show_help
else
    main "$@"
fi