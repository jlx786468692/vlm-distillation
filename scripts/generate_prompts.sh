#!/bin/bash
# Prompt自动生成快速启动脚本

echo "========================================"
echo "  Prompt自动生成工具"
echo "========================================"
echo ""

# 检查数据目录
if [ ! -d "data/coco/val2014" ]; then
    echo "⚠ COCO数据集未找到"
    echo ""
    echo "请先下载COCO val2014数据集："
    echo "  mkdir -p data/coco"
    echo "  wget http://images.cocodataset.org/zips/val2014.zip"
    echo "  unzip val2014.zip -d data/coco/"
    echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    echo "  unzip annotations_trainval2014.zip -d data/coco/"
    echo ""
    exit 1
fi

# 检查标注文件
if [ ! -f "data/coco/annotations/captions_val2014.json" ]; then
    echo "❌ COCO标注文件未找到"
    echo "请下载标注文件："
    echo "  wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    echo "  unzip annotations_trainval2014.zip -d data/coco/"
    exit 1
fi

echo "✓ COCO数据集已就绪"
echo ""

# 询问用户选择任务
echo "请选择要生成prompt的任务："
echo "  1. VQA"
echo "  2. Detection"
echo "  3. 全部"
echo ""
read -p "请输入选项 (1/2/3): " task_choice

case $task_choice in
    1)
        TASK="vqa"
        ;;
    2)
        TASK="detection"
        ;;
    3)
        TASK="all"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

# 询问分析样本数
echo ""
read -p "请输入分析样本数 (默认1000): " num_samples
NUM_SAMPLES=${num_samples:-1000}

echo ""
echo "========================================"
echo "  开始生成Prompt"
echo "========================================"
echo ""
echo "任务类型: $TASK"
echo "分析样本数: $NUM_SAMPLES"
echo ""

# Step 1: 分析数据模式
echo "Step 1: 分析数据模式..."
python scripts/simple_prompt_generator.py \
    --mode analyze \
    --coco_root data/coco/val2014 \
    --annotation data/coco/annotations/captions_val2014.json \
    --num_samples $NUM_SAMPLES

if [ $? -ne 0 ]; then
    echo "❌ 数据分析失败"
    exit 1
fi

echo ""
echo "✓ 数据分析完成"
echo ""

# Step 2: 生成Prompt
echo "Step 2: 生成Prompt..."
python scripts/simple_prompt_generator.py \
    --mode generate \
    --task $TASK \
    --coco_root data/coco/val2014 \
    --annotation data/coco/annotations/captions_val2014.json \
    --num_samples $NUM_SAMPLES

if [ $? -ne 0 ]; then
    echo "❌ Prompt生成失败"
    exit 1
fi

echo ""
echo "========================================"
echo "  ✓ Prompt生成成功"
echo "========================================"
echo ""
echo "生成的Prompt已保存到:"
echo "  configs/generated_prompts/"
echo ""
echo "主配置文件已更新:"
echo "  configs/prompts_en.yaml"
echo ""
echo "下一步:"
echo "  1. 检查生成的prompt: configs/prompts_en.yaml"
echo "  2. 测试prompt效果: python scripts/simple_prompt_generator.py --mode test --test_image <图像路径>"
echo "  3. 运行标签生成: python src/distillation/distiller.py"
echo ""