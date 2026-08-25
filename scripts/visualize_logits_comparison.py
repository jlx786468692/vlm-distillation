#!/usr/bin/env python
"""
Logits处理流程可视化（更新版）
============================

只做调用和展示，不做逻辑处理：
- 调用 QuestionClassifier 获取问题类型
- 调用 VQAClosedLabelGenerator 生成软硬标签和CoT
- 展示过滤前后的token
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from typing import Dict, Any

from src.models.teacher_model import TeacherModel
from src.utils.config import ConfigManager
from src.classification.question_classifier import QuestionClassifier
from src.distillation.vqa_closed_label_generator import VQAClosedLabelGenerator


def visualize_logits_comparison(
    image_path: str,
    question: str,
    ground_truth: str = None,
    candidate_pool: list = None
):
    """
    可视化logits处理流程（调用现有模块）

    Args:
        image_path: 图像路径
        question: 问题
        ground_truth: 真实答案（可选）
        candidate_pool: 候选池（可选，用于choice问题）
    """
    print("\n" + "="*160)
    print("LOGITS处理流程可视化（完整流程）".center(160))
    print("="*160)

    # ====================
    # 1. 初始化模块
    # ====================
    print("\n【步骤1】初始化模块...")

    config = ConfigManager()

    # 初始化Teacher模型
    print("  - 初始化Teacher模型...")
    teacher = TeacherModel(config)

    # 初始化问题分类器
    print("  - 初始化问题分类器...")
    classifier = QuestionClassifier(
        model_path=config.get("classification.model_path", "models/bart-large-mnli"),
        confidence_threshold=config.get("classification.confidence_threshold", 0.7)
    )

    # 初始化标签生成器
    print("  - 初始化标签生成器...")
    label_generator = VQAClosedLabelGenerator(
        teacher_model=teacher,
        config=config
    )

    # ====================
    # 2. 问题分类
    # ====================
    print("\n【步骤2】问题分类...")

    classification_result = classifier.classify(question)

    question_type = classification_result.question_type.value
    confidence = classification_result.confidence
    method = classification_result.method

    print(f"\n  问题: '{question}'")
    print(f"  问题类型: {question_type} (置信度: {confidence:.2f}, 方法: {method})")

    # ====================
    # 3. 生成标签和CoT
    # ====================
    print("\n【步骤3】生成软硬标签和CoT...")

    # 调用标签生成器
    result = label_generator.generate_labels(
        image_path=image_path,
        question=question,
        question_type=question_type,
        candidate_pool=candidate_pool,
        ground_truth=ground_truth
    )

    # ====================
    # 4. 展示结果
    # ====================
    print("\n" + "="*160)
    print("【结果展示】".center(160))
    print("="*160)

    # 4.1 问题信息
    print("\n┌" + "─"*158 + "┐")
    print("│ 【问题信息】".ljust(159) + "│")
    print("├" + "─"*158 + "┤")
    print(f"│ 图像路径: {Path(image_path).name}".ljust(159) + "│")
    print(f"│ 问题: {question}".ljust(159) + "│")
    print(f"│ 问题类型: {question_type} (置信度: {confidence:.2f})".ljust(159) + "│")
    print(f"│ 分类方法: {method}".ljust(159) + "│")
    if ground_truth:
        print(f"│ 真实答案: {ground_truth}".ljust(159) + "│")
    if candidate_pool:
        print(f"│ 候选池: {candidate_pool}".ljust(159) + "│")
    print("└" + "─"*158 + "┘")

    # 4.2 硬标签
    hard_label = result.get('hard_label', {})
    print("\n┌" + "─"*158 + "┐")
    print("│ 【硬标签】".ljust(159) + "│")
    print("├" + "─"*158 + "┤")
    print(f"│ 答案: {hard_label.get('answer', 'N/A')}".ljust(159) + "│")
    print(f"│ 置信度: {hard_label.get('confidence', 0.0):.4f}".ljust(159) + "│")
    print("└" + "─"*158 + "┘")

    # 4.3 软标签
    soft_label = result.get('soft_label', {})
    answer_distribution = soft_label.get('answer_distribution', {})

    print("\n┌" + "─"*158 + "┐")
    print("│ 【软标签】".ljust(159) + "│")
    print("├" + "─"*158 + "┤")
    print(f"│ 主答案: {soft_label.get('primary_answer', 'N/A')}".ljust(159) + "│")

    # 按概率排序显示答案分布
    if answer_distribution:
        print("│ 答案分布:".ljust(159) + "│")
        sorted_answers = sorted(answer_distribution.items(), key=lambda x: x[1], reverse=True)

        # 显示前10个答案
        for i, (answer, prob) in enumerate(sorted_answers[:10], 1):
            bar = "█" * int(prob * 50)  # 可视化概率条
            print(f"│   {i:>2}. {answer:<20} {prob:>6.4f}  {bar}".ljust(159) + "│")

        if len(sorted_answers) > 10:
            print(f"│   ... 还有 {len(sorted_answers)-10} 个答案".ljust(159) + "│")

    print("└" + "─"*158 + "┘")

    # 4.4 CoT推理
    cot_reasoning = result.get('cot_reasoning', {})
    reasoning_paragraph = cot_reasoning.get('reasoning_paragraph', '')

    print("\n┌" + "─"*158 + "┐")
    print("│ 【CoT推理】".ljust(159) + "│")
    print("├" + "─"*158 + "┤")

    if reasoning_paragraph:
        # 分段显示CoT（每行不超过158字符）
        words = reasoning_paragraph.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 <= 155:
                line += " " + word if line else word
            else:
                print(f"│ {line}".ljust(159) + "│")
                line = word
        if line:
            print(f"│ {line}".ljust(159) + "│")
    else:
        print("│ 无CoT生成".ljust(159) + "│")

    print("└" + "─"*158 + "┘")

    # 4.5 Token过滤信息（如果有的话）
    # 注意：这部分信息需要从label_generator内部获取
    # 由于我们只做调用，这里展示结果中已有的信息

    print("\n┌" + "─"*158 + "┐")
    print("│ 【配置信息】".ljust(159) + "│")
    print("├" + "─"*158 + "┤")

    temperature = config.get("distillation.soft_labels.temperature", 4)
    top_k_logits = config.get("distillation.soft_labels.top_k_logits", 50)

    print(f"│ 软标签温度: T={temperature}".ljust(159) + "│")
    print(f"│ Top-K logits: {top_k_logits}".ljust(159) + "│")
    print(f"│ 候选池类型: {question_type}".ljust(159) + "│")

    print("└" + "─"*158 + "┘")

    # 4.6 完整结果（JSON格式）
    print("\n┌" + "─"*158 + "┐")
    print("│ 【完整结果（JSON）】".ljust(159) + "│")
    print("├" + "─"*158 + "┤")

    # 格式化JSON输出
    json_str = json.dumps(result, indent=2, ensure_ascii=False)

    # 分行显示
    for line in json_str.split('\n'):
        if len(line) > 155:
            # 超长行截断
            line = line[:152] + "..."
        print(f"│ {line}".ljust(159) + "│")

    print("└" + "─"*158 + "┘")

    # 关闭分类器
    classifier.close()

    print("\n" + "="*160)
    print("✅ 可视化完成".center(160))
    print("="*160)


def main():
    """主函数"""
    # ====================
    # 测试用例1：颜色问题
    # ====================
    print("\n" + "="*160)
    print("测试用例1: 颜色问题".center(160))
    print("="*160)

    image_path = "data/coco/val2014/COCO_val2014_000000051314.jpg"
    question = "What is the color of the water?"
    ground_truth = "green"

    if Path(image_path).exists():
        visualize_logits_comparison(
            image_path=image_path,
            question=question,
            ground_truth=ground_truth
        )
    else:
        print(f"⚠️  图像不存在: {image_path}")

    # ====================
    # 测试用例2：是否问题
    # ====================
    print("\n" + "="*160)
    print("测试用例2: 是否问题".center(160))
    print("="*160)

    image_path = "data/coco/val2014/COCO_val2014_000000545000.jpg"
    question = "Is the fire hydrant red?"
    ground_truth = "yes"

    if Path(image_path).exists():
        visualize_logits_comparison(
            image_path=image_path,
            question=question,
            ground_truth=ground_truth
        )
    else:
        print(f"⚠️  图像不存在: {image_path}")

    # ====================
    # 测试用例3：选择问题
    # ====================
    print("\n" + "="*160)
    print("测试用例3: 选择问题".center(160))
    print("="*160)

    image_path = "data/coco/val2014/COCO_val2014_000000051314.jpg"
    question = "Is it day or night?"
    ground_truth = "day"
    candidate_pool = ["day", "night"]

    if Path(image_path).exists():
        visualize_logits_comparison(
            image_path=image_path,
            question=question,
            ground_truth=ground_truth,
            candidate_pool=candidate_pool
        )
    else:
        print(f"⚠️  图像不存在: {image_path}")


if __name__ == "__main__":
    main()