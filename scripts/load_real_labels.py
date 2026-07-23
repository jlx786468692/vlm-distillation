"""
真实标签数据加载器
==================

从 outputs/merged/ 目录加载真实的硬标签和软标签数据。

数据结构：
- 每个图像一个JSON文件
- 包含 hard_label（硬标签）、soft_label（软标签）、cot_reasoning（CoT推理）

使用方法：
    from scripts.load_real_labels import RealLabelLoader

    loader = RealLabelLoader("outputs/merged/")
    samples = loader.load_vqa_samples(num_samples=100)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


# ==================
# 答案标准化工具
# ==================

# 数字到英文的映射
NUMBER_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
}

# 英文到数字的映射
WORD_TO_NUMBER = {v: k for k, v in NUMBER_TO_WORD.items()}


def normalize_answer(answer: str, target_format: str = 'word') -> str:
    """
    标准化答案格式

    Args:
        answer: 原始答案（可能是 "1" 或 "one"）
        target_format: 目标格式，'word'（英文）或 'number'（阿拉伯数字）

    Returns:
        标准化后的答案

    Examples:
        >>> normalize_answer('1', 'word')
        'one'
        >>> normalize_answer('one', 'number')
        '1'
    """
    answer = answer.strip().lower()

    # 如果是空值，直接返回
    if not answer:
        return answer

    # 转换为英文格式
    if target_format == 'word':
        # 如果已经是英文，直接返回
        if answer in WORD_TO_NUMBER:
            return answer
        # 如果是数字，转换为英文
        if answer in NUMBER_TO_WORD:
            return NUMBER_TO_WORD[answer]
        # 其他情况（如 "yes", "no" 等）直接返回
        return answer

    # 转换为数字格式
    elif target_format == 'number':
        if answer in NUMBER_TO_WORD:
            return answer  # 已经是数字
        if answer in WORD_TO_NUMBER:
            return WORD_TO_NUMBER[answer]  # 英文转数字
        return answer

    return answer


def normalize_distribution_keys(distribution: Dict[str, float], target_format: str = 'word') -> Dict[str, float]:
    """
    标准化概率分布的键

    Args:
        distribution: 原始概率分布，如 {'one': 0.25, 'two': 0.17}
        target_format: 目标格式，'word' 或 'number'

    Returns:
        标准化后的概率分布
    """
    return {
        normalize_answer(key, target_format): value
        for key, value in distribution.items()
    }


class RealLabelLoader:
    """
    真实标签数据加载器

    从 outputs/merged/ 目录加载硬标签和软标签数据
    """

    def __init__(self, merged_dir: str = "outputs/merged/"):
        """
        初始化加载器

        Args:
            merged_dir: 合并后的标签数据目录
        """
        self.merged_dir = Path(merged_dir)

        if not self.merged_dir.exists():
            raise FileNotFoundError(f"标签目录不存在: {merged_dir}")

        # 加载摘要文件
        self.summary = self._load_summary()

        print(f"✓ 加载真实标签数据")
        print(f"  目录: {merged_dir}")
        print(f"  总图像数: {self.summary.get('total_images', 0)}")
        print(f"  任务类型: {', '.join(self.summary.get('tasks', []))}")

    def _load_summary(self) -> Dict:
        """加载摘要文件"""
        summary_path = self.merged_dir / "merged_summary.json"

        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        return {}

    def load_vqa_samples(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        加载VQA标签数据

        Args:
            num_samples: 加载样本数（None表示加载全部）

        Returns:
            VQA样本列表，每个样本包含：
            - image_path: 图像路径
            - question: 问题
            - hard_label: 硬标签（答案和置信度）
            - soft_label: 软标签（概率分布）
            - cot_reasoning: CoT推理（如果有）
        """
        samples = []

        # 获取所有JSON文件（排除摘要文件）
        json_files = list(self.merged_dir.glob("COCO_val2014_*.json"))

        if num_samples:
            json_files = json_files[:num_samples]

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 提取VQA数据
                if 'tasks' in data and 'vqa' in data['tasks']:
                    vqa_data = data['tasks']['vqa']

                    sample = {
                        'image_id': data.get('image_id'),
                        'image_path': data.get('image_path'),
                        'question': vqa_data.get('question'),
                        'hard_label': vqa_data.get('hard_label', {}),
                        'soft_label': vqa_data.get('soft_label', {}),
                        'cot_reasoning': vqa_data.get('cot_reasoning', {}),
                        'metadata': data.get('metadata', {})
                    }

                    samples.append(sample)

            except Exception as e:
                print(f"⚠ 加载文件失败 {json_file}: {e}")
                continue

        print(f"✓ 加载 {len(samples)} 个VQA样本")
        return samples

    def load_detection_samples(self, num_samples: Optional[int] = None) -> List[Dict]:
        """
        加载Detection标签数据

        Args:
            num_samples: 加载样本数（None表示加载全部）

        Returns:
            Detection样本列表
        """
        samples = []

        json_files = list(self.merged_dir.glob("COCO_val2014_*.json"))

        if num_samples:
            json_files = json_files[:num_samples]

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'tasks' in data and 'detection' in data['tasks']:
                    detection_data = data['tasks']['detection']

                    sample = {
                        'image_id': data.get('image_id'),
                        'image_path': data.get('image_path'),
                        'hard_label': detection_data.get('hard_label', {}),
                        'soft_label': detection_data.get('soft_label', {}),
                        'cot_reasoning': detection_data.get('cot_reasoning', {}),
                        'metadata': data.get('metadata', {})
                    }

                    samples.append(sample)

            except Exception as e:
                print(f"⚠ 加载文件失败 {json_file}: {e}")
                continue

        print(f"✓ 加载 {len(samples)} 个Detection样本")
        return samples

    def get_statistics(self) -> Dict:
        """
        获取标签统计信息

        Returns:
            统计信息字典
        """
        vqa_samples = self.load_vqa_samples()
        detection_samples = self.load_detection_samples()

        stats = {
            'total_images': len(vqa_samples),
            'vqa': {
                'total': len(vqa_samples),
                'with_hard_label': sum(1 for s in vqa_samples if s['hard_label']),
                'with_soft_label': sum(1 for s in vqa_samples if s['soft_label']),
                'with_cot': sum(1 for s in vqa_samples if s['cot_reasoning']),
                'avg_confidence': 0.0
            },
            'detection': {
                'total': len(detection_samples),
                'with_hard_label': sum(1 for s in detection_samples if s['hard_label']),
                'with_soft_label': sum(1 for s in detection_samples if s['soft_label']),
                'with_cot': sum(1 for s in detection_samples if s['cot_reasoning'])
            }
        }

        # 计算平均置信度
        if vqa_samples:
            confidences = [
                s['hard_label'].get('confidence', 0)
                for s in vqa_samples
                if s['hard_label']
            ]
            if confidences:
                stats['vqa']['avg_confidence'] = sum(confidences) / len(confidences)

        return stats

    def to_dspy_format(self, samples: List[Dict], target_format: str = 'word') -> List[Dict]:
        """
        转换为DSPy格式

        Args:
            samples: 样本列表
            target_format: 答案格式，'word'（英文）或 'number'（数字）

        Returns:
            DSPy格式的样本列表

        🔧 关键改进：标准化答案格式，确保 primary_answer 和 allowed_answers 一致
        """
        try:
            import dspy
            DSPY_AVAILABLE = True
        except ImportError:
            DSPY_AVAILABLE = False
            print("⚠ DSPy未安装，返回普通字典格式")

        dspy_samples = []

        for sample in samples:
            # 提取硬标签
            hard_label = sample.get('hard_label', {})
            primary_answer_raw = hard_label.get('answer', '')

            # 提取软标签
            soft_label = sample.get('soft_label', {})
            answer_distribution_raw = soft_label.get('answer_distribution', {})

            # 🔧 关键：标准化答案格式（统一为英文或数字）
            # 标准化概率分布的键
            answer_distribution = normalize_distribution_keys(answer_distribution_raw, target_format)
            allowed_answers = list(answer_distribution.keys())

            # 标准化 primary_answer
            # 优先使用软标签的 primary_answer（如果存在）
            primary_answer_raw = soft_label.get('primary_answer', primary_answer_raw)
            primary_answer = normalize_answer(primary_answer_raw, target_format)

            # 🔧 验证：确保 primary_answer 在 allowed_answers 中
            if primary_answer not in allowed_answers:
                # 尝试找到最相似的答案
                primary_answer_normalized = normalize_answer(primary_answer, target_format)
                if primary_answer_normalized in allowed_answers:
                    primary_answer = primary_answer_normalized
                else:
                    # 如果还是不匹配，使用概率最高的答案
                    if answer_distribution:
                        primary_answer = max(answer_distribution.items(), key=lambda x: x[1])[0]

            # 格式化概率分布
            distribution_str = ', '.join([
                f"{ans}:{prob:.2f}"
                for ans, prob in answer_distribution.items()
            ])

            if DSPY_AVAILABLE:
                # 创建DSPy Example
                dspy_sample = dspy.Example(
                    image_path=sample['image_path'],
                    question=sample['question'],
                    allowed_answers=', '.join(allowed_answers),
                    primary_answer=primary_answer,
                    answer_distribution=distribution_str,
                    observation="",  # 由模型生成
                    analysis="",
                    conclusion=""
                ).with_inputs('image_path', 'question', 'allowed_answers', 'primary_answer', 'answer_distribution')

                dspy_samples.append(dspy_sample)
            else:
                # 返回普通字典
                dspy_samples.append({
                    'image_path': sample['image_path'],
                    'question': sample['question'],
                    'allowed_answers': allowed_answers,
                    'primary_answer': primary_answer,
                    'answer_distribution': answer_distribution,
                    'distribution_str': distribution_str
                })

        return dspy_samples


def main():
    """测试加载器"""
    import argparse

    parser = argparse.ArgumentParser(description="真实标签数据加载器")
    parser.add_argument('--dir', type=str, default='outputs/merged/',
                        help='标签数据目录')
    parser.add_argument('--task', type=str, default='vqa', choices=['vqa', 'detection', 'stats'],
                        help='任务类型')
    parser.add_argument('--num', type=int, default=None,
                        help='加载数量')

    args = parser.parse_args()

    # 初始化加载器
    loader = RealLabelLoader(args.dir)

    if args.task == 'stats':
        # 显示统计信息
        stats = loader.get_statistics()
        print("\n" + "="*60)
        print("标签统计信息")
        print("="*60)
        print(f"总图像数: {stats['total_images']}")
        print(f"\nVQA任务:")
        print(f"  总样本: {stats['vqa']['total']}")
        print(f"  有硬标签: {stats['vqa']['with_hard_label']}")
        print(f"  有软标签: {stats['vqa']['with_soft_label']}")
        print(f"  有CoT: {stats['vqa']['with_cot']}")
        print(f"  平均置信度: {stats['vqa']['avg_confidence']:.3f}")
        print(f"\nDetection任务:")
        print(f"  总样本: {stats['detection']['total']}")
        print(f"  有硬标签: {stats['detection']['with_hard_label']}")
        print(f"  有软标签: {stats['detection']['with_soft_label']}")
        print(f"  有CoT: {stats['detection']['with_cot']}")
        print("="*60)

    elif args.task == 'vqa':
        # 加载VQA样本
        samples = loader.load_vqa_samples(args.num)

        if samples:
            print("\n示例数据:")
            print("-"*60)
            sample = samples[0]
            print(f"图像: {sample['image_path']}")
            print(f"问题: {sample['question']}")
            print(f"硬标签: {sample['hard_label']}")
            print(f"软标签: {sample['soft_label']}")

    elif args.task == 'detection':
        # 加载Detection样本
        samples = loader.load_detection_samples(args.num)

        if samples:
            print("\n示例数据:")
            print("-"*60)
            sample = samples[0]
            print(f"图像: {sample['image_path']}")
            print(f"硬标签: {sample['hard_label']}")


if __name__ == "__main__":
    main()