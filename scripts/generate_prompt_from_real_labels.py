#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于真实标签的Prompt生成器（修复版）
====================================

使用 outputs/merged/ 下的真实硬标签和软标签数据生成优化的prompt。

使用方法：
    cd /data/workspace2/jlx/workspace/vlm-distillation
    python -m scripts.generate_prompt_from_real_labels --task vqa --num_samples 100

    或者：
    python scripts/generate_prompt_from_real_labels.py --task vqa --num_samples 100
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入工具模块
try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


class RealLabelLoader:
    """
    真实标签数据加载器（简化版，避免依赖问题）
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
            VQA样本列表
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


class RealLabelPromptGenerator:
    """
    基于真实标签的Prompt生成器
    """

    def __init__(self, labels_dir: str = "outputs/merged/"):
        """
        初始化生成器

        Args:
            labels_dir: 标签数据目录
        """
        self.loader = RealLabelLoader(labels_dir)
        self.logger = get_logger()
        self.output_dir = Path("configs/generated_prompts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_vqa_prompt_from_real_labels(self, num_samples: int = 100) -> Dict[str, str]:
        """
        基于真实VQA标签生成prompt

        Args:
            num_samples: 使用样本数

        Returns:
            生成的prompt字典
        """
        self.logger.info(f"加载 {num_samples} 个真实VQA标签样本")

        # 加载真实标签
        samples = self.loader.load_vqa_samples(num_samples)

        if not samples:
            self.logger.warning("没有找到VQA标签数据")
            return {}

        # 分析数据
        analysis = self._analyze_vqa_labels(samples)

        # 生成prompt
        prompts = self._generate_prompts_from_analysis(analysis, samples)

        # 保存prompt
        self._save_prompts(prompts, 'vqa_real_labels')

        return prompts

    def _analyze_vqa_labels(self, samples: List[Dict]) -> Dict:
        """
        分析VQA标签数据
        """
        from collections import Counter

        analysis = {
            'total_samples': len(samples),
            'question_types': Counter(),
            'answer_types': Counter(),
            'avg_confidence': 0.0,
            'common_questions': [],
            'answer_distribution_examples': []
        }

        confidences = []

        for sample in samples:
            # 分析问题类型
            question = sample.get('question', '').lower()

            if 'how many' in question:
                analysis['question_types']['count'] += 1
            elif 'what color' in question:
                analysis['question_types']['color'] += 1
            elif 'is there' in question or 'are there' in question:
                analysis['question_types']['existence'] += 1
            elif 'what' in question:
                analysis['question_types']['what'] += 1
            else:
                analysis['question_types']['other'] += 1

            # 分析答案
            hard_label = sample.get('hard_label', {})
            if hard_label:
                answer = hard_label.get('answer', '')
                analysis['answer_types'][answer] += 1

                confidence = hard_label.get('confidence', 0)
                confidences.append(confidence)

            # 收集软标签示例
            soft_label = sample.get('soft_label', {})
            if soft_label:
                answer_dist = soft_label.get('answer_distribution', {})
                if answer_dist:
                    analysis['answer_distribution_examples'].append({
                        'question': question,
                        'distribution': answer_dist,
                        'primary_answer': hard_label.get('answer', '')
                    })

        # 计算平均置信度
        if confidences:
            analysis['avg_confidence'] = sum(confidences) / len(confidences)

        self.logger.info(f"✓ 分析完成: {analysis['total_samples']} 个样本")
        self.logger.info(f"  问题类型分布: {dict(analysis['question_types'])}")
        self.logger.info(f"  平均置信度: {analysis['avg_confidence']:.3f}")

        return analysis

    def _generate_prompts_from_analysis(self, analysis: Dict, samples: List[Dict]) -> Dict[str, str]:
        """
        基于分析结果生成prompt
        """
        # 提取常见答案
        common_answers = [ans for ans, count in analysis['answer_types'].most_common(20)]

        # 提取示例
        examples = self._create_examples(samples[:5])

        # 生成system prompt
        system_prompt = f"""TASK: Answer the question by selecting from the given answer list based on probability distribution.

CRITICAL RULES:
1. OBSERVATION: ONLY describe visual features relevant to distinguishing between the allowed answers
   - DO NOT describe unrelated background or scenery
   - Focus on features that help choose from the answer list
   - Use the probability distribution as reference: higher probability answers should be prioritized

2. ANALYSIS: Use the probability distribution to guide reasoning
   - Start from the highest probability answer
   - Compare visual evidence against each candidate answer
   - Explain why certain answers are more likely than others
   - Distribution format: "answer:probability, answer:probability, ..."

3. CONCLUSION: Select ONE answer from the allowed answers list
   - MUST be exactly one of the allowed answers
   - Match the format of allowed answers exactly
   - Based on visual evidence AND probability distribution

COMMON ANSWERS (from real data):
{', '.join(common_answers[:15])}

QUALITY STANDARDS:
- Observation: Pure visual facts relevant to the answer options
- Analysis: Use probability distribution to prioritize reasoning
- Conclusion: Clear answer from the allowed list

REAL EXAMPLES (from actual label data):

{examples}"""

        # 生成user prompt模板
        user_prompt = """Question: {question}

ALLOWED ANSWERS: {allowed_answers}
PRIMARY ANSWER (from hard label): {primary_answer}
PROBABILITY DISTRIBUTION (from soft label): {answer_distribution}

INSTRUCTIONS:
1. Observation: Describe ONLY visual features that help distinguish between these answers
2. Analysis: Use the probability distribution to reason about which answer best matches
3. Conclusion: Select ONE answer from the allowed answers list

Observation: Focus on answer-relevant features

Analysis: Use probability distribution to guide reasoning

Conclusion: Final Answer: {primary_answer}"""

        return {
            'system': system_prompt,
            'user_template': user_prompt,
            'metadata': {
                'source': 'real_labels',
                'num_samples': analysis['total_samples'],
                'avg_confidence': analysis['avg_confidence'],
                'generated_at': datetime.now().isoformat()
            }
        }

    def _create_examples(self, samples: List[Dict]) -> str:
        """
        创建示例字符串
        """
        examples = []

        for i, sample in enumerate(samples, 1):
            question = sample.get('question', '')

            # 硬标签
            hard_label = sample.get('hard_label', {})
            primary_answer = hard_label.get('answer', '')

            # 软标签
            soft_label = sample.get('soft_label', {})
            answer_dist = soft_label.get('answer_distribution', {})
            allowed_answers = list(answer_dist.keys())  # ✅ 从键提取

            # CoT推理（如果有）
            cot = sample.get('cot_reasoning', {})
            structured = cot.get('structured_reasoning', {})

            if not structured:
                continue

            example = f"""Example {i}:
Question: {question}
Allowed answers: {', '.join(allowed_answers)}
Distribution: {', '.join([f"{ans}:{prob:.2f}" for ans, prob in answer_dist.items()])}

Observation: {structured.get('observation', 'N/A')[:200]}...
Analysis: {structured.get('analysis', 'N/A')[:200]}...
Conclusion: {structured.get('conclusion', 'N/A')}
"""

            examples.append(example)

        return '\n'.join(examples) if examples else "No examples available"

    def _save_prompts(self, prompts: Dict[str, str], task: str):
        """
        保存生成的prompt
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 🔧 使用独特的文件名（真实标签版本）
        output_file = self.output_dir / f"real_labels_{task}_prompts_{timestamp}.yaml"

        # 构建YAML数据
        yaml_data = {
            'task': task,
            'source': 'real_labels',
            'generated_by': 'Real Label Prompt Generator',
            'timestamp': timestamp,
            'metadata': prompts.get('metadata', {}),
            'prompts': {
                'system': prompts.get('system', ''),
                'user_template': prompts.get('user_template', '')
            }
        }

        # 保存到YAML
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"✓ Prompt已保存到: {output_file}")

        # 🔧 打印生成的prompt
        self._print_prompts(prompts, task, output_file)

    def _print_prompts(self, prompts: Dict[str, str], task: str, output_file: Path):
        """
        打印生成的prompt

        Args:
            prompts: prompt字典
            task: 任务类型
            output_file: 输出文件路径
        """
        print("\n" + "="*80)
        print(f"生成的 {task.upper()} Prompt [真实标签版本]")
        print("="*80)
        print(f"📁 保存位置: {output_file}")
        print("-"*80)

        system_prompt = prompts.get('system', '')
        user_prompt = prompts.get('user_template', '')
        metadata = prompts.get('metadata', {})

        print("\n📝 System Prompt:")
        print("-"*80)
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)

        print("\n📝 User Template:")
        print("-"*80)
        print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)

        print("\n📊 数据统计:")
        print("-"*80)
        print(f"  样本数: {metadata.get('num_samples', 'N/A')}")
        print(f"  平均置信度: {metadata.get('avg_confidence', 'N/A'):.3f}" if isinstance(metadata.get('avg_confidence'), (int, float)) else f"  平均置信度: {metadata.get('avg_confidence', 'N/A')}")
        print(f"  生成时间: {metadata.get('generated_at', 'N/A')}")

        print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于真实标签的Prompt生成器")
    parser.add_argument('--task', type=str, default='vqa', choices=['vqa', 'detection'],
                        help='任务类型')
    parser.add_argument('--labels_dir', type=str, default='outputs/merged/',
                        help='标签数据目录')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='使用样本数')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("基于真实标签的Prompt生成器")
    print("="*60)
    print(f"任务类型: {args.task}")
    print(f"标签目录: {args.labels_dir}")
    print(f"样本数: {args.num_samples}")
    print("="*60 + "\n")

    # 生成prompt
    generator = RealLabelPromptGenerator(args.labels_dir)

    if args.task == 'vqa':
        prompts = generator.generate_vqa_prompt_from_real_labels(args.num_samples)

        if prompts:
            print("\n生成的Prompt:")
            print("-"*60)
            print("System Prompt (前500字符):")
            print(prompts['system'][:1500] + "...")
            print("\nUser Template:")
            print(prompts['user_template'])

    print("\n" + "="*60)
    print("✓ Prompt生成完成")
    print("="*60)


if __name__ == "__main__":
    main()