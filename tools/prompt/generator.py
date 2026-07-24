#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prompt生成器 - 完整版
===================

支持策略：
- real_labels: 基于真实标签生成
- pattern_based: 基于数据模式生成
- dspy: DSPy优化生成

使用方式：
    python -m tools.prompt.generator --strategy real_labels
    python -m tools.prompt.generator --strategy pattern_based
    python -m tools.prompt.generator --strategy dspy
"""

import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter
import textwrap

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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


class PromptGenerator:
    """Prompt生成器 - 支持多种策略"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        self.source_dir = Path(config.get('source_dir', 'outputs/merged'))
        self.num_samples = config.get('num_samples', 100)
        self.strategy = config.get('strategy', 'real_labels')
        self.logger = get_logger()

    def generate(self) -> Dict[str, Any]:
        """
        生成Prompt

        Returns:
            Prompt字典
        """
        self.logger.info(f"📊 使用 {self.strategy} 策略生成Prompt")
        self.logger.info(f"  数据源: {self.source_dir}")
        self.logger.info(f"  样本数: {self.num_samples}")

        if self.strategy == 'real_labels':
            return self._generate_from_real_labels()
        elif self.strategy == 'pattern_based':
            return self._generate_from_patterns()
        elif self.strategy == 'dspy':
            return self._generate_with_dspy()
        elif self.strategy == 'dspy_fewshot':
            return self._generate_with_dspy_fewshot()
        else:
            raise ValueError(f"未知策略: {self.strategy}")

    def _generate_from_real_labels(self) -> Dict[str, Any]:
        """基于真实标签生成Prompt"""
        # 加载真实标签
        samples = self._load_real_labels()

        if not samples:
            self.logger.warning("没有找到真实标签数据")
            return {}

        # 分析数据
        analysis = self._analyze_samples(samples)

        # 生成Prompt
        prompts = self._build_prompts(analysis, samples)

        self.logger.info("✓ Prompt生成完成")
        return prompts

    def _generate_from_patterns(self) -> Dict[str, Any]:
        """基于数据模式生成Prompt"""
        self.logger.info("基于数据模式生成Prompt")
        # TODO: 实现基于模式的生成
        return self._generate_from_real_labels()

    def _generate_with_dspy(self) -> Dict[str, Any]:
        """使用DSPy优化生成Prompt（MIPROv2方法）"""
        try:
            import dspy
            DSPY_AVAILABLE = True
        except ImportError:
            self.logger.warning("DSPy未安装，使用real_labels策略")
            self.logger.info("安装DSPy: pip install dspy-ai")
            return self._generate_from_real_labels()

        self.logger.info("使用DSPy MIPROv2优化生成Prompt")

        # 加载真实标签作为训练数据
        samples = self._load_real_labels()
        if not samples:
            self.logger.warning("没有找到训练数据")
            return {}

        # 创建DSPy优化prompt（MIPROv2风格）
        prompts = self._build_dspy_prompts(samples)

        self.logger.info("✓ DSPy MIPROv2 Prompt生成完成")
        return prompts

    def _generate_with_dspy_fewshot(self) -> Dict[str, Any]:
        """
        使用DSPy基于真实数据的Few-Shot方法生成Prompt

        核心思路：
        1. 从真实数据中选择高质量示例
        2. 构建few-shot prompt模板
        3. 自动优化示例的选择和排列
        """
        try:
            import dspy
            DSPY_AVAILABLE = True
        except ImportError:
            self.logger.warning("DSPy未安装，使用real_labels策略")
            self.logger.info("安装DSPy: pip install dspy-ai")
            return self._generate_from_real_labels()

        self.logger.info("使用DSPy Few-Shot方法生成Prompt")

        # 加载真实标签数据
        samples = self._load_real_labels()
        if not samples:
            self.logger.warning("没有找到训练数据")
            return {}

        # 筛选高质量示例
        quality_samples = self._select_quality_samples(samples)

        if not quality_samples:
            self.logger.warning("没有找到高质量示例")
            return self._generate_from_real_labels()

        # 构建Few-Shot Prompt
        prompts = self._build_fewshot_prompts(quality_samples)

        self.logger.info(f"✓ DSPy Few-Shot Prompt生成完成（使用{len(quality_samples)}个示例）")
        return prompts

    def _select_quality_samples(self, samples: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        选择高质量示例

        选择标准：
        1. 置信度高
        2. 有完整的CoT推理
        3. 答案分布清晰
        """
        quality_samples = []

        for sample in samples:
            # 检查硬标签置信度
            hard_label = sample.get('hard_label', {})
            confidence = hard_label.get('confidence', 0)

            if confidence < 0.7:  # 只选择高置信度样本
                continue

            # 检查是否有完整的CoT推理
            cot = sample.get('cot_reasoning', {})
            structured = cot.get('structured_reasoning', {})

            if not structured:
                continue

            # 检查是否有软标签分布
            soft_label = sample.get('soft_label', {})
            answer_dist = soft_label.get('answer_distribution', {})

            if not answer_dist or len(answer_dist) < 2:
                continue

            # 符合条件，添加到列表
            quality_samples.append({
                'question': sample.get('question', ''),
                'hard_label': hard_label,
                'soft_label': soft_label,
                'cot_reasoning': cot,
                'quality_score': confidence
            })

        # 按质量分数排序
        quality_samples.sort(key=lambda x: x['quality_score'], reverse=True)

        # 返回Top-K
        return quality_samples[:top_k]

    def _build_fewshot_prompts(self, quality_samples: List[Dict]) -> Dict[str, Any]:
        """
        构建Few-Shot Prompt

        包含：
        1. 任务说明
        2. 多个高质量示例
        3. 明确的输出格式要求
        """
        # 构建示例部分
        fewshot_examples = []
        for i, sample in enumerate(quality_samples[:5], 1):
            question = sample['question']
            hard_label = sample['hard_label']
            primary_answer = hard_label.get('answer', '')

            soft_label = sample['soft_label']
            answer_dist = soft_label.get('answer_distribution', {})
            allowed_answers = list(answer_dist.keys())

            cot = sample['cot_reasoning']
            structured = cot.get('structured_reasoning', {})

            # 格式化概率分布
            dist_str = ', '.join([f"{ans}:{prob:.2f}" for ans, prob in answer_dist.items()])

            example = f"""EXAMPLE {i}:
Question: {question}
Allowed Answers: {', '.join(allowed_answers)}
Probability Distribution: {dist_str}

Observation: {structured.get('observation', '')}

Analysis: {structured.get('analysis', '')}

Conclusion: {structured.get('conclusion', '')}

---"""
            fewshot_examples.append(example)

        fewshot_block = '\n\n'.join(fewshot_examples)

        # 构建System Prompt
        system_prompt = f"""TASK: Answer visual questions by selecting from the given answer list based on probability distribution.

METHODOLOGY:
You will be given:
1. A question about an image
2. A list of allowed answers
3. A probability distribution over these answers
4. The primary (correct) answer

Your job is to:
1. OBSERVE: Describe ONLY visual features relevant to distinguishing between the allowed answers
2. ANALYZE: Use the probability distribution to reason about which answer best matches
3. CONCLUDE: Select ONE answer from the allowed answers list

QUALITY EXAMPLES (from real data):
{fewshot_block}

OUTPUT FORMAT:
Observation: [Visual features relevant to answer choices]
Analysis: [Reasoning using probability distribution]
Conclusion: Final Answer: [one answer from allowed list]

IMPORTANT:
- Focus ONLY on features that help choose between the allowed answers
- Use the probability distribution as a guide (higher probability = more likely)
- Always end with "Final Answer: X" where X is from the allowed answers"""

        # 构建User Prompt
        user_prompt = """Question: {question}

Allowed Answers: {allowed_answers}
Probability Distribution: {answer_distribution}
Primary Answer: {primary_answer}

Now provide your reasoning:

Observation:"""

        return {
            'prompts': {
                'cot': {
                    'vqa_system': system_prompt,
                    'vqa_user': user_prompt
                }
            },
            'metadata': {
                'source': 'dspy_fewshot',
                'num_samples': len(quality_samples),
                'generated_at': datetime.now().isoformat(),
                'method': 'few-shot learning from real data'
            }
        }

    def _load_real_labels(self) -> List[Dict]:
        """加载真实标签"""
        samples = []

        if not self.source_dir.exists():
            self.logger.warning(f"数据源目录不存在: {self.source_dir}")
            return samples

        # 加载JSON文件
        json_files = list(self.source_dir.glob("COCO_val2014_*.json"))[:self.num_samples]

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'tasks' in data and 'vqa' in data['tasks']:
                    vqa_data = data['tasks']['vqa']
                    samples.append({
                        'image_id': data.get('image_id'),
                        'question': vqa_data.get('question'),
                        'hard_label': vqa_data.get('hard_label', {}),
                        'soft_label': vqa_data.get('soft_label', {}),
                        'cot_reasoning': vqa_data.get('cot_reasoning', {})
                    })
            except Exception as e:
                self.logger.warning(f"加载失败 {json_file}: {e}")
                continue

        self.logger.info(f"✓ 加载 {len(samples)} 个样本")
        return samples

    def _analyze_samples(self, samples: List[Dict]) -> Dict:
        """分析样本"""
        analysis = {
            'total': len(samples),
            'answer_counter': Counter(),
            'question_types': Counter(),
            'avg_confidence': 0.0
        }

        confidences = []

        for sample in samples:
            hard_label = sample.get('hard_label', {})
            if hard_label:
                answer = hard_label.get('answer', '')
                if answer:
                    analysis['answer_counter'][answer] += 1

                confidence = hard_label.get('confidence', 0)
                if confidence:
                    confidences.append(confidence)

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

        if confidences:
            analysis['avg_confidence'] = sum(confidences) / len(confidences)

        self.logger.info(f"✓ 分析完成: {analysis['total']} 个样本")
        self.logger.info(f"  问题类型分布: {dict(analysis['question_types'])}")
        self.logger.info(f"  平均置信度: {analysis['avg_confidence']:.3f}")

        return analysis

    def _build_prompts(self, analysis: Dict, samples: List[Dict]) -> Dict[str, Any]:
        """构建Prompt"""
        common_answers = [ans for ans, _ in analysis['answer_counter'].most_common(20)]
        examples = self._create_examples(samples[:5])

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

        user_prompt = """Question: {question}

ALLOWED ANSWERS: {allowed_answers}
PRIMARY ANSWER (from hard label): {primary_answer}
PROBABILITY DISTRIBUTION (from soft label): {answer_distribution}

INSTRUCTIONS:
1. Observation: Describe ONLY visual features that help distinguish between these answers
2. Analysis: Use the probability distribution to reason about which answer best matches
3. Conclusion: Select ONE answer from the allowed answers list

Observation: [Focus on answer-relevant features]

Analysis: [Use probability distribution to guide reasoning]

Conclusion: Final Answer: {primary_answer}"""

        return {
            'prompts': {
                'cot': {
                    'vqa_system': system_prompt,
                    'vqa_user': user_prompt
                }
            },
            'metadata': {
                'source': 'real_labels',
                'num_samples': analysis['total'],
                'avg_confidence': analysis['avg_confidence'],
                'generated_at': datetime.now().isoformat()
            }
        }

    def _build_dspy_prompts(self, samples: List[Dict]) -> Dict[str, Any]:
        """构建DSPy优化的Prompt"""
        # DSPy优化的prompt模板
        system_prompt = """TASK: Answer the question by selecting from the given answer list.

CRITICAL RULES:
1. OBSERVATION: ONLY describe visual features relevant to distinguishing between the allowed answers
   - DO NOT describe unrelated background or scenery
   - Focus on features that help choose from the answer list
   - Be specific about colors, counts, positions ONLY if relevant to the answer options

2. ANALYSIS: Use the probability distribution to guide reasoning
   - Start from the highest probability answer
   - Compare visual evidence against each candidate answer
   - Explain why certain answers are more likely than others

3. CONCLUSION: Select ONE answer from the allowed answers list
   - MUST be exactly one of: {allowed_answers}
   - Match the format of allowed answers exactly

FORBIDDEN BEHAVIORS:
- DO NOT describe unrelated background scenery
- DO NOT mention objects not relevant to the answer choice
- DO NOT speculate beyond the visual evidence
- DO NOT use forbidden words: appear, seem, look like, might, probably

EXAMPLES:

Example 1 - Binary Choice:
Question: Is there a dog in the image?
Allowed answers: yes, no
Distribution: yes:0.85, no:0.10

Observation: I see a four-legged animal with fur and floppy ears in the center.
Analysis: The distribution suggests 'yes' (85%). The animal has distinctive dog features: four legs, fur, floppy ears. The 'no' option (10%) contradicts the visual evidence.
Conclusion: Final Answer: yes

Example 2 - Color Choice:
Question: What color is the car?
Allowed answers: red, blue, green, black, white
Distribution: red:0.70, blue:0.15, green:0.10

Observation: The car has a bright, warm-toned paint color.
Analysis: Starting from the highest probability (red: 70%), I check if the car's color matches red characteristics. The visual evidence shows a warm, reddish hue.
Conclusion: Final Answer: red"""

        user_prompt = """Question: {question}

ALLOWED ANSWERS: {allowed_answers}
PRIMARY ANSWER: {primary_answer}
PROBABILITY DISTRIBUTION: {answer_distribution}

INSTRUCTIONS:
1. Observation: Describe ONLY visual features that help distinguish between these answers
2. Analysis: Use the probability distribution to reason about which answer best matches
3. Conclusion: Select ONE answer from the allowed answers list

Observation: [Focus on answer-relevant visual features]

Analysis: [Use probability distribution to guide reasoning]

Conclusion: Final Answer: {primary_answer}"""

        # 创建示例
        examples = self._create_examples(samples[:3])

        return {
            'prompts': {
                'cot': {
                    'vqa_system': system_prompt,
                    'vqa_user': user_prompt
                }
            },
            'metadata': {
                'source': 'dspy_optimized',
                'num_samples': len(samples),
                'generated_at': datetime.now().isoformat()
            },
            'examples': examples
        }

    def _create_examples(self, samples: List[Dict]) -> str:
        """创建示例字符串"""
        examples = []

        for i, sample in enumerate(samples, 1):
            question = sample.get('question', '')
            hard_label = sample.get('hard_label', {})
            primary_answer = hard_label.get('answer', '')
            soft_label = sample.get('soft_label', {})
            answer_dist = soft_label.get('answer_distribution', {})
            allowed_answers = list(answer_dist.keys())

            cot = sample.get('cot_reasoning', {})
            structured = cot.get('structured_reasoning', {})

            if not structured or not allowed_answers:
                continue

            dist_str = ', '.join([f"{ans}:{prob:.2f}" for ans, prob in answer_dist.items()])

            example = f"""Example {i}:
Question: {question}
Allowed answers: {', '.join(allowed_answers)}
Distribution: {dist_str}

Observation: {structured.get('observation', 'N/A')}
Analysis: {structured.get('analysis', 'N/A')}
Conclusion: {structured.get('conclusion', 'N/A')}
"""
            examples.append(example)

        return '\n'.join(examples) if examples else "No examples available"


def save_prompts_yaml(prompts: Dict[str, Any], output_path: Path):
    """
    保存Prompt到YAML文件（处理换行和转义）

    Args:
        prompts: Prompt字典
        output_path: 输出路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 自定义YAML Dumper，处理多行字符串
    class YamlDumper(yaml.SafeDumper):
        pass

    def str_representer(dumper, data):
        """处理字符串的YAML表示"""
        if '\n' in data:
            # 多行字符串使用块样式
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    YamlDumper.add_representer(str, str_representer)

    # 保存YAML
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(prompts, f, Dumper=YamlDumper, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✓ Prompt已保存: {output_path}")


def main():
    """独立执行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Prompt生成器")
    parser.add_argument('--config', default='configs/tools.yaml')
    parser.add_argument('--strategy',
                        choices=['real_labels', 'pattern_based', 'dspy', 'dspy_fewshot'],
                        default='real_labels',
                        help='生成策略: real_labels(默认), pattern_based, dspy(MIPROv2), dspy_fewshot(Few-Shot)')
    parser.add_argument('--source_dir', default='outputs/merged')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output', default='outputs/prompts/vqa_en.yaml')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # 合并参数
    prompt_config = config.get('prompt_generation', {})
    if args.strategy:
        prompt_config['strategy'] = args.strategy
    if args.source_dir:
        prompt_config['source_dir'] = args.source_dir
    if args.num_samples:
        prompt_config['num_samples'] = args.num_samples

    # 生成
    generator = PromptGenerator(prompt_config)
    prompts = generator.generate()

    # 保存（使用自定义YAML保存方法）
    output_path = Path(args.output)
    save_prompts_yaml(prompts, output_path)


if __name__ == "__main__":
    main()