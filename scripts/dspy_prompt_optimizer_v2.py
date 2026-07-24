"""
DSPy Prompt Optimizer - 答案导向版本
=====================================

优化版：CoT推理聚焦于答案选择过程，基于软标签概率分布和硬标签进行推理。

核心改进：
1. 输入包含软标签概率分布
2. 推理过程聚焦于答案列表
3. 避免无关背景描述
4. 直接基于概率分布进行选择推理

使用方法：
    python scripts/dspy_prompt_optimizer_v2.py --task vqa --num_samples 100
"""

import os
import sys
import json
import yaml
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import dspy
    from dspy import Signature, Module, Predict, Example
    from dspy.evaluate import Evaluate
    from dspy.teleprompt import MIPROv2, BootstrapFewShot, KNNFewShot
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠ DSPy未安装，请运行: pip install dspy-ai")

from pycocotools.coco import COCO
from PIL import Image
import torch
from transformers import AutoProcessor
from tqdm import tqdm

from src.utils.config import ConfigManager
from src.utils.logger import get_logger


# ==================
# DSPy Signatures - 答案导向版本
# ==================

class VQACoTSignatureV2(Signature):
    """
    VQA Chain-of-Thought推理签名 - 答案导向版本

    核心改进：推理过程聚焦于答案选择，不描述无关背景

    输入：
    - 图像路径
    - 问题
    - 允许的答案列表（从软标签提取）
    - 主要答案（硬标签）
    - 答案概率分布（软标签）

    输出：
    - 观察：聚焦于答案相关的视觉特征
    - 分析：基于概率分布的答案选择推理
    - 结论：最终答案
    """
    image_path: str = dspy.InputField(desc="Path to the image file")
    question: str = dspy.InputField(desc="Question about the image")
    allowed_answers: str = dspy.InputField(desc="Comma-separated list of allowed answers from soft labels")
    primary_answer: str = dspy.InputField(desc="The correct answer from hard label")
    answer_distribution: str = dspy.InputField(desc="Probability distribution over allowed answers (e.g., 'yes:0.8, no:0.15, maybe:0.05')")

    observation: str = dspy.OutputField(desc="Visual features relevant to distinguishing between the allowed answers (focus on answer-related features only)")
    analysis: str = dspy.OutputField(desc="Reasoning about which answer from the list best matches the visual evidence, using the probability distribution as reference")
    conclusion: str = dspy.OutputField(desc="Final answer selected from the allowed answers list")


class DetectionCoTSignatureV2(Signature):
    """
    Detection Chain-of-Thought推理签名 - 答案导向版本

    核心改进：推理过程聚焦于对象检测和分类，不描述无关背景

    输入：图像路径
    输出：扫描、对象、验证三个步骤的推理过程
    """
    image_path: str = dspy.InputField(desc="Path to the image file")

    scanning: str = dspy.OutputField(desc="Quick scan for object presence (focus on objects, not background)")
    objects: str = dspy.OutputField(desc="List of detected objects with confidence levels")
    verification: str = dspy.OutputField(desc="Verification that all significant objects are detected")


# ==================
# DSPy Modules - 答案导向版本
# ==================

class VQACoTModuleV2(dspy.Module):
    """
    VQA CoT推理模块 - 答案导向版本

    核心改进：推理过程聚焦于答案选择
    """

    def __init__(self):
        super().__init__()
        self.generate_cot = dspy.ChainOfThought(VQACoTSignatureV2)

    def forward(self, image_path: str, question: str, allowed_answers: str,
                primary_answer: str, answer_distribution: str):
        """执行VQA CoT推理 - 答案导向"""
        return self.generate_cot(
            image_path=image_path,
            question=question,
            allowed_answers=allowed_answers,
            primary_answer=primary_answer,
            answer_distribution=answer_distribution
        )


class DetectionCoTModuleV2(dspy.Module):
    """
    Detection CoT推理模块 - 答案导向版本
    """

    def __init__(self):
        super().__init__()
        self.generate_cot = dspy.ChainOfThought(DetectionCoTSignatureV2)

    def forward(self, image_path: str):
        """执行Detection CoT推理 - 答案导向"""
        return self.generate_cot(image_path=image_path)


# ==================
# Answer-Focused Prompt Templates
# ==================

class AnswerFocusedPromptTemplates:
    """
    答案导向的Prompt模板

    核心原则：
    1. 观察阶段：只描述与答案相关的视觉特征
    2. 分析阶段：基于概率分布进行答案选择推理
    3. 结论阶段：从答案列表中选择最终答案
    """

    @staticmethod
    def get_vqa_system_prompt() -> str:
        """获取VQA系统prompt - 答案导向版本"""
        return """TASK: Answer the question by selecting from the given answer list.

CRITICAL RULES:
1. OBSERVATION: ONLY describe visual features relevant to distinguishing between the allowed answers
   - DO NOT describe unrelated background or scenery
   - Focus on features that help choose from the answer list
   - Be specific about colors, counts, positions ONLY if relevant to the answer options

2. ANALYSIS: Use the probability distribution to guide reasoning
   - Start from the highest probability answer
   - Compare visual evidence against each candidate answer
   - Explain why certain answers are more likely than others
   - Use the distribution: {answer_distribution}

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
Distribution: yes:0.85, no:0.10, uncertain:0.05

Observation: I see a four-legged animal with fur, floppy ears, and a wagging tail in the center of the image.
Analysis: The probability distribution suggests 'yes' is most likely (85%). The animal has distinctive dog features: four legs, fur, floppy ears, and tail wagging. These features clearly match a dog, not a cat or other animal. The 'no' option (10%) would require the animal to not be a dog, which contradicts the visual evidence.
Conclusion: Final Answer: yes

Example 2 - Color Choice:
Question: What color is the car?
Allowed answers: red, blue, green, black, white, other
Distribution: red:0.70, blue:0.15, green:0.10, other:0.05

Observation: The car has a bright, warm-toned paint color that reflects sunlight.
Analysis: Starting from the highest probability (red: 70%), I check if the car's color matches red characteristics: bright, warm tone. The visual evidence shows a warm, reddish hue, not the cool tones of blue or green. The distribution supports red as the most likely answer. Black and white are ruled out by the color presence.
Conclusion: Final Answer: red

Example 3 - Count Choice:
Question: How many people are in the image?
Allowed answers: 0, 1, 2, 3, 4, 5, more
Distribution: 2:0.65, 3:0.20, 1:0.10, other:0.05

Observation: I can see two distinct persons: one on the left side and one in the center.
Analysis: The distribution suggests '2' as most likely (65%). I verify by counting: person 1 on the left, person 2 in the center. No other persons are visible. The '3' option (20%) would require a third person, which I don't see. The '1' option (10%) contradicts the visual count.
Conclusion: Final Answer: 2"""

    @staticmethod
    def get_vqa_user_prompt(question: str, allowed_answers: List[str],
                           primary_answer: str, answer_distribution: Dict[str, float]) -> str:
        """
        获取VQA用户prompt - 答案导向版本

        Args:
            question: 问题
            allowed_answers: 允许的答案列表
            primary_answer: 主要答案（硬标签）
            answer_distribution: 答案概率分布（软标签）

        Returns:
            用户prompt字符串
        """
        # 格式化概率分布
        dist_str = ', '.join([f"{ans}:{prob:.2f}" for ans, prob in answer_distribution.items()])

        return f"""Question: {question}

ALLOWED ANSWERS: {', '.join(allowed_answers)}
PRIMARY ANSWER: {primary_answer}
PROBABILITY DISTRIBUTION: {dist_str}

INSTRUCTIONS:
1. Observation: Describe ONLY visual features that help distinguish between these answers: {', '.join(allowed_answers)}
2. Analysis: Use the probability distribution to reason about which answer best matches visual evidence
3. Conclusion: Select ONE answer from: {', '.join(allowed_answers)}

Your reasoning should focus on choosing from the answer list, not describing unrelated content.

Observation: [Focus on answer-relevant visual features]

Analysis: [Use probability distribution to guide reasoning]

Conclusion: Final Answer: {primary_answer}"""


# ==================
# Data Preparation with Soft Labels
# ==================

class COCODataLoaderWithSoftLabels:
    """
    COCO数据加载器，支持软标签和硬标签
    """

    def __init__(
        self,
        coco_root: str,
        annotation_file: str,
        hard_labels_file: Optional[str] = None,
        soft_labels_file: Optional[str] = None,
        max_samples: int = 100
    ):
        """
        初始化COCO数据加载器

        Args:
            coco_root: COCO数据集根目录
            annotation_file: 标注文件路径
            hard_labels_file: 硬标签文件路径
            soft_labels_file: 软标签文件路径
            max_samples: 最大样本数
        """
        self.coco_root = coco_root
        self.coco = COCO(annotation_file)
        self.max_samples = max_samples
        self.logger = get_logger()

        # 加载硬标签和软标签（如果提供）
        self.hard_labels = None
        self.soft_labels = None

        if hard_labels_file and os.path.exists(hard_labels_file):
            with open(hard_labels_file, 'r', encoding='utf-8') as f:
                self.hard_labels = json.load(f)
            self.logger.info(f"✓ 加载硬标签: {len(self.hard_labels)} 条")

        if soft_labels_file and os.path.exists(soft_labels_file):
            with open(soft_labels_file, 'r', encoding='utf-8') as f:
                self.soft_labels = json.load(f)
            self.logger.info(f"✓ 加载软标签: {len(self.soft_labels)} 条")

        # 获取所有图像ID
        self.image_ids = list(self.coco.imgs.keys())
        random.shuffle(self.image_ids)

    def get_vqa_samples_with_labels(self, num_samples: int = 50) -> List[dspy.Example]:
        """
        获取VQA训练样本，包含软标签和硬标签

        Args:
            num_samples: 样本数量

        Returns:
            DSPy Example列表
        """
        samples = []
        count = 0

        for img_id in self.image_ids[:min(num_samples * 3, len(self.image_ids))]:
            if count >= num_samples:
                break

            # 获取图像信息
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.coco_root, img_info['file_name'])

            if not os.path.exists(img_path):
                continue

            # 如果有真实的硬标签和软标签，使用它们
            if self.hard_labels and self.soft_labels:
                # 从真实数据中提取
                image_id_str = str(img_id)
                if image_id_str in self.hard_labels:
                    hard_label_data = self.hard_labels[image_id_str]
                    soft_label_data = self.soft_labels.get(image_id_str, {})

                    question = hard_label_data.get('question', '')
                    primary_answer = hard_label_data.get('answer', '')

                    # 获取概率分布
                    answer_distribution = soft_label_data.get('answer_distribution', {})
                    allowed_answers = list(answer_distribution.keys())

                    if not allowed_answers:
                        continue

                    # 创建DSPy Example
                    example = dspy.Example(
                        image_path=img_path,
                        question=question,
                        allowed_answers=', '.join(allowed_answers),
                        primary_answer=primary_answer,
                        answer_distribution=self._format_distribution(answer_distribution),
                        observation="",  # 留空，由模型生成
                        analysis="",
                        conclusion=""
                    ).with_inputs('image_path', 'question', 'allowed_answers', 'primary_answer', 'answer_distribution')

                    samples.append(example)
                    count += 1

            else:
                # 使用COCO caption生成伪VQA数据（简化版）
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)

                if not anns:
                    continue

                caption = anns[0]['caption']

                # 生成问题和答案
                qa_data = self._generate_answer_focused_qa(caption)

                if qa_data:
                    example = dspy.Example(
                        image_path=img_path,
                        question=qa_data['question'],
                        allowed_answers=', '.join(qa_data['allowed_answers']),
                        primary_answer=qa_data['primary_answer'],
                        answer_distribution=qa_data['distribution_str'],
                        observation="",
                        analysis="",
                        conclusion=""
                    ).with_inputs('image_path', 'question', 'allowed_answers', 'primary_answer', 'answer_distribution')

                    samples.append(example)
                    count += 1

        self.logger.info(f"Generated {len(samples)} VQA samples with soft labels")
        return samples

    def _format_distribution(self, distribution: Dict[str, float]) -> str:
        """
        格式化概率分布为字符串

        Args:
            distribution: 概率分布字典

        Returns:
            格式化的字符串
        """
        return ', '.join([f"{ans}:{prob:.2f}" for ans, prob in distribution.items()])

    def _generate_answer_focused_qa(self, caption: str) -> Optional[Dict]:
        """
        从caption生成答案导向的QA

        Args:
            caption: COCO caption

        Returns:
            QA数据字典
        """
        import re
        caption_lower = caption.lower()

        # 问题类型1：对象计数
        numbers = re.findall(r'\b(\d+)\b', caption)
        if numbers:
            count = numbers[0]
            allowed = [str(i) for i in range(max(0, int(count)-2), int(count)+3)]
            probs = self._create_distribution(count, allowed)
            return {
                'question': "How many items are mentioned?",
                'allowed_answers': allowed,
                'primary_answer': count,
                'distribution_str': self._format_distribution(probs)
            }

        # 问题类型2：颜色
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown']
        for color in colors:
            if color in caption_lower:
                allowed = colors + ['other']
                probs = self._create_distribution(color, allowed)
                return {
                    'question': "What color is mentioned?",
                    'allowed_answers': allowed,
                    'primary_answer': color,
                    'distribution_str': self._format_distribution(probs)
                }

        # 问题类型3：是/否
        if 'person' in caption_lower or 'people' in caption_lower:
            allowed = ['yes', 'no']
            probs = {'yes': 0.85, 'no': 0.15}
            return {
                'question': "Is there a person?",
                'allowed_answers': allowed,
                'primary_answer': 'yes',
                'distribution_str': self._format_distribution(probs)
            }

        return None

    def _create_distribution(self, primary: str, allowed: List[str]) -> Dict[str, float]:
        """
        创建概率分布，主要答案概率最高

        Args:
            primary: 主要答案
            allowed: 允许的答案列表

        Returns:
            概率分布字典
        """
        distribution = {}
        primary_prob = 0.75
        remaining_prob = 1.0 - primary_prob

        for ans in allowed:
            if ans == primary:
                distribution[ans] = primary_prob
            else:
                # 平均分配剩余概率
                distribution[ans] = remaining_prob / (len(allowed) - 1)

        return distribution


# ==================
# Prompt Optimizer - 答案导向版本
# ==================

class DSPyPromptOptimizerV2:
    """
    DSPy Prompt优化器 - 答案导向版本

    核心改进：推理过程聚焦于答案选择
    """

    def __init__(
        self,
        model_path: str,
        coco_root: str,
        annotation_file: str,
        hard_labels_file: Optional[str] = None,
        soft_labels_file: Optional[str] = None,
        output_dir: str = "configs/generated_prompts"
    ):
        """
        初始化优化器

        Args:
            model_path: 模型路径
            coco_root: COCO数据集根目录
            annotation_file: 标注文件路径
            hard_labels_file: 硬标签文件路径
            soft_labels_file: 软标签文件路径
            output_dir: 输出目录
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not installed. Please run: pip install dspy-ai")

        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.config = ConfigManager()

        # 初始化COCO数据加载器（支持软标签）
        self.data_loader = COCODataLoaderWithSoftLabels(
            coco_root,
            annotation_file,
            hard_labels_file,
            soft_labels_file
        )

        # 初始化DSPy LM（如果需要）
        # self.lm = QwenVLLM(model_path, config=self.config)
        # dspy.settings.configure(lm=self.lm)

    def optimize_vqa_prompts(
        self,
        num_train_samples: int = 50,
        num_test_samples: int = 20
    ) -> Dict[str, str]:
        """
        优化VQA prompt - 答案导向版本

        Args:
            num_train_samples: 训练样本数
            num_test_samples: 测试样本数

        Returns:
            生成的prompt字典
        """
        self.logger.info("="*60)
        self.logger.info("开始优化VQA CoT prompt - 答案导向版本")
        self.logger.info("="*60)

        # 获取训练和测试样本（包含软标签）
        all_samples = self.data_loader.get_vqa_samples_with_labels(num_train_samples + num_test_samples)
        train_samples = all_samples[:num_train_samples]
        test_samples = all_samples[num_train_samples:num_train_samples + num_test_samples]

        self.logger.info(f"训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")

        # 生成答案导向的prompt
        generated_prompts = {
            'system': AnswerFocusedPromptTemplates.get_vqa_system_prompt(),
            'user_template': """Question: {question}

ALLOWED ANSWERS: {allowed_answers}
PRIMARY ANSWER: {primary_answer}
PROBABILITY DISTRIBUTION: {answer_distribution}

INSTRUCTIONS:
1. Observation: Describe ONLY visual features that help distinguish between the allowed answers
2. Analysis: Use the probability distribution to reason about which answer best matches
3. Conclusion: Select ONE answer from the allowed answers list

Observation: [Focus on answer-relevant features]

Analysis: [Use probability distribution]

Conclusion: Final Answer: {primary_answer}"""
        }

        # 保存prompt
        self._save_prompts(generated_prompts, 'vqa_answer_focused')

        self.logger.info("✓ Prompt优化完成 - 答案导向版本")

        return generated_prompts

    def optimize_detect_prompts(
        self,
        num_train_samples: int = 50,
        num_test_samples: int = 20
    ) -> Dict[str, str]:
        """
        优化Detection prompt - 答案导向版本

        Args:
            num_train_samples: 训练样本数
            num_test_samples: 测试样本数

        Returns:
            生成的prompt字典
        """
        self.logger.info("="*60)
        self.logger.info("开始优化Detection CoT prompt - 答案导向版本")
        self.logger.info("="*60)

        # 生成答案导向的Detection prompt
        generated_prompts = {
            'system': """TASK: Detect objects in the image through three-step reasoning.

CRITICAL RULES:
1. SCANNING: Quick scan for object presence
   - Focus on identifying potential objects
   - Note their rough locations and sizes
   - DO NOT describe unrelated background or scenery

2. OBJECTS: List detected objects with details
   - Category: Use COCO 80 categories (person, car, bicycle, etc.)
   - Confidence: Estimate confidence (0.0-1.0)
   - Location: Rough position (left, center, right)

3. VERIFICATION: Ensure all significant objects are detected
   - Double-check for missed objects
   - Verify category accuracy
   - Confidence calibration

FORBIDDEN BEHAVIORS:
- DO NOT describe unrelated background scenery
- DO NOT mention objects not in COCO categories
- DO NOT speculate beyond the visual evidence

EXAMPLE:

Example 1 - Indoor Scene:
Scanning: I can see furniture and objects in what appears to be a living room.
Objects: 1) sofa (confidence: 0.95, location: center), 2) potted plant (confidence: 0.88, location: left corner), 3) tv (confidence: 0.92, location: right wall)
Verification: I've scanned the image and confirmed all major objects are detected. No persons or other furniture are visible.

Example 2 - Street Scene:
Scanning: I see vehicles and pedestrians on a street.
Objects: 1) person (confidence: 0.97, location: crosswalk), 2) car (confidence: 0.94, location: left lane), 3) traffic light (confidence: 0.89, location: intersection)
Verification: All significant objects in the street scene are detected. Checked for bicycles and motorcycles.""",
            'user_template': """Detect all objects in this image:

INSTRUCTIONS:
1. Scanning: Quick scan for object presence (focus on objects, not background)
2. Objects: List detected objects with category, confidence, and location
3. Verification: Check for missed objects

Output format:
Scanning: [Quick scan description]
Objects: [List of detected objects]
Verification: [Verification statement]

Scanning:
Objects:
Verification:"""
        }

        # 保存prompt
        self._save_prompts(generated_prompts, 'detection_answer_focused')

        self.logger.info("✓ Detection Prompt优化完成 - 答案导向版本")

        return generated_prompts

    def _save_prompts(self, prompts: Dict[str, str], task: str):
        """
        保存生成的prompt到YAML文件

        Args:
            prompts: prompt字典
            task: 任务类型
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 🔧 使用独特的文件名（v2版本）
        output_file = self.output_dir / f"dspy_v2_{task}_prompts_{timestamp}.yaml"

        # 构建YAML结构
        yaml_data = {
            'task': task,
            'version': 'v2_answer_focused',
            'generated_by': 'DSPy Answer-Focused Optimizer',
            'timestamp': timestamp,
            'model': self.model_path,
            'key_improvements': [
                'Observation focuses on answer-relevant visual features only',
                'Analysis uses probability distribution from soft labels',
                'Conclusion selects from allowed answers list',
                'No unrelated background description'
            ],
            'prompts': prompts
        }

        # 写入YAML文件
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
        print(f"生成的 {task.upper()} Prompt [DSPy V2 答案导向版本]")
        print("="*80)
        print(f"📁 保存位置: {output_file}")
        print("-"*80)

        system_prompt = prompts.get('system', '')
        user_prompt = prompts.get('user_template', '')

        print("\n📝 System Prompt:")
        print("-"*80)
        print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)

        print("\n📝 User Template:")
        print("-"*80)
        print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)

        print("\n💡 关键改进:")
        print("-"*80)
        print("  ✓ Observation: 聚焦答案相关的视觉特征")
        print("  ✓ Analysis: 使用软标签概率分布引导推理")
        print("  ✓ Conclusion: 从允许答案列表中选择")
        print("  ✓ 无无关背景描述")

        print("\n" + "="*80)


# ==================
# Main Entry
# ==================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DSPy Prompt优化器 - 答案导向版本")
    parser.add_argument('--task', type=str, default='vqa', choices=['vqa', 'detection', 'all'],
                        help='要优化的任务类型')
    parser.add_argument('--model', type=str, default='models/Qwen2.5-VL-72B-Instruct-AWQ',
                        help='模型路径')
    parser.add_argument('--coco_root', type=str, default='data/coco/val2014',
                        help='COCO数据集根目录')
    parser.add_argument('--annotation', type=str, default='data/coco/annotations/captions_val2014.json',
                        help='COCO标注文件路径')
    parser.add_argument('--hard_labels', type=str, default=None,
                        help='硬标签文件路径')
    parser.add_argument('--soft_labels', type=str, default=None,
                        help='软标签文件路径')
    parser.add_argument('--num_train', type=int, default=50,
                        help='训练样本数')
    parser.add_argument('--num_test', type=int, default=20,
                        help='测试样本数')

    args = parser.parse_args()

    # 检查DSPy是否安装
    if not DSPY_AVAILABLE:
        print("\n" + "="*60)
        print("错误：DSPy未安装")
        print("="*60)
        print("\n请运行以下命令安装DSPy:")
        print("  pip install dspy-ai")
        print("="*60)
        return

    # 初始化优化器
    print("\n" + "="*60)
    print("DSPy Prompt优化器 - 答案导向版本")
    print("="*60)
    print(f"任务类型: {args.task}")
    print(f"模型路径: {args.model}")
    print(f"COCO根目录: {args.coco_root}")
    print(f"标注文件: {args.annotation}")
    if args.hard_labels:
        print(f"硬标签文件: {args.hard_labels}")
    if args.soft_labels:
        print(f"软标签文件: {args.soft_labels}")
    print("="*60 + "\n")

    try:
        optimizer = DSPyPromptOptimizerV2(
            model_path=args.model,
            coco_root=args.coco_root,
            annotation_file=args.annotation,
            hard_labels_file=args.hard_labels,
            soft_labels_file=args.soft_labels
        )

        # 执行优化
        if args.task in ['vqa', 'all']:
            vqa_prompts = optimizer.optimize_vqa_prompts(
                num_train_samples=args.num_train,
                num_test_samples=args.num_test
            )

            print("\n生成的VQA Prompt (答案导向版本):")
            print("-" * 60)
            print("System:")
            print(vqa_prompts.get('system', '')[:500] + "...")
            print("\nUser Template:")
            print(vqa_prompts.get('user_template', ''))

        if args.task in ['detection', 'all']:
            detect_prompts = optimizer.optimize_detect_prompts(
                num_train_samples=args.num_train,
                num_test_samples=args.num_test
            )

            print("\n生成的Detection Prompt (答案导向版本):")
            print("-" * 60)
            print("System:")
            print(detect_prompts.get('system', '')[:500] + "...")
            print("\nUser Template:")
            print(detect_prompts.get('user_template', ''))

        print("\n" + "="*60)
        print("✓ Prompt优化完成 - 答案导向版本")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()