"""
DSPy Prompt Optimizer
=====================

使用DSPy框架自动优化VQA和Detection任务的prompt。

核心功能：
1. 定义DSPy签名（Signature）- 描述输入输出字段
2. 创建DSPy模块（Module）- 封装prompt模板
3. 使用DSPy优化器（Optimizer）- 通过示例学习最优prompt
4. 保存生成的prompt到YAML配置文件

使用方法：
    python scripts/dspy_prompt_optimizer.py --task vqa --num_samples 100
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
# DSPy Signatures
# ==================

class VQACoTSignature(Signature):
    """
    VQA Chain-of-Thought推理签名

    输入：图像路径、问题、允许的答案列表、主要答案
    输出：观察、分析、结论三个步骤的推理过程
    """
    image_path: str = dspy.InputField(desc="Path to the image file")
    question: str = dspy.InputField(desc="Question about the image")
    allowed_answers: str = dspy.InputField(desc="Comma-separated list of allowed answers")
    primary_answer: str = dspy.InputField(desc="The correct answer")

    observation: str = dspy.OutputField(desc="What you see in the image (objects, colors, counts, positions)")
    analysis: str = dspy.OutputField(desc="Logical connection between observations and the question")
    conclusion: str = dspy.OutputField(desc="Clear final answer matching one of the allowed answers")


class DetectionCoTSignature(Signature):
    """
    Detection Chain-of-Thought推理签名

    输入：图像路径
    输出：扫描、对象、验证三个步骤的推理过程
    """
    image_path: str = dspy.InputField(desc="Path to the image file")

    scanning: str = dspy.OutputField(desc="Overall scene description")
    objects: str = dspy.OutputField(desc="List of detected objects with descriptions")
    verification: str = dspy.OutputField(desc="Confirmation of complete detection")


# ==================
# DSPy Modules
# ==================

class VQACoTModule(dspy.Module):
    """
    VQA CoT推理模块

    封装prompt模板和推理逻辑
    """

    def __init__(self):
        super().__init__()
        self.generate_cot = dspy.ChainOfThought(VQACoTSignature)

    def forward(self, image_path: str, question: str, allowed_answers: str, primary_answer: str):
        """执行VQA CoT推理"""
        return self.generate_cot(
            image_path=image_path,
            question=question,
            allowed_answers=allowed_answers,
            primary_answer=primary_answer
        )


class DetectionCoTModule(dspy.Module):
    """
    Detection CoT推理模块
    """

    def __init__(self):
        super().__init__()
        self.generate_cot = dspy.ChainOfThought(DetectionCoTSignature)

    def forward(self, image_path: str):
        """执行Detection CoT推理"""
        return self.generate_cot(image_path=image_path)


# ==================
# Custom LM for Qwen-VL
# ==================

class QwenVLLM(dspy.LM):
    """
    自定义DSPy LM类，适配Qwen-VL模型

    由于Qwen-VL是多模态模型，需要特殊处理图像输入
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[ConfigManager] = None,
        **kwargs
    ):
        """初始化Qwen-VL模型"""
        super().__init__(model=model_path, **kwargs)

        self.config = config or ConfigManager()
        self.logger = get_logger()

        # 加载模型和processor
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载Qwen-VL模型"""
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

        self.logger.info(f"Loading Qwen-VL model from {model_path}")

        # 检查是否使用AWQ量化
        use_awq = self.config.get("model.use_awq", False)

        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        torch_dtype = torch.bfloat16

        if use_awq:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True
            )

        self.logger.info("Qwen-VL model loaded successfully")

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        执行推理

        Args:
            prompt: 输入提示（可能包含图像路径）
            **kwargs: 其他参数

        Returns:
            模型生成的文本
        """
        # 解析prompt，提取图像路径和文本
        image_path, text_prompt = self._parse_prompt(prompt)

        # 准备输入
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt}
                    ]
                }
            ]
        else:
            # 纯文本输入（用于生成prompt模板）
            messages = [
                {"role": "user", "content": text_prompt}
            ]

        # 应用chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 处理输入
        inputs = self.processor(
            text=[text],
            images=[image] if image_path else None,
            padding=True,
            return_tensors="pt"
        )

        # 移动到设备
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_new_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取assistant回复
        if 'assistant' in generated_text:
            response = generated_text.split('assistant')[-1].strip()
        else:
            response = generated_text.strip()

        return response

    def _parse_prompt(self, prompt: str) -> Tuple[Optional[str], str]:
        """
        解析prompt，提取图像路径

        DSPy生成的prompt可能包含特殊格式的图像路径标记

        Args:
            prompt: 输入提示

        Returns:
            (image_path, text_prompt) 元组
        """
        # 尝试匹配常见的图像路径格式
        import re

        # 格式1: [IMAGE: path/to/image.jpg]
        match = re.search(r'\[IMAGE:\s*([^\]]+)\]', prompt)
        if match:
            image_path = match.group(1).strip()
            text_prompt = prompt.replace(match.group(0), '').strip()
            return image_path, text_prompt

        # 格式2: Image: path/to/image.jpg
        match = re.search(r'Image:\s*([^\n]+)', prompt)
        if match:
            image_path = match.group(1).strip()
            text_prompt = prompt.replace(match.group(0), '').strip()
            return image_path, text_prompt

        # 格式3: 从prompt中提取文件路径
        # 匹配常见的图像扩展名
        match = re.search(r'([^\s]+\.(?:jpg|jpeg|png|bmp|gif))', prompt, re.IGNORECASE)
        if match and os.path.exists(match.group(1)):
            image_path = match.group(1)
            text_prompt = prompt.replace(image_path, '').strip()
            return image_path, text_prompt

        # 没有找到图像路径，返回纯文本
        return None, prompt


# ==================
# Data Preparation
# ==================

class COCODataLoader:
    """
    COCO数据加载器，为DSPy准备训练样本
    """

    def __init__(
        self,
        coco_root: str,
        annotation_file: str,
        max_samples: int = 100
    ):
        """
        初始化COCO数据加载器

        Args:
            coco_root: COCO数据集根目录
            annotation_file: 标注文件路径
            max_samples: 最大样本数
        """
        self.coco_root = coco_root
        self.coco = COCO(annotation_file)
        self.max_samples = max_samples
        self.logger = get_logger()

        # 获取所有图像ID
        self.image_ids = list(self.coco.imgs.keys())
        random.shuffle(self.image_ids)

    def get_vqa_samples(self, num_samples: int = 50) -> List[dspy.Example]:
        """
        获取VQA训练样本

        使用COCO captions作为伪VQA数据
        （实际应用中应使用VQA v2.0标注）

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

            # 获取标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            if not anns:
                continue

            # 从caption生成伪VQA问题
            caption = anns[0]['caption']

            # 生成简单问题（示例）
            questions = self._generate_questions_from_caption(caption)

            for question_data in questions:
                question = question_data['question']
                answer = question_data['answer']
                allowed_answers = question_data['allowed_answers']

                # 创建DSPy Example
                example = dspy.Example(
                    image_path=img_path,
                    question=question,
                    allowed_answers=', '.join(allowed_answers),
                    primary_answer=answer,
                    observation="",  # 留空，由模型生成
                    analysis="",
                    conclusion=""
                ).with_inputs('image_path', 'question', 'allowed_answers', 'primary_answer')

                samples.append(example)
                count += 1

                if count >= num_samples:
                    break

        self.logger.info(f"Generated {len(samples)} VQA samples")
        return samples

    def get_detection_samples(self, num_samples: int = 50) -> List[dspy.Example]:
        """
        获取Detection训练样本

        Args:
            num_samples: 样本数量

        Returns:
            DSPy Example列表
        """
        samples = []
        count = 0

        for img_id in self.image_ids[:min(num_samples * 2, len(self.image_ids))]:
            if count >= num_samples:
                break

            # 获取图像信息
            img_info = self.coco.imgs[img_id]
            img_path = os.path.join(self.coco_root, img_info['file_name'])

            if not os.path.exists(img_path):
                continue

            # 获取标注
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            if not anns:
                continue

            # 生成检测结果描述
            objects = []
            for ann in anns[:5]:  # 限制对象数量
                cat_name = self.coco.cats[ann['category_id']]['name']
                objects.append(cat_name)

            objects_desc = ', '.join(objects)

            # 创建DSPy Example
            example = dspy.Example(
                image_path=img_path,
                scanning=f"This image contains: {objects_desc}",
                objects=f"Detected objects: {objects_desc}",
                verification="All objects have been detected"
            ).with_inputs('image_path')

            samples.append(example)
            count += 1

        self.logger.info(f"Generated {len(samples)} Detection samples")
        return samples

    def _generate_questions_from_caption(self, caption: str) -> List[Dict]:
        """
        从caption生成伪VQA问题和答案

        这是一个简化版本，实际应用中应使用真实的VQA标注

        Args:
            caption: COCO caption

        Returns:
            问题答案字典列表
        """
        # 简化实现：生成一些通用问题
        questions = []

        # 问题类型1：计数
        import re
        numbers = re.findall(r'\b(\d+)\b', caption)
        if numbers:
            questions.append({
                'question': f"How many items are mentioned in the description?",
                'answer': numbers[0],
                'allowed_answers': numbers + ['multiple', 'several', 'many']
            })

        # 问题类型2：颜色
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'orange']
        caption_lower = caption.lower()
        for color in colors:
            if color in caption_lower:
                questions.append({
                    'question': f"What color is mentioned in the image?",
                    'answer': color,
                    'allowed_answers': colors + ['other']
                })
                break

        # 问题类型3：是/否问题
        if 'person' in caption_lower or 'people' in caption_lower:
            questions.append({
                'question': "Is there a person in the image?",
                'answer': 'yes',
                'allowed_answers': ['yes', 'no']
            })

        if 'indoor' in caption_lower:
            questions.append({
                'question': "Is this an indoor scene?",
                'answer': 'yes',
                'allowed_answers': ['yes', 'no']
            })

        # 如果没有生成问题，添加一个通用问题
        if not questions:
            questions.append({
                'question': "What is the main subject in this image?",
                'answer': 'object',
                'allowed_answers': ['object', 'person', 'animal', 'scene', 'other']
            })

        return questions


# ==================
# Prompt Optimizer
# ==================

class DSPyPromptOptimizer:
    """
    DSPy Prompt优化器

    使用DSPy框架自动优化VQA和Detection任务的prompt
    """

    def __init__(
        self,
        model_path: str,
        coco_root: str,
        annotation_file: str,
        output_dir: str = "configs/generated_prompts"
    ):
        """
        初始化优化器

        Args:
            model_path: 模型路径
            coco_root: COCO数据集根目录
            annotation_file: 标注文件路径
            output_dir: 输出目录
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is not installed. Please run: pip install dspy-ai")

        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger()
        self.config = ConfigManager()

        # 初始化COCO数据加载器
        self.data_loader = COCODataLoader(coco_root, annotation_file)

        # 初始化DSPy LM
        self.lm = QwenVLLM(model_path, config=self.config)
        dspy.settings.configure(lm=self.lm)

    def optimize_vqa_prompts(
        self,
        num_train_samples: int = 50,
        num_test_samples: int = 20,
        optimization_steps: int = 10
    ) -> Dict[str, str]:
        """
        优化VQA prompt

        Args:
            num_train_samples: 训练样本数
            num_test_samples: 测试样本数
            optimization_steps: 优化步数

        Returns:
            生成的prompt字典
        """
        self.logger.info("="*60)
        self.logger.info("开始优化VQA CoT prompt")
        self.logger.info("="*60)

        # 获取训练和测试样本
        all_samples = self.data_loader.get_vqa_samples(num_train_samples + num_test_samples)
        train_samples = all_samples[:num_train_samples]
        test_samples = all_samples[num_train_samples:num_train_samples + num_test_samples]

        self.logger.info(f"训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")

        # 创建VQA CoT模块
        vqa_module = VQACoTModule()

        # 定义评估函数
        def validate_vqa_cot(example, pred, trace=None):
            """验证VQA CoT质量"""
            score = 0.0

            # 检查是否包含必需的关键词
            if 'observation' in pred.observation.lower() or len(pred.observation) > 20:
                score += 0.3

            if 'analysis' in pred.analysis.lower() or len(pred.analysis) > 20:
                score += 0.3

            if 'conclusion' in pred.conclusion.lower() or len(pred.conclusion) > 20:
                score += 0.3

            # 检查结论中是否包含答案
            if example.primary_answer.lower() in pred.conclusion.lower():
                score += 0.1

            return score >= 0.6

        # 使用BootstrapFewShot优化器
        self.logger.info("开始prompt优化...")

        optimizer = BootstrapFewShot(
            metric=validate_vqa_cot,
            max_bootstrapped_demos=5,
            max_labeled_demos=10
        )

        try:
            optimized_module = optimizer.compile(
                vqa_module,
                trainset=train_samples
            )

            self.logger.info("✓ Prompt优化完成")

            # 提取生成的prompt
            generated_prompts = self._extract_prompts_from_module(optimized_module, 'vqa')

            # 保存prompt
            self._save_prompts(generated_prompts, 'vqa')

            return generated_prompts

        except Exception as e:
            self.logger.error(f"Prompt优化失败: {e}")
            self.logger.info("使用默认prompt模板")

            # 返回默认prompt
            return self._get_default_prompts('vqa')

    def optimize_detection_prompts(
        self,
        num_train_samples: int = 50,
        num_test_samples: int = 20,
        optimization_steps: int = 10
    ) -> Dict[str, str]:
        """
        优化Detection prompt

        Args:
            num_train_samples: 训练样本数
            num_test_samples: 测试样本数
            optimization_steps: 优化步数

        Returns:
            生成的prompt字典
        """
        self.logger.info("="*60)
        self.logger.info("开始优化Detection CoT prompt")
        self.logger.info("="*60)

        # 获取训练和测试样本
        all_samples = self.data_loader.get_detection_samples(num_train_samples + num_test_samples)
        train_samples = all_samples[:num_train_samples]
        test_samples = all_samples[num_train_samples:num_train_samples + num_test_samples]

        self.logger.info(f"训练样本: {len(train_samples)}, 测试样本: {len(test_samples)}")

        # 创建Detection CoT模块
        detection_module = DetectionCoTModule()

        # 定义评估函数
        def validate_detection_cot(example, pred, trace=None):
            """验证Detection CoT质量"""
            score = 0.0

            # 检查是否包含必需的关键词
            if len(pred.scanning) > 20:
                score += 0.3

            if len(pred.objects) > 20:
                score += 0.3

            if len(pred.verification) > 10:
                score += 0.3

            # 检查objects字段是否包含对象列表
            if 'detected' in pred.objects.lower() or 'found' in pred.objects.lower():
                score += 0.1

            return score >= 0.6

        # 使用BootstrapFewShot优化器
        self.logger.info("开始prompt优化...")

        optimizer = BootstrapFewShot(
            metric=validate_detection_cot,
            max_bootstrapped_demos=5,
            max_labeled_demos=10
        )

        try:
            optimized_module = optimizer.compile(
                detection_module,
                trainset=train_samples
            )

            self.logger.info("✓ Prompt优化完成")

            # 提取生成的prompt
            generated_prompts = self._extract_prompts_from_module(optimized_module, 'detection')

            # 保存prompt
            self._save_prompts(generated_prompts, 'detection')

            return generated_prompts

        except Exception as e:
            self.logger.error(f"Prompt优化失败: {e}")
            self.logger.info("使用默认prompt模板")

            # 返回默认prompt
            return self._get_default_prompts('detection')

    def _extract_prompts_from_module(self, module: dspy.Module, task: str) -> Dict[str, str]:
        """
        从优化后的模块中提取prompt

        Args:
            module: DSPy模块
            task: 任务类型

        Returns:
            prompt字典
        """
        prompts = {}

        try:
            # 尝试提取prompt模板
            if hasattr(module, 'generate_cot'):
                # 获取signature的instruction
                if hasattr(module.generate_cot, 'signature'):
                    signature = module.generate_cot.signature
                    prompts['instruction'] = signature.instructions if hasattr(signature, 'instructions') else ""

                # 获取demonstrations
                if hasattr(module.generate_cot, 'demos'):
                    demos = module.generate_cot.demos
                    prompts['demonstrations'] = self._format_demonstrations(demos)

        except Exception as e:
            self.logger.warning(f"提取prompt时出错: {e}")

        # 如果提取失败，使用默认prompt
        if not prompts:
            prompts = self._get_default_prompts(task)

        return prompts

    def _format_demonstrations(self, demos: List) -> str:
        """
        格式化示例

        Args:
            demos: DSPy示例列表

        Returns:
            格式化的字符串
        """
        formatted = []

        for i, demo in enumerate(demos):
            formatted.append(f"\nExample {i+1}:")

            if hasattr(demo, 'question'):
                formatted.append(f"Question: {demo.question}")

            if hasattr(demo, 'observation'):
                formatted.append(f"Observation: {demo.observation}")

            if hasattr(demo, 'analysis'):
                formatted.append(f"Analysis: {demo.analysis}")

            if hasattr(demo, 'conclusion'):
                formatted.append(f"Conclusion: {demo.conclusion}")

        return '\n'.join(formatted)

    def _get_default_prompts(self, task: str) -> Dict[str, str]:
        """
        获取默认prompt（作为备选）

        Args:
            task: 任务类型

        Returns:
            prompt字典
        """
        if task == 'vqa':
            return {
                'system': """TASK: Answer visual questions through three-step reasoning.

OUTPUT FORMAT (exactly 3 paragraphs):
Observation: Describe what you see (objects, colors, counts, positions)
Analysis: Connect observations to the question
Conclusion: State the answer

HARD CONSTRAINTS:
1. ANSWER SOURCE: Final Answer MUST be one word from the allowed answers list.
2. FORBIDDEN WORDS: appear, seem, look like, suggest, possible, probably, might, typical, characteristic, resemble, align with
3. BLOCKED CONTENT: If target is blocked/unclear, write "cannot recognize".
4. NO META-CONTENT: No JSON, no braces, no quotes. Only plain sentences.

QUALITY STANDARD:
Observation: What you actually see (no inference)
Analysis: Logical connection (no speculation)
Conclusion: Clear answer (matches allowed answers)""",

                'user': """Question: {question}
Allowed answers: {allowed_answers}
Required answer: {primary_answer}

Write exactly three paragraphs:

Observation:
Analysis:
Conclusion: Final Answer: {primary_answer}"""
            }

        elif task == 'detection':
            return {
                'system': """TASK: Find objects through three-step process.

OUTPUT FORMAT (exactly 3 paragraphs):
Scanning: What you see overall
Objects: List of detected items
Verification: Confirm detection complete

HARD CONSTRAINTS:
1. Reasoning only - NO JSON output in text
2. No coordinates in reasoning (handled separately)
3. Plain sentences, no meta-formatting

QUALITY STANDARD:
Scanning: Scene overview
Objects: Specific items found
Verification: Check for missed items""",

                'user': """Find objects in this image:

Scanning:
Objects:
Verification:"""
            }

        return {}

    def _save_prompts(self, prompts: Dict[str, str], task: str):
        """
        保存生成的prompt到YAML文件

        Args:
            prompts: prompt字典
            task: 任务类型
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"dspy_{task}_prompts_{timestamp}.yaml"

        # 构建YAML结构
        yaml_data = {
            'task': task,
            'generated_by': 'DSPy',
            'timestamp': timestamp,
            'model': self.model_path,
            'prompts': {
                'system': prompts.get('system', ''),
                'user': prompts.get('user', ''),
                'instruction': prompts.get('instruction', ''),
                'demonstrations': prompts.get('demonstrations', '')
            }
        }

        # 写入YAML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"✓ Prompt已保存到: {output_file}")

        # 同时更新主配置文件
        self._update_main_config(prompts, task)

    def _update_main_config(self, prompts: Dict[str, str], task: str):
        """
        更新主配置文件

        Args:
            prompts: prompt字典
            task: 任务类型
        """
        config_path = Path("configs/prompts_en.yaml")

        if not config_path.exists():
            self.logger.warning(f"配置文件不存在: {config_path}")
            return

        # 读取现有配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # 更新prompt
        if 'prompts' not in config_data:
            config_data['prompts'] = {}
        if 'cot' not in config_data['prompts']:
            config_data['prompts']['cot'] = {}

        if task == 'vqa':
            config_data['prompts']['cot']['vqa_system'] = prompts.get('system', '')
            config_data['prompts']['cot']['vqa_user'] = prompts.get('user', '')
        elif task == 'detection':
            config_data['prompts']['cot']['detection_system'] = prompts.get('system', '')
            config_data['prompts']['cot']['detection_user'] = prompts.get('user', '')

        # 写回配置文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f'.yaml.backup_{timestamp}')

        # 备份原文件
        import shutil
        shutil.copy(config_path, backup_path)
        self.logger.info(f"✓ 原配置已备份到: {backup_path}")

        # 写入新配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"✓ 配置文件已更新: {config_path}")


# ==================
# Main Entry
# ==================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DSPy Prompt优化器")
    parser.add_argument('--task', type=str, default='vqa', choices=['vqa', 'detection', 'all'],
                        help='要优化的任务类型')
    parser.add_argument('--model', type=str, default='models/Qwen2.5-VL-72B-Instruct-AWQ',
                        help='模型路径')
    parser.add_argument('--coco_root', type=str, default='data/coco/val2014',
                        help='COCO数据集根目录')
    parser.add_argument('--annotation', type=str, default='data/coco/annotations/captions_val2014.json',
                        help='COCO标注文件路径')
    parser.add_argument('--num_train', type=int, default=50,
                        help='训练样本数')
    parser.add_argument('--num_test', type=int, default=20,
                        help='测试样本数')
    parser.add_argument('--optimization_steps', type=int, default=10,
                        help='优化步数')

    args = parser.parse_args()

    # 检查DSPy是否安装
    if not DSPY_AVAILABLE:
        print("\n" + "="*60)
        print("错误：DSPy未安装")
        print("="*60)
        print("\n请运行以下命令安装DSPy:")
        print("  pip install dspy-ai")
        print("\n或者:")
        print("  pip install dspy-ai[display]")
        print("="*60)
        return

    # 初始化优化器
    print("\n" + "="*60)
    print("DSPy Prompt优化器")
    print("="*60)
    print(f"任务类型: {args.task}")
    print(f"模型路径: {args.model}")
    print(f"COCO根目录: {args.coco_root}")
    print(f"标注文件: {args.annotation}")
    print(f"训练样本数: {args.num_train}")
    print(f"测试样本数: {args.num_test}")
    print("="*60 + "\n")

    try:
        optimizer = DSPyPromptOptimizer(
            model_path=args.model,
            coco_root=args.coco_root,
            annotation_file=args.annotation
        )

        # 执行优化
        if args.task in ['vqa', 'all']:
            vqa_prompts = optimizer.optimize_vqa_prompts(
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                optimization_steps=args.optimization_steps
            )

            print("\n生成的VQA Prompt:")
            print("-" * 60)
            print("System:")
            print(vqa_prompts.get('system', ''))
            print("\nUser:")
            print(vqa_prompts.get('user', ''))

        if args.task in ['detection', 'all']:
            detection_prompts = optimizer.optimize_detection_prompts(
                num_train_samples=args.num_train,
                num_test_samples=args.num_test,
                optimization_steps=args.optimization_steps
            )

            print("\n生成的Detection Prompt:")
            print("-" * 60)
            print("System:")
            print(detection_prompts.get('system', ''))
            print("\nUser:")
            print(detection_prompts.get('user', ''))

        print("\n" + "="*60)
        print("✓ Prompt优化完成")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()