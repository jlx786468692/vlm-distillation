"""
Simplified Prompt Generator
============================

简化版的prompt生成器，使用模板化和少样本学习生成优化的prompt。

核心思路：
1. 分析COCO数据集中的样本，提取常见模式
2. 使用预定义模板 + 示例生成prompt
3. 通过人工反馈迭代优化

使用方法：
    python scripts/simple_prompt_generator.py --task vqa --num_samples 100 --mode generate
    python scripts/simple_prompt_generator.py --task detection --mode optimize
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
from collections import Counter
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

from src.utils.config import ConfigManager
from src.utils.logger import get_logger


class PatternAnalyzer:
    """
    分析COCO数据模式，提取常见问题类型和答案模式
    """

    def __init__(self, coco_root: str, annotation_file: str):
        """
        初始化模式分析器

        Args:
            coco_root: COCO数据集根目录
            annotation_file: 标注文件路径
        """
        self.coco_root = coco_root
        self.coco = COCO(annotation_file)
        self.logger = get_logger()

    def analyze_vqa_patterns(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        分析VQA模式

        从COCO captions中提取常见模式：
        - 对象类型及频率
        - 颜色分布
        - 场景类型
        - 动作类型

        Args:
            num_samples: 分析样本数

        Returns:
            模式字典
        """
        self.logger.info(f"分析VQA模式，样本数: {num_samples}")

        patterns = {
            'object_types': Counter(),
            'colors': Counter(),
            'scenes': Counter(),
            'actions': Counter(),
            'question_types': Counter()
        }

        # 常见颜色词
        color_words = ['red', 'blue', 'green', 'yellow', 'black', 'white',
                       'brown', 'orange', 'pink', 'purple', 'gray', 'grey']

        # 常见场景词
        scene_words = ['indoor', 'outdoor', 'street', 'room', 'kitchen',
                       'bedroom', 'bathroom', 'park', 'beach', 'office']

        # 常见动作词
        action_words = ['standing', 'sitting', 'walking', 'running', 'eating',
                       'drinking', 'playing', 'working', 'holding', 'wearing']

        # 分析caption
        image_ids = list(self.coco.imgs.keys())[:num_samples]

        for img_id in tqdm(image_ids, desc="分析模式"):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                caption = ann.get('caption', '').lower()

                # 提取对象（从COCO类别）
                for cat_id in self.coco.cats:
                    cat_name = self.coco.cats[cat_id]['name']
                    if cat_name in caption:
                        patterns['object_types'][cat_name] += 1

                # 提取颜色
                for color in color_words:
                    if color in caption:
                        patterns['colors'][color] += 1

                # 提取场景
                for scene in scene_words:
                    if scene in caption:
                        patterns['scenes'][scene] += 1

                # 提取动作
                for action in action_words:
                    if action in caption:
                        patterns['actions'][action] += 1

        # 生成问题类型统计
        patterns['question_types'] = Counter({
            'what': sum(1 for img_id in image_ids for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)) if 'what' in ann.get('caption', '').lower()),
            'how many': patterns['object_types']['person'] + patterns['object_types'].get('people', 0),
            'color': sum(patterns['colors'].values()),
            'location': sum(patterns['scenes'].values()),
        })

        self.logger.info("✓ 模式分析完成")
        return patterns

    def generate_question_templates(self, patterns: Dict) -> List[Dict]:
        """
        基于分析结果生成问题模板

        Args:
            patterns: 模式字典

        Returns:
            问题模板列表
        """
        templates = []

        # 问题类型1：对象识别
        top_objects = [obj for obj, count in patterns['object_types'].most_common(10)]
        for obj in top_objects:
            templates.append({
                'question': f"Is there a {obj} in the image?",
                'type': 'yes_no',
                'expected_answer': f"yes/no",
                'reasoning_template': {
                    'observation': f"I can see the image clearly. There is/there isn't a {obj} visible.",
                    'analysis': f"The question asks about the presence of {obj}. Based on the visual evidence, I need to confirm if {obj} is present.",
                    'conclusion': f"Final Answer: yes/no"
                }
            })

        # 问题类型2：颜色
        top_colors = [color for color, count in patterns['colors'].most_common(5)]
        for color in top_colors:
            templates.append({
                'question': "What color is the main object?",
                'type': 'color',
                'expected_answer': color,
                'reasoning_template': {
                    'observation': f"I can see objects in the image. The main object appears to be {color} in color.",
                    'analysis': f"The question asks about the color of the main object. Looking at the visual features, the dominant color is {color}.",
                    'conclusion': f"Final Answer: {color}"
                }
            })

        # 问题类型3：计数
        templates.append({
            'question': "How many people are in the image?",
            'type': 'count',
            'expected_answer': "number",
            'reasoning_template': {
                'observation': "I can see people in the image. Let me count them: 1, 2, 3...",
                'analysis': "The question asks for the number of people. I need to count all visible persons.",
                'conclusion': "Final Answer: X"
            }
        })

        # 问题类型4：场景
        top_scenes = [scene for scene, count in patterns['scenes'].most_common(5)]
        for scene in top_scenes:
            templates.append({
                'question': "What kind of scene is this?",
                'type': 'scene',
                'expected_answer': scene,
                'reasoning_template': {
                    'observation': f"The image shows an {scene} environment with typical features.",
                    'analysis': f"Based on the visual elements (furniture, lighting, objects), this appears to be an {scene} scene.",
                    'conclusion': f"Final Answer: {scene}"
                }
            })

        return templates


class PromptTemplateGenerator:
    """
    基于模式分析生成优化的prompt模板
    """

    def __init__(self, output_dir: str = "configs/generated_prompts"):
        """
        初始化模板生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def generate_vqa_system_prompt(self, patterns: Dict) -> str:
        """
        生成VQA系统prompt

        基于分析结果，生成包含：
        - 明确的任务定义
        - 结构化的输出格式
        - 基于真实数据的示例
        - 硬性约束

        Args:
            patterns: 模式字典

        Returns:
            系统prompt字符串
        """
        # 提取常见对象和场景
        top_objects = [obj for obj, count in patterns['object_types'].most_common(10)]
        top_scenes = [scene for scene, count in patterns['scenes'].most_common(5)]
        top_colors = [color for color, count in patterns['colors'].most_common(5)]

        system_prompt = f"""TASK: Answer visual questions through structured three-step reasoning.

OUTPUT FORMAT (exactly 3 paragraphs):
Observation: Describe what you see (objects, colors, counts, positions)
Analysis: Connect observations to the question
Conclusion: State the answer clearly

HARD CONSTRAINTS:
1. ANSWER FORMAT: Final answer MUST be ONE word or number from allowed answers
2. NO SPECULATION: Use only what you actually see in the image
3. FORBIDDEN WORDS: appear, seem, look like, suggest, possible, probably, might
4. NO META-CONTENT: No JSON, no braces, no quotes, no markdown
5. BE SPECIFIC: Use concrete descriptions, not vague phrases

COMMON OBJECTS (for reference):
{', '.join(top_objects[:10])}

COMMON SCENES:
{', '.join(top_scenes[:5])}

COMMON COLORS:
{', '.join(top_colors[:5])}

QUALITY STANDARD:
Observation: Pure visual facts (what, where, how many)
Analysis: Logical reasoning connecting observation to question
Conclusion: Clear, concise answer matching allowed answers

EXAMPLES:

Example 1 - Object Detection:
Question: Is there a dog in the image?
Allowed answers: yes, no
Observation: I see a living room with a couch, a coffee table, and an animal sitting on the floor.
Analysis: The animal has four legs, fur, and distinctive dog features like floppy ears and a wagging tail. This is clearly a dog.
Conclusion: Final Answer: yes

Example 2 - Color Recognition:
Question: What color is the car?
Allowed answers: red, blue, green, black, white, other
Observation: There is a vehicle parked on the street. The vehicle has a glossy finish and reflects light.
Analysis: Looking at the car's paint, it has a bright, warm tone. The color is clearly red, not orange or any other color.
Conclusion: Final Answer: red

Example 3 - Counting:
Question: How many people are in the image?
Allowed answers: 0, 1, 2, 3, 4, 5, more
Observation: I can see multiple people in this scene. Let me count: one person on the left, two people in the center, and one person on the right.
Analysis: Adding them up: 1 + 2 + 1 = 4 people total. Each person is clearly visible and distinguishable.
Conclusion: Final Answer: 4
"""

        return system_prompt

    def generate_vqa_user_prompt(self) -> str:
        """
        生成VQA用户prompt模板

        Returns:
            用户prompt字符串
        """
        user_prompt = """Question: {question}
Allowed answers: {allowed_answers}
Required answer: {primary_answer}

Write exactly three paragraphs:

Observation:
[Describe what you see in the image]

Analysis:
[Connect your observations to the question]

Conclusion:
Final Answer: {primary_answer}"""

        return user_prompt

    def generate_detection_system_prompt(self, patterns: Dict) -> str:
        """
        生成Detection系统prompt

        Args:
            patterns: 模式字典

        Returns:
            系统prompt字符串
        """
        top_objects = [obj for obj, count in patterns['object_types'].most_common(15)]

        system_prompt = f"""TASK: Detect and describe objects through three-step reasoning.

OUTPUT FORMAT (exactly 3 paragraphs):
Scanning: Overall scene description
Objects: List of detected objects with descriptions
Verification: Confirmation of detection completeness

HARD CONSTRAINTS:
1. REASONING ONLY: NO JSON, NO coordinates in text output
2. BE THOROUGH: Scan systematically from left to right, top to bottom
3. BE SPECIFIC: Describe each object's location and characteristics
4. NO META-CONTENT: Plain text only, no formatting symbols

COMMON OBJECTS TO DETECT:
{', '.join(top_objects)}

QUALITY STANDARD:
Scanning: Scene overview (indoor/outdoor, main focus)
Objects: Specific items with locations (e.g., "a red car on the left")
Verification: Check for missed objects

EXAMPLES:

Example 1 - Indoor Scene:
Scanning: This is an indoor living room scene. The image shows furniture and household items arranged in a typical living space.
Objects: I detect: (1) a brown leather couch in the center, (2) a wooden coffee table in front of the couch, (3) a television mounted on the wall, (4) a potted plant in the corner, (5) a lamp on the side table.
Verification: I have scanned the entire image systematically. All major objects have been identified and described.

Example 2 - Outdoor Street:
Scanning: This is an outdoor street scene with vehicles and pedestrians.
Objects: I detect: (1) a red car parked on the left side, (2) a person walking on the sidewalk, (3) a street lamp, (4) a bicycle leaning against a post, (5) a trash can on the corner.
Verification: The scene has been fully scanned. No additional significant objects are visible.
"""

        return system_prompt

    def generate_detection_user_prompt(self) -> str:
        """
        生成Detection用户prompt模板

        Returns:
            用户prompt字符串
        """
        user_prompt = """Detect all objects in this image:

Scanning:
[Describe the overall scene]

Objects:
[List detected objects with locations]

Verification:
[Confirm detection completeness]"""

        return user_prompt

    def save_prompts(self, prompts: Dict[str, str], task: str) -> str:
        """
        保存生成的prompt到YAML文件

        Args:
            prompts: prompt字典
            task: 任务类型

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"optimized_{task}_prompts_{timestamp}.yaml"

        # 构建YAML结构
        yaml_data = {
            'task': task,
            'generated_by': 'Pattern-Based Generator',
            'timestamp': timestamp,
            'prompts': prompts
        }

        # 写入YAML文件
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"✓ Prompt已保存到: {output_file}")
        return str(output_file)

    def update_main_config(self, prompts: Dict[str, str], task: str):
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

        # 备份原文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_path.with_suffix(f'.yaml.backup_{timestamp}')

        import shutil
        shutil.copy(config_path, backup_path)
        self.logger.info(f"✓ 原配置已备份到: {backup_path}")

        # 写入新配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"✓ 配置文件已更新: {config_path}")


class PromptTester:
    """
    测试生成的prompt效果
    """

    def __init__(self, model_path: str):
        """
        初始化测试器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.logger = get_logger()

        # 加载teacher model
        try:
            from src.models.teacher_model import TeacherModel
            self.teacher = TeacherModel()
            self.logger.info("✓ Teacher模型加载成功")
        except Exception as e:
            self.logger.error(f"Teacher模型加载失败: {e}")
            self.teacher = None

    def test_vqa_prompt(
        self,
        image_path: str,
        question: str,
        system_prompt: str,
        user_prompt: str,
        allowed_answers: List[str],
        primary_answer: str
    ) -> Dict[str, Any]:
        """
        测试VQA prompt

        Args:
            image_path: 图像路径
            question: 问题
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            allowed_answers: 允许的答案
            primary_answer: 主要答案

        Returns:
            测试结果
        """
        if not self.teacher:
            return {'error': 'Teacher model not loaded'}

        # 构造prompt
        formatted_user_prompt = user_prompt.format(
            question=question,
            allowed_answers=', '.join(allowed_answers),
            primary_answer=primary_answer
        )

        # 调用模型
        result = self.teacher.inference_vqa(
            image=image_path,
            question=formatted_user_prompt,
            generate_cot=True,
            return_logits=False
        )

        # 解析结果
        full_response = result.get('full_response', '')

        # 提取三段内容
        observation, analysis, conclusion = self._parse_cot_response(full_response)

        # 验证质量
        quality_score = self._evaluate_cot_quality(observation, analysis, conclusion)

        return {
            'full_response': full_response,
            'observation': observation,
            'analysis': analysis,
            'conclusion': conclusion,
            'quality_score': quality_score,
            'expected_answer': primary_answer,
            'matched': primary_answer.lower() in conclusion.lower()
        }

    def _parse_cot_response(self, response: str) -> Tuple[str, str, str]:
        """
        解析CoT响应，提取三段内容

        Args:
            response: 模型响应

        Returns:
            (observation, analysis, conclusion) 元组
        """
        observation = ""
        analysis = ""
        conclusion = ""

        # 查找关键词
        if 'observation' in response.lower():
            match = re.search(r'Observation[:\s]*(.*?)(?=Analysis|$)', response, re.IGNORECASE | re.DOTALL)
            if match:
                observation = match.group(1).strip()

        if 'analysis' in response.lower():
            match = re.search(r'Analysis[:\s]*(.*?)(?=Conclusion|$)', response, re.IGNORECASE | re.DOTALL)
            if match:
                analysis = match.group(1).strip()

        if 'conclusion' in response.lower():
            match = re.search(r'Conclusion[:\s]*(.*?)(?=$)', response, re.IGNORECASE | re.DOTALL)
            if match:
                conclusion = match.group(1).strip()

        return observation, analysis, conclusion

    def _evaluate_cot_quality(self, observation: str, analysis: str, conclusion: str) -> float:
        """
        评估CoT质量

        Args:
            observation: 观察段落
            analysis: 分析段落
            conclusion: 结论段落

        Returns:
            质量分数 (0-1)
        """
        score = 0.0

        # 检查长度
        if len(observation) >= 20:
            score += 0.25
        if len(analysis) >= 20:
            score += 0.25
        if len(conclusion) >= 10:
            score += 0.25

        # 检查内容质量
        forbidden_words = ['seem', 'appear', 'might', 'probably', 'maybe']
        all_text = (observation + analysis + conclusion).lower()

        if not any(word in all_text for word in forbidden_words):
            score += 0.1

        if 'final answer' in conclusion.lower():
            score += 0.15

        return min(score, 1.0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版Prompt生成器")
    parser.add_argument('--task', type=str, default='vqa', choices=['vqa', 'detection', 'all'],
                        help='要生成prompt的任务类型')
    parser.add_argument('--coco_root', type=str, default='data/coco/val2014',
                        help='COCO数据集根目录')
    parser.add_argument('--annotation', type=str, default='data/coco/annotations/captions_val2014.json',
                        help='COCO标注文件路径')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='分析样本数（用于模式提取）')
    parser.add_argument('--mode', type=str, default='generate', choices=['analyze', 'generate', 'test'],
                        help='运行模式: analyze(分析模式), generate(生成prompt), test(测试prompt)')
    parser.add_argument('--test_image', type=str, default=None,
                        help='测试图像路径（用于test模式）')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("简化版Prompt生成器")
    print("="*60)
    print(f"任务类型: {args.task}")
    print(f"COCO根目录: {args.coco_root}")
    print(f"标注文件: {args.annotation}")
    print(f"运行模式: {args.mode}")
    print("="*60 + "\n")

    # 检查文件是否存在
    if not os.path.exists(args.annotation):
        print(f"❌ 标注文件不存在: {args.annotation}")
        print("\n请先下载COCO数据集:")
        print("  1. 下载 val2014 图像: http://images.cocodataset.org/zips/val2014.zip")
        print("  2. 下载标注文件: http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
        print("  3. 解压到 data/coco/ 目录")
        return

    # 模式1：分析数据模式
    if args.mode == 'analyze':
        analyzer = PatternAnalyzer(args.coco_root, args.annotation)
        patterns = analyzer.analyze_vqa_patterns(args.num_samples)

        print("\n分析结果:")
        print("-" * 60)
        print("常见对象:", [obj for obj, count in patterns['object_types'].most_common(10)])
        print("常见颜色:", [color for color, count in patterns['colors'].most_common(5)])
        print("常见场景:", [scene for scene, count in patterns['scenes'].most_common(5)])
        print("常见动作:", [action for action, count in patterns['actions'].most_common(5)])

        # 保存分析结果
        output_file = Path("configs/generated_prompts/pattern_analysis.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'object_types': dict(patterns['object_types'].most_common(50)),
                'colors': dict(patterns['colors'].most_common(20)),
                'scenes': dict(patterns['scenes'].most_common(20)),
                'actions': dict(patterns['actions'].most_common(20))
            }, f, indent=2)

        print(f"\n✓ 分析结果已保存到: {output_file}")

    # 模式2：生成prompt
    elif args.mode == 'generate':
        # 先分析模式
        analyzer = PatternAnalyzer(args.coco_root, args.annotation)
        patterns = analyzer.analyze_vqa_patterns(args.num_samples)

        # 生成prompt
        generator = PromptTemplateGenerator()

        if args.task in ['vqa', 'all']:
            print("\n" + "="*60)
            print("生成VQA Prompt")
            print("="*60)

            vqa_prompts = {
                'system': generator.generate_vqa_system_prompt(patterns),
                'user': generator.generate_vqa_user_prompt()
            }

            print("\n生成的System Prompt:")
            print("-" * 60)
            print(vqa_prompts['system'])

            print("\n生成的User Prompt:")
            print("-" * 60)
            print(vqa_prompts['user'])

            # 保存
            output_path = generator.save_prompts(vqa_prompts, 'vqa')
            print(f"\n✓ VQA Prompt已保存到: {output_path}")

            # 更新主配置
            generator.update_main_config(vqa_prompts, 'vqa')

        if args.task in ['detection', 'all']:
            print("\n" + "="*60)
            print("生成Detection Prompt")
            print("="*60)

            detection_prompts = {
                'system': generator.generate_detection_system_prompt(patterns),
                'user': generator.generate_detection_user_prompt()
            }

            print("\n生成的System Prompt:")
            print("-" * 60)
            print(detection_prompts['system'])

            print("\n生成的User Prompt:")
            print("-" * 60)
            print(detection_prompts['user'])

            # 保存
            output_path = generator.save_prompts(detection_prompts, 'detection')
            print(f"\n✓ Detection Prompt已保存到: {output_path}")

            # 更新主配置
            generator.update_main_config(detection_prompts, 'detection')

        print("\n" + "="*60)
        print("✓ Prompt生成完成")
        print("="*60)

    # 模式3：测试prompt
    elif args.mode == 'test':
        if not args.test_image:
            print("❌ 测试模式需要指定 --test_image 参数")
            return

        if not os.path.exists(args.test_image):
            print(f"❌ 测试图像不存在: {args.test_image}")
            return

        # 加载生成的prompt
        config_path = Path("configs/prompts_en.yaml")
        if not config_path.exists():
            print(f"❌ 配置文件不存在: {config_path}")
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        tester = PromptTester("models/Qwen2.5-VL-72B-Instruct-AWQ")

        if args.task in ['vqa', 'all']:
            system_prompt = config_data['prompts']['cot']['vqa_system']
            user_prompt = config_data['prompts']['cot']['vqa_user']

            # 测试用例
            test_cases = [
                {
                    'question': 'Is there a person in the image?',
                    'allowed_answers': ['yes', 'no'],
                    'primary_answer': 'yes'
                },
                {
                    'question': 'What color is the main object?',
                    'allowed_answers': ['red', 'blue', 'green', 'black', 'white', 'other'],
                    'primary_answer': 'other'
                }
            ]

            print("\n测试VQA Prompt:")
            print("-" * 60)

            for i, test_case in enumerate(test_cases, 1):
                print(f"\n测试用例 {i}:")
                print(f"问题: {test_case['question']}")

                result = tester.test_vqa_prompt(
                    image_path=args.test_image,
                    question=test_case['question'],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    allowed_answers=test_case['allowed_answers'],
                    primary_answer=test_case['primary_answer']
                )

                print(f"\n观察: {result.get('observation', 'N/A')[:100]}...")
                print(f"分析: {result.get('analysis', 'N/A')[:100]}...")
                print(f"结论: {result.get('conclusion', 'N/A')}")
                print(f"质量分数: {result.get('quality_score', 0):.2f}")
                print(f"答案匹配: {'✓' if result.get('matched') else '✗'}")


if __name__ == "__main__":
    main()