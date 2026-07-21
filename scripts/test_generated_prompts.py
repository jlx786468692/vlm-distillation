"""
Prompt测试示例
==============

使用生成的prompt进行简单的CoT生成测试。

使用方法：
    python scripts/test_generated_prompts.py --image <图像路径> --question "What color is the car?"
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
import torch

from src.models.teacher_model import TeacherModel
from src.utils.config import ConfigManager
from src.utils.logger import getLogger


class PromptTester:
    """测试生成的prompt效果"""

    def __init__(self, config_path: str = "configs/prompts_en.yaml"):
        """
        初始化测试器

        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigManager()
        self.logger = getLogger()

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.prompts_config = yaml.safe_load(f)

        # 加载teacher模型
        self.logger.info("加载Teacher模型...")
        self.teacher = TeacherModel()
        self.logger.info("✓ Teacher模型加载成功")

    def test_vqa_prompt(
        self,
        image_path: str,
        question: str,
        allowed_answers: List[str] = None,
        primary_answer: str = None
    ):
        """
        测试VQA prompt

        Args:
            image_path: 图像路径
            question: 问题
            allowed_answers: 允许的答案列表
            primary_answer: 主要答案
        """
        print("\n" + "="*60)
        print("测试VQA Prompt")
        print("="*60)
        print(f"图像: {image_path}")
        print(f"问题: {question}")
        if allowed_answers:
            print(f"允许答案: {', '.join(allowed_answers)}")
        if primary_answer:
            print(f"主要答案: {primary_answer}")
        print("="*60)

        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            return

        # 获取CoT推理
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            generate_cot=True,
            return_logits=False,
            primary_answer=primary_answer,
            allowed_answers=allowed_answers
        )

        # 解析结果
        full_response = result.get('full_response', '')

        print("\n【完整响应】")
        print("-"*60)
        print(full_response)
        print("-"*60)

        # 解析三段内容
        observation, analysis, conclusion = self._parse_cot_response(full_response)

        print("\n【结构化解析】")
        print("-"*60)
        print(f"Observation:\n{observation}\n")
        print(f"Analysis:\n{analysis}\n")
        print(f"Conclusion:\n{conclusion}\n")
        print("-"*60)

        # 质量评估
        quality_score = self._evaluate_quality(observation, analysis, conclusion)

        print("\n【质量评估】")
        print("-"*60)
        print(f"观察长度: {len(observation)} 字符")
        print(f"分析长度: {len(analysis)} 字符")
        print(f"结论长度: {len(conclusion)} 字符")
        print(f"质量分数: {quality_score:.2f} / 1.00")

        # 检查是否包含禁止词汇
        forbidden_words = ['seem', 'appear', 'might', 'probably', 'maybe', 'suggest']
        all_text = (observation + analysis + conclusion).lower()
        found_forbidden = [word for word in forbidden_words if word in all_text]

        if found_forbidden:
            print(f"⚠ 发现禁止词汇: {', '.join(found_forbidden)}")
        else:
            print("✓ 未发现禁止词汇")

        # 检查答案匹配
        if primary_answer:
            matched = primary_answer.lower() in conclusion.lower()
            print(f"{'✓' if matched else '✗'} 答案匹配: {primary_answer}")

        print("="*60)

        return {
            'full_response': full_response,
            'observation': observation,
            'analysis': analysis,
            'conclusion': conclusion,
            'quality_score': quality_score
        }

    def test_detection_prompt(self, image_path: str):
        """
        测试Detection prompt

        Args:
            image_path: 图像路径
        """
        print("\n" + "="*60)
        print("测试Detection Prompt")
        print("="*60)
        print(f"图像: {image_path}")
        print("="*60)

        # 检查图像是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图像不存在: {image_path}")
            return

        # 获取CoT推理
        result = self.teacher.inference_detection(
            image=image_path,
            generate_cot=True,
            return_logits=False
        )

        # 解析结果
        full_response = result.get('full_response', '')

        print("\n【完整响应】")
        print("-"*60)
        print(full_response)
        print("-"*60)

        # 解析三段内容
        scanning, objects, verification = self._parse_detection_response(full_response)

        print("\n【结构化解析】")
        print("-"*60)
        print(f"Scanning:\n{scanning}\n")
        print(f"Objects:\n{objects}\n")
        print(f"Verification:\n{verification}\n")
        print("-"*60)

        print("="*60)

        return {
            'full_response': full_response,
            'scanning': scanning,
            'objects': objects,
            'verification': verification
        }

    def _parse_cot_response(self, response: str):
        """解析VQA CoT响应"""
        import re

        observation = ""
        analysis = ""
        conclusion = ""

        # 提取assistant回复
        if 'assistant' in response:
            response = response.split('assistant')[-1]

        # 查找关键词
        patterns = {
            'observation': [
                r'Observation\s*:\s*(.*?)(?=Analysis|$)',
                r'Step\s*1\s*:\s*(.*?)(?=Step\s*2|$)'
            ],
            'analysis': [
                r'Analysis\s*:\s*(.*?)(?=Conclusion|$)',
                r'Step\s*2\s*:\s*(.*?)(?=Step\s*3|$)'
            ],
            'conclusion': [
                r'Conclusion\s*:\s*(.*?)(?=$)',
                r'Step\s*3\s*:\s*(.*?)(?=$)'
            ]
        }

        for section, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if content:
                        if section == 'observation':
                            observation = content
                        elif section == 'analysis':
                            analysis = content
                        elif section == 'conclusion':
                            conclusion = content
                        break

        return observation, analysis, conclusion

    def _parse_detection_response(self, response: str):
        """解析Detection CoT响应"""
        import re

        scanning = ""
        objects = ""
        verification = ""

        # 提取assistant回复
        if 'assistant' in response:
            response = response.split('assistant')[-1]

        # 查找关键词
        patterns = {
            'scanning': [
                r'Scanning\s*:\s*(.*?)(?=Objects|$)',
                r'Step\s*1\s*:\s*(.*?)(?=Step\s*2|$)'
            ],
            'objects': [
                r'Objects\s*:\s*(.*?)(?=Verification|$)',
                r'Step\s*2\s*:\s*(.*?)(?=Step\s*3|$)'
            ],
            'verification': [
                r'Verification\s*:\s*(.*?)(?=$)',
                r'Step\s*3\s*:\s*(.*?)(?=$)'
            ]
        }

        for section, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if content:
                        if section == 'scanning':
                            scanning = content
                        elif section == 'objects':
                            objects = content
                        elif section == 'verification':
                            verification = content
                        break

        return scanning, objects, verification

    def _evaluate_quality(self, observation: str, analysis: str, conclusion: str) -> float:
        """评估CoT质量"""
        score = 0.0

        # 长度评分
        if len(observation) >= 20:
            score += 0.25
        if len(analysis) >= 20:
            score += 0.25
        if len(conclusion) >= 10:
            score += 0.25

        # 格式评分
        if 'final answer' in conclusion.lower():
            score += 0.15

        # 禁止词汇检查
        forbidden_words = ['seem', 'appear', 'might', 'probably', 'maybe', 'suggest']
        all_text = (observation + analysis + conclusion).lower()

        if not any(word in all_text for word in forbidden_words):
            score += 0.10

        return min(score, 1.0)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试生成的Prompt")
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--task', type=str, default='vqa', choices=['vqa', 'detection'], help='任务类型')
    parser.add_argument('--question', type=str, default='What is in the image?', help='VQA问题')
    parser.add_argument('--answers', type=str, default='', help='允许的答案（逗号分隔）')
    parser.add_argument('--primary', type=str, default='', help='主要答案')

    args = parser.parse_args()

    # 创建测试器
    tester = PromptTester()

    # 执行测试
    if args.task == 'vqa':
        allowed_answers = args.answers.split(',') if args.answers else None
        primary_answer = args.primary if args.primary else None

        tester.test_vqa_prompt(
            image_path=args.image,
            question=args.question,
            allowed_answers=allowed_answers,
            primary_answer=primary_answer
        )

    elif args.task == 'detection':
        tester.test_detection_prompt(image_path=args.image)


if __name__ == "__main__":
    main()