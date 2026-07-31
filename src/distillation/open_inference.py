"""
开放样本推理器（官方标准）
==========================

严格按照官方 batch_infer.py 标准实现开放样本的单阶段推理。

官方标准核心设计：
1. 无闭合候选集约束：不加载 allowed_answers，不限制输出词汇
2. 仅单阶段推理，无阶段 1 logits 提取：不计算、不存储软硬标签
3. Prompt 极简，无答案列表、无概率分布注入
4. 仅要求基于图像完整回答问题，不强制三段式 CoT

使用方式：
    from src.distillation.open_inference import OpenSampleInferencer

    inferencer = OpenSampleInferencer(teacher_model, config)
    result = inferencer.infer(image_path, question)
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class OpenSampleInferencer:
    """
    开放样本推理器（官方标准）

    官方标准：
    - 单阶段推理
    - 无 logits 输出（output_scores=False）
    - 自由文本生成
    - 无候选集约束
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None
    ):
        """
        初始化开放样本推理器

        Args:
            teacher_model: 教师模型实例
            config: 配置管理器
        """
        self.teacher = teacher_model
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # 官方标准：开放样本生成配置
        self.max_new_tokens = self.config.get('teacher.max_new_tokens', 512)  # 允许长文本
        self.temperature = self.config.get('teacher.temperature', 0.1)  # 低温度，稳定输出
        self.do_sample = self.config.get('teacher.do_sample', False)  # 贪婪解码

        # 🔧 新增：从配置文件读取开放 prompt（支持自定义）
        self.open_prompt = self.config.get(
            'prompts.open.vqa',
            """You are a vision assistant, answer the question truthfully based on the image, provide complete natural language explanation, no single-word limited answer.

Question: {question}

Please provide a comprehensive answer based on what you observe in the image."""
        )

        self.logger.info("✓ 开放样本推理器初始化完成（官方标准）")
        self.logger.info(f"  - max_new_tokens: {self.max_new_tokens}")
        self.logger.info(f"  - temperature: {self.temperature}")
        self.logger.info(f"  - output_scores: False（官方标准）")

    def generate_vqa_open(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        开放问答推理（官方标准）

        核心原则：
        - 无候选集 → 不生成 soft/hard 标签
        - 自由长文本回答无需强制结构化推理 → 不生成 CoT
        - 仅输出完整自然语言 answer

        Args:
            image_path: 图像路径
            question: 问题文本
            image_id: 图像ID（可选）

        Returns:
            {
                "answer": "完整自然语言回答（唯一监督文本）",
                "img_path": "图像路径",
                "question": "问题文本",
                "question_type": "open_descriptive",
                "inference_mode": "open"
            }

        Example:
            >>> result = inferencer.generate_vqa_open(
                    "image.jpg",
                    "Why might someone from PETA be upset about this picture?"
                )
            >>> print(result['answer'])
            "PETA is an animal rights organization that opposes animal exploitation
            for entertainment or tourism. The image shows an elephant being ridden
            by tourists. Elephants used for rides often endure harsh training,
            confinement and physical strain, which violates animal welfare standards.
            This exploitative use of elephants would make PETA advocates upset."
        """
        self.logger.debug(f"开放问答推理: {question}")

        # ───────────────────────────────────────────────────────
        # 官方标准：使用极简 Prompt
        # 无答案列表、无概率分布注入、无候选集约束
        # ───────────────────────────────────────────────────────

        # 调用教师模型推理（官方标准参数）
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=False,  # ✅ 官方标准：不输出 logits
            generate_cot=False,   # ✅ 官方标准：不强制三段式 CoT
            custom_prompt=self.open_prompt,  # 使用开放 prompt
            is_open_question=True  # ✅ 关键修复：开放问题返回完整答案
        )

        # ───────────────────────────────────────────────────────
        # 官方标准：仅保留完整自然语言回答
        # ───────────────────────────────────────────────────────

        raw_answer = result.get('answer', '')

        # 🔧 修复：推理阶段不做清洗，直接返回原始答案
        # 清洗逻辑由下游 RewardModelScorer 处理
        # 参考日志：这里不需要进行字符串检测

        output = {
            "answer": raw_answer,  # 原始答案，清洗由下游处理
            "img_path": image_path,
            "question": question,
            "question_type": "open_descriptive",
            "inference_mode": "open"
        }

        if image_id:
            output["image_id"] = image_id

        self.logger.info(f"✓ 开放问答推理完成，回答长度: {len(output['answer'])} 字符")

        return output

    # ✅ 别名：保持向后兼容
    infer = generate_vqa_open

    def batch_infer(
        self,
        samples: list
    ) -> list:
        """
        批量推理开放样本（官方标准）

        官方优化：无需提取 logits，可直接使用 vLLM 连续批处理加速

        Args:
            samples: 样本列表，每个样本包含 {"image_path": str, "question": str}

        Returns:
            结果列表
        """
        results = []

        for sample in samples:
            image_path = sample.get("image_path")
            question = sample.get("question")
            image_id = sample.get("image_id")

            result = self.infer(image_path, question, image_id)
            results.append(result)

        self.logger.info(f"✓ 批量推理完成，处理 {len(results)} 个开放样本")

        return results


# ============================================================
# 官方标准：开放问答输出格式示例
# ============================================================

OUTPUT_EXAMPLE = '''
开放问答输出格式（官方标准）：
{
  "question_type": "open_descriptive",
  "question": "Why might someone from PETA be upset about this picture?",
  "answer": "PETA is an animal rights organization that opposes animal exploitation for entertainment or tourism. The image shows an elephant being ridden by tourists. Elephants used for rides often endure harsh training, confinement and physical strain, which violates animal welfare standards. This exploitative use of elephants would make PETA advocates upset."
}

核心原则：
- 无候选集 → 不生成 soft/hard 标签
- 自由长文本回答无需强制结构化推理 → 不生成 CoT
- 仅输出完整自然语言 answer（唯一监督文本）

对比闭合问答输出格式：
{
  "question_type": "counting",
  "question": "How many people are in the image?",
  "hard_label": {"answer": "two", "confidence": 0.92},
  "soft_label": {
    "answer_distribution": {"two": 0.85, "one": 0.08, "three": 0.05},
    "primary_answer": "two",
    "allowed_answers": ["one", "two", "three"]
  },
  "cot_reasoning": {
    "structured_reasoning": {
      "observation": "Looking at the image...",
      "analysis": "Analyzing the scene...",
      "conclusion": "Therefore, there are two people."
    }
  }
}

关键差异：
闭合问答：hard_label + soft_label + CoT（三重监督）
开放问答：仅 answer（单一监督）
'''


# ============================================================
# 测试代码
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("开放样本推理器测试（官方标准）")
    print("="*70)

    print("\n官方标准核心设计：")
    print("  1. 无闭合候选集约束")
    print("  2. 仅单阶段推理，无 logits 提取")
    print("  3. Prompt 极简，无答案列表")
    print("  4. 仅生成自由文本回答")

    print("\n输出格式示例：")
    print(OUTPUT_EXAMPLE)

    print("\n官方标准参数配置：")
    print("  max_new_tokens: 512  # 允许长文本描述")
    print("  temperature: 0.1     # 低温度，稳定输出")
    print("  output_scores: False # 关键：不输出 logits")
    print("  do_sample: False     # 贪婪解码")

    print("\n" + "="*70)