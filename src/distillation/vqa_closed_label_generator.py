"""
VQA闭合问题标签生成器（官方标准）
================================

按照官方开源流水线标准实现闭合问题的软硬标签生成。

官方流程（严格顺序）：
步骤1：前置分流 + 加载候选词列表
步骤2：阶段1推理 - 提取候选词原始logits（关键裁剪，T=0贪婪解码）
步骤3：温度缩放 + softmax归一化（软标签）
步骤4：由软标签推导硬标签（置信度用 T=1）
步骤5：阶段2推理 - 生成CoT

核心要点：
- 软标签 → 硬标签（不是反过来）
- 只提取候选词logits（裁剪）
- 推理生成：T=0（贪婪解码，确定性）
- 软标签温度缩放：从配置读取（用于知识蒸馏）
- 硬标签置信度：T=1 原始logits直接softmax（反映模型真实置信度）
- CoT生成：从配置读取单独的温度参数
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.answer_normalizer import normalize_answer


class VQAClosedLabelGenerator:
    """
    VQA闭合问题标签生成器（官方标准）

    严格按照官方流程：
    1. 加载候选词列表
    2. 提取候选词logits
    3. 温度缩放 + softmax
    4. 从软标签推导硬标签
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None
    ):
        """
        初始化标签生成器

        Args:
            teacher_model: 教师模型实例
            config: 配置管理器
        """
        self.teacher = teacher_model
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # ───────────────────────────────────────────────────────
        # 温度参数：从配置中读取
        # ───────────────────────────────────────────────────────
        # 软标签温度缩放（用于知识蒸馏，软化分布）
        self.soft_label_temperature = self.config.get(
            "distillation.soft_labels.temperature", 4.0
        )

        # CoT生成温度（用于采样生成思维链）
        self.cot_temperature = self.config.get(
            "distillation.cot.temperature", 0.1
        )

        # 候选词列表（官方标准）
        self.candidate_sets = {
            'binary': ['yes', 'no'],
            'counting': ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'],
            'color': ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'black', 'white', 'brown', 'gray'],
            'location': ['left', 'right', 'top', 'bottom', 'center', 'middle', 'front', 'back']
        }

        self.logger.info("✓ VQA闭合问题标签生成器初始化完成")
        self.logger.info(f"  - 软标签温度缩放: T={self.soft_label_temperature}")
        self.logger.info(f"  - CoT生成温度: T={self.cot_temperature}")
        self.logger.info(f"  - 候选集: binary={len(self.candidate_sets['binary'])}, counting={len(self.candidate_sets['counting'])}, color={len(self.candidate_sets['color'])}")

    def generate_labels(
        self,
        image_path: str,
        question: str,
        question_type: str,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        生成软硬标签（官方标准流程）

        步骤：
        1. 加载候选词列表
        2. 推理并提取候选词logits
        3. 温度缩放 + softmax (软标签)
        4. 从软标签推导硬标签（置信度用 T=1 原始logits计算）

        Args:
            image_path: 图像路径
            question: 问题文本
            question_type: 问题类型（binary/counting/color/location）
            image_id: 图像ID

        Returns:
            {
                'hard_label': {'answer': str, 'confidence': float},
                'soft_label': {
                    'answer_distribution': Dict[str, float],
                    'primary_answer': str,
                    'allowed_answers': List[str]
                }
            }
        """
        self.logger.info(f"[Label Gen] 开始生成标签，问题类型: {question_type}")

        # ───────────────────────────────────────────────────────
        # 步骤1：加载候选词列表（官方标准）
        # ───────────────────────────────────────────────────────
        candidate_answers = self._get_candidate_answers(question_type)

        if not candidate_answers:
            self.logger.warning(f"[Label Gen] 无候选词列表，问题类型: {question_type}")
            return None

        self.logger.info(f"[Label Gen] 候选词列表: {candidate_answers}")

        # ───────────────────────────────────────────────────────
        # 步骤2：阶段1推理 - 提取候选词原始logits（关键裁剪）
        # ───────────────────────────────────────────────────────
        # 推理获取完整logits
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,  # 获取logits
            generate_cot=False,
            candidate_answers=candidate_answers  # 传入候选词（用于prompt）
        )

        # 提取候选词logits
        logits_data = result.get('logits', {})
        candidate_logits = self._extract_candidate_logits(
            logits_data,
            candidate_answers
        )

        if not candidate_logits:
            self.logger.warning("[Label Gen] 无法提取候选词logits")
            return None

        self.logger.info(f"[Label Gen] 提取的候选词logits: {candidate_logits}")

        # ───────────────────────────────────────────────────────
        # 步骤3：温度缩放 + softmax归一化（官方标准，用于软标签）
        # ───────────────────────────────────────────────────────
        soft_label = self._compute_soft_label(candidate_logits, candidate_answers)

        self.logger.info(f"[Label Gen] 软标签分布: {soft_label['answer_distribution']}")

        # ───────────────────────────────────────────────────────
        # 步骤4：由软标签推导硬标签
        # 🔧 关键：置信度使用 T=1 原始logits直接softmax计算
        # ───────────────────────────────────────────────────────
        hard_label = self._derive_hard_label(soft_label, candidate_logits)

        self.logger.info(f"[Label Gen] 硬标签: answer={hard_label['answer']}, confidence={hard_label['confidence']:.4f}")

        return {
            'hard_label': hard_label,
            'soft_label': soft_label,
            'candidate_pool': candidate_answers  # 🔧 新增：输出候选答案池
        }

    def _get_candidate_answers(self, question_type: str) -> List[str]:
        """
        获取候选词列表（官方标准）

        Args:
            question_type: 问题类型

        Returns:
            候选答案列表
        """
        # 标准化问题类型
        type_mapping = {
            'yes_no': 'binary',
            'binary': 'binary',
            'counting': 'counting',
            'color': 'color',
            'location': 'location'
        }

        normalized_type = type_mapping.get(question_type, question_type)

        return self.candidate_sets.get(normalized_type, [])

    def _extract_candidate_logits(
        self,
        logits_data: Dict[str, Any],
        candidate_answers: List[str]
    ) -> Dict[str, float]:
        """
        提取候选词logits（官方核心优化）

        关键裁剪逻辑：
        - 遍历候选词，用tokenizer转为token id
        - 只提取候选id对应的logits值
        - 丢弃词表其余所有token的logits

        ✅ 官方标准：处理原始logits，不是概率

        Args:
            logits_data: 原始logits数据（包含raw_logits）
            candidate_answers: 候选答案列表

        Returns:
            {候选词: logit值}
        """
        candidate_logits = {}

        # ───────────────────────────────────────────────────────
        # ✅ 官方标准：获取原始logits
        # ───────────────────────────────────────────────────────
        raw_logits = logits_data.get('raw_logits')

        if raw_logits is None:
            # 回退：使用top-k logits
            top_k_indices = logits_data.get('top_k_indices')
            top_k_values = logits_data.get('top_k_values')

            if top_k_indices is None or top_k_values is None:
                self.logger.warning("[Logits Extract] logits数据为空")
                return {}

            # 处理维度
            if top_k_indices.dim() >= 2:
                top_k_indices = top_k_indices[0]
                top_k_values = top_k_values[0]

            # 从top-k中提取候选词logits
            for candidate in candidate_answers:
                candidate_lower = candidate.lower()
                token_ids = self.teacher.tokenizer.encode(candidate_lower, add_special_tokens=False)

                if not token_ids:
                    continue

                # 遍历top-k，查找匹配的token
                for idx, token_id in enumerate(top_k_indices):
                    if token_id.item() in token_ids:
                        logit_value = top_k_values[idx].item()
                        candidate_logits[candidate_lower] = logit_value
                        self.logger.debug(f"[Logits Extract] '{candidate_lower}': {logit_value:.4f}")
                        break

            return candidate_logits

        # ───────────────────────────────────────────────────────
        # ✅ 官方标准：从完整logits中提取候选词logits
        # ───────────────────────────────────────────────────────
        # raw_logits形状：[num_tokens, vocab_size] 或 [batch, num_tokens, vocab_size]
        if raw_logits.dim() == 3:
            # [batch, num_tokens, vocab_size] -> 取第一个batch
            raw_logits = raw_logits[0]

        # 取第一个token位置的logits（用于答案预测）
        # 形状：[vocab_size]
        first_token_logits = raw_logits[0]

        # ───────────────────────────────────────────────────────
        # 官方核心：遍历候选词，提取对应的logits
        # ───────────────────────────────────────────────────────
        for candidate in candidate_answers:
            candidate_lower = candidate.lower()

            # 用tokenizer编码候选词
            token_ids = self.teacher.tokenizer.encode(candidate_lower, add_special_tokens=False)

            if not token_ids:
                self.logger.warning(f"[Logits Extract] 无法编码候选词: '{candidate}'")
                continue

            # 取第一个token的logit（对于多token答案，可能需要更复杂的逻辑）
            first_token_id = token_ids[0]

            # 提取logit值
            logit_value = first_token_logits[first_token_id].item()

            candidate_logits[candidate_lower] = logit_value

            self.logger.debug(f"[Logits Extract] '{candidate_lower}': logit={logit_value:.4f}")

        return candidate_logits

    def _compute_soft_label(
        self,
        candidate_logits: Dict[str, float],
        candidate_answers: List[str]
    ) -> Dict[str, Any]:
        """
        温度缩放 + softmax归一化

        公式：p_i = softmax(z_i / T)
        T 从配置读取（distillation.soft_labels.temperature）

        Args:
            candidate_logits: {候选词: logit值}
            candidate_answers: 候选答案列表

        Returns:
            {
                'answer_distribution': {候选词: 概率},
                'primary_answer': str,
                'allowed_answers': List[str]
            }
        """
        if not candidate_logits:
            return {
                'answer_distribution': {},
                'primary_answer': '',
                'allowed_answers': []
            }

        # ───────────────────────────────────────────────────────
        # 温度缩放：logits / T（从配置读取）
        # ───────────────────────────────────────────────────────
        # 转换为tensor
        candidates = list(candidate_logits.keys())
        logits = torch.tensor([candidate_logits[c] for c in candidates])

        # 温度缩放：logits / T
        logits_scaled = logits / self.soft_label_temperature

        # ───────────────────────────────────────────────────────
        # 官方标准：softmax归一化
        # ───────────────────────────────────────────────────────
        probs = F.softmax(logits_scaled, dim=0)

        # 构建概率分布
        answer_distribution = {c: probs[i].item() for i, c in enumerate(candidates)}

        # 确保概率和为1（官方标准）
        total_prob = sum(answer_distribution.values())
        if abs(total_prob - 1.0) > 0.01:
            self.logger.warning(f"[Soft Label] 概率和不等于1: {total_prob:.4f}，重新归一化")
            answer_distribution = {k: v/total_prob for k, v in answer_distribution.items()}

        # ───────────────────────────────────────────────────────
        # 找到概率最大的候选词（primary_answer）
        # ───────────────────────────────────────────────────────
        primary_answer = max(answer_distribution.items(), key=lambda x: x[1])[0]

        return {
            'answer_distribution': answer_distribution,
            'primary_answer': primary_answer,
            'allowed_answers': candidates  # 所有候选词
        }

    def _derive_hard_label(
        self,
        soft_label: Dict[str, Any],
        candidate_logits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        由软标签推导硬标签

        步骤：
        1. 遍历answer_distribution，取概率最大值对应的候选文本
        2. 该候选文本标准化后存入hard_label["answer"]
        3. hard_label["confidence"]使用 T=1 原始logits直接softmax计算

        🔧 关键改进：
        - 软标签使用 T=3.0 温度缩放（用于知识蒸馏）
        - 硬标签置信度使用 T=1 原始logits（反映模型真实置信度）

        Args:
            soft_label: 软标签字典
            candidate_logits: 原始候选词logits

        Returns:
            {'answer': str, 'confidence': float}
        """
        answer_distribution = soft_label['answer_distribution']
        primary_answer = soft_label['primary_answer']

        # ───────────────────────────────────────────────────────
        # 🔧 关键改进：使用 T=1 原始logits计算置信度
        # ───────────────────────────────────────────────────────
        # 构建与answer_distribution相同顺序的logits tensor
        candidates = list(answer_distribution.keys())

        if not candidates or not candidate_logits:
            # 回退：使用软标签概率
            max_prob = answer_distribution.get(primary_answer, 0.0)
            return {
                'answer': primary_answer,
                'confidence': max_prob
            }

        # 提取原始logits（按照candidates的顺序）
        logits_list = []
        for c in candidates:
            logit_val = candidate_logits.get(c, candidate_logits.get(c.lower(), 0.0))
            logits_list.append(logit_val)

        logits_tensor = torch.tensor(logits_list)

        # 🔧 T=1：原始logits直接softmax（不缩放）
        probs_t1 = F.softmax(logits_tensor, dim=0)

        # 找到primary_answer在candidates中的索引
        primary_idx = candidates.index(primary_answer) if primary_answer in candidates else 0

        # 获取T=1置信度
        confidence_t1 = probs_t1[primary_idx].item()

        self.logger.debug(
            f"[Hard Label] T=1 置信度计算: primary={primary_answer}, "
            f"confidence={confidence_t1:.4f} "
            f"(T={self.soft_label_temperature} soft_label prob={answer_distribution.get(primary_answer, 0.0):.4f})"
        )

        return {
            'answer': primary_answer,
            'confidence': confidence_t1
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VQA闭合问题标签生成器测试（官方标准）")
    print("="*70)

    print("\n官方标准流程：")
    print("  步骤1：前置分流 + 加载候选词列表")
    print("  步骤2：阶段1推理 - 提取候选词原始logits（T=0贪婪解码）")
    print("  步骤3：温度缩放 + softmax归一化（软标签，T从配置读取）")
    print("  步骤4：由软标签推导硬标签（置信度用 T=1）")
    print("  步骤5：阶段2推理 - 生成CoT（T从配置读取）")

    print("\n关键参数（从配置读取）：")
    print("  - 推理生成温度: T=0（贪婪解码，确定性）")
    print("  - 软标签温度缩放: T=4（默认，用于知识蒸馏）")
    print("  - 硬标签置信度: T=1（原始logits直接softmax）")
    print("  - CoT生成温度: T=0.1（默认，低温度采样）")
    print("  - 候选集裁剪: 只保留5-20个候选词logits")
    print("  - 顺序: 软标签 → 硬标签（不是反过来）")

    print("\n" + "="*70)