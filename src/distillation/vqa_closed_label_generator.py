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

🔧 候选池策略：
- closed_choice：动态从问题中提取候选池（如 "A or B?" → ["A", "B"]）
- closed_yesno：固定候选池 ["yes", "no"]（从配置读取）
- closed_enumerate：无固定候选池，使用弱约束prompt（不强制从列表选）
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.answer_normalizer import normalize_answer

# 🔧 新增：导入答案标准化模块（遵守三条红线）
try:
    from .answer_handler import normalize_hard_label, clean_soft_label
    ANSWER_HANDLER_AVAILABLE = True
except ImportError:
    ANSWER_HANDLER_AVAILABLE = False
    normalize_hard_label = None
    clean_soft_label = None


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

        # ───────────────────────────────────────────────────────
        # 🔧 候选池策略（新方案）
        # ───────────────────────────────────────────────────────
        # - closed_choice：动态从问题中提取（无需预定义）
        # - closed_yesno：固定候选池（从配置读取）
        # - closed_enumerate：无固定候选池（弱约束prompt）
        # ───────────────────────────────────────────────────────

        # closed_yesno 的固定候选池（从配置读取）
        self.yes_no_candidates = self.config.get(
            "candidate_pools.yes_no", ["yes", "no"]
        )

        # 🔧 防御性检查：确保yes_no候选池只包含["yes", "no"]
        if self.yes_no_candidates != ["yes", "no"]:
            self.logger.warning(
                f"⚠️ yes_no候选池配置异常: {self.yes_no_candidates}，"
                f"预期应该是 ['yes', 'no']"
            )
            # 强制修正
            self.yes_no_candidates = ["yes", "no"]
            self.logger.info(f"已自动修正为: {self.yes_no_candidates}")

        self.logger.info("✓ VQA闭合问题标签生成器初始化完成")
        self.logger.info(f"  - 软标签温度缩放: T={self.soft_label_temperature}")
        self.logger.info(f"  - CoT生成温度: T={self.cot_temperature}")
        self.logger.info(f"  - closed_yesno候选池: {self.yes_no_candidates}")
        self.logger.info(f"  - closed_enumerate候选池: 无固定候选池（使用弱约束prompt）")

    def generate_labels(
        self,
        image_path: str,
        question: str,
        question_type: str,
        image_id: Optional[str] = None,
        candidate_pool: Optional[List[str]] = None,  # 动态候选答案池（用于 CHOICE 类型）
        ground_truth: Optional[str] = None,  # COCO标注答案作为硬标签
        gt_consistency: Optional[float] = 1.0,  # 🔧 新增：GT置信度
        gt_dist: Optional[Dict[str, float]] = None  # 🔧 新增：GT答案分布
    ) -> Dict[str, Any]:
        """
        生成软硬标签（官方标准流程）

        🔧 新方案：
        - 硬标签：直接使用COCO标注（ground truth）
        - 置信度：使用 gt_consistency（过滤no后最常见答案的频率）
        - 软标签：教师模型推理得到概率分布

        步骤：
        1. 加载候选词列表
        2. 推理并提取候选词logits
        3. 温度缩放 + softmax (软标签)
        4. 硬标签来自COCO标注，置信度使用 gt_consistency

        Args:
            image_path: 图像路径
            question: 问题文本
            question_type: 问题类型（binary/counting/color/location/choice）
            image_id: 图像ID
            candidate_pool: 动态候选答案池（用于 CHOICE 类型，如 ["day", "night"]）
            ground_truth: COCO标注答案（用作硬标签）
            gt_consistency: GT置信度（过滤no后最常见答案的频率，范围0-1）
            gt_dist: GT答案分布（过滤no后的答案频率分布）

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

        # 🔧 新增：转换问题类型为枚举（用于答案标准化）
        from ..classification.question_classifier import QuestionType
        try:
            question_type_enum = QuestionType(question_type)
        except ValueError:
            # 如果转换失败，使用默认类型
            self.logger.warning(f"[Label Gen] 无效的问题类型: {question_type}，使用默认类型")
            question_type_enum = QuestionType.OPEN

        # 🔧 新增：判断是否为强候选池类型
        # closed_choice / closed_yesno → 强候选池（MUST pick from list）
        # closed_enumerate → 弱候选池（MAY consider list）
        major_category = question_type_enum.to_major_category()
        is_strong_pool = major_category in [QuestionType.CLOSED_CHOICE, QuestionType.CLOSED_YESNO]

        self.logger.info(f"[Label Gen] 候选池类型: {'强候选池' if is_strong_pool else '弱候选池'} ({major_category.value})")

        # ───────────────────────────────────────────────────────
        # 步骤1：加载候选词列表（新方案）
        # ───────────────────────────────────────────────────────
        # 🔧 候选池策略：
        # - closed_choice：动态从问题中提取候选池（如 "A or B?" → ["A", "B"]）
        # - closed_yesno：固定候选池 ["yes", "no"]
        # - closed_enumerate：无固定候选池，使用弱约束prompt
        # ───────────────────────────────────────────────────────

        # 🔧 新增：优先使用动态候选答案池（CHOICE 类型）
        if candidate_pool:
            candidate_answers = candidate_pool
            self.logger.info(f"[Label Gen] 使用动态候选答案池: {candidate_answers}")
        else:
            candidate_answers = self._get_candidate_answers(question_type)

        # 🔧 防御性检查：确保candidate_answers不包含问题类型名称
        if candidate_answers:
            invalid_values = ['closed_enumerate', 'open', 'closed_choice', 'closed_yesno']
            invalid_items = [x for x in candidate_answers if x in invalid_values]
            if invalid_items:
                self.logger.warning(
                    f"[Label Gen] ⚠️ candidate_answers包含无效值: {invalid_items}，"
                    f"这可能是bug！原始列表: {candidate_answers}"
                )
                # 移除无效项
                candidate_answers = [x for x in candidate_answers if x not in invalid_values]
                self.logger.warning(f"[Label Gen] 已过滤无效项，剩余: {candidate_answers}")

        # 🔧 关键判断：是否有固定候选池
        if not candidate_answers:
            # ───────────────────────────────────────────────────────
            # closed_enumerate（counting/color/location）
            # ───────────────────────────────────────────────────────
            # 策略：
            # 1. 使用弱约束prompt（不强制从列表选）
            # 2. 生成完整的CoT推理
            # 3. 生成软标签分布（用于知识蒸馏）
            # 4. 在数据清洗阶段做过滤
            # ───────────────────────────────────────────────────────

            self.logger.info(f"[Label Gen] closed_enumerate类型（{question_type}）")
            self.logger.info(f"[Label Gen] 使用弱约束prompt + 生成软标签")

            # 🔧 步骤1：硬标签生成
            # 🔧 新方案：硬标签直接使用COCO标注（ground_truth）
            # 置信度使用 gt_consistency（过滤no后最常见答案的频率）
            if ground_truth:
                # 使用COCO标注作为硬标签
                answer = ground_truth
                confidence = gt_consistency if gt_consistency is not None else 1.0
                self.logger.info(
                    f"[Label Gen] 硬标签来自COCO标注: {answer} "
                    f"(置信度: {confidence:.4f})"
                )

                # 🔧 新增：记录GT分布（用于分析）
                if gt_dist:
                    self.logger.debug(f"[Label Gen] GT分布: {gt_dist}")

                # 仍然需要教师模型推理获取logits（用于软标签）
                result = self.teacher.inference_vqa(
                    image=image_path,
                    question=question,
                    return_logits=True,
                    generate_cot=False,
                    candidate_answers=None,  # 无候选池约束
                    is_strong_pool=False  # 弱约束
                )
            else:
                # 回退：使用教师模型推理生成硬标签（如果没有COCO标注）
                self.logger.warning(f"[Label Gen] 无COCO标注，使用教师模型推理生成硬标签")
                result = self.teacher.inference_vqa(
                    image=image_path,
                    question=question,
                    return_logits=True,
                    generate_cot=False,
                    candidate_answers=None,  # 无候选池约束
                    is_strong_pool=False  # 弱约束
                )

                # 提取硬标签
                answer = result.get('answer', '')
                confidence = result.get('confidence', 0.0)

            self.logger.info(f"[Label Gen] 硬标签: {answer} (置信度: {confidence:.4f})")

            # 🔧 步骤2：软标签生成（使用logits）
            # 对于closed_enumerate，我们从top-k logits中提取概率分布
            logits_data = result.get('logits', {})

            # 提取top-k候选词的logits（用于软标签）
            top_k_logits = self._extract_top_k_logits(logits_data, top_k=50)

            if top_k_logits:
                # 温度缩放 + softmax（生成软标签）
                soft_label = self._compute_soft_label_from_logits(top_k_logits)

                self.logger.info(f"[Label Gen] 软标签分布（top-5）: {dict(list(soft_label['answer_distribution'].items())[:5])}")
            else:
                # 如果无法提取logits，使用硬标签作为软标签
                soft_label = {
                    'answer_distribution': {answer: confidence},
                    'primary_answer': answer,
                    'allowed_answers': [answer]
                }
                self.logger.warning(f"[Label Gen] 无法提取logits，使用硬标签作为软标签")

            # 🔧 返回完整结果（包含软标签）
            return {
                'hard_label': {'answer': answer, 'confidence': confidence},
                'soft_label': soft_label,  # ✅ 包含软标签
                'candidate_pool': None  # 无固定候选池
            }

        self.logger.info(f"[Label Gen] 使用固定候选池: {candidate_answers}")

        # 🔧 步骤2：推理获取logits（用于软标签）
        # 🔧 关键：根据候选池类型选择合适的prompt
        # - 强候选池（closed_choice/closed_yesno）：使用强约束prompt（MUST pick from list）
        # - 弱候选池（closed_enumerate）：使用弱约束prompt（MAY consider list）
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,  # 获取logits
            generate_cot=False,
            candidate_answers=candidate_answers,  # 传入候选词（用于prompt）
            is_strong_pool=is_strong_pool  # 根据候选池类型选择prompt
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
        # 步骤4：硬标签生成
        # 🔧 新方案：硬标签直接使用COCO标注（ground_truth）
        # 置信度使用 gt_consistency（过滤no后最常见答案的频率）
        # ───────────────────────────────────────────────────────
        if ground_truth:
            # 使用COCO标注作为硬标签
            hard_label = {'answer': ground_truth, 'confidence': gt_consistency if gt_consistency is not None else 1.0}
            self.logger.info(
                f"[Label Gen] 硬标签来自COCO标注: {ground_truth} "
                f"(置信度: {hard_label['confidence']:.4f})"
            )

            # 🔧 新增：记录GT分布（用于分析）
            if gt_dist:
                self.logger.debug(f"[Label Gen] GT分布: {gt_dist}")
        else:
            # 回退：从软标签推导硬标签
            self.logger.warning(f"[Label Gen] 无COCO标注，从软标签推导硬标签")
            hard_label = self._derive_hard_label(soft_label, candidate_logits)
            self.logger.info(
                f"[Label Gen] 硬标签: answer={hard_label['answer']}, "
                f"confidence={hard_label['confidence']:.4f}"
            )

        return {
            'hard_label': hard_label,
            'soft_label': soft_label,
            'candidate_pool': candidate_answers  # 🔧 新增：输出候选答案池
        }

    def _get_candidate_answers(self, question_type: str) -> Optional[List[str]]:
        """
        获取候选词列表（新方案）

        🔧 候选池策略：
        - closed_yesno：固定候选池 ["yes", "no"]（从配置读取）
        - closed_choice：动态从问题中提取（在generate_labels中处理）
        - closed_enumerate：无固定候选池，返回None（使用弱约束prompt）

        Args:
            question_type: 问题类型

        Returns:
            候选答案列表，或None（表示无固定候选池）
        """
        # 标准化问题类型
        type_mapping = {
            'yes_no': 'binary',
            'binary': 'binary',
            'counting': 'counting',
            'color': 'color',
            'location': 'location',
            'choice': 'choice'
        }

        normalized_type = type_mapping.get(question_type, question_type)

        # closed_yesno：返回固定候选池
        if normalized_type == 'binary':
            return self.yes_no_candidates

        # closed_choice：动态提取（在generate_labels中处理）
        if normalized_type == 'choice':
            # 返回None，表示需要在generate_labels中动态提取
            return None

        # closed_enumerate：无固定候选池
        # counting/color/location 使用弱约束prompt，不强制从列表选
        if normalized_type in ['counting', 'color', 'location']:
            # 🔧 返回None，表示无固定候选池
            # 模型将使用弱约束prompt（MAY consider list）
            return None

        # 未知类型：返回None
        self.logger.warning(f"[Candidate Pool] 未知问题类型: {question_type}")
        return None

    def _extract_top_k_logits(
        self,
        logits_data: Dict[str, Any],
        top_k: int = 50
    ) -> Dict[str, float]:
        """
        从logits中提取top-k候选词及其概率

        用于closed_enumerate的软标签生成

        Args:
            logits_data: logits数据
            top_k: 提取的候选词数量

        Returns:
            {候选词: logit值}
        """
        candidate_logits = {}

        # 获取原始logits
        raw_logits = logits_data.get('raw_logits')
        top_k_indices = logits_data.get('top_k_indices')
        top_k_values = logits_data.get('top_k_values')

        if raw_logits is not None:
            # 从完整logits中提取
            if raw_logits.dim() == 3:
                raw_logits = raw_logits[0]

            # 取第一个token位置的logits
            first_token_logits = raw_logits[0]

            # 获取top-k
            top_k_values, top_k_indices = torch.topk(first_token_logits, min(top_k, first_token_logits.size(0)))

            for idx, (value, token_id) in enumerate(zip(top_k_values, top_k_indices)):
                token = self.teacher.tokenizer.decode([token_id.item()])
                token = token.strip().lower()

                # 过滤无效token
                if token and len(token) > 0 and not token.startswith('<') and not token.startswith('['):
                    candidate_logits[token] = value.item()

        elif top_k_indices is not None and top_k_values is not None:
            # 从top-k数据中提取
            if top_k_indices.dim() >= 2:
                top_k_indices = top_k_indices[0]
                top_k_values = top_k_values[0]

            for idx, (token_id, logit_value) in enumerate(zip(top_k_indices[:top_k], top_k_values[:top_k])):
                token = self.teacher.tokenizer.decode([token_id.item()])
                token = token.strip().lower()

                if token and len(token) > 0 and not token.startswith('<') and not token.startswith('['):
                    candidate_logits[token] = logit_value.item()

        return candidate_logits

    def _compute_soft_label_from_logits(
        self,
        candidate_logits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        从logits生成软标签分布（温度缩放 + softmax）

        Args:
            candidate_logits: {候选词: logit值}

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

        import torch
        import torch.nn.functional as F

        # 转换为tensor
        candidates = list(candidate_logits.keys())
        logits_list = [candidate_logits[c] for c in candidates]
        logits_tensor = torch.tensor(logits_list, dtype=torch.float32)

        # 温度缩放
        scaled_logits = logits_tensor / self.soft_label_temperature

        # Softmax归一化
        probs = F.softmax(scaled_logits, dim=0)

        # 构建概率分布
        answer_distribution = {}
        for i, candidate in enumerate(candidates):
            prob = probs[i].item()
            if prob > 0.001:  # 过滤极小概率
                answer_distribution[candidate] = prob

        # 找到primary answer
        primary_answer = max(answer_distribution.items(), key=lambda x: x[1])[0]

        # allowed_answers（top-10）
        sorted_candidates = sorted(answer_distribution.items(), key=lambda x: x[1], reverse=True)
        allowed_answers = [c for c, p in sorted_candidates[:10]]

        return {
            'answer_distribution': answer_distribution,
            'primary_answer': primary_answer,
            'allowed_answers': allowed_answers
        }

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