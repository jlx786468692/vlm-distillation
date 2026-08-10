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
from ..utils.vqa_token_filter import VQATokenFilter  # 🔧 新增：导入噪声过滤器

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

        # 🔧 新增：初始化噪声过滤器
        self.token_filter = VQATokenFilter()
        self.logger.info("✓ 噪声过滤器初始化成功")

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
        gt_consistency: Optional[float] = 1.0,  # GT置信度
        gt_dist: Optional[Dict[str, float]] = None  # GT答案分布
    ) -> Dict[str, Any]:
        """
        生成软硬标签和CoT（一次推理）

        🔧 新方案：一次推理同时生成软标签和CoT
        - 硬标签：直接使用COCO标注（ground truth）
        - 置信度：使用 gt_consistency
        - 软标签：从推理logits获取概率分布
        - CoT：从推理结果提取推理段落

        Args:
            image_path: 图像路径
            question: 问题文本
            question_type: 问题类型（binary/counting/color/location/choice/open）
            image_id: 图像ID
            candidate_pool: 动态候选答案池（用于 CHOICE 类型）
            ground_truth: COCO标注答案（用作硬标签）
            gt_consistency: GT置信度
            gt_dist: GT答案分布

        Returns:
            {
                'hard_label': {'answer': str, 'confidence': float},
                'soft_label': {
                    'answer_distribution': Dict[str, float],
                    'primary_answer': str,
                    'allowed_answers': List[str]
                },
                'cot_reasoning': {
                    'reasoning_paragraph': str,
                    'answer': str
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
            self.logger.info(f"[Label Gen] 使用弱约束prompt + 一次推理生成软标签和CoT")

            # 🔧 步骤1：硬标签生成（来自COCO标注）
            if ground_truth:
                # 使用COCO标注作为硬标签
                answer = ground_truth
                confidence = gt_consistency if gt_consistency is not None else 1.0
                self.logger.info(
                    f"[Label Gen] 硬标签来自COCO标注: {answer} "
                    f"(置信度: {confidence:.4f})"
                )

                if gt_dist:
                    self.logger.debug(f"[Label Gen] GT分布: {gt_dist}")

                # ───────────────────────────────────────────────────────
                # 🔧 正确方案：一次推理同时获取logits和CoT
                # ───────────────────────────────────────────────────────
                # 关键：提取【答案】标记之后的logits，而不是第一个token的logits
                # ───────────────────────────────────────────────────────
                self.logger.info("[Label Gen] 一次推理：生成答案、logits和CoT")

                result = self.teacher.inference_vqa(
                    image=image_path,
                    question=question,
                    return_logits=True,  # ✅ 获取logits（包含完整序列）
                    generate_cot=True,   # ✅ 同时生成CoT
                    primary_answer=ground_truth,  # 使用GT作为参考答案
                    allowed_answers=None,
                    candidate_answers=None,  # 无候选池约束
                    is_strong_pool=False,  # 弱约束
                    question_type=question_type
                )
            else:
                # 回退：使用教师模型推理生成硬标签
                self.logger.warning(f"[Label Gen] 无COCO标注，使用教师模型推理")

                # 一次推理：生成答案、logits和CoT
                result = self.teacher.inference_vqa(
                    image=image_path,
                    question=question,
                    return_logits=True,  # ✅ 获取logits
                    generate_cot=True,   # ✅ 同时生成CoT
                    candidate_answers=None,
                    is_strong_pool=False,
                    question_type=question_type
                )

                # 提取硬标签
                answer = result.get('answer', '')
                confidence = result.get('confidence', 0.0)

            self.logger.info(f"[Label Gen] 硬标签: {answer} (置信度: {confidence:.4f})")

            # ───────────────────────────────────────────────────────
            # 步骤2：软标签生成（从logits）
            # ───────────────────────────────────────────────────────
            # 🔧 关键：传递完整的logits数据（包含sequences和scores）
            logits_data = result.get('logits', {})
            sequences = result.get('sequences')
            scores = logits_data.get('scores')

            # 添加INFO级别的调试日志（确保能看到）
            self.logger.info(f"[Label Gen DEBUG] result keys: {list(result.keys())}")
            self.logger.info(f"[Label Gen DEBUG] logits_data keys: {list(logits_data.keys())}")
            self.logger.info(f"[Label Gen DEBUG] sequences存在: {sequences is not None}")
            self.logger.info(f"[Label Gen DEBUG] scores存在: {scores is not None}")

            # 构建完整的logits数据
            logits_data['sequences'] = sequences

            top_k_logits = self._extract_top_k_logits(
                logits_data,
                top_k=50,
                question_type=question_type  # 🔧 传入问题类型
            )

            if top_k_logits:
                soft_label = self._compute_soft_label_from_logits(top_k_logits)
                self.logger.info(f"[Label Gen] 软标签分布（top-5）: {dict(list(soft_label['answer_distribution'].items())[:5])}")
            else:
                soft_label = {
                    'answer_distribution': {answer: confidence},
                    'primary_answer': answer,
                    'allowed_answers': [answer]
                }
                self.logger.warning(f"[Label Gen] 无法提取logits，使用硬标签作为软标签")

            # ───────────────────────────────────────────────────────
            # 步骤3：CoT提取（从推理结果）
            # ───────────────────────────────────────────────────────
            full_response = result.get('full_response', '')
            cot_reasoning = self._extract_cot_from_response(full_response)

            if cot_reasoning.get('reasoning_paragraph'):
                self.logger.info(f"[Label Gen] CoT提取成功: {len(cot_reasoning['reasoning_paragraph'])} 字符")
            else:
                self.logger.warning(f"[Label Gen] CoT提取失败，返回空结构")

            # ───────────────────────────────────────────────────────
            # 返回完整结果（包含软标签和CoT）
            # ───────────────────────────────────────────────────────
            return {
                'hard_label': {'answer': answer, 'confidence': confidence},
                'soft_label': soft_label,
                'cot_reasoning': cot_reasoning,  # 🔧 新增：CoT推理
                'candidate_pool': None
            }

        self.logger.info(f"[Label Gen] 使用固定候选池: {candidate_answers}")

        # ───────────────────────────────────────────────────────
        # 🔧 正确方案：一次推理同时获取logits和CoT
        # ───────────────────────────────────────────────────────
        # 关键：提取【答案】标记之后的logits，而不是第一个token的logits
        # ───────────────────────────────────────────────────────
        self.logger.info("[Label Gen] 一次推理：生成答案、logits和CoT")

        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,  # ✅ 获取logits（包含完整序列）
            generate_cot=True,   # ✅ 同时生成CoT
            primary_answer=ground_truth,  # 使用GT作为参考
            allowed_answers=candidate_answers,
            candidate_answers=candidate_answers,  # 传入候选词（用于prompt）
            is_strong_pool=is_strong_pool,  # 根据候选池类型选择prompt
            question_type=question_type
        )

        # 提取候选词logits
        # 🔧 关键：传递完整的logits数据（包含sequences和scores）
        logits_data = result.get('logits', {})
        sequences = result.get('sequences')
        scores = logits_data.get('scores')

        # 添加INFO级别的调试日志（确保能看到）
        self.logger.info(f"[Label Gen DEBUG] result keys: {list(result.keys())}")
        self.logger.info(f"[Label Gen DEBUG] logits_data keys: {list(logits_data.keys())}")
        self.logger.info(f"[Label Gen DEBUG] sequences存在: {sequences is not None}")
        self.logger.info(f"[Label Gen DEBUG] scores存在: {scores is not None}")
        if sequences:
            self.logger.info(f"[Label Gen DEBUG] sequences长度: {len(sequences) if hasattr(sequences, '__len__') else 'N/A'}")

        # 构建完整的logits数据
        logits_data['sequences'] = sequences

        candidate_logits = self._extract_candidate_logits(
            logits_data,
            candidate_answers,
            question_type=question_type  # 🔧 新增：传入问题类型
        )

        if not candidate_logits:
            self.logger.warning("[Label Gen] 无法提取候选词logits")
            return None

        self.logger.info(f"[Label Gen] 提取的候选词logits: {candidate_logits}")

        # ───────────────────────────────────────────────────────
        # 步骤2：温度缩放 + softmax归一化（用于软标签）
        # ───────────────────────────────────────────────────────
        soft_label = self._compute_soft_label(candidate_logits, candidate_answers)

        self.logger.info(f"[Label Gen] 软标签分布: {soft_label['answer_distribution']}")

        # ───────────────────────────────────────────────────────
        # 步骤3：硬标签生成
        # 🔧 新方案：硬标签直接使用COCO标注（ground_truth）
        # ───────────────────────────────────────────────────────
        if ground_truth:
            # 使用COCO标注作为硬标签
            hard_label = {'answer': ground_truth, 'confidence': gt_consistency if gt_consistency is not None else 1.0}
            self.logger.info(
                f"[Label Gen] 硬标签来自COCO标注: {ground_truth} "
                f"(置信度: {hard_label['confidence']:.4f})"
            )

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

        # ───────────────────────────────────────────────────────
        # 步骤4：CoT提取（从推理结果）
        # ───────────────────────────────────────────────────────
        full_response = result.get('full_response', '')
        cot_reasoning = self._extract_cot_from_response(full_response)

        if cot_reasoning.get('reasoning_paragraph'):
            self.logger.info(f"[Label Gen] CoT提取成功: {len(cot_reasoning['reasoning_paragraph'])} 字符")
        else:
            self.logger.warning(f"[Label Gen] CoT提取失败，返回空结构")

        # ───────────────────────────────────────────────────────
        # 返回完整结果（包含软标签和CoT）
        # ───────────────────────────────────────────────────────
        return {
            'hard_label': hard_label,
            'soft_label': soft_label,
            'cot_reasoning': cot_reasoning,  # 🔧 新增：CoT推理
            'candidate_pool': candidate_answers
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
        top_k: int = 50,
        question_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        从logits中提取候选词（优化版V2 - 无候选池）

        🔧 优化流程：
        1. 全局 logits / T
        2. 全词表变体合并（LogSumExp）
           - 数字：1/one/first/_1/'1 → 1
           - 颜色：red/Red/RED → red
           - 实体：hot dog/hotdog → hot dog
        3. Top-P + max_k 动态截断（过滤长尾噪声）
        4. 返回合并后的候选词logits（softmax在_compute_soft_label_from_logits中执行）

        Args:
            logits_data: logits数据（包含scores和sequences）
            top_k: 最大候选词数量（硬上限）
            question_type: 问题类型（counting/color/location/open）

        Returns:
            {候选词: logit值}（温度缩放+合并后，未归一化）
        """
        candidate_logits = {}

        # ───────────────────────────────────────────────────────
        # 🔧 步骤1：正确提取答案位置的logits
        # ───────────────────────────────────────────────────────
        sequences = logits_data.get('sequences')
        scores = logits_data.get('scores') or logits_data.get('raw_logits')

        self.logger.info(f"[Logits Extract V2] 开始无候选池提取，问题类型: {question_type}")

        if sequences is None or scores is None:
            self.logger.warning("[Logits Extract V2] 缺少sequences或scores")
            return {}

        # 解码序列
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.cpu().tolist()
        if len(sequences) > 0 and isinstance(sequences[0], list):
            sequences = sequences[0]

        # ───────────────────────────────────────────────────────
        # 🔧 步骤2：查找 [Answer] 标记位置
        # ───────────────────────────────────────────────────────
        answer_marker = "[Answer]"
        answer_marker_tokens = self.teacher.tokenizer.encode(answer_marker, add_special_tokens=False)

        # 计算num_logits和input_token_count
        if hasattr(scores, 'shape'):
            if scores.dim() == 3:
                num_logits = scores.shape[0]
            elif scores.dim() == 2:
                num_logits = scores.shape[0]
            else:
                num_logits = 1
        else:
            num_logits = len(scores)

        input_token_count = len(sequences) - num_logits

        # 只在生成的部分查找（跳过输入prompt）
        generated_sequences = sequences[input_token_count:]

        # 从后往前查找
        answer_start_pos = None
        for i in range(len(generated_sequences) - len(answer_marker_tokens), -1, -1):
            if generated_sequences[i:i+len(answer_marker_tokens)] == answer_marker_tokens:
                answer_start_pos = input_token_count + i + len(answer_marker_tokens)
                self.logger.info(f"[Logits Extract V2] 找到 [Answer] 标记，答案位置: {answer_start_pos}")
                break

        if answer_start_pos is None:
            self.logger.warning("[Logits Extract V2] 未找到 [Answer] 标记，使用最后一个token")
            answer_start_pos = len(sequences) - 1

        # 调整索引
        if answer_start_pos < input_token_count:
            self.logger.error(f"[Logits Extract V2] 答案位置{answer_start_pos}在输入prompt范围内")
            return {}

        logits_index = answer_start_pos - input_token_count

        if logits_index >= num_logits:
            self.logger.error(f"[Logits Extract V2] logits索引 {logits_index} 超出范围 {num_logits}")
            return {}

        # 提取答案位置的完整logits [vocab_size]
        if hasattr(scores, 'dim') and scores.dim() >= 2:
            if scores.dim() == 3:
                raw_logits = scores[logits_index, 0]
            else:
                raw_logits = scores[logits_index]
        else:
            raw_logits = scores[logits_index]
            if hasattr(raw_logits, 'dim') and raw_logits.dim() >= 2:
                raw_logits = raw_logits[0]

        self.logger.info(f"[Logits Extract V2] 提取logits，形状: {raw_logits.shape}")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤3：Temperature Scaling（在原始logit域）
        # ───────────────────────────────────────────────────────
        scaled_logits = raw_logits / self.soft_label_temperature
        self.logger.info(f"[Logits Extract V2] 温度缩放完成，T={self.soft_label_temperature}")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤1：三层过滤（非英文 + 残缺子词 + 黑名单）
        # ───────────────────────────────────────────────────────
        # 过滤顺序：
        # 1. 非英文Token过滤（俄语西里尔、中日韩等）
        # 2. 残缺子词过滤（explo, anthrop等片段）
        # 3. 黑名单过滤（特殊token、标点、停用词等）
        # ───────────────────────────────────────────────────────

        # 🔧 步骤1.1：对 Top-1000 token 做 decode（存储完整元组）
        top_k_decode = min(1000, scaled_logits.size(0))
        top_values, top_indices = torch.topk(scaled_logits, top_k_decode)

        # 解码 Top-K tokens（存储完整元组）
        decoded_token_tuples = []  # [(token_id, token_str, logit_value)]
        for idx, (token_id, logit_value) in enumerate(zip(top_indices, top_values)):
            token = self.teacher.tokenizer.decode([token_id.item()])
            token = token.strip()

            # 🔧 硬规则过滤：跳过特殊token
            if not token or token.startswith('<') or token.startswith('['):
                continue

            # 🔧 新增过滤1：非英文Token过滤
            # 过滤俄语西里尔、中日韩等非英文字符
            if hasattr(self, 'token_filter') and not self.token_filter.is_english_token(token, question_type):
                self.logger.debug(f"[Non-English Filter] 过滤非英文token: '{token}'")
                continue

            # 🔧 新增过滤2：残缺子词过滤
            # 过滤 explo, anthrop 等残缺片段（需要token_id和tokenizer）
            if hasattr(self, 'token_filter') and not self.token_filter.is_complete_word_token(
                token_id.item(), token, self.teacher.tokenizer
            ):
                self.logger.debug(f"[Subword Filter] 过滤残缺子词: '{token}'")
                continue

            # 🔧 过滤3：黑名单过滤
            # 应用 VQATokenFilter 的黑名单过滤
            if hasattr(self, 'token_filter') and not self.token_filter.is_valid_token(token):
                self.logger.debug(f"[Hard Filter] 过滤噪声token: '{token}'")
                continue

            # 存储完整元组：(token_id, token_str, logit_value)
            decoded_token_tuples.append((token_id.item(), token, logit_value.item()))

        self.logger.info(
            f"[Logits Extract V2] 三层过滤后: {len(decoded_token_tuples)} 个token "
            f"(非英文+残缺子词+黑名单)"
        )

        # 转换为字典格式（后续处理）
        decoded_tokens = {item[1]: item[2] for item in decoded_token_tuples}

        # ───────────────────────────────────────────────────────
        # 🔧 步骤2：在线实时 Canon（基本标准化 + 数字归一化）
        # ───────────────────────────────────────────────────────
        # 🔧 处理流程：
        # 1. 去空格前缀/后缀
        # 2. 去尾标点
        # 3. 小写
        # 4. 全角数字转半角（４ → 4）
        # 5. 检查equivalent_tokens映射（数字归一化）
        #
        # ❌ 不做单复数归一化（避免"male"→"mal"的错误）
        # ───────────────────────────────────────────────────────

        token_to_canonical = {}  # token -> canonical_key 映射

        for token in decoded_tokens.keys():
            # 基本标准化（小写+去标点）
            canonical_key = token.strip().lower()

            # 🔧 DEBUG: 如果token包含引号，记录处理过程
            if '"' in token or "'" in token or '"' in token or '"' in token:
                self.logger.debug(f"[Canon DEBUG] 原始token: '{token}'")

            # 去尾标点（扩展列表）
            while canonical_key and canonical_key[-1] in [
                '.', ',', '!', '?', ';', ':',  # 基本标点
                '"', "'", '"', '"', ''', ''', '`',  # 引号（ASCII + Unicode + 反引号）
                ')', ']', '}', '>',  # 闭合括号
                '-', '+', '=', '*', '#', '@', '%', '/', '\\', '|', '~', '$', '£', '€', '¥'  # 特殊符号 + 货币符号
            ]:
                canonical_key = canonical_key[:-1]

            # 去前缀标点（扩展列表）
            while canonical_key and canonical_key[0] in [
                '"', "'", '"', '"', ''', ''', '`',  # 引号（ASCII + Unicode + 反引号）
                '(', '[', '{', '<',  # 开放括号
                '-', '+', '=', '*', '#', '@', '%', '/', '\\', '|', '~', '_', '$', '£', '€', '¥'  # 特殊符号 + 货币符号
            ]:
                canonical_key = canonical_key[1:]

            # 🔧 DEBUG: 如果处理前后不同，记录
            if '"' in token or "'" in token or '"' in token or '"' in token:
                self.logger.debug(f"[Canon DEBUG] 处理后: '{canonical_key}'")

            # 🔧 新增：全角数字转半角
            # 全角数字：０１２３４５６７８９ → 半角：0123456789
            fullwidth_digits = '０１２３４５６７８９'
            halfwidth_digits = '0123456789'
            for fw, hw in zip(fullwidth_digits, halfwidth_digits):
                canonical_key = canonical_key.replace(fw, hw)

            # 🔧 新增：检查equivalent_tokens映射（数字归一化）
            # 例：four → 4, one → 1, grey → gray
            if hasattr(self, 'token_filter') and hasattr(self.token_filter, 'equivalent_tokens'):
                if canonical_key in self.token_filter.equivalent_tokens:
                    original_key = canonical_key
                    canonical_key = self.token_filter.equivalent_tokens[canonical_key]
                    # 🔧 DEBUG: 记录映射转换
                    if original_key != canonical_key:
                        self.logger.debug(f"[Canon Mapping] '{original_key}' → '{canonical_key}'")

            # 🔧 新增：数字类问题特殊处理 - 强制归一化为阿拉伯数字
            # 🔧 问题：equivalent_tokens 配置中 "one": ["1", ...]
            # 🔧 结果："1" -> "one"（方向反了），我们需要 "one" -> "1"
            # 🔧 解决：counting 类型问题，强制将单词数字转为阿拉伯数字
            if question_type == 'counting':
                number_word_to_digit = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                    'eighteen': '18', 'nineteen': '19', 'twenty': '20'
                }
                if canonical_key in number_word_to_digit:
                    original_key = canonical_key
                    canonical_key = number_word_to_digit[canonical_key]
                    self.logger.debug(f"[Number Normalization] '{original_key}' → '{canonical_key}' (counting任务)")

            token_to_canonical[token] = canonical_key

        self.logger.info(f"[Logits Extract V2] Canon处理完成（基本标准化+数字归一化）")

        # ───────────────────────────────────────────────────────
        # 🔧 新增：问题类型感知的Token过滤
        # ───────────────────────────────────────────────────────
        # 规则：
        # - 如果问题类型是binary/yes_no：保留yes/no及其同义词
        # - 如果问题类型不是binary/yes_no：丢弃yes/no及其同义词（避免噪音）
        #
        # 原因：
        # - counting/color/location等任务不应该出现yes/no
        # - yes/no可能是模型的无意义输出或错误判断
        #
        # 🔧 修复：QuestionType.BINARY = "yes_no"（不是"binary"）
        # ───────────────────────────────────────────────────────

        # 检查是否是Yes/No任务（兼容多种写法）
        is_binary_task = question_type in ['binary', 'yes_no', 'closed_yesno'] if question_type else False

        if not is_binary_task and hasattr(self, 'token_filter'):
            # 获取yes/no及其所有同义词
            yes_no_tokens = set()

            # 从task_whitelists获取binary候选词
            if 'binary' in self.token_filter.task_whitelists:
                yes_no_tokens.update(self.token_filter.task_whitelists['binary'])

            # 从equivalent_tokens获取yes/no的所有变体
            for canonical in ['yes', 'no']:
                if canonical in self.token_filter.equivalent_tokens:
                    yes_no_tokens.add(canonical)
                # 查找所有映射到yes/no的变体
                if hasattr(self.token_filter, 'canonical_to_variants'):
                    if canonical in self.token_filter.canonical_to_variants:
                        yes_no_tokens.update(self.token_filter.canonical_to_variants[canonical])

            # 过滤掉yes/no及其变体
            filtered_token_to_canonical = {}
            for token, canonical in token_to_canonical.items():
                if canonical in yes_no_tokens:
                    self.logger.debug(f"[Task-Aware Filter] 过滤yes/no token: '{token}' → '{canonical}' (任务类型: {question_type})")
                else:
                    filtered_token_to_canonical[token] = canonical

            # 更新映射
            token_to_canonical = filtered_token_to_canonical
            self.logger.info(f"[Task-Aware Filter] 过滤yes/no后: {len(token_to_canonical)} 个token (任务: {question_type})")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤3：按 canonical_key 分组
        # ───────────────────────────────────────────────────────
        # 把所有得到相同 canonical_key 的 token 放到同一组
        # ───────────────────────────────────────────────────────

        canonical_groups = {}  # canonical_key -> [tokens]

        for token, canonical_key in token_to_canonical.items():
            if canonical_key not in canonical_groups:
                canonical_groups[canonical_key] = []
            canonical_groups[canonical_key].append(token)

        self.logger.info(f"[Logits Extract V2] 分组完成: {len(canonical_groups)} 个canonical组")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤4：LogSumExp 组内合并
        # ───────────────────────────────────────────────────────
        # 同一 canonical_key 的 token logits 做 LogSumExp 合并
        # 例: "cat" (logit=2.0) + "Cat" (logit=1.5) + "cats" (logit=1.2)
        #     -> merged_logit = log(exp(2.0) + exp(1.5) + exp(1.2))
        # ───────────────────────────────────────────────────────

        canonical_logits = {}

        for canonical_key, tokens in canonical_groups.items():
            # 收集该组所有 token 的 logits
            token_logits = [decoded_tokens[token] for token in tokens]

            if len(token_logits) == 1:
                # 只有一个token，直接取值
                canonical_logits[canonical_key] = token_logits[0]
            else:
                # 多个token，LogSumExp合并
                logits_tensor = torch.tensor(token_logits)
                merged_logit = torch.logsumexp(logits_tensor, dim=0).item()
                canonical_logits[canonical_key] = merged_logit

        self.logger.info(f"[Logits Extract V2] LogSumExp合并完成: {len(canonical_logits)} 个canonical logit")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤5：问题类型过滤（可选）
        # ───────────────────────────────────────────────────────
        if question_type == 'counting':
            # 只保留数字相关token
            filtered_logits = {}
            for token, logit in canonical_logits.items():
                is_number = (
                    token.isdigit() or
                    token in self.token_filter.task_whitelists.get('count', set())
                )
                if is_number:
                    filtered_logits[token] = logit
            canonical_logits = filtered_logits
            self.logger.info(f"[Logits Extract V2] counting类型过滤: {len(canonical_logits)} 个数字token")

        elif question_type == 'color':
            # 只保留颜色相关token
            filtered_logits = {}
            for token, logit in canonical_logits.items():
                if token in self.token_filter.task_whitelists.get('color', set()):
                    filtered_logits[token] = logit
            canonical_logits = filtered_logits
            self.logger.info(f"[Logits Extract V2] color类型过滤: {len(canonical_logits)} 个颜色token")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤6：Top-P + max_k 动态截断（开放集专用）
        # ───────────────────────────────────────────────────────
        # Top-P (0.90)：过滤海量长尾噪声
        # max_k=50：硬上限，保安全
        # 构建"动态候选集"
        # ───────────────────────────────────────────────────────

        if not canonical_logits:
            self.logger.warning("[Logits Extract V2] 合并后无候选词")
            return {}

        # 将logits转换为tensor
        tokens = list(canonical_logits.keys())
        logits_tensor = torch.tensor([canonical_logits[t] for t in tokens])

        # 先做softmax得到概率分布
        probs = F.softmax(logits_tensor, dim=0)

        # 排序（降序）
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_probs = probs[sorted_indices]

        # Top-P截断（累积概率达到p时停止）
        top_p = 0.90  # 可配置参数
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff_index = (cumsum_probs <= top_p).sum().item()

        # 限制最大数量为max_k
        max_k = min(top_k, len(tokens))
        cutoff_index = min(cutoff_index + 1, max_k)

        # 提取top-p候选词
        top_indices = sorted_indices[:cutoff_index]

        for idx in top_indices:
            token = tokens[idx.item()]
            logit_value = canonical_logits[token]
            candidate_logits[token] = logit_value

        self.logger.info(
            f"[Logits Extract V2] Top-P截断: {len(candidate_logits)} 个候选词 "
            f"(Top-P={top_p}, max_k={max_k})"
        )

        return candidate_logits

    def _compute_soft_label_from_logits(
        self,
        candidate_logits: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        软标签生成（无候选池，唯一归一化步骤）

        🔧 优化流程（V2）：
        - 输入：已经过 Temperature Scaling + 全词表变体合并的logits
        - 输出：在动态候选集上做唯一一次softmax归一化

        Args:
            candidate_logits: {候选词: logit值}（已温度缩放+合并）

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

        self.logger.info(f"[Soft Label V2] 开始归一化，输入logits: {len(candidate_logits)} 个候选词")

        # ───────────────────────────────────────────────────────
        # 🔧 唯一一次归一化：softmax（在动态候选集上）
        # ───────────────────────────────────────────────────────
        candidates = list(candidate_logits.keys())
        logits_tensor = torch.tensor([candidate_logits[c] for c in candidates])

        # 🔧 优化：唯一一次softmax归一化
        probs = F.softmax(logits_tensor, dim=0)

        # 构建概率分布
        answer_distribution = {c: probs[i].item() for i, c in enumerate(candidates)}

        # 确保概率和为1（官方标准）
        total_prob = sum(answer_distribution.values())
        if abs(total_prob - 1.0) > 0.01:
            self.logger.warning(f"[Soft Label V2] 概率和不等于1: {total_prob:.4f}，重新归一化")
            answer_distribution = {k: v/total_prob for k, v in answer_distribution.items()}

        # 找到概率最大的候选词（primary_answer）
        primary_answer = max(answer_distribution.items(), key=lambda x: x[1])[0]

        self.logger.info(
            f"[Soft Label V2] 归一化完成，primary_answer: '{primary_answer}', "
            f"分布（top-5）: {dict(list(answer_distribution.items())[:5])}"
        )

        return {
            'answer_distribution': answer_distribution,
            'primary_answer': primary_answer,
            'allowed_answers': candidates
        }

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
        candidate_answers: List[str],
        question_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        提取候选词logits并做同义词合并（优化版）

        🔧 优化流程：
        1. 获取 Raw Logits [vocab_size]
        2. Temperature Scaling: logits / T
        3. 同义词合并 LogSumExp（logit域，不经softmax）
        4. 返回合并后的候选词logits（softmax在_compute_soft_label中执行）

        ✅ 核心改进：
        - 使用 LogSumExp 合并同义词/变体（1/one/１）
        - 只在候选子集上做一次softmax归一化

        Args:
            logits_data: 原始logits数据（包含scores和sequences）
            candidate_answers: 候选答案列表
            question_type: 问题类型（用于任务感知过滤）

        Returns:
            {候选词: logit值}（温度缩放+合并后，未归一化）
        """
        candidate_logits = {}

        # ───────────────────────────────────────────────────────
        # 🔧 步骤1：正确提取答案位置的logits
        # ───────────────────────────────────────────────────────
        sequences = logits_data.get('sequences')
        scores = logits_data.get('scores') or logits_data.get('raw_logits')

        self.logger.info(f"[Logits Extract V2] 开始优化版提取，候选词: {candidate_answers}")

        if sequences is None or scores is None:
            self.logger.warning("[Logits Extract V2] 缺少sequences或scores")
            return {}

        # 解码序列
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.cpu().tolist()
        if len(sequences) > 0 and isinstance(sequences[0], list):
            sequences = sequences[0]

        # ───────────────────────────────────────────────────────
        # 🔧 步骤2：查找 [Answer] 标记位置
        # ───────────────────────────────────────────────────────
        answer_marker = "[Answer]"
        answer_marker_tokens = self.teacher.tokenizer.encode(answer_marker, add_special_tokens=False)

        # 从后往前查找（避免误匹配输入prompt中的示例）
        answer_start_pos = None
        for i in range(len(sequences) - len(answer_marker_tokens), -1, -1):
            if sequences[i:i+len(answer_marker_tokens)] == answer_marker_tokens:
                answer_start_pos = i + len(answer_marker_tokens)
                self.logger.info(f"[Logits Extract V2] 找到 [Answer] 标记，答案位置: {answer_start_pos}")
                break

        if answer_start_pos is None:
            self.logger.warning("[Logits Extract V2] 未找到 [Answer] 标记，使用最后一个token")
            answer_start_pos = len(sequences) - 1

        # ───────────────────────────────────────────────────────
        # 🔧 步骤3：调整索引（sequences索引 → scores索引）
        # ───────────────────────────────────────────────────────
        # 计算num_logits和input_token_count
        if hasattr(scores, 'shape'):
            if scores.dim() == 3:
                num_logits = scores.shape[0]
            elif scores.dim() == 2:
                num_logits = scores.shape[0]
            else:
                num_logits = 1
        else:
            num_logits = len(scores)

        input_token_count = len(sequences) - num_logits

        if answer_start_pos >= input_token_count:
            logits_index = answer_start_pos - input_token_count
        else:
            self.logger.error(f"[Logits Extract V2] 答案位置{answer_start_pos}在输入prompt范围内")
            return {}

        if logits_index >= num_logits:
            self.logger.error(f"[Logits Extract V2] logits索引 {logits_index} 超出范围 {num_logits}")
            return {}

        # 提取答案位置的完整logits [vocab_size]
        if hasattr(scores, 'dim') and scores.dim() >= 2:
            if scores.dim() == 3:
                raw_logits = scores[logits_index, 0]  # [vocab_size]
            else:
                raw_logits = scores[logits_index]
        else:
            raw_logits = scores[logits_index]
            if hasattr(raw_logits, 'dim') and raw_logits.dim() >= 2:
                raw_logits = raw_logits[0]

        self.logger.info(f"[Logits Extract V2] 提取logits，形状: {raw_logits.shape}")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤4：Temperature Scaling（在原始logit域）
        # ───────────────────────────────────────────────────────
        scaled_logits = raw_logits / self.soft_label_temperature
        self.logger.info(f"[Logits Extract V2] 温度缩放完成，T={self.soft_label_temperature}")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤1：三层过滤（非英文 + 残缺子词 + 黑名单）
        # ───────────────────────────────────────────────────────
        # 过滤顺序：
        # 1. 非英文Token过滤（俄语西里尔、中日韩等）
        # 2. 残缺子词过滤（explo, anthrop等片段）
        # 3. 黑名单过滤（特殊token、标点、停用词等）
        # ───────────────────────────────────────────────────────

        # 🔧 步骤1.1：对 Top-1000 token 做 decode（存储完整元组）
        top_k_decode = min(1000, scaled_logits.size(0))
        top_values, top_indices = torch.topk(scaled_logits, top_k_decode)

        # 解码 Top-K tokens（存储完整元组）
        decoded_token_tuples = []  # [(token_id, token_str, logit_value)]
        for idx, (token_id, logit_value) in enumerate(zip(top_indices, top_values)):
            token = self.teacher.tokenizer.decode([token_id.item()])
            token = token.strip()

            # 🔧 硬规则过滤：跳过特殊token
            if not token or token.startswith('<') or token.startswith('['):
                continue

            # 🔧 新增过滤1：非英文Token过滤
            # 过滤俄语西里尔、中日韩等非英文字符
            if hasattr(self, 'token_filter') and not self.token_filter.is_english_token(token, question_type):
                self.logger.debug(f"[Non-English Filter] 过滤非英文token: '{token}'")
                continue

            # 🔧 新增过滤2：残缺子词过滤
            # 过滤 explo, anthrop 等残缺片段（需要token_id和tokenizer）
            if hasattr(self, 'token_filter') and not self.token_filter.is_complete_word_token(
                token_id.item(), token, self.teacher.tokenizer
            ):
                self.logger.debug(f"[Subword Filter] 过滤残缺子词: '{token}'")
                continue

            # 🔧 过滤3：黑名单过滤
            # 应用 VQATokenFilter 的黑名单过滤
            if hasattr(self, 'token_filter') and not self.token_filter.is_valid_token(token):
                self.logger.debug(f"[Hard Filter] 过滤噪声token: '{token}'")
                continue

            # 存储完整元组：(token_id, token_str, logit_value)
            decoded_token_tuples.append((token_id.item(), token, logit_value.item()))

        self.logger.info(
            f"[Logits Extract V2] 三层过滤后: {len(decoded_token_tuples)} 个token "
            f"(非英文+残缺子词+黑名单)"
        )

        # 转换为字典格式（后续处理）
        decoded_tokens = {item[1]: item[2] for item in decoded_token_tuples}

        # ───────────────────────────────────────────────────────
        # 🔧 步骤2：在线实时 Canon（基本标准化 + 数字归一化）
        # ───────────────────────────────────────────────────────
        # 🔧 处理流程：
        # 1. 去空格前缀/后缀
        # 2. 去尾标点
        # 3. 小写
        # 4. 全角数字转半角（４ → 4）
        # 5. 检查equivalent_tokens映射（数字归一化）
        #
        # ❌ 不做单复数归一化（避免"male"→"mal"的错误）
        # ───────────────────────────────────────────────────────

        token_to_canonical = {}  # token -> canonical_key 映射

        for token in decoded_tokens.keys():
            # 基本标准化（小写+去标点）
            canonical_key = token.strip().lower()

            # 🔧 DEBUG: 如果token包含引号，记录处理过程
            if '"' in token or "'" in token or '"' in token or '"' in token:
                self.logger.debug(f"[Canon DEBUG] 原始token: '{token}'")

            # 去尾标点（扩展列表）
            while canonical_key and canonical_key[-1] in [
                '.', ',', '!', '?', ';', ':',  # 基本标点
                '"', "'", '"', '"', ''', ''', '`',  # 引号（ASCII + Unicode + 反引号）
                ')', ']', '}', '>',  # 闭合括号
                '-', '+', '=', '*', '#', '@', '%', '/', '\\', '|', '~', '$', '£', '€', '¥'  # 特殊符号 + 货币符号
            ]:
                canonical_key = canonical_key[:-1]

            # 去前缀标点（扩展列表）
            while canonical_key and canonical_key[0] in [
                '"', "'", '"', '"', ''', ''', '`',  # 引号（ASCII + Unicode + 反引号）
                '(', '[', '{', '<',  # 开放括号
                '-', '+', '=', '*', '#', '@', '%', '/', '\\', '|', '~', '_', '$', '£', '€', '¥'  # 特殊符号 + 货币符号
            ]:
                canonical_key = canonical_key[1:]

            # 🔧 DEBUG: 如果处理前后不同，记录
            if '"' in token or "'" in token or '"' in token or '"' in token:
                self.logger.debug(f"[Canon DEBUG] 处理后: '{canonical_key}'")

            # 🔧 新增：全角数字转半角
            # 全角数字：０１２３４５６７８９ → 半角：0123456789
            fullwidth_digits = '０１２３４５６７８９'
            halfwidth_digits = '0123456789'
            for fw, hw in zip(fullwidth_digits, halfwidth_digits):
                canonical_key = canonical_key.replace(fw, hw)

            # 🔧 新增：检查equivalent_tokens映射（数字归一化）
            # 例：four → 4, one → 1, grey → gray
            if hasattr(self, 'token_filter') and hasattr(self.token_filter, 'equivalent_tokens'):
                if canonical_key in self.token_filter.equivalent_tokens:
                    original_key = canonical_key
                    canonical_key = self.token_filter.equivalent_tokens[canonical_key]
                    # 🔧 DEBUG: 记录映射转换
                    if original_key != canonical_key:
                        self.logger.debug(f"[Canon Mapping] '{original_key}' → '{canonical_key}'")

            # 🔧 新增：数字类问题特殊处理 - 强制归一化为阿拉伯数字
            # 🔧 问题：equivalent_tokens 配置中 "one": ["1", ...]
            # 🔧 结果："1" -> "one"（方向反了），我们需要 "one" -> "1"
            # 🔧 解决：counting 类型问题，强制将单词数字转为阿拉伯数字
            if question_type == 'counting':
                number_word_to_digit = {
                    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                    'eighteen': '18', 'nineteen': '19', 'twenty': '20'
                }
                if canonical_key in number_word_to_digit:
                    original_key = canonical_key
                    canonical_key = number_word_to_digit[canonical_key]
                    self.logger.debug(f"[Number Normalization] '{original_key}' → '{canonical_key}' (counting任务)")

            token_to_canonical[token] = canonical_key

        self.logger.info(f"[Logits Extract V2] Canon处理完成（基本标准化+数字归一化）")

        # ───────────────────────────────────────────────────────
        # 🔧 新增：问题类型感知的Token过滤
        # ───────────────────────────────────────────────────────
        # 规则：
        # - 如果问题类型是binary/yes_no：保留yes/no及其同义词
        # - 如果问题类型不是binary/yes_no：丢弃yes/no及其同义词（避免噪音）
        #
        # 原因：
        # - counting/color/location等任务不应该出现yes/no
        # - yes/no可能是模型的无意义输出或错误判断
        #
        # 🔧 修复：QuestionType.BINARY = "yes_no"（不是"binary"）
        # ───────────────────────────────────────────────────────

        # 检查是否是Yes/No任务（兼容多种写法）
        is_binary_task = question_type in ['binary', 'yes_no', 'closed_yesno'] if question_type else False

        if not is_binary_task and hasattr(self, 'token_filter'):
            # 获取yes/no及其所有同义词
            yes_no_tokens = set()

            # 从task_whitelists获取binary候选词
            if 'binary' in self.token_filter.task_whitelists:
                yes_no_tokens.update(self.token_filter.task_whitelists['binary'])

            # 从equivalent_tokens获取yes/no的所有变体
            for canonical in ['yes', 'no']:
                if canonical in self.token_filter.equivalent_tokens:
                    yes_no_tokens.add(canonical)
                # 查找所有映射到yes/no的变体
                if hasattr(self.token_filter, 'canonical_to_variants'):
                    if canonical in self.token_filter.canonical_to_variants:
                        yes_no_tokens.update(self.token_filter.canonical_to_variants[canonical])

            # 过滤掉yes/no及其变体
            filtered_token_to_canonical = {}
            for token, canonical in token_to_canonical.items():
                if canonical in yes_no_tokens:
                    self.logger.debug(f"[Task-Aware Filter] 过滤yes/no token: '{token}' → '{canonical}' (任务类型: {question_type})")
                else:
                    filtered_token_to_canonical[token] = canonical

            # 更新映射
            token_to_canonical = filtered_token_to_canonical
            self.logger.info(f"[Task-Aware Filter] 过滤yes/no后: {len(token_to_canonical)} 个token (任务: {question_type})")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤3：按 canonical_key 分组
        # ───────────────────────────────────────────────────────
        # 把所有得到相同 canonical_key 的 token 放到同一组
        # ───────────────────────────────────────────────────────

        canonical_groups = {}  # canonical_key -> [tokens]

        for token, canonical_key in token_to_canonical.items():
            if canonical_key not in canonical_groups:
                canonical_groups[canonical_key] = []
            canonical_groups[canonical_key].append(token)

        self.logger.info(f"[Logits Extract V2] 分组完成: {len(canonical_groups)} 个canonical组")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤4：LogSumExp 组内合并
        # ───────────────────────────────────────────────────────
        # 同一 canonical_key 的 token logits 做 LogSumExp 合并
        # 例: "cat" (logit=2.0) + "Cat" (logit=1.5) + "cats" (logit=1.2)
        #     -> merged_logit = log(exp(2.0) + exp(1.5) + exp(1.2))
        # ───────────────────────────────────────────────────────

        canonical_logits = {}

        for canonical_key, tokens in canonical_groups.items():
            # 收集该组所有 token 的 logits
            token_logits = [decoded_tokens[token] for token in tokens]

            if len(token_logits) == 1:
                # 只有一个token，直接取值
                canonical_logits[canonical_key] = token_logits[0]
            else:
                # 多个token，LogSumExp合并
                logits_tensor = torch.tensor(token_logits)
                merged_logit = torch.logsumexp(logits_tensor, dim=0).item()
                canonical_logits[canonical_key] = merged_logit

        self.logger.info(f"[Logits Extract V2] LogSumExp合并完成: {len(canonical_logits)} 个canonical logit")

        # ───────────────────────────────────────────────────────
        # 🔧 步骤5：候选锚定过滤（闭合集专用）
        # ───────────────────────────────────────────────────────
        # 仅保留命中候选集的 token
        # 未命中的直接丢弃（如 "the" 不命中任何候选）
        # 此步骤等价于闭合集的"隐式 Top-P"
        #
        # 🔧 关键修复：有候选集的闭合问题，不改变候选池的形式
        # - 候选池是 ["male", "female"] → 软标签保持 ["male", "female"]
        # - 候选池是 ["males", "females"] → 软标签保持 ["males", "females"]
        # - 不需要单复数处理，候选池就是标准答案
        # ───────────────────────────────────────────────────────

        # 🔧 修复：候选池不做canon()处理，保持原样
        candidate_set = set(candidate_answers)

        candidate_logits = {}
        for canonical, logit in canonical_logits.items():
            if canonical in candidate_set:
                candidate_logits[canonical] = logit
            else:
                self.logger.debug(f"[Candidate Filter] 过滤非候选词: '{canonical}' (logit={logit:.4f})")

        # 🔧 补充：未命中的候选赋 logit = -1e9
        for candidate in candidate_set:
            if candidate not in candidate_logits:
                candidate_logits[candidate] = -1e9
                self.logger.warning(f"[Candidate Fill] 候选词 '{candidate}' 未命中，赋值 -1e9")

        self.logger.info(
            f"[Logits Extract V2] 候选锚定完成: {len(candidate_logits)} 个候选词"
        )
        return candidate_logits

    def _compute_soft_label(
        self,
        candidate_logits: Dict[str, float],
        candidate_answers: List[str]
    ) -> Dict[str, Any]:
        """
        软标签生成（唯一归一化步骤）

        🔧 优化流程（V2）：
        - 输入：已经过 Temperature Scaling + LogSumExp合并的logits
        - 输出：在候选子集上做唯一一次softmax归一化

        公式：p_i = softmax(z_i)

        Args:
            candidate_logits: {候选词: logit值}（已温度缩放+合并）
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

        self.logger.info(f"[Soft Label V2] 开始归一化，输入logits: {candidate_logits}")

        # ───────────────────────────────────────────────────────
        # 🔧 唯一一次归一化：softmax（在合并后的候选子集上）
        # ───────────────────────────────────────────────────────
        candidates = list(candidate_logits.keys())
        logits_tensor = torch.tensor([candidate_logits[c] for c in candidates])

        # 🔧 优化：唯一一次softmax归一化
        probs = F.softmax(logits_tensor, dim=0)

        # 构建概率分布
        answer_distribution = {c: probs[i].item() for i, c in enumerate(candidates)}

        # 确保概率和为1（官方标准）
        total_prob = sum(answer_distribution.values())
        if abs(total_prob - 1.0) > 0.01:
            self.logger.warning(f"[Soft Label V2] 概率和不等于1: {total_prob:.4f}，重新归一化")
            answer_distribution = {k: v/total_prob for k, v in answer_distribution.items()}

        # 找到概率最大的候选词（primary_answer）
        primary_answer = max(answer_distribution.items(), key=lambda x: x[1])[0]

        self.logger.info(f"[Soft Label V2] 归一化完成，分布: {answer_distribution}")

        return {
            'answer_distribution': answer_distribution,
            'primary_answer': primary_answer,
            'allowed_answers': candidates
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

    def _extract_cot_from_response(self, full_response: str) -> Dict[str, str]:
        """
        从推理响应中提取CoT（两段式格式）

        🔧 统一格式：[Reasoning]...[Answer]...
        - 内在逻辑：观察→分析→结论
        - 外在形式：两段式

        Args:
            full_response: 完整的推理响应文本

        Returns:
            {
                'reasoning_paragraph': str,
                'answer': str
            }
        """
        import re

        cot_reasoning = {
            'reasoning_paragraph': '',
            'answer': ''
        }

        if not full_response or len(full_response.strip()) < 10:
            self.logger.warning("[CoT Extract] Empty or too short response")
            return cot_reasoning

        # 提取assistant回复
        if 'assistant' in full_response:
            response_part = full_response.split('assistant')[-1]
        else:
            response_part = full_response

        # ───────────────────────────────────────────────────────
        # 尝试匹配两段式格式：[Reasoning]...[Answer]...
        # ───────────────────────────────────────────────────────
        reasoning_match = re.search(r'\[Reasoning\]\s*(.*?)(?=\[Answer\]|$)', response_part, re.DOTALL)
        answer_match = re.search(r'\[Answer\]\s*(.*?)(?=\n\n|$)', response_part, re.DOTALL)

        if reasoning_match and answer_match:
            # 新格式：两段式
            cot_reasoning['reasoning_paragraph'] = reasoning_match.group(1).strip()
            cot_reasoning['answer'] = answer_match.group(1).strip()

            self.logger.debug(
                f"[CoT Extract] Two-part format found: "
                f"reasoning={len(cot_reasoning['reasoning_paragraph'])} chars, "
                f"answer={len(cot_reasoning['answer'])} chars"
            )

        else:
            # ───────────────────────────────────────────────────────
            # 回退：尝试旧的Observation/Analysis/Conclusion格式
            # ───────────────────────────────────────────────────────
            obs_match = re.search(r'Observation:\s*(.*?)(?=Analysis:|Conclusion:|$)', response_part, re.DOTALL | re.IGNORECASE)
            ana_match = re.search(r'Analysis:\s*(.*?)(?=Conclusion:|$)', response_part, re.DOTALL | re.IGNORECASE)
            con_match = re.search(r'Conclusion:\s*(.*?)(?=\n\n|Observation:|Analysis:|$)', response_part, re.DOTALL | re.IGNORECASE)

            if obs_match and ana_match:
                # 合并observation和analysis为reasoning_paragraph
                cot_reasoning['reasoning_paragraph'] = f"{obs_match.group(1).strip()} {ana_match.group(1).strip()}"

            if con_match:
                cot_reasoning['answer'] = con_match.group(1).strip()

            if cot_reasoning['reasoning_paragraph'] or cot_reasoning['answer']:
                self.logger.debug("[CoT Extract] Old three-part format found and converted")
            else:
                # 最终回退：使用整个响应
                self.logger.warning("[CoT Extract] No structured format found, using full response")
                cot_reasoning['reasoning_paragraph'] = response_part.strip()

        return cot_reasoning


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