"""
VQA Soft Label Generator
========================

VQA任务的软标签生成器，继承基类复用公共逻辑。
"""

import torch
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_soft_label_gen import BaseSoftLabelGenerator
from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.vqa_token_filter import VQATokenFilter
from ..utils.answer_normalizer import normalize_answer


class VQASoftLabelGenerator(BaseSoftLabelGenerator):
    """
    VQA任务的软标签生成器。

    继承基类复用公共逻辑：
    - 温度缩放 (_apply_temperature)
    - Top-K概率提取 (_get_top_k_probabilities)
    - 数据序列化 (_make_serializable)
    - 文件保存 (save_soft_labels)
    - 数据验证 (validate_soft_labels)
    - 统计信息 (get_statistics)

    VQA特有逻辑：
    - Token过滤（截断词、噪音等）
    - 答案标准化
    - 硬标签保底策略
    - 分布优化
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        Initialize VQA Soft Label Generator.

        Args:
            teacher_model: Teacher model instance
            config: Configuration manager
            temperature: 温度参数（覆盖配置）
            top_k: Top-K参数（覆盖配置）
        """
        # 🔧 调用基类初始化
        super().__init__(
            teacher_model=teacher_model,
            config=config,
            temperature=temperature,
            top_k=top_k
        )

        # VQA特有参数
        self.min_probability = self.config.get("distillation.soft_labels.min_probability", 0.01)

        # 🔧 初始化VQA Token过滤器
        try:
            self.token_filter = VQATokenFilter()
            self.logger.info("✓ VQA Token过滤器初始化成功")
        except Exception as e:
            self.logger.warning(f"VQA Token过滤器初始化失败: {e}，将不使用任务适配过滤")
            self.token_filter = None

    def generate_vqa_soft_labels(
        self,
        image_path: str,
        question: str,
        image_id: Optional[str] = None,
        answer_candidates: Optional[List[str]] = None,
        hard_label_result: Optional[Dict[str, Any]] = None,
        cot_result: Optional[Dict[str, Any]] = None  # 保留参数兼容，但不再使用
    ) -> Dict[str, Any]:
        """
        Generate soft labels for VQA.

        新方案：使用 hard_label 中的真实 logits，不从 CoT 获取

        Args:
            image_path: Path to image
            question: Question text
            image_id: Image identifier
            answer_candidates: List of possible answer candidates (optional)
            hard_label_result: hard_label 结果（包含 answer 和 logits）
            cot_result: 保留参数兼容，但不再使用

        Returns:
            Soft label dictionary
        """
        self.logger.debug(f"Generating VQA soft labels for image {image_id}")

        # 🔧 从 hard_label 获取 logits 和答案信息
        if hard_label_result and 'logits' in hard_label_result:
            logits_data = hard_label_result['logits']
            primary_answer_raw = hard_label_result.get('answer', '')

            # 🔧 标准化答案格式：将数字转换为英文单词，确保与 answer_distribution 键一致
            # 例如：'1' -> 'one', '2' -> 'two'
            primary_answer = normalize_answer(primary_answer_raw, target_format='word')

            if primary_answer != primary_answer_raw:
                self.logger.debug(f"[Answer Normalization] '{primary_answer_raw}' -> '{primary_answer}'")

            # 🔧 传入 primary_answer，确保分布合理
            # 🔧 新增：传入question用于上下文感知过滤
            distribution = self._process_vqa_logits(
                logits_data,
                answer_candidates,
                primary_answer=primary_answer,
                question=question
            )

            # 🔧 提取合法答案列表（用于 CoT 限定答案范围）
            allowed_answers = list(distribution.keys())

            soft_label = {
                'answer_distribution': distribution,
                'primary_answer': primary_answer,
                'allowed_answers': allowed_answers  # 🔧 新增：合法答案列表
            }
            return soft_label

        # 如果没有 logits，调用模型获取
        if hard_label_result and 'answer' in hard_label_result:
            primary_answer_raw = hard_label_result['answer']

            # 🔧 标准化答案格式：将数字转换为英文单词
            primary_answer = normalize_answer(primary_answer_raw, target_format='word')

            if primary_answer != primary_answer_raw:
                self.logger.debug(f"[Answer Normalization] '{primary_answer_raw}' -> '{primary_answer}'")

            # 简化分布
            confidence = hard_label_result.get('confidence', 0.5)
            main_prob = min(confidence, 0.98)

            distribution = {
                primary_answer.lower(): main_prob,
                'other': 1.0 - main_prob
            }
            allowed_answers = list(distribution.keys())  # ✅ 添加 allowed_answers

            soft_label = {
                'answer_distribution': distribution,
                'primary_answer': primary_answer,
                'allowed_answers': allowed_answers  # ✅ 同源
            }
            return soft_label

        # 如果没有 hard_label，调用模型
        self.logger.warning(f"No hard_label result provided, calling inference_vqa")
        result = self.teacher.inference_vqa(
            image=image_path,
            question=question,
            return_logits=True,
            generate_cot=False
        )

        primary_answer = result.get('answer', '')

        if 'logits' in result:
            logits_data = result['logits']
            # 🔧 新增：传入question用于上下文感知过滤
            distribution = self._process_vqa_logits(
                logits_data,
                answer_candidates,
                question=question
            )
            allowed_answers = list(distribution.keys())  # ✅ 同源

            soft_label = {
                'answer_distribution': distribution,
                'primary_answer': primary_answer,
                'allowed_answers': allowed_answers  # ✅ 添加
            }
        else:
            distribution = {
                result.get('answer', 'unknown').lower(): 1.0
            }
            allowed_answers = list(distribution.keys())  # ✅ 同源

            soft_label = {
                'answer_distribution': distribution,
                'primary_answer': primary_answer,
                'allowed_answers': allowed_answers  # ✅ 添加
            }

        return soft_label

    def _process_vqa_logits(
        self,
        logits_data: Dict[str, torch.Tensor],
        answer_candidates: Optional[List[str]] = None,
        primary_answer: Optional[str] = None,
        confidence: Optional[float] = None,
        question: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Process VQA logits into answer probability distribution.

        改进：
        1. 输入是原始logits（不是概率）
        2. 应用温度缩放后再计算softmax
        3. 标准公式：soft_probs = softmax(logits / temperature)
        4. 🔧 新增：在Logits层级应用Token白名单过滤（字符串判断 + Logits过滤）
        5. 🔧 新增：topk兜底策略（当白名单过滤后为空时）

        Args:
            logits_data: Dictionary with logits (top_k_indices/top_k_values)
            answer_candidates: Optional list of answer candidates
            primary_answer: 模型给出的主要答案（用于验证和保留）
            confidence: 答案的置信度（用于验证）
            question: 问题文本（用于上下文感知过滤）

        Returns:
            Dictionary mapping answers to probabilities
        """
        distribution = {}

        # 方法1：从原始logits提取top-k，应用温度缩放，然后计算概率
        if 'top_k_indices' in logits_data and 'top_k_values' in logits_data:
            token_indices = logits_data['top_k_indices']
            token_logits = logits_data['top_k_values']  # 🔧 现在是logits，不是概率

            # 🔧 修复3：空值防护，阻断 None.dim() 崩溃
            if token_indices is None or token_logits is None:
                self.logger.warning("[VQA Logits] token_indices or token_logits is None, returning empty distribution")
                return distribution

            # 🔧 修复：正确处理不同维度的tensor
            # 期望形状：[num_tokens, top_k] 或 [top_k]
            if token_indices.dim() >= 1 and token_logits.dim() >= 1:
                # 添加调试日志
                self.logger.debug(f"[VQA Logits] token_indices shape: {token_indices.shape}, token_logits shape: {token_logits.shape}")

                # 取第一个位置（答案的第一个 token）
                if token_indices.dim() == 1:
                    # 已经是 [top_k] 形状，直接使用
                    first_token_indices = token_indices
                    first_token_logits = token_logits
                    self.logger.debug(f"[VQA Logits] Using 1D tensor directly, shape: {first_token_indices.shape}")
                elif token_indices.dim() == 2:
                    # [num_tokens, top_k] 形状，取第一个token
                    first_token_indices = token_indices[0]
                    first_token_logits = token_logits[0]
                    self.logger.debug(f"[VQA Logits] Taking first token from 2D tensor, shape: {first_token_indices.shape}")
                else:
                    # [batch, num_tokens, top_k] 形状，取第一个batch的第一个token
                    first_token_indices = token_indices[0, 0]
                    first_token_logits = token_logits[0, 0]
                    self.logger.debug(f"[VQA Logits] Taking first batch, first token from 3D tensor, shape: {first_token_indices.shape}")

                # 🔧 Step 1: 应用温度缩放到logits
                # 标准公式：soft_probs = softmax(logits / temperature)
                scaled_logits = first_token_logits / self.temperature

                # ===== 🔧 三层防护策略（VQA过滤标准实践） =====
                # 第一层：黑名单（核心防线）- 拦截不可能作为单字答案的Token
                # 第二层：硬标签保护（安全网）- 确保正确答案永不丢失
                # 第三层：Top-K兜底（多样性保障）- 防止过滤后分布过于稀疏
                # ==========================================================

                valid_token_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)
                primary_answer_lower = primary_answer.lower() if primary_answer else None

                # 🔧 新增：获取hard_label对应的token ID（用于第二层保护）
                hard_label_token_ids = set()
                if primary_answer_lower:
                    try:
                        # 将主答案编码为token ID
                        encoded_ids = self.teacher.tokenizer.encode(primary_answer_lower, add_special_tokens=False)
                        hard_label_token_ids = set(encoded_ids)
                        self.logger.debug(f"[Hard Label Protection] Primary answer '{primary_answer}' -> token IDs: {hard_label_token_ids}")
                    except Exception as e:
                        self.logger.warning(f"[Hard Label Protection] Failed to encode primary answer: {e}")

                self.logger.debug(f"[Blacklist Filter] Filtering {len(first_token_indices)} tokens...")

                for i, token_id in enumerate(first_token_indices):
                    try:
                        # 解码token ID到字符串
                        token_str = self.teacher.tokenizer.decode([token_id.item()]).strip()

                        # ===== 🔧 第二层：硬标签保护（安全网） =====
                        # 无论黑名单如何，强制将hard_label_id加入放行列表
                        # 原因：极少数情况下，正确答案的Token可能因为词表构造原因被黑名单误伤
                        if token_id.item() in hard_label_token_ids:
                            valid_token_mask[i] = True
                            self.logger.debug(f"[Hard Label Protection] ✓ Reserved hard label token: '{token_str}' (ID: {token_id.item()})")
                            continue

                        # ===== 🔧 第一层：黑名单（核心防线） =====
                        # 使用VQATokenFilter判断是否有效（已包含BPE碎片、特殊Token、标点等）
                        if self.token_filter and self.token_filter.is_valid_token(token_str, question):
                            valid_token_mask[i] = True
                            self.logger.debug(f"[Blacklist Filter] ✓ Valid token: '{token_str}'")
                        else:
                            self.logger.debug(f"[Blacklist Filter] ✗ Filtered out: '{token_str}'")
                    except Exception as e:
                        self.logger.warning(f"[Blacklist Filter] Failed to decode token {token_id}: {e}")

                # ===== 🔧 第三层：Top-K兜底（多样性保障） =====
                # 如果过滤后剩余Token少于N个（如10个），从原始Top-K中补充
                min_valid_tokens = 10  # 最少保留的有效token数量

                num_valid = valid_token_mask.sum().item()
                self.logger.info(f"[Blacklist Filter] {num_valid}/{len(first_token_indices)} tokens passed blacklist filter")

                # 🔧 第三层逻辑：如果过滤后少于min_valid_tokens个，从Top-K补充
                if num_valid < min_valid_tokens and num_valid > 0:
                    # 有有效token，但数量不足，需要补充
                    self.logger.info(f"[Top-K Fallback] Only {num_valid} tokens remaining, supplementing from Top-{self.top_k}")

                    # 从原始Top-K中补充未被黑名单拦截的token
                    # 计算原始概率分布
                    token_probs_raw = torch.softmax(scaled_logits, dim=-1)

                    # 按概率排序，取Top-50（或更多）
                    top_k_fallback = min(self.top_k * 2, len(first_token_indices))  # 取2倍的top_k作为候选集
                    top_k_indices = torch.topk(token_probs_raw, top_k_fallback).indices

                    # 补充逻辑：从Top-K中添加未被过滤的token
                    for idx in top_k_indices:
                        if not valid_token_mask[idx]:
                            # 检查这个token是否在黑名单中
                            try:
                                token_id = first_token_indices[idx]
                                token_str = self.teacher.tokenizer.decode([token_id.item()]).strip()

                                # 使用较宽松的过滤策略（只过滤绝对噪音）
                                # 注意：这里不使用上下文感知，避免过度过滤
                                if self.token_filter and self.token_filter.is_valid_token(token_str, None):
                                    valid_token_mask[idx] = True
                                    self.logger.debug(f"[Top-K Fallback] + Supplement token: '{token_str}'")

                                    # 检查是否达到最小数量
                                    if valid_token_mask.sum().item() >= min_valid_tokens:
                                        break
                            except Exception as e:
                                self.logger.warning(f"[Top-K Fallback] Failed to decode token: {e}")

                    self.logger.info(f"[Top-K Fallback] After supplementation: {valid_token_mask.sum().item()} tokens")

                # 🔧 应用mask到logits（将无效token的logits设为极小值）
                if num_valid > 0 or valid_token_mask.sum().item() > 0:
                    # 有有效token，应用过滤
                    # 将无效token的logits设为-1e9（softmax后会接近0）
                    scaled_logits_filtered = scaled_logits.clone()
                    scaled_logits_filtered[~valid_token_mask] = -1e9

                    # 计算softmax得到概率
                    token_probs = torch.softmax(scaled_logits_filtered, dim=-1)
                else:
                    # 极端情况：所有token都被过滤（不应该发生，因为有硬标签保护）
                    self.logger.warning(f"[Emergency Fallback] All tokens filtered! Using raw top-{self.top_k}")

                    # 回退到原始top-k策略：保留概率最高的k个token
                    token_probs = torch.softmax(scaled_logits, dim=-1)

                    # 取top-k（保留原始配置的top_k数量）
                    top_k = min(self.top_k, len(token_probs))
                    top_k_indices = torch.topk(token_probs, top_k).indices

                    # 只保留top-k的概率，其余置零
                    token_probs_filtered = torch.zeros_like(token_probs)
                    token_probs_filtered[top_k_indices] = token_probs[top_k_indices]
                    token_probs = token_probs_filtered

                # 🔧 Step 5: 提取并解码
                items = []
                for idx, prob_val in zip(first_token_indices, token_probs):
                    # 过滤掉概率太小的
                    if prob_val < 0.001:  # 过滤掉小于 0.1% 的
                        continue
                    try:
                        word = self.teacher.tokenizer.decode([idx.item()])
                        word = word.strip().lower()

                        # 过滤特殊 token
                        if word and word not in ['<s>', '</s>', '<pad>', '<|im', '|>', '<|', '|>', 'the', 'a', 'an']:
                            # 🔧 新增：过滤下标和上标字符（单字符且非数字）
                            # 这些字符通常是噪音，如：₀₁₂₃₄₅₆₇₈₉ 和 ⁰¹²³⁴⁵⁶⁷⁸⁹
                            if len(word) == 1 and not word.isdigit():
                                # 检查是否是下标或上标数字/字母
                                # Unicode范围：
                                # - 下标数字：U+2080-U+2089
                                # - 上标数字：U+2070, U+00B9, U+00B2, U+00B3, U+2074-U+2079
                                # - 下标字母：U+2090-U+209C
                                # - 上标字母：U+1D43-U+1DBF
                                char_code = ord(word)
                                is_subscript = (0x2080 <= char_code <= 0x2089 or  # 下标数字
                                                0x2090 <= char_code <= 0x209C)    # 下标字母
                                is_superscript = (0x2070 == char_code or           # 上标0
                                                  char_code == 0x00B9 or          # 上标1
                                                  0x00B2 <= char_code <= 0x00B3 or # 上标2-3
                                                  0x2074 <= char_code <= 0x2079 or # 上标4-9
                                                  0x1D43 <= char_code <= 0x1DBF)   # 上标字母
                                if is_subscript or is_superscript:
                                    self.logger.debug(f"[Token Filter] Filtered out subscript/superscript: '{word}' (U+{char_code:04X})")
                                    continue

                            # 🔧 新增：验证token完整性（可选，更严格的检查）
                            # 检查解码后的word是否对应单个token（防止BPE碎片）
                            # 方法：重新编码，检查是否能得到相同的token ID
                            try:
                                re_encoded_ids = self.teacher.tokenizer.encode(word, add_special_tokens=False)

                                # 如果重新编码后：
                                # 1. 只得到一个token ID
                                # 2. 且该ID与原ID一致
                                # 则认为是完整单词，保留
                                # 否则认为是BPE碎片，跳过
                                if len(re_encoded_ids) == 1 and re_encoded_ids[0] == idx.item():
                                    # 是完整单词
                                    if len(word) > 1 or word.isdigit():
                                        items.append((word, float(prob_val)))
                                else:
                                    # 是BPE碎片，跳过（除非是硬标签）
                                    primary_answer_lower = primary_answer.lower() if primary_answer else None
                                    if primary_answer_lower and word == primary_answer_lower:
                                        # 硬标签即使是碎片也保留（安全网）
                                        items.append((word, float(prob_val)))
                                        self.logger.debug(f"[Token Validation] Reserved hard label fragment: '{word}'")
                                    else:
                                        self.logger.debug(f"[Token Validation] Skipped BPE fragment: '{word}' (ID: {idx.item()})")
                            except Exception as e:
                                # 如果验证失败，保守地保留（避免过度过滤）
                                if len(word) > 1 or word.isdigit():
                                    items.append((word, float(prob_val)))
                    except Exception:
                        pass

                # 🔧 合并相同词的概率（如 'one' + 'One' = 'one'）
                word_probs = {}
                for word, prob in items:
                    if word in word_probs:
                        word_probs[word] += prob  # 合并
                    else:
                        word_probs[word] = prob

                # 🔧 关键改进：多Token答案标准化（如 'hot' -> 'hotdog'）
                # 问题：多Token答案（如"hotdog"）被分词成["hot", "dog"]
                # 软标签分布只包含第一个token"hot"的概率，导致主答案丢失
                # 解决：将第一个token映射到完整答案
                if primary_answer_lower:
                    try:
                        # 检查答案是否是多Token
                        primary_token_ids = self.teacher.tokenizer.encode(primary_answer_lower, add_special_tokens=False)

                        if len(primary_token_ids) > 1:
                            # 是多Token答案，获取第一个token
                            first_token = self.teacher.tokenizer.decode([primary_token_ids[0]]).strip().lower()

                            # 如果第一个token在分布中，将其概率转移到完整答案
                            if first_token in word_probs and primary_answer_lower not in word_probs:
                                prob = word_probs.pop(first_token)
                                word_probs[primary_answer_lower] = prob
                                self.logger.debug(
                                    f"[Multi-Token Normalization] Mapped '{first_token}' -> '{primary_answer_lower}' (prob: {prob:.4f})"
                                )
                    except Exception as e:
                        self.logger.warning(f"[Multi-Token Normalization] Failed: {e}")

                # 🔧 关键改进：硬标签保底策略（防止分布过于平均）
                primary_answer_lower = primary_answer.lower() if primary_answer else None

                if primary_answer_lower and primary_answer_lower in word_probs:
                    current_prob = word_probs[primary_answer_lower]

                    # 🔧 如果硬标签概率太低，强制提升到最小阈值
                    # 这确保了正确答案至少占有意义的概率
                    min_hard_label_prob = 0.25  # 最小25%概率

                    if current_prob < min_hard_label_prob:
                        # 从其他token的概率中"借"一部分给硬标签
                        deficit = min_hard_label_prob - current_prob
                        total_other_prob = sum(word_probs.values()) - current_prob

                        if total_other_prob > 0:
                            # 按比例从其他token扣减
                            reduction_ratio = min(deficit / total_other_prob, 0.5)  # 最多扣减50%

                            for word in list(word_probs.keys()):
                                if word != primary_answer_lower:
                                    word_probs[word] *= (1.0 - reduction_ratio)

                            # 提升硬标签概率
                            word_probs[primary_answer_lower] = min_hard_label_prob

                            self.logger.debug(
                                f"[Hard Label Boost] Boosted '{primary_answer_lower}' from {current_prob:.4f} to {min_hard_label_prob:.4f}"
                            )

                # 🔧 关键改进：Top-K过滤（防止分布过于稀疏）
                # 只保留概率最高的K个token，减少噪音
                sorted_items = sorted(word_probs.items(), key=lambda x: x[1], reverse=True)

                # 保留策略：
                # 1. 硬标签永远保留
                # 2. Top-20保留
                # 3. 概率>0.01的保留
                filtered_items = []
                for i, (word, prob) in enumerate(sorted_items):
                    # 硬标签强制保留
                    if primary_answer_lower and word == primary_answer_lower:
                        filtered_items.append((word, prob))
                        continue

                    # Top-20保留
                    if i < 20:
                        filtered_items.append((word, prob))
                        continue

                    # 高概率保留
                    if prob > 0.01:
                        filtered_items.append((word, prob))
                        continue

                # 构建 distribution（使用过滤后的items）
                for word, prob in filtered_items:
                    distribution[word] = prob

        # 方法2：如果没有 logits，使用 confidence 构建
        elif primary_answer and confidence:
            main_prob = min(confidence, 0.95)
            distribution[primary_answer.lower()] = main_prob
            remaining = 1.0 - main_prob
            if remaining > 0:
                distribution['other'] = remaining

        # 🔧 归一化分布
        if distribution:
            total_prob = sum(distribution.values())
            if total_prob > 0:
                distribution = {k: v / total_prob for k, v in distribution.items()}

        # ===== 🔧 新增：合并等价token的概率（如 '1' 和 'one'） =====
        if distribution and self.token_filter:
            distribution = self.token_filter.merge_equivalent_tokens(distribution)
            self.logger.debug(f"[Token Merge] After merging equivalent tokens: {len(distribution)} unique answers")

        # ===== 🔧 字符串层级二次过滤（可选，作为安全网） =====
        if distribution and self.token_filter:
            # 使用过滤器再次确认（确保万无一失）
            distribution = self.token_filter.filter_distribution(
                distribution=distribution,
                question=question,
                primary_answer=primary_answer,
                min_prob=0.001,
                max_answers=50
            )

            self.logger.debug(
                f"[Token Filter] After secondary filtering: {len(distribution)} tokens remaining, "
                f"primary_answer='{primary_answer}' with prob={distribution.get(primary_answer.lower(), 0):.4f}"
            )

        # ===== 🔧 第四层：任务适配过滤（白名单） =====
        # 根据问题类型应用白名单，过滤掉不属于该任务类型的token
        if distribution and primary_answer and question and self.token_filter:
            # 推断任务类型
            task_type = self.token_filter.infer_task_type(question, primary_answer)

            # 应用任务白名单过滤
            distribution = self.token_filter.filter_by_task_type(
                distribution=distribution,
                task_type=task_type,
                hard_label=primary_answer,
                preserve_hard_label=True
            )

            self.logger.info(
                f"[Task Filter] Task type: {task_type}, "
                f"tokens after filtering: {len(distribution)}, "
                f"primary_answer='{primary_answer}' with prob={distribution.get(primary_answer.lower(), 0):.4f}"
            )

        return distribution

    def generate_batch_soft_labels(
        self,
        batch_data: Dict[str, Any],
        tasks: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Generate soft labels for batch of images.

        Args:
            batch_data: Batch data dictionary
            tasks: Tasks to process

        Returns:
            Dictionary with soft labels per task
        """
        results = {task: [] for task in tasks}

        for img_data in batch_data['images']:
            image_id = img_data['id']
            image_path = img_data['path']

            self.logger.info(f"Processing image {image_id} for soft labels")

            if 'vqa' in tasks:
                questions = batch_data['annotations']['vqa'].get(image_id, [])
                for q_data in questions:
                    question = q_data.get('question', '')
                    soft_label = self.generate_vqa_soft_labels(
                        image_path=image_path,
                        question=question,
                        image_id=image_id
                    )
                    results['vqa'].append(soft_label)

        return results

    # ==================
    # 继承自基类的方法（已删除重复实现）
    # ==================
    # 以下方法已从基类 BaseSoftLabelGenerator 继承：
    # - _apply_temperature(): 应用温度缩放
    # - _get_top_k_probabilities(): 提取Top-K概率
    # - save_soft_labels(): 保存软标签到文件
    # - _make_serializable(): 转换为可序列化格式
    # - validate_soft_labels(): 验证数据有效性
    # - get_statistics(): 计算统计信息
    # - __repr__(): 字符串表示
    #
    # 如需VQA特定的验证逻辑，可覆盖validate_soft_labels方法：
    #
    # def validate_soft_labels(self, soft_labels: Dict[str, Any]) -> bool:
    #     """验证VQA软标签"""
    #     if not super().validate_soft_labels(soft_labels):
    #         return False
    #     return 'answer_distribution' in soft_labels

    def __repr__(self) -> str:
        """字符串表示"""
        return f"VQASoftLabelGenerator(teacher={self.teacher.model_name}, temp={self.temperature}, top_k={self.top_k})"


# 🔧 兼容性别名（保持向后兼容）
# 旧代码中使用 SoftLabelGenerator 的地方，会自动使用 VQASoftLabelGenerator
SoftLabelGenerator = VQASoftLabelGenerator