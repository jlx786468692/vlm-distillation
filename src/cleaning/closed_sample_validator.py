"""
闭合样本校验模块（官方标准）
============================

实现两条新增校验规则：
- 校验A：三元自洽校验（hard_label / soft_label.primary_answer / cot.conclusion）
- 校验B：GT真值与Hard标签语义一致性校验

支持 strict_closed_mode 开关控制是否升级为一票否决
"""

import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path


class ClosedSampleValidator:
    """
    闭合样本校验器

    执行校验A和校验B
    """

    def __init__(self, config: Optional[Any] = None, logger: Optional[logging.Logger] = None):
        """
        初始化校验器

        Args:
            config: 配置管理器
            logger: 日志记录器（可选，如果不提供则使用模块默认logger）
        """
        # 🔧 修复：优先使用传入的 logger，确保日志统一
        self.logger = logger if logger else logging.getLogger(__name__)
        self.config = config

        # 从配置读取 strict_closed_mode 开关
        if self.config:
            self.strict_closed_mode = self.config.get('cleaning.strict_closed_mode', False)
        else:
            self.strict_closed_mode = False  # 默认：重度扣分，不一票否决

        # 初始化同义词词典
        self._init_synonym_dict()

        # MNLI模型（延迟加载）
        self.nli_model = None
        self.nli_tokenizer = None
        self._nli_loaded = False

        self.logger.info("✓ 闭合样本校验器初始化完成")
        self.logger.info(f"  - strict_closed_mode: {self.strict_closed_mode}")

    def _init_synonym_dict(self):
        """
        初始化同义词词典
        """
        # 内置同义词词典（常见等价词）
        self.synonym_dict = {
            # 颜色同义词
            'gray': 'grey',
            'grey': 'gray',
            'color': 'colour',

            # 数字同义词
            'one': '1',
            'two': '2',
            'three': '3',
            'four': '4',
            'five': '5',

            # 常见同义词
            'yes': 'true',
            'no': 'false',
            'true': 'yes',
            'false': 'no',
        }

        self.logger.debug(f"同义词词典: {len(self.synonym_dict)} 组")

    def normalize(self, text: str) -> str:
        """
        文本归一化

        流程：
        1. 小写
        2. 去首尾空格
        3. 清除标点

        Args:
            text: 原始文本

        Returns:
            归一化后的文本
        """
        if not text:
            return ''

        # 小写
        text = text.lower()

        # 去首尾空格
        text = text.strip()

        # 清除标点（只保留字母、数字、空格）
        text = re.sub(r'[^\w\s]', '', text)

        # 再次去除多余空格
        text = ' '.join(text.split())

        return text

    def check_synonym_equivalence(self, text1: str, text2: str) -> bool:
        """
        检查两个文本是否同义

        流程：
        1. 归一化
        2. 查询同义词词典

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            是否同义
        """
        norm1 = self.normalize(text1)
        norm2 = self.normalize(text2)

        # 完全一致
        if norm1 == norm2:
            return True

        # 查询同义词词典
        # 检查 norm1 的同义词是否等于 norm2
        if norm1 in self.synonym_dict:
            if self.synonym_dict[norm1] == norm2:
                return True

        if norm2 in self.synonym_dict:
            if self.synonym_dict[norm2] == norm1:
                return True

        return False

    def _load_nli_model(self):
        """
        加载MNLI模型（延迟加载）
        """
        if self._nli_loaded:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            model_path = "models/bart-large-mnli"

            self.logger.info(f"加载MNLI模型: {model_path}")

            self.nli_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_path)

            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                self.nli_model = self.nli_model.to('cuda')

            self._nli_loaded = True
            self.logger.info("✓ MNLI模型加载成功")

        except Exception as e:
            self.logger.warning(f"MNLI模型加载失败: {e}")
            self._nli_loaded = True

    def check_semantic_equivalence(self, text1: str, text2: str, threshold: float = 0.65) -> bool:
        """
        检查两个文本语义等价（使用MNLI）

        Args:
            text1: 文本1
            text2: 文本2
            threshold: entailment阈值（默认0.65）

        Returns:
            是否语义等价
        """
        # 先检查同义词词典
        if self.check_synonym_equivalence(text1, text2):
            return True

        # 延迟加载MNLI模型
        if not self._nli_loaded:
            self._load_nli_model()

        # 如果模型加载失败，降级为字符串匹配
        if self.nli_model is None or self.nli_tokenizer is None:
            self.logger.warning("MNLI模型不可用，降级为字符串匹配")
            return self.normalize(text1) == self.normalize(text2)

        try:
            import torch

            # 构造输入
            inputs = self.nli_tokenizer(
                text1,
                text2,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )

            # 移动到设备
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                logits = outputs.logits

                # 获取entailment概率
                # BART-MNLI: [contradiction, neutral, entailment]
                probs = torch.softmax(logits, dim=-1)
                entailment_prob = probs[0, 2].item()  # entailment在第2位

            self.logger.debug(f"MNLI entailment: {entailment_prob:.3f}")

            return entailment_prob > threshold

        except Exception as e:
            self.logger.warning(f"MNLI推理失败: {e}，降级为字符串匹配")
            return self.normalize(text1) == self.normalize(text2)

    # ============================================================
    # 校验A：三元自洽校验
    # ============================================================

    def validate_triple_consistency(
        self,
        hard_label_answer: str,
        soft_label_primary_answer: str,
        cot_conclusion: str,
        candidate_pool: List[str]
    ) -> Tuple[bool, str]:
        """
        校验A：三元自洽校验（以硬标签为基准）

        🔧 修改：以硬标签（GT）为基准，检查软标签和CoT是否与硬标签一致
        - hard_label.answer vs soft_label.primary_answer（语义相同即可）
        - hard_label.answer vs cot.answer（语义相同即可）

        流程：
        ① 归一化字符串完全一致 → 通过
        ② 不一致，查询同义词词典等价 → 通过
        ③ 词典不匹配，调用MNLI语义等价校验 → 通过
        ④ 额外强约束：cot_norm必须存在于归一后的candidate_pool

        Args:
            hard_label_answer: 硬标签答案（GT，基准）
            soft_label_primary_answer: 软标签主答案
            cot_conclusion: CoT结论
            candidate_pool: 候选池

        Returns:
            (是否一致, 不一致原因)
        """
        self.logger.info("【校验A】开始以硬标签为基准的一致性校验...")

        # 归一化
        hard_norm = self.normalize(hard_label_answer)
        soft_norm = self.normalize(soft_label_primary_answer)
        cot_norm = self.normalize(cot_conclusion)

        self.logger.info(f"  - 硬标签(GT): {hard_norm}")
        self.logger.info(f"  - 软标签: {soft_norm}")
        self.logger.info(f"  - CoT结论: {cot_norm}")

        # ───────────────────────────────────────────────────────
        # ④ 强约束：cot_norm 必须存在于归一后的 candidate_pool
        # ───────────────────────────────────────────────────────
        if candidate_pool:
            candidate_pool_norm = [self.normalize(c) for c in candidate_pool]

            if cot_norm not in candidate_pool_norm:
                reason = f"CoT结论不在候选池内: '{cot_norm}' not in {candidate_pool_norm}"
                self.logger.warning(f"【校验A失败】{reason}")
                return False, reason

        # ───────────────────────────────────────────────────────
        # 🔧 修改：以硬标签为基准，检查两项一致性
        # ───────────────────────────────────────────────────────
        # 1. hard vs soft（必须通过）
        # 2. hard vs cot（必须通过）
        # ───────────────────────────────────────────────────────

        # ① 字符串完全一致检查
        hard_soft_match = (hard_norm == soft_norm)
        hard_cot_match = (hard_norm == cot_norm)

        if hard_soft_match and hard_cot_match:
            self.logger.info("✓ 软标签和CoT均与硬标签完全一致（字符串匹配），校验通过")
            return True, ""

        # ② 查询同义词词典等价
        if not hard_soft_match:
            hard_soft_match = self.check_synonym_equivalence(hard_norm, soft_norm)

        if not hard_cot_match:
            hard_cot_match = self.check_synonym_equivalence(hard_norm, cot_norm)

        if hard_soft_match and hard_cot_match:
            self.logger.info("✓ 软标签和CoT均与硬标签同义（词典匹配），校验通过")
            return True, ""

        # ③ 调用MNLI语义等价校验
        self.logger.info("  部分不一致，调用MNLI语义等价校验...")

        # hard vs soft
        if not hard_soft_match:
            hard_soft_match = self.check_semantic_equivalence(hard_label_answer, soft_label_primary_answer)

        # hard vs cot
        if not hard_cot_match:
            hard_cot_match = self.check_semantic_equivalence(hard_label_answer, cot_conclusion)

        # 两项都通过才算校验通过
        if hard_soft_match and hard_cot_match:
            self.logger.info("✓ 软标签和CoT均与硬标签语义等价（MNLI），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # 不一致
        # ───────────────────────────────────────────────────────
        reasons = []
        if not hard_soft_match:
            reasons.append(f"硬标签≠软标签: '{hard_norm}' vs '{soft_norm}'")
        if not hard_cot_match:
            reasons.append(f"硬标签≠CoT: '{hard_norm}' vs '{cot_norm}'")

        reason = "与硬标签不一致: " + ", ".join(reasons)
        self.logger.warning(f"【校验A失败】{reason}")

        return False, reason

    # ============================================================
    # 校验B：GT真值与Hard标签语义一致性校验
    # ============================================================

    def parse_answer_to_set(self, answer: str) -> set:
        """
        将答案拆解为集合（支持多答案情况）

        示例：
        - "red" → {red}
        - "red and black" → {red, black}
        - "red, blue, green" → {red, blue, green}
        - "red or blue" → {red, blue}

        Args:
            answer: 答案文本

        Returns:
            答案集合
        """
        # 归一化
        answer_norm = self.normalize(answer)

        if not answer_norm:
            return set()

        # 拆分关键词
        split_keywords = [
            ' and ',   # red and black
            ' or ',    # red or blue
            ',',       # red, blue, green
            ';',       # red; blue
            '/'        # red/blue
        ]

        # 尝试拆分
        parts = [answer_norm]
        for keyword in split_keywords:
            if keyword in answer_norm:
                parts = answer_norm.split(keyword)
                break

        # 清理并去重
        result = set()
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                # 同义词归一化
                normalized = self.synonym_dict.get(cleaned, cleaned)
                result.add(normalized)

        return result

    def validate_gt_hard_consistency(
        self,
        ground_truth: str,
        hard_label_answer: str
    ) -> Tuple[bool, str]:
        """
        校验B：GT真值与Hard标签语义一致性校验（支持多答案）

        🔧 已废弃：硬标签现在直接使用COCO标注，此校验不再需要
        - 硬标签来源：直接从COCO标注获取（ground truth）
        - 检验B目的：验证教师模型预测的硬标签与标注一致性
        - 现状：硬标签本身就是标注，检验B永远通过
        - 保留原因：历史记录和可能的回退场景

        校验目标：教师生成硬标签与原始数据集标注答案相符

        🔧 改进：答案集合拆解 + 子集判断

        规则（行业通用，大量COCO VQA蒸馏项目在用）：
        ✅ 允许：预测集合 ⊆ GT集合（只漏颜色，不新增不存在颜色）
        ❌ 拒绝：预测集合包含GT不存在的颜色（凭空多出颜色）
        ❌ 拒绝：GT是单色，预测输出多色（无中生有增加属性）

        示例：
        - "red and black" vs "red" → ✅ 通过（{red} ⊆ {red, black}）
        - "red and black" vs "red, yellow" → ❌ 拒绝（yellow不在GT中）
        - "red" vs "red and black" → ❌ 拒绝（GT是单色，预测多色）

        Args:
            ground_truth: 数据集标注答案
            hard_label_answer: 教师生成的硬标签

        Returns:
            (是否一致, 不一致原因)
        """
        self.logger.warning("【校验B】已废弃 - 硬标签直接使用COCO标注，此校验不再需要")
        # 直接返回通过（因为硬标签本身就是标注）
        return True, ""
        self.logger.info("【校验B】开始GT与Hard一致性校验...")

        # 归一化
        gt_norm = self.normalize(ground_truth)
        hard_norm = self.normalize(hard_label_answer)

        self.logger.info(f"  - ground_truth: {gt_norm}")
        self.logger.info(f"  - hard_label: {hard_norm}")

        # ───────────────────────────────────────────────────────
        # ① 字符串完全一致 → 通过
        # ───────────────────────────────────────────────────────
        if gt_norm == hard_norm:
            self.logger.info("✓ GT与Hard完全一致（字符串匹配），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # ② 查同义词等价 → 通过
        # ───────────────────────────────────────────────────────
        if self.check_synonym_equivalence(gt_norm, hard_norm):
            self.logger.info("✓ GT与Hard同义（词典匹配），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # 🔧 新增：答案集合拆解 + 子集判断
        # ───────────────────────────────────────────────────────
        gt_set = self.parse_answer_to_set(ground_truth)
        hard_set = self.parse_answer_to_set(hard_label_answer)

        self.logger.info(f"  - GT集合: {gt_set}")
        self.logger.info(f"  - Hard集合: {hard_set}")

        # 规则1：✅ 预测集合 ⊆ GT集合 → 通过
        if hard_set.issubset(gt_set):
            self.logger.info(f"✓ Hard集合是GT集合的子集，校验通过")
            return True, ""

        # 规则2：❌ GT是单色，预测是多色 → 拒绝（无中生有）
        if len(gt_set) == 1 and len(hard_set) > 1:
            reason = f"GT是单色'{gt_set}'，预测多色'{hard_set}'（无中生有）"
            self.logger.warning(f"【校验B失败】{reason}")
            return False, reason

        # 规则3：❌ 预测集合包含GT不存在的颜色 → 拒绝（凭空多出颜色）
        extra_colors = hard_set - gt_set
        if extra_colors:
            reason = f"预测包含GT不存在的颜色: {extra_colors}（凭空多出颜色）"
            self.logger.warning(f"【校验B失败】{reason}")
            return False, reason

        # ───────────────────────────────────────────────────────
        # ③ 调用MNLI语义等价（阈值>0.65）
        # ───────────────────────────────────────────────────────
        if self.check_semantic_equivalence(ground_truth, hard_label_answer, threshold=0.65):
            self.logger.info("✓ GT与Hard语义等价（MNLI），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # 不一致
        # ───────────────────────────────────────────────────────
        reason = f"GT与Hard语义不等价: '{gt_norm}' vs '{hard_norm}'"
        self.logger.warning(f"【校验B失败】{reason}")

        return False, reason

    # ============================================================
    # 完整校验流程
    # ============================================================

    def validate_closed_sample(
        self,
        sample: Dict,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        完整校验样本（区分闭合问题和开放问题）

        执行顺序：
        1. 判断问题类型（根据candidate_pool是否存在）
        2. 闭合问题：三元校验（hard_label、soft_label、cot）
        3. 开放问题：两元校验（hard_label、cot）

        Args:
            sample: 样本数据
            ground_truth: 数据集标注答案（可选）

        Returns:
            {
                'validation_a': {'is_consistent': bool, 'reason': str},
                'validation_b': {'is_consistent': bool, 'reason': str},
                'overall_valid': bool,
                'deductions': Dict,
                'question_type': str  # 'closed' or 'open'
            }
        """
        vqa_data = sample.get('tasks', {}).get('vqa', {})

        result = {
            'validation_a': {'is_consistent': True, 'reason': ''},
            'validation_b': {'is_consistent': True, 'reason': ''},
            'overall_valid': True,
            'deductions': {},
            'question_type': 'closed'  # 默认闭合问题
        }

        # ───────────────────────────────────────────────────────
        # 判断问题类型（根据candidate_pool）
        # ───────────────────────────────────────────────────────
        candidate_pool = vqa_data.get('candidate_pool', [])
        is_closed_question = bool(candidate_pool)  # 有候选池 = 闭合问题

        hard_label_answer = vqa_data.get('hard_label', {}).get('answer', '')
        soft_label_primary_answer = vqa_data.get('soft_label', {}).get('primary_answer', '')
        cot_conclusion = vqa_data.get('cot_reasoning', {}).get('answer', '')

        # ───────────────────────────────────────────────────────
        # 校验A：根据问题类型执行不同校验
        # ───────────────────────────────────────────────────────
        if is_closed_question:
            # 闭合问题：三元校验（hard_label、soft_label、cot）
            result['question_type'] = 'closed'
            self.logger.info("[校验A] 闭合问题：执行三元校验（hard_label、soft_label、cot）")

            is_consistent_a, reason_a = self.validate_triple_consistency(
                hard_label_answer,
                soft_label_primary_answer,
                cot_conclusion,
                candidate_pool
            )

            result['validation_a'] = {
                'is_consistent': is_consistent_a,
                'reason': reason_a
            }

            # 处罚
            if not is_consistent_a:
                if self.strict_closed_mode:
                    # 一票否决
                    result['deductions']['validation_a'] = {
                        'veto': True,
                        'reason': reason_a
                    }
                else:
                    # 重度扣分
                    result['deductions']['validation_a'] = {
                        'deduction': 22,
                        'reason': reason_a
                    }

                result['overall_valid'] = False

        else:
            # 开放问题：只校验hard_label和cot（不校验soft_label）
            result['question_type'] = 'open'
            self.logger.info("[校验A] 开放问题：只校验hard_label和cot（不校验soft_label）")

            # 只校验hard_label和cot是否一致
            hard_norm = self.normalize(hard_label_answer)
            cot_norm = self.normalize(cot_conclusion)

            self.logger.info(f"  - hard_label: {hard_norm}")
            self.logger.info(f"  - cot: {cot_norm}")
            self.logger.info(f"  - soft_label.primary_answer: {self.normalize(soft_label_primary_answer)}（不校验）")

            # 检查hard_label和cot是否一致
            is_consistent = self.check_synonym_equivalence(hard_label_answer, cot_conclusion)

            if not is_consistent:
                # 如果不一致，尝试MNLI语义等价校验
                is_consistent = self.check_semantic_equivalence(hard_label_answer, cot_conclusion)

            if is_consistent:
                self.logger.info("✓ hard_label和cot一致，校验通过")
                result['validation_a'] = {
                    'is_consistent': True,
                    'reason': ''
                }
            else:
                reason = f"hard_label与cot不一致: '{hard_norm}' vs '{cot_norm}'"
                self.logger.warning(f"【校验A失败】{reason}")
                result['validation_a'] = {
                    'is_consistent': False,
                    'reason': reason
                }

                # 处罚（比闭合问题轻）
                if self.strict_closed_mode:
                    result['deductions']['validation_a'] = {
                        'veto': True,
                        'reason': reason
                    }
                else:
                    # 中度扣分（比闭合问题的22分轻）
                    result['deductions']['validation_a'] = {
                        'deduction': 15,
                        'reason': reason
                    }

                result['overall_valid'] = False

        # ───────────────────────────────────────────────────────
        # 校验B：GT真值与Hard标签校验
        # 🔧 新方案：硬标签直接使用COCO标注，检验B不再需要
        # ───────────────────────────────────────────────────────
        # if ground_truth:
        #     is_consistent_b, reason_b = self.validate_gt_hard_consistency(
        #         ground_truth,
        #         hard_label_answer
        #     )
        #
        #     result['validation_b'] = {
        #         'is_consistent': is_consistent_b,
        #         'reason': reason_b
        #     }
        #
        #     # 处罚
        #     if not is_consistent_b:
        #         if self.strict_closed_mode:
        #             # 一票否决
        #             result['deductions']['validation_b'] = {
        #                 'veto': True,
        #                 'reason': reason_b
        #             }
        #         else:
        #             # 重度扣分
        #             result['deductions']['validation_b'] = {
        #                 'deduction': 20,
        #                 'reason': reason_b
        #             }
        #
        #         result['overall_valid'] = False
        #
        # self.logger.info("【校验B】已跳过（硬标签直接使用COCO标注，无需检验）")

        return result


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("闭合样本校验器测试")
    print("="*70)

    print("\n校验规则：")
    print("  A. 三元自洽校验（hard/soft/cot）")
    print("  B. GT真值与Hard标签校验（支持多答案）")

    print("\n开关控制：")
    print("  strict_closed_mode=false（默认）：重度扣分")
    print("  strict_closed_mode=true（严格）：一票否决")

    print("\n扣分规则：")
    print("  A校验失败：-22分")
    print("  B校验失败：-20分")

    print("\n校验B多答案规则：")
    print("  ✅ 预测集合 ⊆ GT集合 → 通过（允许漏检）")
    print("  ❌ 预测包含GT不存在的颜色 → 拒绝（凭空多出）")
    print("  ❌ GT单色，预测多色 → 拒绝（无中生有）")

    print("\n" + "="*70)

    # 🔧 新增：多答案校验测试
    print("\n多答案校验测试：")

    validator = ClosedSampleValidator()

    # 测试1：子集规则（✅ 通过）
    gt1 = "red and black"
    hard1 = "red"
    is_valid1, reason1 = validator.validate_gt_hard_consistency(gt1, hard1)
    status1 = "✓" if is_valid1 else "✗"
    print(f"\n{status1} 测试1: GT='{gt1}', Hard='{hard1}'")
    print(f"  结果: {'通过' if is_valid1 else '拒绝'}")
    if not is_valid1:
        print(f"  原因: {reason1}")

    # 测试2：凭空多出颜色（❌ 拒绝）
    gt2 = "red and black"
    hard2 = "red and yellow"
    is_valid2, reason2 = validator.validate_gt_hard_consistency(gt2, hard2)
    status2 = "✓" if is_valid2 else "✗"
    print(f"\n{status2} 测试2: GT='{gt2}', Hard='{hard2}'")
    print(f"  结果: {'通过' if is_valid2 else '拒绝'}")
    if not is_valid2:
        print(f"  原因: {reason2}")

    # 测试3：无中生有（❌ 拒绝）
    gt3 = "red"
    hard3 = "red and black"
    is_valid3, reason3 = validator.validate_gt_hard_consistency(gt3, hard3)
    status3 = "✓" if is_valid3 else "✗"
    print(f"\n{status3} 测试3: GT='{gt3}', Hard='{hard3}'")
    print(f"  结果: {'通过' if is_valid3 else '拒绝'}")
    if not is_valid3:
        print(f"  原因: {reason3}")

    print("\n" + "="*70)