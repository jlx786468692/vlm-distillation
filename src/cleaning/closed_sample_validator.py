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
        校验A：三元自洽校验（hard_label / soft_label.primary_answer / cot.conclusion）

        校验目标：教师模型输出内部标签自洽，三者表达同一个候选答案

        流程：
        ① 三者归一字符串完全一致 → 通过
        ② 不一致，查询同义词词典全部等价 → 通过
        ③ 词典不匹配，调用MNLI两两语义等价校验；三组两两全部等价才算通过
        ④ 额外强约束：cot_norm必须存在于归一后的candidate_pool

        Args:
            hard_label_answer: 硬标签答案
            soft_label_primary_answer: 软标签主答案
            cot_conclusion: CoT结论
            candidate_pool: 候选池

        Returns:
            (是否一致, 不一致原因)
        """
        self.logger.info("【校验A】开始三元自洽校验...")

        # 归一化
        hard_norm = self.normalize(hard_label_answer)
        soft_norm = self.normalize(soft_label_primary_answer)
        cot_norm = self.normalize(cot_conclusion)

        self.logger.info(f"  - hard_label: {hard_norm}")
        self.logger.info(f"  - soft_label: {soft_norm}")
        self.logger.info(f"  - cot_conclusion: {cot_norm}")

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
        # ① 三者归一字符串完全一致 → 通过
        # ───────────────────────────────────────────────────────
        if hard_norm == soft_norm == cot_norm:
            self.logger.info("✓ 三元完全一致（字符串匹配），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # ② 查询同义词词典全部等价 → 通过
        # ───────────────────────────────────────────────────────
        hard_soft_synonym = self.check_synonym_equivalence(hard_norm, soft_norm)
        hard_cot_synonym = self.check_synonym_equivalence(hard_norm, cot_norm)
        soft_cot_synonym = self.check_synonym_equivalence(soft_norm, cot_norm)

        if hard_soft_synonym and hard_cot_synonym and soft_cot_synonym:
            self.logger.info("✓ 三元同义（词典匹配），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # ③ 调用MNLI两两语义等价校验
        # ───────────────────────────────────────────────────────
        self.logger.info("  三者不完全一致，调用MNLI语义等价校验...")

        # hard vs soft
        hard_soft_semantic = self.check_semantic_equivalence(hard_label_answer, soft_label_primary_answer)

        # hard vs cot
        hard_cot_semantic = self.check_semantic_equivalence(hard_label_answer, cot_conclusion)

        # soft vs cot
        soft_cot_semantic = self.check_semantic_equivalence(soft_label_primary_answer, cot_conclusion)

        # 三组两两全部等价才算通过
        if hard_soft_semantic and hard_cot_semantic and soft_cot_semantic:
            self.logger.info("✓ 三元语义等价（MNLI），校验通过")
            return True, ""

        # ───────────────────────────────────────────────────────
        # 不一致
        # ───────────────────────────────────────────────────────
        reasons = []
        if not hard_soft_synonym:
            reasons.append(f"hard≠soft: '{hard_norm}' vs '{soft_norm}'")
        if not hard_cot_synonym:
            reasons.append(f"hard≠cot: '{hard_norm}' vs '{cot_norm}'")
        if not soft_cot_synonym:
            reasons.append(f"soft≠cot: '{soft_norm}' vs '{cot_norm}'")

        reason = "三元不自洽: " + ", ".join(reasons)
        self.logger.warning(f"【校验A失败】{reason}")

        return False, reason

    # ============================================================
    # 校验B：GT真值与Hard标签语义一致性校验
    # ============================================================

    def validate_gt_hard_consistency(
        self,
        ground_truth: str,
        hard_label_answer: str
    ) -> Tuple[bool, str]:
        """
        校验B：GT真值与Hard标签语义一致性校验

        校验目标：教师生成硬标签与原始数据集标注答案相符

        流程：
        ① 字符串完全一致 → 通过
        ② 不一致，查同义词等价 → 通过
        ③ 调用MNLI判断语义等价（阈值>0.65）

        Args:
            ground_truth: 数据集标注答案
            hard_label_answer: 教师生成的硬标签

        Returns:
            (是否一致, 不一致原因)
        """
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
        完整校验闭合样本

        执行顺序：
        1. 校验A：三元自洽校验
        2. 校验B：GT真值与Hard标签校验

        Args:
            sample: 样本数据
            ground_truth: 数据集标注答案（可选）

        Returns:
            {
                'validation_a': {'is_consistent': bool, 'reason': str},
                'validation_b': {'is_consistent': bool, 'reason': str},
                'overall_valid': bool,
                'deductions': Dict
            }
        """
        vqa_data = sample.get('tasks', {}).get('vqa', {})

        result = {
            'validation_a': {'is_consistent': True, 'reason': ''},
            'validation_b': {'is_consistent': True, 'reason': ''},
            'overall_valid': True,
            'deductions': {}
        }

        # ───────────────────────────────────────────────────────
        # 校验A：三元自洽校验
        # ───────────────────────────────────────────────────────
        hard_label_answer = vqa_data.get('hard_label', {}).get('answer', '')
        soft_label_primary_answer = vqa_data.get('soft_label', {}).get('primary_answer', '')
        cot_conclusion = vqa_data.get('cot_reasoning', {}).get('structured_reasoning', {}).get('conclusion', '')
        candidate_pool = vqa_data.get('candidate_pool', [])

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

        # ───────────────────────────────────────────────────────
        # 校验B：GT真值与Hard标签校验
        # ───────────────────────────────────────────────────────
        if ground_truth:
            is_consistent_b, reason_b = self.validate_gt_hard_consistency(
                ground_truth,
                hard_label_answer
            )

            result['validation_b'] = {
                'is_consistent': is_consistent_b,
                'reason': reason_b
            }

            # 处罚
            if not is_consistent_b:
                if self.strict_closed_mode:
                    # 一票否决
                    result['deductions']['validation_b'] = {
                        'veto': True,
                        'reason': reason_b
                    }
                else:
                    # 重度扣分
                    result['deductions']['validation_b'] = {
                        'deduction': 20,
                        'reason': reason_b
                    }

                result['overall_valid'] = False

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
    print("  B. GT真值与Hard标签校验")

    print("\n开关控制：")
    print("  strict_closed_mode=false（默认）：重度扣分")
    print("  strict_closed_mode=true（严格）：一票否决")

    print("\n扣分规则：")
    print("  A校验失败：-22分")
    print("  B校验失败：-20分")

    print("\n" + "="*70)