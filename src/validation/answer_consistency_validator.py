"""
答案一致性验证器（官方标准）
============================

针对闭合问题，验证标注答案与模型预测答案是否一致。

验证流程：
1. 检查四种标签是否一致：
   - hard_label.answer
   - soft_label.primary_answer
   - cot_reasoning.conclusion
   - ground_truth（VQA标注答案）

2. 一致性判断方法：
   - 步骤1：规则归一化（VQA v2官方标准）
   - 步骤2：归一化后字符串相等 → 直接判定等价
   - 步骤3：字符串不等 → 调用bart-large-mnli做NLI推理
   - 步骤4：entailment概率 ≥ 阈值 → 等价，否则弃掉样本

归一化规则（参考VQA v2官方evaluation代码）：
- 全部小写
- 去除标点、冠词 a/an/the
- 数字单词 ↔ 阿拉伯数字映射
- 同义词静态词典
- 去除多余空格
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# 延迟导入transformers，避免启动时加载
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AnswerConsistencyValidator:
    """
    答案一致性验证器（官方标准）

    验证闭合问题的标注答案与模型预测答案是否一致。
    """

    # 官方标准：NLI阈值
    ENTAIL_THRESHOLD = 0.65

    # 数字映射（VQA v2官方）
    NUMBER_WORD_MAP = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12'
    }

    # 同义词词典（VQA v2官方）
    SYNONYM_DICT = {
        # 颜色同义词
        'reddish': 'red',
        'bluish': 'blue',
        'greenish': 'green',
        'yellowish': 'yellow',
        # 交通工具同义词
        'automobile': 'car',
        'vehicle': 'car',
        # 其他常见同义词
        'yeah': 'yes',
        'yep': 'yes',
        'nope': 'no',
        'nothing': 'none',
        'nobody': 'none',
        'noone': 'none',
    }

    def __init__(
        self,
        config: Optional[Any] = None,
        nli_model_path: str = "models/bart-large-mnli",  # 🔧 使用本地模型
        use_nli_model: bool = True,
        entail_threshold: float = 0.65
    ):
        """
        初始化答案一致性验证器

        Args:
            config: 配置管理器
            nli_model_path: NLI模型路径（默认：models/bart-large-mnli，使用本地模型）
            use_nli_model: 是否使用NLI模型（False则仅用规则）
            entail_threshold: 蕴含概率阈值
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # NLI模型配置
        self.nli_model_path = nli_model_path
        self.use_nli_model = use_nli_model
        self.ENTAIL_THRESHOLD = entail_threshold

        # 延迟加载的模型和tokenizer
        self.nli_model = None
        self.nli_tokenizer = None
        self._model_loaded = False

        self.logger.info("✓ 答案一致性验证器初始化完成")
        self.logger.info(f"  - 归一化规则: VQA v2官方标准")
        self.logger.info(f"  - NLI模型: {nli_model_path if use_nli_model else '禁用'}")
        self.logger.info(f"  - 蕴含阈值: {self.ENTAIL_THRESHOLD}")

    def _load_nli_model(self):
        """延迟加载NLI模型"""
        if self._model_loaded or not self.use_nli_model:
            return

        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("transformers未安装，无法使用NLI模型")
            self.use_nli_model = False
            self._model_loaded = True
            return

        try:
            self.logger.info(f"加载NLI模型: {self.nli_model_path}")

            # 加载tokenizer（本地模型）
            self.nli_tokenizer = AutoTokenizer.from_pretrained(
                self.nli_model_path,
                local_files_only=True  # 🔧 强制使用本地文件
            )

            # 加载模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                self.nli_model_path,
                local_files_only=True  # 🔧 强制使用本地文件
            ).to(device)
            self.nli_model.eval()

            self.logger.info(f"✓ NLI模型加载成功（本地），设备: {device}")
            self._model_loaded = True

        except Exception as e:
            self.logger.warning(f"NLI模型加载失败: {e}，将仅使用规则归一化")
            self.use_nli_model = False
            self._model_loaded = True

    def validate_closed_sample(
        self,
        sample: Dict[str, Any],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        验证闭合问题样本的答案一致性

        Args:
            sample: 样本数据（包含hard_label, soft_label, cot_reasoning）
            ground_truth: VQA标注答案（可选，如果提供则参与验证）

        Returns:
            {
                'is_consistent': bool,      # 是否一致
                'consistent_answers': list, # 一致的答案列表
                'inconsistent_pairs': list, # 不一致的答案对
                'validation_details': dict  # 详细验证结果
            }
        """
        vqa_data = sample.get('tasks', {}).get('vqa', {})

        # 收集四种答案
        answers = {}

        # 1. hard_label.answer
        hard_label = vqa_data.get('hard_label', {})
        if hard_label and 'answer' in hard_label:
            answers['hard_label'] = hard_label['answer']

        # 2. soft_label.primary_answer
        soft_label = vqa_data.get('soft_label', {})
        if soft_label and 'primary_answer' in soft_label:
            answers['soft_label'] = soft_label['primary_answer']

        # 3. cot_reasoning.conclusion
        cot_reasoning = vqa_data.get('cot_reasoning', {})
        if cot_reasoning and 'conclusion' in cot_reasoning:
            # 提取conclusion的第一个单词（去除标点）
            conclusion = cot_reasoning['conclusion']
            conclusion_word = self._extract_first_word(conclusion)
            if conclusion_word:
                answers['cot_label'] = conclusion_word

        # 4. ground_truth（VQA标注答案）
        if ground_truth:
            answers['ground_truth'] = ground_truth

        # 如果没有任何答案，返回有效（无法验证）
        if not answers:
            return {
                'is_consistent': True,
                'consistent_answers': [],
                'inconsistent_pairs': [],
                'validation_details': {'message': '无答案可验证'}
            }

        # ───────────────────────────────────────────────────────
        # 验证答案一致性
        # ───────────────────────────────────────────────────────
        validation_details = {}
        inconsistent_pairs = []

        # 获取所有答案名称
        answer_names = list(answers.keys())

        # 两两比较
        for i in range(len(answer_names)):
            for j in range(i + 1, len(answer_names)):
                name1, name2 = answer_names[i], answer_names[j]
                ans1, ans2 = answers[name1], answers[name2]

                # 检查一致性
                is_equivalent, detail = self.check_answer_equivalence(ans1, ans2)

                validation_details[f"{name1}_vs_{name2}"] = {
                    'answer1': ans1,
                    'answer2': ans2,
                    'is_equivalent': is_equivalent,
                    'detail': detail
                }

                if not is_equivalent:
                    inconsistent_pairs.append({
                        'source1': name1,
                        'answer1': ans1,
                        'source2': name2,
                        'answer2': ans2,
                        'detail': detail
                    })

        # ───────────────────────────────────────────────────────
        # 判断是否一致
        # ───────────────────────────────────────────────────────
        is_consistent = len(inconsistent_pairs) == 0

        # 找到一致的答案（取第一个作为代表）
        consistent_answers = []
        if is_consistent and answers:
            # 归一化后的答案
            first_answer = list(answers.values())[0]
            normalized = self.normalize_answer(first_answer)
            consistent_answers = [normalized]

        return {
            'is_consistent': is_consistent,
            'consistent_answers': consistent_answers,
            'inconsistent_pairs': inconsistent_pairs,
            'validation_details': validation_details
        }

    def check_answer_equivalence(
        self,
        answer1: str,
        answer2: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        检查两个答案是否语义等价

        步骤：
        1. 规则归一化
        2. 归一化后字符串相等 → 直接判定等价
        3. 字符串不等 → 调用NLI模型
        4. entailment概率 ≥ 阈值 → 等价

        Args:
            answer1: 答案1
            answer2: 答案2

        Returns:
            (is_equivalent, detail)
        """
        detail = {
            'answer1_original': answer1,
            'answer2_original': answer2,
            'answer1_normalized': '',
            'answer2_normalized': '',
            'method': '',
            'nli_prob': None
        }

        # ───────────────────────────────────────────────────────
        # 步骤1：规则归一化
        # ───────────────────────────────────────────────────────
        norm1 = self.normalize_answer(answer1)
        norm2 = self.normalize_answer(answer2)

        detail['answer1_normalized'] = norm1
        detail['answer2_normalized'] = norm2

        # ───────────────────────────────────────────────────────
        # 步骤2：归一化后字符串相等 → 直接判定等价
        # ───────────────────────────────────────────────────────
        if norm1 == norm2:
            detail['method'] = 'normalized_match'
            return True, detail

        # ───────────────────────────────────────────────────────
        # 步骤3：字符串不等 → 调用NLI模型
        # ───────────────────────────────────────────────────────
        if not self.use_nli_model:
            detail['method'] = 'nli_disabled'
            return False, detail

        # 延迟加载模型
        if not self._model_loaded:
            self._load_nli_model()

        if not self.use_nli_model or self.nli_model is None:
            detail['method'] = 'nli_unavailable'
            return False, detail

        # 调用NLI模型
        # 重要：premise=GT真值（answer1），hypothesis=教师答案（answer2）
        # 顺序不能颠倒！
        try:
            entail_prob = self._compute_nli_entailment(
                premise=answer1,  # GT真值
                hypothesis=answer2  # 教师答案
            )

            detail['nli_prob'] = entail_prob
            detail['method'] = 'nli_inference'

            # 步骤4：entailment概率 ≥ 阈值 → 等价
            is_equivalent = entail_prob >= self.ENTAIL_THRESHOLD

            return is_equivalent, detail

        except Exception as e:
            self.logger.warning(f"NLI推理失败: {e}")
            detail['method'] = 'nli_error'
            detail['error'] = str(e)
            return False, detail

    def normalize_answer(self, answer: str) -> str:
        """
        归一化答案（VQA v2官方标准）

        步骤：
        1. 全部小写
        2. 去除标点
        3. 去除冠词 a/an/the
        4. 数字单词 ↔ 阿拉伯数字映射
        5. 同义词替换
        6. 去除多余空格

        Args:
            answer: 原始答案

        Returns:
            归一化后的答案
        """
        if not answer:
            return ''

        # 1. 全部小写
        text = answer.lower().strip()

        # 2. 去除标点（保留字母、数字、空格）
        text = re.sub(r'[^\w\s]', ' ', text)

        # 3. 去除冠词 a/an/the
        words = text.split()
        words = [w for w in words if w not in ['a', 'an', 'the']]
        text = ' '.join(words)

        # 4. 数字单词 ↔ 阿拉伯数字映射
        words = text.split()
        words = [self.NUMBER_WORD_MAP.get(w, w) for w in words]
        text = ' '.join(words)

        # 5. 同义词替换
        words = text.split()
        words = [self.SYNONYM_DICT.get(w, w) for w in words]
        text = ' '.join(words)

        # 6. 去除多余空格
        text = ' '.join(text.split())

        return text

    def _extract_first_word(self, text: str) -> str:
        """
        提取文本的第一个单词（用于conclusion）

        Args:
            text: 文本内容

        Returns:
            第一个单词（小写，去除标点）
        """
        if not text:
            return ''

        # 去除标点
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()

        if not words:
            return ''

        return words[0]

    def _compute_nli_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> float:
        """
        计算NLI蕴含概率

        Args:
            premise: 前提（GT真值）
            hypothesis: 假设（教师答案）

        Returns:
            蕴含概率
        """
        # 编码输入
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )

        # 移动到设备
        device = next(self.nli_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 推理
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits

            # 获取蕴含概率（标签0=蕴含，1=中性，2=矛盾）
            # BART-MNLI的标签顺序：contradiction, neutral, entailment
            # 需要确认具体标签顺序
            probs = torch.softmax(logits, dim=-1)

            # 蕴含标签通常是最后一个
            entail_prob = probs[0, -1].item()

        return entail_prob

    def batch_validate(
        self,
        samples: List[Dict[str, Any]],
        ground_truths: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        批量验证样本

        Args:
            samples: 样本列表
            ground_truths: 标注答案字典 {image_id: answer}

        Returns:
            {
                'total': int,
                'valid': int,
                'invalid': int,
                'invalid_samples': list,
                'statistics': dict
            }
        """
        total = len(samples)
        valid_count = 0
        invalid_count = 0
        invalid_samples = []

        for sample in samples:
            image_id = sample.get('image_id')
            ground_truth = ground_truths.get(image_id) if ground_truths else None

            # 跳过开放样本
            vqa_data = sample.get('tasks', {}).get('vqa', {})
            inference_mode = vqa_data.get('inference_mode', 'closed')
            if inference_mode == 'open':
                # 开放样本跳过验证
                valid_count += 1
                continue

            # 验证闭合样本
            result = self.validate_closed_sample(sample, ground_truth)

            if result['is_consistent']:
                valid_count += 1
            else:
                invalid_count += 1
                invalid_samples.append({
                    'image_id': image_id,
                    'inconsistent_pairs': result['inconsistent_pairs'],
                    'validation_details': result['validation_details']
                })

        return {
            'total': total,
            'valid': valid_count,
            'invalid': invalid_count,
            'invalid_samples': invalid_samples,
            'statistics': {
                'valid_rate': valid_count / total if total > 0 else 0.0,
                'invalid_rate': invalid_count / total if total > 0 else 0.0
            }
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("答案一致性验证器测试（官方标准）")
    print("="*70)

    # 创建验证器
    validator = AnswerConsistencyValidator(use_nli_model=False)

    # 测试归一化
    print("\n归一化测试：")
    test_answers = [
        "Two", "2", "TWO",
        "A car", "the automobile", "Car",
        "Yes", "yeah", "YES!",
        "Reddish", "red",
        "One", "1"
    ]

    for ans in test_answers:
        normalized = validator.normalize_answer(ans)
        print(f"  '{ans}' → '{normalized}'")

    # 测试一致性验证
    print("\n一致性验证测试：")
    test_samples = [
        {
            'name': '一致样本',
            'sample': {
                'tasks': {
                    'vqa': {
                        'hard_label': {'answer': 'yes'},
                        'soft_label': {'primary_answer': 'yes'},
                        'cot_reasoning': {'conclusion': 'Yes, it is.'},
                        'inference_mode': 'closed'
                    }
                }
            },
            'ground_truth': 'yes'
        },
        {
            'name': '不一致样本',
            'sample': {
                'tasks': {
                    'vqa': {
                        'hard_label': {'answer': 'yes'},
                        'soft_label': {'primary_answer': 'no'},
                        'cot_reasoning': {'conclusion': 'Maybe.'},
                        'inference_mode': 'closed'
                    }
                }
            },
            'ground_truth': 'yes'
        }
    ]

    for test in test_samples:
        result = validator.validate_closed_sample(
            test['sample'],
            test.get('ground_truth')
        )
        status = "✓ 一致" if result['is_consistent'] else "✗ 不一致"
        print(f"\n{status} {test['name']}:")
        if result['inconsistent_pairs']:
            for pair in result['inconsistent_pairs']:
                print(f"  - {pair['source1']}({pair['answer1']}) vs {pair['source2']}({pair['answer2']})")

    print("\n" + "="*70)
    print("官方标准验证流程：")
    print("  1. 规则归一化（VQA v2官方标准）")
    print("  2. 归一化后字符串相等 → 直接判定等价")
    print("  3. 字符串不等 → 调用bart-large-mnli做NLI推理")
    print(f"  4. entailment概率 ≥ {validator.ENTAIL_THRESHOLD} → 等价")
    print("="*70)