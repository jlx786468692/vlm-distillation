"""
VQA问题分类器（官方标准）
=========================

严格按照官方标准实现分层分类：

官方分流判定标准：
1. 满足以下任意一类，判定为 open_descriptive 开放样本：
   - 因果推理问句：why / what reason / cause / how come
   - 描述类问句：describe this picture / what can you see / what is happening
   - 开放式抽象问答：无固定枚举答案、无法用单个单词作答
   - BART-MNLI 零样本分类最高置信类别为 "This is an open descriptive question"，且置信度 ≥ 0.7
   - 若模型置信度 < 0.7，统一兜底归入开放样本，绝不强行塞入闭合候选集

2. 否则判定为闭合样本：counting / color / yes_no / location

分类流程：
- 第一层：规则匹配（CPU，无GPU占用）
- 第二层：BART-MNLI 模型兜底（仅处理规则无法判定的样本）
- 置信度过滤：< 0.7 归为开放样本
"""

import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml


class QuestionType(str, Enum):
    """问题类型枚举（官方标准）"""
    COUNT = "counting"       # 官方命名
    COLOR = "color"
    BINARY = "yes_no"        # 官方命名
    LOCATION = "location"
    OPEN = "open_descriptive"  # 官方命名


@dataclass
class ClassificationResult:
    """分类结果数据类"""
    question_type: QuestionType
    confidence: float
    method: str  # "rule" 或 "model"
    model_scores: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "question_type": self.question_type.value,
            "confidence": self.confidence,
            "method": self.method,
            "model_scores": self.model_scores
        }


class QuestionClassifier:
    """
    分层问题分类器（官方标准）

    官方分流逻辑：
    - 闭合分支：counting / color / yes_no / location
    - 开放分支：open_descriptive

    规则优先级：
    1. 因果推理问句（强制开放）
    2. 描述类问句（强制开放）
    3. 闭合问句（计数/颜色/是非/位置）
    4. 模型兜底（置信度 < 0.7 归为开放）
    """

    # BART-MNLI 候选标签（官方标准）
    CANDIDATE_LABELS = [
        "This is a counting question",
        "This is a color question",
        "This is a yes/no question",
        "This is a location question",
        "This is an open descriptive question"
    ]

    # 标签到问题类型的映射
    LABEL_TO_TYPE = {
        "This is a counting question": QuestionType.COUNT,
        "This is a color question": QuestionType.COLOR,
        "This is a yes/no question": QuestionType.BINARY,
        "This is a location question": QuestionType.LOCATION,
        "This is an open descriptive question": QuestionType.OPEN
    }

    def __init__(
        self,
        model_path: str = "models/bart-large-mnli",
        config_path: Optional[str] = None,
        device: str = "cuda",
        confidence_threshold: float = 0.7,
        enable_model: bool = True
    ):
        """
        初始化问题分类器

        Args:
            model_path: BART-MNLI模型路径
            config_path: 配置文件路径（可选）
            device: 运行设备
            confidence_threshold: 模型置信度阈值（官方标准：0.7）
            enable_model: 是否启用模型推理
        """
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.enable_model = enable_model

        # 加载配置（如果提供）
        self.config = self._load_config(config_path) if config_path else {}

        # 初始化规则关键词库（官方标准）
        self._init_rules()

        # 初始化模型（延迟加载）
        self.model = None
        self.tokenizer = None

        print(f"✓ 问题分类器初始化完成（官方标准）")
        print(f"  - 规则匹配: 已启用")
        print(f"  - 模型兜底: {'已启用' if enable_model else '已禁用'}")
        print(f"  - 置信度阈值: {confidence_threshold}（官方标准）")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_rules(self):
        """
        初始化规则关键词库（官方标准）

        优先级：
        1. 因果推理问句（强制开放）
        2. 描述类问句（强制开放）
        3. 闭合问句（计数/颜色/是非/位置）
        """
        # ───────────────────────────────────────────────────────
        # 官方标准：开放样本强制判定规则
        # ───────────────────────────────────────────────────────

        # 因果推理问句（官方标准）
        self.causal_reasoning_keywords = [
            "why", "what reason", "cause", "how come",  # 官方标准
            "why is", "why are", "why do", "why does",  # 变体
            "what is the reason", "what causes"         # 变体
        ]

        # 描述类问句（官方标准）
        self.descriptive_keywords = [
            "describe", "what can you see", "what is happening",  # 官方标准
            "what do you see", "describe this picture",           # 官方标准
            "explain", "tell me about", "what is going on"       # 变体
        ]

        # 开放式抽象问答关键词（官方标准补充）
        self.open_abstract_keywords = [
            "what kind", "what type", "what sort",  # 无法用单个单词作答
            "what is the story", "what is happening",
            "how does", "what makes",  # 抽象问答
            "what is", "what are"  # ✅ 新增：物体识别问题（如 "What is she sitting on?"）
        ]

        # ───────────────────────────────────────────────────────
        # 官方标准：闭合样本判定规则
        # ───────────────────────────────────────────────────────

        # 计数问句
        self.count_keywords = [
            "how many", "number of", "count",
            "how much", "quantity of", "total number"
        ]

        # 颜色问句
        self.color_keywords = [
            "what color", "what colour", "which color",
            "colour is", "color is"
        ]

        # 是非问句（需要检查是否以这些词开头）
        self.yes_no_keywords = [
            "is there", "are there", "is it", "are they",  # 官方标准
            "does", "do you", "can you", "could you",
            "is ", "are ", "was ", "were "
        ]

        # 位置问句
        self.location_keywords = [
            "where", "location", "position",
            "which part", "which side", "on the left", "on the right"
        ]

    def _rule_match(self, question: str) -> Tuple[Optional[QuestionType], float]:
        """
        规则匹配（第一层，官方标准）

        ✅ 修复：调整优先级，先检查闭合问句，再检查开放问句

        优先级：
        1. 颜色问句（闭合）- 包含 "color" 或 "colour"
        2. 计数问句（闭合）- 包含 "how many", "number of"
        3. 位置问句（闭合）- 包含 "where", "location"
        4. 是非问句（闭合）- 以 "is there", "are there" 开头
        5. 因果推理问句（开放）- 包含 "why", "what reason"
        6. 描述类问句（开放）- 包含 "describe", "what can you see"
        7. 开放式抽象问答（开放）- 包含 "what is", "what are"

        Args:
            question: 问题文本

        Returns:
            (问题类型, 置信度) 或 (None, 0.0) 表示规则未命中
        """
        question_lower = question.lower().strip()

        # ───────────────────────────────────────────────────────
        # ✅ 优先级1：颜色问句（闭合）
        # 最高优先级，避免被其他规则误判
        # ───────────────────────────────────────────────────────
        for kw in self.color_keywords:
            if kw in question_lower:
                return QuestionType.COLOR, 1.0

        # 额外检查：如果问题包含 "color" 或 "colour" 一词
        if "color" in question_lower or "colour" in question_lower:
            # 排除包含其他明确类型的词（如 "how many colors"）
            if not any(ckw in question_lower for ckw in ["how many", "number of", "count"]):
                return QuestionType.COLOR, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级2：计数问句（闭合）
        # ───────────────────────────────────────────────────────
        for kw in self.count_keywords:
            if kw in question_lower:
                return QuestionType.COUNT, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级3：位置问句（闭合）
        # ───────────────────────────────────────────────────────
        for kw in self.location_keywords:
            if kw in question_lower:
                return QuestionType.LOCATION, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级4：是非问句（闭合）
        # ───────────────────────────────────────────────────────
        for kw in self.yes_no_keywords:
            if question_lower.startswith(kw):
                return QuestionType.BINARY, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级5：因果推理问句（开放）
        # ───────────────────────────────────────────────────────
        for kw in self.causal_reasoning_keywords:
            if kw in question_lower:
                return QuestionType.OPEN, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级6：描述类问句（开放）
        # ───────────────────────────────────────────────────────
        for kw in self.descriptive_keywords:
            if kw in question_lower:
                return QuestionType.OPEN, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级7：开放式抽象问答（开放）
        # 包括物体识别问题（如 "What is she sitting on?"）
        # ───────────────────────────────────────────────────────
        for kw in self.open_abstract_keywords:
            if kw in question_lower:
                return QuestionType.OPEN, 1.0

        # ───────────────────────────────────────────────────────
        # 规则未命中：需要模型兜底
        # ───────────────────────────────────────────────────────
        return None, 0.0

    def _load_model(self):
        """延迟加载BART-MNLI模型"""
        if self.model is not None:
            return

        if not self.enable_model:
            raise RuntimeError("模型推理已禁用，请设置 enable_model=True")

        print(f"✓ 加载BART-MNLI模型: {self.model_path}")

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ 模型加载完成，设备: {self.device}")

    def _model_inference(self, question: str) -> Tuple[QuestionType, float, Dict[str, float]]:
        """
        模型推理（第二层）- 零样本分类（NLI格式）

        官方标准：
        - 最高置信类别为 "This is an open descriptive question"，且置信度 ≥ 0.7 → 开放样本
        - 若模型置信度 < 0.7，统一兜底归入开放样本，绝不强行塞入闭合候选集

        Args:
            question: 问题文本

        Returns:
            (问题类型, 置信度, 各类别分数字典)
        """
        self._load_model()

        # ───────────────────────────────────────────────────────
        # 零样本分类（NLI格式）
        # ───────────────────────────────────────────────────────
        # BART-MNLI是NLI模型，输出3个类别：
        # - entailment (索引0): 文本蕴含
        # - neutral (索引1): 中性
        # - contradiction (索引2): 矛盾
        #
        # 零样本分类方法：
        # 对每个候选标签，构建 (question, label) pair
        # 计算entailment概率，选择概率最高的标签
        # ───────────────────────────────────────────────────────

        scores = {}

        # 对每个候选标签计算entailment概率
        for label in self.CANDIDATE_LABELS:
            # 构建NLI格式的输入
            # premise: question
            # hypothesis: label
            inputs = self.tokenizer(
                question,
                label,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)

            # 推理
            with torch.no_grad():
                logits = self.model(**inputs).logits
                # 取entailment概率（索引0）
                entailment_prob = torch.softmax(logits, dim=-1)[0, 0].item()

            scores[label] = entailment_prob

        # 找到最高分类别
        max_label = max(scores, key=scores.get)
        max_score = scores[max_label]
        question_type = self.LABEL_TO_TYPE[max_label]

        # ───────────────────────────────────────────────────────
        # 官方标准：置信度 < 0.7，统一归为开放样本
        # ───────────────────────────────────────────────────────
        if max_score < self.confidence_threshold:
            # 官方标准：绝不强行塞入闭合候选集
            return QuestionType.OPEN, max_score, scores

        return question_type, max_score, scores

    def classify(
        self,
        question: str,
        return_scores: bool = False
    ) -> ClassificationResult:
        """
        分类问题（官方标准）

        流程：
        1. 规则匹配（优先级：因果推理 > 描述类 > 闭合问句）
        2. 模型推理（置信度 < 0.7 归为开放）

        Args:
            question: 问题文本
            return_scores: 是否返回模型各类别分数

        Returns:
            ClassificationResult对象
        """
        # 第一层：规则匹配
        rule_type, rule_conf = self._rule_match(question)

        if rule_type is not None:
            return ClassificationResult(
                question_type=rule_type,
                confidence=rule_conf,
                method="rule",
                model_scores=None
            )

        # 第二层：模型推理（如果启用）
        if not self.enable_model:
            # 模型未启用，归为开放样本
            return ClassificationResult(
                question_type=QuestionType.OPEN,
                confidence=0.0,
                method="fallback",
                model_scores=None
            )

        try:
            model_type, model_conf, model_scores = self._model_inference(question)

            return ClassificationResult(
                question_type=model_type,
                confidence=model_conf,
                method="model",
                model_scores=model_scores if return_scores else None
            )

        except Exception as e:
            print(f"⚠ 模型推理失败: {e}，归为开放样本")
            return ClassificationResult(
                question_type=QuestionType.OPEN,
                confidence=0.0,
                method="error",
                model_scores=None
            )

    def close(self):
        """释放模型资源"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("✓ 模型资源已释放")


# ===== 使用示例 =====
if __name__ == "__main__":
    # 初始化分类器
    classifier = QuestionClassifier(
        model_path="models/bart-large-mnli",
        confidence_threshold=0.7
    )

    # 测试问题（官方标准示例）
    test_questions = [
        # 因果推理问句（强制开放）
        ("Why might someone from PETA be upset about this picture?", QuestionType.OPEN),

        # 描述类问句（强制开放）
        ("Describe this picture.", QuestionType.OPEN),
        ("What can you see in the image?", QuestionType.OPEN),

        # 开放式抽象问答（强制开放）
        ("What kind of sandwich is this?", QuestionType.OPEN),

        # 计数问句（闭合）
        ("How many people are in the image?", QuestionType.COUNT),

        # 颜色问句（闭合）
        ("What color is the car?", QuestionType.COLOR),

        # 是非问句（闭合）
        ("Is there a dog in the image?", QuestionType.BINARY),

        # 位置问句（闭合）
        ("Where is the cat located?", QuestionType.LOCATION),
    ]

    print("\n" + "="*70)
    print("官方标准测试")
    print("="*70)

    for question, expected_type in test_questions:
        result = classifier.classify(question)

        status = "✓" if result.question_type == expected_type else "✗"
        print(f"\n{status} 问题: {question}")
        print(f"  预期: {expected_type.value}")
        print(f"  实际: {result.question_type.value}")
        print(f"  置信度: {result.confidence:.2f}")
        print(f"  方法: {result.method}")

    classifier.close()