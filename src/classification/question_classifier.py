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
    """问题类型枚举（官方标准 + 4大类映射）"""
    # 细分类型（保留原有逻辑）
    COUNT = "counting"       # 计数问题
    COLOR = "color"          # 颜色问题
    BINARY = "yes_no"        # 是否问题
    LOCATION = "location"    # 位置问题
    CHOICE = "choice"        # 选择型闭合问题（A or B?）
    OPEN = "open_descriptive"  # 开放式描述问题

    # 4大类（用于答案标准化）
    CLOSED_CHOICE = "closed_choice"       # 选择型闭合问题（强候选池）
    CLOSED_YESNO = "closed_yesno"         # 是否型闭合问题（固定候选池）
    CLOSED_ENUMERATE = "closed_enumerate" # 枚举型闭合问题（弱候选池）
    OPEN_TYPE = "open"                    # 开放问题（无候选池）

    def to_major_category(self) -> 'QuestionType':
        """
        将细分类型映射到4大类

        Returns:
            4大类问题类型

        Examples:
            >>> QuestionType.COUNT.to_major_category()
            QuestionType.CLOSED_ENUMERATE
            >>> QuestionType.CHOICE.to_major_category()
            QuestionType.CLOSED_CHOICE

        🔧 修复：location 答案空间开放，应归类为开放问题
        """
        mapping = {
            QuestionType.CHOICE: QuestionType.CLOSED_CHOICE,
            QuestionType.BINARY: QuestionType.CLOSED_YESNO,
            QuestionType.COUNT: QuestionType.CLOSED_ENUMERATE,
            QuestionType.COLOR: QuestionType.CLOSED_ENUMERATE,
            # 🔧 修复：location 答案空间开放，改为开放问题
            # 原因：位置描述可以是任意自然语言表达，如：
            # - "on the left side of the table"
            # - "behind the tree in the background"
            # - "next to the red car parked on the street"
            # 不像颜色/计数有固定的候选集
            QuestionType.LOCATION: QuestionType.OPEN_TYPE,  # 🔧 修改：CLOSED_ENUMERATE → OPEN_TYPE
            QuestionType.OPEN: QuestionType.OPEN_TYPE,
            # 4大类映射到自身
            QuestionType.CLOSED_CHOICE: QuestionType.CLOSED_CHOICE,
            QuestionType.CLOSED_YESNO: QuestionType.CLOSED_YESNO,
            QuestionType.CLOSED_ENUMERATE: QuestionType.CLOSED_ENUMERATE,
            QuestionType.OPEN_TYPE: QuestionType.OPEN_TYPE,
        }
        return mapping.get(self, QuestionType.OPEN_TYPE)


@dataclass
class ClassificationResult:
    """分类结果数据类"""
    question_type: QuestionType      # 细分类型（如counting/color）
    confidence: float
    method: str  # "rule" 或 "model"
    model_scores: Optional[Dict[str, float]] = None
    candidate_pool: Optional[List[str]] = None  # 候选答案池（用于选择型问题）

    def get_major_category(self) -> QuestionType:
        """
        获取4大类问题类型（用于答案标准化）

        Returns:
            4大类问题类型

        Examples:
            >>> result = ClassificationResult(question_type=QuestionType.COUNT, ...)
            >>> result.get_major_category()
            QuestionType.CLOSED_ENUMERATE
        """
        return self.question_type.to_major_category()

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "question_type": self.question_type.value,
            "major_category": self.get_major_category().value,  # 新增：4大类
            "confidence": self.confidence,
            "method": self.method,
            "model_scores": self.model_scores,
            "candidate_pool": self.candidate_pool
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
        0. 选择型闭合问题（A or B?）- 优先级最高，低成本识别
        1. 因果推理问句（强制开放）
        2. 描述类问句（强制开放）
        3. 闭合问句（计数/颜色/是非/位置）
        """
        # ───────────────────────────────────────────────────────
        # 🔧 新增：选择型闭合问题正则模式（优先级最高）
        # ───────────────────────────────────────────────────────
        # 适用句式：
        # - Are A or B?
        # - Is it X or Y?
        # - Was this picture taken during the day or night? ← 🔧 新增过去式
        # - X or Y?
        # - Choose X or Y?
        # - Which one, X or Y?
        # ───────────────────────────────────────────────────────
        self.choice_patterns = [
            # Pattern 1: "Are A or B?" / "Is A or B?" / "Was A or B?" / "Were A or B?"
            # 🔧 修复：添加过去式（was/were）
            r'^(?:are|is|was|were)\s+(.+?)\s+or\s+(.+?)\??$',

            # Pattern 2: "Is it X or Y?" / "Was it X or Y?"
            # 🔧 修复：添加过去式
            r'^(?:is|was)\s+it\s+(.+?)\s+or\s+(.+?)\??$',

            # Pattern 3: "Choose X or Y?"
            r'^choose\s+(.+?)\s+or\s+(.+?)\??$',

            # Pattern 4: "Which one, X or Y?"
            r'^which\s+one[,\s]+(.+?)\s+or\s+(.+?)\??$',

            # Pattern 5: "A or B?" (最简单的形式)
            r'^(.+?)\s+or\s+(.+?)\?$',
        ]

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
            "what is", "what are",  # ✅ 新增：物体识别问题（如 "What is she sitting on?"）
            # 🔧 新增：年龄、时间等估算类问题（答案为描述性数字）
            "what age", "how old",       # 年龄估算问题
            "about what age",            # "About what age is this person?"
            "what time", "which time",   # 时间问题（答案可能是描述）
            "what year", "which year",   # 年份问题
            "what season", "which season"  # 季节问题
        ]

        # ───────────────────────────────────────────────────────
        # 官方标准：闭合样本判定规则
        # ───────────────────────────────────────────────────────

        # 计数问句
        self.count_keywords = [
            "how many", "number of", "count",
            "how much", "quantity of", "total number",
            # 🔧 新增："What number" 句式（如 "What number is on the train?"）
            "what number",  # 匹配 "What number is on..."
        ]

        # 颜色问句
        self.color_keywords = [
            "what color", "what colour", "which color",
            "colour is", "color is",
            # 🔧 新增：交通灯颜色问题（答案为颜色）
            "traffic light is on",  # "What traffic light is on?" → red/green/yellow
            "which traffic light",  # "Which traffic light is on?"
            "light is on",          # "What light is on?"
            "color is the",         # "What color is the...?"
            "color of the",         # "What color of the...?"
        ]

        # 是非问句（需要检查是否以这些词开头）
        # 🔧 修复：添加所有常见的情态动词开头
        self.yes_no_keywords = [
            # 存在类问句
            "is there", "are there", "is it", "are they",
            "was there", "were there", "was it", "were they",

            # 情态动词问句（Would/Should/Could/Might等）
            "would", "should", "could", "might", "will", "shall",
            "would you", "should you", "could you", "might you", "will you",
            "would it", "should it", "could it", "might it", "will it",
            "would there", "should there", "could there", "might there", "will there",

            # 助动词问句
            "does", "do you", "do they", "does it", "do these", "do those",
            "did", "did you", "did it", "did they",

            # 能力/许可问句
            "can you", "can it", "can they", "can we",
            "may", "may you", "may it",

            # 状态问句
            "is ", "are ", "was ", "were ",
            "has ", "have ", "had "
        ]

        # 位置问句
        self.location_keywords = [
            "where", "location", "position",
            "which part", "which side", "on the left", "on the right"
        ]

    def _match_choice_pattern(self, question: str) -> Tuple[bool, Optional[List[str]]]:
        """
        🔧 新增：匹配选择型闭合问题（优先级最高）

        适用句式：
        - "Are the tomatoes sliced or diced?"
        - "Is it red or blue?"
        - "Choose A or B?"
        - "Which one, X or Y?"
        - "Is this meant for males or females to use?" ← 🔧 新增长句式

        Args:
            question: 问题文本

        Returns:
            (是否匹配, 候选答案列表)
            例如：(True, ["sliced", "diced"]) 或 (True, ["males", "females"])
        """
        question_clean = question.strip()

        # ───────────────────────────────────────────────────────
        # 方法1：精确匹配（短句式）
        # ───────────────────────────────────────────────────────
        # 按优先级尝试每个模式
        for pattern in self.choice_patterns:
            match = re.match(pattern, question_clean, re.IGNORECASE)

            if match:
                # 提取两个候选答案
                option1 = match.group(1).strip()
                option2 = match.group(2).strip()

                # 过滤掉无效候选（如过长的句子）
                if len(option1.split()) <= 3 and len(option2.split()) <= 3:
                    candidates = [option1.lower(), option2.lower()]

                    # 去重
                    candidates = list(dict.fromkeys(candidates))

                    return True, candidates

        # ───────────────────────────────────────────────────────
        # 方法2：检测长句式中的 "X or Y" 结构
        # ───────────────────────────────────────────────────────
        # 例如："Is this meant for males or females to use?"
        # 提取：males, females
        # ───────────────────────────────────────────────────────

        # 检测问题是否包含 "X or Y" 结构
        # 使用更宽松的匹配：查找单词 + or + 单词
        loose_pattern = r'\b([a-zA-Z]+)\s+or\s+([a-zA-Z]+)\b'
        matches = re.findall(loose_pattern, question_clean, re.IGNORECASE)

        if matches:
            # 遍历所有匹配，寻找有效的候选词
            for option1, option2 in matches:
                option1 = option1.lower().strip()
                option2 = option2.lower().strip()

                # 过滤掉介词、冠词等无关词
                stop_words = ['is', 'are', 'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'this', 'that', 'it']

                if option1 not in stop_words and option2 not in stop_words:
                    # 检查是否是有效的名词/形容词（长度 > 2）
                    if len(option1) > 2 and len(option2) > 2:
                        candidates = [option1, option2]

                        # 去重
                        candidates = list(dict.fromkeys(candidates))

                        return True, candidates

        return False, None

    def _rule_match(self, question: str) -> Tuple[Optional[QuestionType], float]:
        """
        规则匹配（第一层，官方标准）

        ✅ 修复：闭合问句优先于开放问句（官方标准）

        注意：选择型闭合问题（A or B?）已在 classify() 中优先处理

        优先级：
        1. 是非问句（闭合）- 以 "is there", "are there", "is this" 等开头（优先级最高）
        2. 颜色问句（闭合）- 包含 "color" 或 "colour"（排除是非问题后）
        3. 计数问句（闭合）- 包含 "how many", "number of"
        4. 位置问句（闭合）- 包含 "where", "location"
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
        # ✅ 优先级1：是非问句（闭合）
        # 🔧 关键：先检查是非问句，避免被其他规则误判
        # 例如："Is this photo in color?" 应该是 yes_no，不是 color
        # ───────────────────────────────────────────────────────
        for kw in self.yes_no_keywords:
            if question_lower.startswith(kw):
                return QuestionType.BINARY, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级2：颜色问句（闭合）
        # 🔧 改进：只有不是是非问句，才检查颜色
        # ───────────────────────────────────────────────────────
        for kw in self.color_keywords:
            if kw in question_lower:
                return QuestionType.COLOR, 1.0

        # 额外检查：如果问题包含 "color" 或 "colour" 一词
        # 🔧 改进：增加更多排除条件，避免是非问句误判
        if "color" in question_lower or "colour" in question_lower:
            # 排除包含其他明确类型的词
            if not any(ckw in question_lower for ckw in [
                "how many", "number of", "count",  # 计数问题
                "is there", "are there", "is it", "are they",  # 是非问题
                "is this", "is that", "are these", "are those"  # 是非问题
            ]):
                return QuestionType.COLOR, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级3：计数问句（闭合）
        # ───────────────────────────────────────────────────────
        for kw in self.count_keywords:
            if kw in question_lower:
                return QuestionType.COUNT, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级4：位置问句（闭合）
        # ───────────────────────────────────────────────────────
        for kw in self.location_keywords:
            if kw in question_lower:
                return QuestionType.LOCATION, 1.0

        # ───────────────────────────────────────────────────────
        # ✅ 优先级5：因果推理问句（开放）
        # 闭合问句检查完毕，开始检查开放问句
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
        0. 选择型闭合问题匹配（优先级最高）- 提取候选答案池
        1. 规则匹配（优先级：因果推理 > 描述类 > 闭合问句）
        2. 模型推理（置信度 < 0.7 归为开放）

        Args:
            question: 问题文本
            return_scores: 是否返回模型各类别分数

        Returns:
            ClassificationResult对象
        """
        # ───────────────────────────────────────────────────────
        # 🔧 优先级0：选择型闭合问题（A or B?）
        # ───────────────────────────────────────────────────────
        is_choice, candidates = self._match_choice_pattern(question)
        if is_choice:
            return ClassificationResult(
                question_type=QuestionType.CHOICE,
                confidence=1.0,
                method="rule",
                model_scores=None,
                candidate_pool=candidates  # 🔧 返回候选答案池
            )

        # 第一层：规则匹配
        rule_type, rule_conf = self._rule_match(question)

        if rule_type is not None:
            return ClassificationResult(
                question_type=rule_type,
                confidence=rule_conf,
                method="rule",
                model_scores=None,
                candidate_pool=None
            )

        # 第二层：模型推理（如果启用）
        if not self.enable_model:
            # 模型未启用，归为开放样本
            return ClassificationResult(
                question_type=QuestionType.OPEN,
                confidence=0.0,
                method="fallback",
                model_scores=None,
                candidate_pool=None
            )

        try:
            model_type, model_conf, model_scores = self._model_inference(question)

            return ClassificationResult(
                question_type=model_type,
                confidence=model_conf,
                method="model",
                model_scores=model_scores if return_scores else None,
                candidate_pool=None
            )

        except Exception as e:
            print(f"⚠ 模型推理失败: {e}，归为开放样本")
            return ClassificationResult(
                question_type=QuestionType.OPEN,
                confidence=0.0,
                method="error",
                model_scores=None,
                candidate_pool=None
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
        # 🔧 新增：选择型闭合问题（优先级最高）
        ("Are the tomatoes sliced or diced?", QuestionType.CHOICE),
        ("Is it red or blue?", QuestionType.CHOICE),
        # 🔧 修复：过去式选择型问题
        ("Was this picture taken during the day or night?", QuestionType.CHOICE),
        ("Were there more cats or dogs?", QuestionType.CHOICE),
        ("Choose A or B?", QuestionType.CHOICE),
        ("Which one, left or right?", QuestionType.CHOICE),
        # 🔧 新增：长句式选择型问题
        ("Is this meant for males or females to use?", QuestionType.CHOICE),  # ✅ 新增
        ("Is this designed for adults or children?", QuestionType.CHOICE),  # ✅ 新增

        # 因果推理问句（强制开放）
        ("Why might someone from PETA be upset about this picture?", QuestionType.OPEN),

        # 描述类问句（强制开放）
        ("Describe this picture.", QuestionType.OPEN),
        ("What can you see in the image?", QuestionType.OPEN),

        # 开放式抽象问答（强制开放）
        ("What kind of sandwich is this?", QuestionType.OPEN),
        # 🔧 新增：年龄估算问题测试
        ("About what age is this person?", QuestionType.OPEN),
        ("How old is the man?", QuestionType.OPEN),
        ("What age is the woman?", QuestionType.OPEN),

        # 计数问句（闭合）
        ("How many people are in the image?", QuestionType.COUNT),
        # 🔧 修复："What number" 句式（计数问题）
        ("What number is on the train?", QuestionType.COUNT),

        # 颜色问句（闭合）
        ("What color is the car?", QuestionType.COLOR),
        # 🔧 修复：包含 "what is" 的颜色问题（优先级正确）
        ("What is the color of the water?", QuestionType.COLOR),
        # 🔧 新增：交通灯颜色问题测试
        ("What traffic light is on?", QuestionType.COLOR),
        ("Which traffic light is on?", QuestionType.COLOR),
        ("What color is the traffic light?", QuestionType.COLOR),

        # 是非问句（闭合）
        ("Is there a dog in the image?", QuestionType.BINARY),
        # 🔧 新增：颜色相关的yes/no问题测试
        ("Is this photo in color?", QuestionType.BINARY),
        ("Is the car red?", QuestionType.BINARY),
        ("Are there colors in this image?", QuestionType.BINARY),

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

        # 🔧 新增：显示候选答案池（如果有）
        if result.candidate_pool:
            print(f"  候选答案池: {result.candidate_pool}")

    classifier.close()