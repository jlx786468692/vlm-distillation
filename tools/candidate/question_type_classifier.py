"""
问题类型分类器
================

使用零样本分类（facebook/bart-large-mnli）对问题类型进行分类。

支持的问题类型：
- count: 计数问题（how many, how much等）
- color: 颜色问题（what color等）
- binary: 二元问题（yes/no问题）
- other: 其他类型问题

使用方式：
    from tools.candidate.question_type_classifier import QuestionTypeClassifier

    classifier = QuestionTypeClassifier()
    result = classifier.classify("How many people are in the image?")
    print(result)  # {'type': 'count', 'confidence': 0.95}
"""

import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionTypeClassifier:
    """
    问题类型分类器（使用零样本分类）

    使用facebook/bart-large-mnli模型进行零样本分类，
    将问题分类为：count, color, binary, other等类型。
    """

    # 问题类型定义（用于零样本分类）
    QUESTION_TYPES = {
        "count": "This is a counting question asking about numbers or quantities",
        "color": "This is a color question asking about colors",
        "binary": "This is a yes/no question asking for confirmation",
        "other": "This is a general question about objects, actions, or attributes"
    }

    # 问题类型候选标签
    TYPE_LABELS = ["count", "color", "binary", "other"]

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        初始化问题类型分类器

        Args:
            model_name: 模型名称，默认使用facebook/bart-large-mnli
            device: 设备类型（cuda/cpu），自动检测如果为None
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.cache_dir = cache_dir

        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🚀 初始化QuestionTypeClassifier")
        logger.info(f"  模型: {self.model_name}")
        logger.info(f"  设备: {self.device}")

        # 加载模型和tokenizer
        self._load_model()

        # 预定义答案集（根据问题类型）
        self._init_predefined_answers()

    def _load_model(self):
        """加载零样本分类模型"""
        try:
            logger.info(f"⏳ 加载模型 {self.model_name}...")

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # 加载模型
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )

            # 移动到指定设备
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ 模型加载完成")

        except Exception as e:
            logger.error(f"✗ 模型加载失败: {e}")
            raise RuntimeError(f"无法加载模型 {self.model_name}: {e}")

    def _init_predefined_answers(self):
        """初始化预定义答案集（根据问题类型）"""
        # 数字答案（计数问题）
        self.count_answers = [
            "zero", "one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"
        ]

        # 颜色答案
        self.color_answers = [
            "red", "green", "blue", "yellow", "orange", "purple",
            "pink", "brown", "black", "white", "gray", "grey",
            "cyan", "magenta", "turquoise", "azure", "silver",
            "golden", "bronze", "maroon", "navy", "teal", "olive", "lime"
        ]

        # 二元答案
        self.binary_answers = ["yes", "no"]

        # 其他常见答案（通用）
        self.common_answers = [
            # 人物
            "man", "woman", "boy", "girl", "child", "person", "people",
            # 动物
            "dog", "cat", "bird", "horse", "cow", "sheep", "pig",
            # 物品
            "car", "bicycle", "motorcycle", "bus", "truck", "train",
            "chair", "table", "desk", "bed", "sofa", "couch",
            "cup", "glass", "bottle", "bowl", "plate", "spoon", "fork",
            "book", "pen", "pencil", "paper", "notebook",
            "phone", "laptop", "computer", "keyboard", "mouse",
            "tree", "flower", "grass", "leaf", "plant",
            # 动作
            "standing", "sitting", "walking", "running", "lying",
            "eating", "drinking", "reading", "writing", "playing",
            # 位置
            "left", "right", "top", "bottom", "center", "middle",
            "front", "back", "side", "corner", "edge",
            # 大小
            "big", "small", "large", "tiny", "huge", "medium",
            "tall", "short", "long", "wide", "narrow",
            # 天气/环境
            "sunny", "cloudy", "rainy", "snowy", "windy",
            "day", "night", "morning", "evening", "afternoon",
            # 室内/室外
            "indoor", "outdoor", "inside", "outside",
            "kitchen", "bedroom", "bathroom", "living room",
            "street", "park", "beach", "mountain", "river"
        ]

        logger.info(f"✓ 预定义答案集初始化完成")
        logger.info(f"  计数答案: {len(self.count_answers)}个")
        logger.info(f"  颜色答案: {len(self.color_answers)}个")
        logger.info(f"  二元答案: {len(self.binary_answers)}个")
        logger.info(f"  通用答案: {len(self.common_answers)}个")

    def classify(
        self,
        question: str,
        return_confidence: bool = True
    ) -> Dict[str, any]:
        """
        对问题进行类型分类

        Args:
            question: 问题文本
            return_confidence: 是否返回置信度

        Returns:
            分类结果字典：
            - type: 问题类型（count/color/binary/other）
            - confidence: 置信度（如果return_confidence=True）
        """
        try:
            # 构建零样本分类输入
            candidate_labels = self.TYPE_LABELS

            # 使用零样本分类
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            # 构建假设文本（用于MNLI）
            hypotheses = [
                f"{self.QUESTION_TYPES[label]}"
                for label in candidate_labels
            ]

            # 对每个候选标签进行分类
            logits_list = []
            for hypothesis in hypotheses:
                hypothesis_inputs = self.tokenizer(
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # 拼接输入
                input_ids = torch.cat([
                    inputs["input_ids"],
                    torch.tensor([[self.tokenizer.sep_token_id]]).to(self.device),
                    hypothesis_inputs["input_ids"]
                ], dim=1)

                attention_mask = torch.cat([
                    inputs["attention_mask"],
                    torch.tensor([[1]]).to(self.device),
                    hypothesis_inputs["attention_mask"]
                ], dim=1)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits

                    # MNLI模型输出3个类别：矛盾、中立、一致
                    # 取一致类别的分数
                    entailment_score = logits[0, 2].item()  # 索引2是一致
                    logits_list.append(entailment_score)

            # 计算概率分布（使用softmax）
            logits_tensor = torch.tensor(logits_list)
            probs = torch.softmax(logits_tensor, dim=0)

            # 获取最可能的类型
            predicted_idx = torch.argmax(probs).item()
            predicted_type = candidate_labels[predicted_idx]
            confidence = probs[predicted_idx].item()

            logger.debug(f"问题: '{question}' -> 类型: {predicted_type} (置信度: {confidence:.4f})")

            result = {
                "type": predicted_type,
            }

            if return_confidence:
                result["confidence"] = confidence

            return result

        except Exception as e:
            logger.error(f"分类失败: {e}")
            # 失败时返回默认类型
            result = {"type": "other"}
            if return_confidence:
                result["confidence"] = 0.0
            return result

    def get_predefined_answers(self, question_type: str) -> List[str]:
        """
        根据问题类型获取预定义答案集

        Args:
            question_type: 问题类型

        Returns:
            预定义答案列表
        """
        if question_type == "count":
            return self.count_answers.copy()
        elif question_type == "color":
            return self.color_answers.copy()
        elif question_type == "binary":
            return self.binary_answers.copy()
        else:
            # 其他类型返回通用答案集
            return (
                self.count_answers +
                self.color_answers +
                self.binary_answers +
                self.common_answers
            )

    def batch_classify(
        self,
        questions: List[str],
        batch_size: int = 8
    ) -> List[Dict[str, any]]:
        """
        批量分类问题类型

        Args:
            questions: 问题列表
            batch_size: 批处理大小

        Returns:
            分类结果列表
        """
        results = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]

            # 批量处理
            for question in batch:
                result = self.classify(question)
                results.append(result)

            logger.info(f"已处理 {min(i+batch_size, len(questions))}/{len(questions)} 个问题")

        return results

    def __repr__(self) -> str:
        """字符串表示"""
        return f"QuestionTypeClassifier(model='{self.model_name}', device='{self.device}')"


# ===== 测试代码 =====
if __name__ == "__main__":
    # 初始化分类器
    classifier = QuestionTypeClassifier()

    # 测试问题
    test_questions = [
        "How many people are in the image?",
        "What color is the car?",
        "Is there a dog in the picture?",
        "What is the man doing?",
        "Where is the cat sitting?",
        "How much does the backpack cost?"
    ]

    print("\n===== 问题类型分类测试 =====")
    for question in test_questions:
        result = classifier.classify(question)
        print(f"问题: {question}")
        print(f"  -> 类型: {result['type']}, 置信度: {result.get('confidence', 0):.4f}")

        # 获取预定义答案
        answers = classifier.get_predefined_answers(result['type'])
        print(f"  -> 预定义答案: {answers[:5]}... (共{len(answers)}个)")
        print()