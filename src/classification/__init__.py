"""
问题分类模块
============

提供VQA问题类型分类功能，支持分层分类策略。

模块：
- QuestionClassifier: 分层问题分类器（规则 + 模型兜底）
- QuestionType: 问题类型枚举
- ClassificationResult: 分类结果数据类

分类类别：
- count: 计数问题
- color: 颜色问题
- binary: 是非问题
- location: 位置问题
- open: 开放式描述问题
"""

from .question_classifier import (
    QuestionClassifier,
    QuestionType,
    ClassificationResult
)

__all__ = [
    'QuestionClassifier',
    'QuestionType',
    'ClassificationResult'
]