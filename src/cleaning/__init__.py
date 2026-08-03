"""
数据清洗模块
============

提供深度数据清洗功能，提升蒸馏数据质量。

现有组件：
- RewardModelScorer: 奖励模型打分器（规则层 + 模型层混合打分）
- RewardModelJudge: 多模态打分模型（Qwen-VL Judge）
- ClosedSampleValidator: 闭合样本校验器（三元自洽 + GT一致性校验）
- DataPartitioner: 数据分区存储器（开源标准）
- ConfidenceController: 置信度占比限流控制器

已清理：
- OpenAnswerCleaner: 已废弃，功能整合到 RewardModelScorer
- DataCleaner: 已废弃，功能整合到 RewardModelScorer
- QwenDataCleaner: 已废弃，功能整合到 RewardModelScorer
- DifferentialCleaner: 已废弃，开放/闭合问题分流在 distillation 模块实现
"""

# 只导入实际存在的模块
from .reward_model_scorer import RewardModelScorer
from .reward_model_judge import RewardModelJudge
from .closed_sample_validator import ClosedSampleValidator
from .data_partitioner import DataPartitioner
from .confidence_controller import ConfidenceController

__all__ = [
    'RewardModelScorer',
    'RewardModelJudge',
    'ClosedSampleValidator',
    'DataPartitioner',
    'ConfidenceController'
]