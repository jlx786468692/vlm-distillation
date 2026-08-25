"""
学生模型评估模块
================

评估蒸馏后学生模型在三个维度上的表现：
1. 蒸馏质量 (distillation_quality) —— 学生是否学到教师精髓
2. 驾驶场景适配 (driving_scenario_eval) —— 安全关键/空间/否定等场景对 GT 准确率
3. 部署效率 (deployment_efficiency) —— 参数量/显存/延迟/吞吐

入口：run_evaluation(cfg)
"""

from .evaluator import run_evaluation, StudentInferencer
from .distillation_quality import evaluate as evaluate_distillation_quality
from .driving_scenario_eval import evaluate as evaluate_driving_scenario
from .deployment_efficiency import evaluate as evaluate_deployment_efficiency

__all__ = [
    "run_evaluation",
    "StudentInferencer",
    "evaluate_distillation_quality",
    "evaluate_driving_scenario",
    "evaluate_deployment_efficiency",
]
