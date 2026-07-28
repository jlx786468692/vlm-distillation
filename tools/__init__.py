"""
工具模块
========

包含Prompt生成、优化、验证，以及候选集生成等工具。

使用方式：
    python -m tools                              # 显示帮助
    python -m tools prompt_generator             # 生成Prompt
    python -m tools candidate_closure            # 三阶段生成候选集
    python -m tools all                          # 运行所有工具

候选集生成三阶段：
    阶段1：构建全局超大候选池（VQA标注 + COCO Caption + 教师模型）
    阶段2：COCO图像分场景（12大场景）
    阶段3：分场景过滤生成局部小闭合集

详细说明：
    - 阶段1-3脚本：tools/candidate/stage1_build_global_pool.py 等
    - 查询器：tools/candidate/scene_candidate_loader.py（运行时查询）
"""

__version__ = "1.1.0"

from .prompt.generator import PromptGenerator
from .candidate.candidate_closure import CandidateClosure

__all__ = [
    'PromptGenerator',
    'CandidateClosure',
]