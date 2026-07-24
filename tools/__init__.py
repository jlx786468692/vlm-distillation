"""
工具模块
========

包含Prompt生成、优化、验证，以及候选集封闭等工具。

使用方式：
    python -m tools                              # 显示帮助
    python -m tools prompt_generator             # 生成Prompt
    python -m tools candidate_closure            # 生成候选集
    python -m tools all                          # 运行所有工具
"""

__version__ = "1.0.0"

from .prompt.generator import PromptGenerator
from .candidate.closure import CandidateClosure

__all__ = [
    'PromptGenerator',
    'CandidateClosure',
]