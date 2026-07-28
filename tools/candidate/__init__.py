"""
候选集工具子模块
===============

包含候选集生成、查询等功能。

核心组件：
- CandidateClosure: 候选集查询器（兼容层）
- SceneCandidateLoader: 场景候选集加载器（底层实现）

生成方式（三阶段）：
    python -m tools candidate_closure

详细脚本：
    - stage1_build_global_pool.py: 构建全局超大候选池
    - stage2_scene_mapping.py: COCO图像分场景
    - stage3_scene_closure.py: 分场景过滤生成局部小闭合集
"""

from .candidate_closure import CandidateClosure
from .scene_candidate_loader import SceneCandidateLoader

__all__ = ['CandidateClosure', 'SceneCandidateLoader']