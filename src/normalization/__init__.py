"""
答案标准化模块
===============

【三条红线 - 绝对禁止】
❌ 红线1：不要把 color 弱候选池映射逻辑用在 GT、教师数据集；映射只上线推理。
❌ 红线2：counting 不要把过滤阈值写进 Prompt 强制模型只能输出该区间数字。
❌ 红线3：软标签（教师文本）不能做语义归一，只允许格式清洗。

【模块说明】
本模块提供分题型的答案标准化处理，严格遵守三条红线。

【核心类】
- AnswerNormalizer: 答案标准化主模块
- CountingNormalizer: counting专用处理器
- ColorNormalizer: color专用处理器
- LocationNormalizer: location专用处理器
- FormatCleaner: 通用格式清洗器

【使用示例】
```python
from src.normalization import AnswerNormalizer

# 初始化
normalizer = AnswerNormalizer()

# GT硬标签标准化
answer, conf = normalizer.normalize_gt("dark blue", "color")
# → ("blue", 1.0)

# 教师软标签清洗
answer, conf = normalizer.clean_teacher_output("dark blue", "color")
# → ("dark blue", 1.0)  # 保留原始语义

# 学生推理后处理（语义归一）
answer, conf = normalizer.validate_for_inference("dark blue", "color")
# → ("blue", 1.0)
```
"""

from .answer_normalizer import AnswerNormalizer
from .counting_normalizer import CountingNormalizer
from .color_normalizer import ColorNormalizer
from .location_normalizer import LocationNormalizer
from .format_cleaner import FormatCleaner

__all__ = [
    'AnswerNormalizer',
    'CountingNormalizer',
    'ColorNormalizer',
    'LocationNormalizer',
    'FormatCleaner'
]