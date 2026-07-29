# 问题分类模块 (Classification Module)

分层问题分类系统，用于VQA数据预处理阶段。

## 模块职责

- **预处理阶段**：在数据蒸馏前，对VQA问题进行类型分类
- **分层策略**：规则优先、模型兜底，兼顾效率与准确性
- **候选集封闭**：为后续软标签过滤提供白名单依据

## 目录结构

```
src/classification/
├── __init__.py              # 模块导出
└── question_classifier.py   # 问题分类器实现
```

## 快速使用

```python
from src.classification import QuestionClassifier, QuestionType

# 初始化分类器
classifier = QuestionClassifier(
    model_path="models/bart-large-mnli",
    confidence_threshold=0.7
)

# 分类问题
question = "What kind of sandwich is this?"
result = classifier.classify(question)

print(f"类型: {result.question_type.value}")  # open
print(f"置信度: {result.confidence}")         # 0.85
print(f"方法: {result.method}")              # model

# 关闭资源
classifier.close()
```

## 分类类别

| 类型 | 描述 | 示例问题 |
|------|------|----------|
| `count` | 计数问题 | "How many people?" |
| `color` | 颜色问题 | "What color is the car?" |
| `binary` | 是非问题 | "Is there a dog?" |
| `location` | 位置问题 | "Where is the cat?" |
| `open` | 开放式问题 | "What kind of sandwich is this?" |

## 与其他模块的关系

```
src/
├── classification/     # 问题分类（预处理）
│   └── question_classifier.py
├── distillation/       # 数据蒸馏（主流程）
├── cleaning/           # 数据清洗（后处理）
└── utils/              # 工具函数
    └── vqa_token_filter.py
```

## 文档

详细使用说明请查看：[docs/question_classifier_usage.md](../../docs/question_classifier_usage.md)

## 示例代码

- [单元测试](../../tests/test_question_classifier.py)
- [集成示例](../../examples/question_classifier_integration.py)