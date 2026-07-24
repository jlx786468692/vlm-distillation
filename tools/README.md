# 🔧 工具模块 (Tools Module)

## 📋 概述

工具模块提供独立的可执行工具，包括：
- **Prompt生成**: 基于真实标签或数据模式生成优化的prompt
- **候选集封闭**: 生成封闭的答案候选集

## 🚀 快速开始

### 运行所有工具

```bash
# 运行所有工具（推荐）
python -m tools all

# 查看帮助
python -m tools --help
```

### 单独运行

```bash
# 只生成Prompt
python -m tools prompt_generator

# 只生成候选集
python -m tools candidate_closure

# 强制重新生成
python -m tools all --force
```

### 使用自定义配置

```bash
# 使用自定义配置文件
python -m tools all --config my_tools.yaml

# 或修改 configs/tools.yaml 后直接运行
python -m tools all
```

---

## 📁 目录结构

```
tools/
├── __init__.py              # 模块接口
├── __main__.py              # 统一执行入口
│
├── prompt/                  # Prompt工具
│   ├── __init__.py
│   └── generator.py         # Prompt生成器
│
└── candidate/               # 候选集工具
    ├── __init__.py
    └── closure.py           # 候选集封闭
```

---

## 📊 输出位置

运行工具后，生成的资源位于：

```
outputs/
├── prompts/                 # 生成的Prompt
│   ├── vqa_en.yaml
│   └── metadata.json
│
└── candidate_sets/          # 生成的候选集
    ├── closure_data.json
    └── metadata.json
```

---

## ⚙️ 配置说明

工具使用独立的配置文件 `configs/tools.yaml`：

```bash
# 查看配置
cat configs/tools.yaml

# 使用默认配置（configs/tools.yaml）
python -m tools all

# 使用自定义配置
python -m tools all --config my_tools.yaml
```

配置文件位置：`configs/tools.yaml`

```yaml
# Prompt生成配置
prompt_generation:
  source_dir: "outputs/merged"       # 数据源目录
  num_samples: 100                   # 样本数
  strategy: "real_labels"            # 策略：real_labels | pattern_based
  output_file: "outputs/prompts/vqa_en.yaml"

# 候选集封闭配置
candidate_closure:
  source_dir: "data/coco/annotations"
  strategy: "frequency_based"        # 策略：frequency_based
  min_frequency: 5                   # 最低频率阈值
  max_candidates: 100                # 最大候选数
  output_file: "outputs/candidate_sets/closure_data.json"
```

**注意**：
- 工具默认使用 `configs/tools.yaml`
- 如果 `tools.yaml` 不存在，会自动使用 `configs/default.yaml`
- 主流程配置仍在 `configs/default.yaml`

---

## 🔧 高级用法

### 方式一：统一入口（推荐）

```bash
# 运行所有工具
python -m tools all

# 单独运行
python -m tools prompt_generator
python -m tools candidate_closure
```

### 方式二：单独执行子模块

```bash
# Prompt生成
python -m tools.prompt.generator --strategy real_labels --num_samples 100

# 候选集封闭
python -m tools.candidate.closure --strategy frequency_based --max_candidates 100
```

### 方式三：在Python代码中使用

```python
from tools import PromptGenerator, CandidateClosure

# Prompt生成
config = {
    'source_dir': 'outputs/merged',
    'num_samples': 100,
    'strategy': 'real_labels'
}
generator = PromptGenerator(config)
prompts = generator.generate()

# 候选集封闭
config = {
    'source_dir': 'data/coco/annotations',
    'strategy': 'frequency_based',
    'max_candidates': 100
}
closure = CandidateClosure(config)
closure_data = closure.generate()
```

---

## 📝 支持的策略

### Prompt生成策略

| 策略 | 说明 | 适用场景 | 推荐度 |
|-----|------|---------|-------|
| `real_labels` | 基于真实标签生成 | 有高质量标注数据时 | ⭐⭐⭐⭐⭐ |
| `dspy_fewshot` | DSPy Few-Shot方法 | 基于真实数据示例 | ⭐⭐⭐⭐⭐ |
| `dspy` | DSPy MIPROv2优化 | 自动优化prompt结构 | ⭐⭐⭐⭐ |
| `pattern_based` | 基于数据模式分析 | 无标注数据时 | ⭐⭐⭐ |

**使用不同策略**:

```bash
# 使用real_labels策略（默认）
python -m tools prompt_generator --strategy real_labels

# 使用DSPy Few-Shot方法（推荐）
python -m tools prompt_generator --strategy dspy_fewshot

# 使用DSPy MIPROv2优化
python -m tools prompt_generator --strategy dspy

# 使用pattern_based策略
python -m tools prompt_generator --strategy pattern_based
```

**策略对比**:

| 特性 | real_labels | dspy_fewshot | dspy | pattern_based |
|-----|------------|--------------|------|--------------|
| 需要 | 真实标签 | 真实标签 | 真实标签 | COCO标注 |
| 质量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 速度 | 快 | 中等 | 慢 | 快 |
| 示例 | ❌ | ✅ 自动选择 | ✅ 自动生成 | ❌ |
| 推荐度 | ✅ 推荐 | ✅ 推荐 | 推荐 | 备用 |

**策略详细说明**:

1. **real_labels**: 直接使用真实标签数据生成prompt，简单快速
2. **dspy_fewshot**: 自动选择高质量示例，构建few-shot prompt（推荐）
3. **dspy**: 使用DSPy的MIPROv2自动优化prompt结构
4. **pattern_based**: 基于数据模式分析生成（备用）

### 候选集封闭策略

| 策略 | 说明 | 适用场景 |
|-----|------|---------|
| `frequency_based` | 基于频率统计 | 通用场景（推荐）|
| `semantic_clustering` | 基于语义聚类 | 需要语义分组时（预留）|

---

## 🔗 与主流程的集成

生成资源后，在主流程中使用：

```yaml
# configs/default.yaml
teacher:
  prompts_config: "outputs/prompts/vqa_en.yaml"

vqa_token_filter:
  candidate_sets: "outputs/candidate_sets/closure_data.json"
```

运行主流程：

```bash
python scripts/run_full_pipeline.py --max_samples 10
```

---

## 🐛 故障排除

### 问题：找不到数据源

**错误**: `数据源目录不存在: outputs/merged`

**解决**: 先运行主流程生成标签数据
```bash
python scripts/run_full_pipeline.py --steps distillation
```

### 问题：生成的Prompt为空

**可能原因**: 样本数配置过大或数据源为空

**解决**: 检查配置并减少样本数
```bash
python -m tools prompt_generator --num_samples 10
```

---

## 📚 相关文档

- [主项目README](../README.md)
- [配置文件说明](../configs/default.yaml)
- [主流程脚本](../scripts/run_full_pipeline.py)

---

**版本**: 1.0.0
**最后更新**: 2026-07-24