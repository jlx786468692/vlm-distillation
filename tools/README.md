# 🔧 工具模块使用说明

## 📋 概述

工具模块提供Prompt生成和候选集封闭功能。

**核心方案**：零样本分类路由 + 分场景独立小闭合集

---

## 🚀 快速开始

### 一键运行（推荐）

```bash
# 运行所有工具
python -m tools all

# 查看帮助
python -m tools --help
```

### 单独运行

```bash
# Prompt生成
python -m tools prompt_generator

# 候选集封闭（自动生成分场景独立小闭合集）
python -m tools candidate_closure
```

---

## 🎯 核心方案

### 零样本分类路由 + 分场景独立小闭合集

**核心流程**：
1. 加载 VQA 训练集标注
2. 零样本分类问题类型（count/color/binary/other）
3. 为每个场景生成独立候选集
4. 输出分场景独立小闭合集

**数据来源**：VQA 训练集标注

**输出文件**：`data/scene_candidates.json`

### 方案优势

| 传统方案 | 新方案 |
|---------|--------|
| 单一候选集（1873个答案） | 分场景独立候选集 ✅ |
| 所有问题都用同一候选集 | 根据问题类型动态选择 ✅ |
| 噪声多、效率低 | 高质量、高效率 ✅ |

### 场景示例

```
计数问题 → count候选集（21个答案）
颜色问题 → color候选集（24个答案）
二元问题 → binary候选集（2个答案）
其他问题 → other候选集（1873个答案）
```

---

## 📚 分场景候选集生成

### 自动生成（推荐）

```bash
# 直接运行，系统自动检测并生成
python -m tools candidate_closure
```

**自动流程**：
1. 检测 `data/scene_candidates.json` 是否存在
2. 如果不存在，从 VQA 训练集生成
3. 初始化候选集封闭系统
4. 演示候选集生成

### 手动生成

```bash
python tools/candidate/generate_vqa_vocab.py \
    --vqa-annotations data/coco/annotations \
    --min-frequency 5 \
    --max-candidates 100
```

### 数据准备

需要下载 VQA v2 训练集标注：

1. 访问 https://visualqa.org/download.html
2. 下载 `v2_Annotations_Train_mscoco.zip`
3. 解压到 `data/coco/annotations/`

---

## 📝 Prompt生成策略

支持4种策略：

| 策略 | 说明 | 推荐度 |
|-----|------|-------|
| `real_labels` | 基于真实标签 | ⭐⭐⭐⭐⭐ |
| `dspy_fewshot` | DSPy Few-Shot | ⭐⭐⭐⭐⭐ |
| `dspy` | DSPy MIPROv2 | ⭐⭐⭐⭐ |
| `pattern_based` | 基于数据模式 | ⭐⭐⭐ |

### 使用不同策略

```bash
# 默认：real_labels
python -m tools prompt_generator

# 推荐：dspy_fewshot
python -m tools prompt_generator --strategy dspy_fewshot

# DSPy优化
python -m tools prompt_generator --strategy dspy

# 无标注数据
python -m tools prompt_generator --strategy pattern_based
```

---

## 📁 输出位置

```
outputs/
├── prompts/
│   └── vqa_en.yaml        # 生成的Prompt
└── candidate_sets/
    └── closure_data.json  # 生成的候选集（已废弃）
```

**新版本输出**：

```
data/
└── scene_candidates.json  # 分场景独立小闭合集 ✅
```

---

## ⚙️ 配置

配置文件：`configs/tools.yaml`

```yaml
prompt_generation:
  strategy: "real_labels"    # 或 dspy_fewshot, dspy, pattern_based
  num_samples: 100

candidate_closure:
  # 🔧 问题类型分类器配置
  classifier_model: "models/bart-large-mnli"  # 零样本分类模型
  enable_classifier: true                        # 是否启用分类器

  # 候选集生成参数
  temperature: 2.0                   # 温度缩放参数
  min_probability: 0.01              # 最小概率阈值
  max_candidates: 100                # 每个场景最大候选数
```

---

## 📚 详细文档

- **核心方案**: [docs/零样本分类路由+分场景独立小闭合集方案.md](../docs/零样本分类路由+分场景独立小闭合集方案.md)
- **完整指南**: [docs/Prompt生成指南.md](../docs/Prompt生成指南.md)
- **配置说明**: [configs/tools.yaml](../configs/tools.yaml)

---

## 💡 推荐用法

### 生产环境

```bash
# 推荐：先运行候选集封闭，生成场景候选集
python -m tools candidate_closure

# 然后运行Prompt生成
python -m tools prompt_generator --strategy dspy_fewshot
```

### 快速原型

```bash
# 推荐：使用默认配置
python -m tools all
```

---

**快速上手**: `python -m tools candidate_closure`