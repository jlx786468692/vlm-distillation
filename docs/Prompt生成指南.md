# Prompt生成完整指南

**重要说明**: 本文档专注于VQA（Visual Question Answering）任务的Prompt生成策略。

## 📋 目录

1. [概述](#概述)
2. [工具模块策略](#工具模块策略)
3. [四种Prompt生成策略](#四种prompt生成策略)
4. [普通方式（模板）](#普通方式模板)
5. [DSPy方式](#dspy方式)
6. [任务导向方式](#任务导向方式)
7. [策略选择建议](#策略选择建议)
8. [VQA任务详细说明](#vqa任务详细说明)
9. [执行方法](#执行方法)
10. [最佳实践](#最佳实践)

---

## 概述

### 本文档内容

本文档整合了所有Prompt生成相关的内容：
- ✅ 四种Prompt生成策略的详细说明和对比
- ✅ 工具模块的使用方法
- ✅ 软标签和硬标签的利用方法
- ✅ 具体执行方法和最佳实践
- ✅ YAML格式优化处理

### 适用人群

- 📌 **算法工程师**：需要生成高质量Prompt用于模型蒸馏
- 📌 **研究人员**：需要理解不同Prompt生成方式的优劣
- 📌 **项目维护者**：需要维护和优化Prompt模板

---

## 工具模块策略

### 工具模块位置

所有Prompt生成工具已整合到 `tools/` 模块：

```
tools/
├── prompt/
│   └── generator.py        # Prompt生成器（支持4种策略）
└── candidate/
    └── closure.py          # 候选集封闭
```

### 执行方式

```bash
# 运行所有工具（推荐）
python -m tools all

# 单独运行Prompt生成
python -m tools prompt_generator --strategy <策略名>

# 单独运行候选集封闭
python -m tools candidate_closure
```

---

## 四种Prompt生成策略

### 策略对比总览

| 策略 | 说明 | 质量 | 速度 | 推荐度 | 适用场景 |
|-----|------|-----|------|-------|---------|
| **real_labels** | 基于真实标签 | ⭐⭐⭐⭐⭐ | 快 | ✅ 推荐 | 有标注数据 |
| **dspy_fewshot** | DSPy Few-Shot | ⭐⭐⭐⭐⭐ | 中等 | ✅ 推荐 | 有标注数据，需要示例 |
| **dspy** | DSPy MIPROv2 | ⭐⭐⭐⭐ | 慢 | 推荐 | 需要优化结构 |
| **pattern_based** | 基于数据模式 | ⭐⭐⭐ | 快 | 备用 | 无标注数据 |

### 策略选择决策树

```
需求分析：
├─ 是否有真实标签数据？
│  ├─ 否 → pattern_based
│  └─ 是 → 继续判断
│
├─ 是否需要few-shot示例？
│  ├─ 是 → dspy_fewshot（推荐）
│  └─ 否 → 继续判断
│
├─ 是否需要快速生成？
│  ├─ 是 → real_labels（推荐）
│  └─ 否 → dspy（进阶优化）
```

---

## 策略详解

### 策略1: real_labels（默认，推荐）

**适用场景**: 有高质量真实标签数据

**核心特点**：
- ✅ 直接使用真实标签数据
- ✅ 速度快，质量高
- ✅ 简单可靠

**使用方法**:
```bash
# 默认使用real_labels
python -m tools prompt_generator

# 或显式指定
python -m tools prompt_generator --strategy real_labels
```

**配置参数**:
```yaml
# configs/tools.yaml
prompt_generation:
  strategy: "real_labels"
  source_dir: "outputs/merged"
  num_samples: 100
  
  real_labels:
    min_confidence: 0.4
```

---

### 策略2: dspy_fewshot（推荐）

**适用场景**: 有真实标签数据，需要few-shot示例

**核心特点**：
- ✅ 自动选择高质量示例（置信度>0.7）
- ✅ 构建结构化few-shot prompt
- ✅ 包含完整的CoT推理示例

**核心逻辑**:
1. 从真实数据中筛选高质量样本
2. 选择标准：高置信度 + 完整CoT + 清晰分布
3. 构建包含5-10个示例的prompt

**输出示例**:
```
包含5-10个高质量真实示例
每个示例都有完整的：
- Question
- Allowed Answers
- Probability Distribution
- Observation
- Analysis
- Conclusion
```

**使用方法**:
```bash
python -m tools prompt_generator --strategy dspy_fewshot
```

**配置参数**:
```yaml
# configs/tools.yaml
prompt_generation:
  strategy: "dspy_fewshot"
  
  dspy_fewshot:
    top_k: 10              # 选择Top-K高质量示例
    min_confidence: 0.7    # 最低置信度阈值
```

---

### 策略3: dspy（MIPROv2优化）

**适用场景**: 需要自动优化prompt结构

**核心特点**：
- ✅ 使用DSPy的MIPROv2方法
- ✅ 自动优化prompt指令和结构
- ⚠️ 需要更多时间和资源

**输出示例**:
```
经过MIPROv2优化的prompt结构
可能包含自动生成的指令优化
```

**使用方法**:
```bash
python -m tools prompt_generator --strategy dspy
```

**配置参数**:
```yaml
prompt_generation:
  strategy: "dspy"
  
  dspy:
    model: "models/Qwen2.5-VL-3B-Instruct"
    num_trials: 10
    metric: "f1_score"
```

---

### 策略4: pattern_based（备用）

**适用场景**: 无标注数据

**核心特点**：
- ✅ 基于数据模式分析
- ✅ 不需要真实标签
- ⚠️ 质量相对较低

**使用方法**:
```bash
python -m tools prompt_generator --strategy pattern_based
```

---

## 普通方式（模板）

### 原理

基于预定义的模板，通过变量填充生成Prompt。

### 配置文件

**位置**: `configs/prompts_en.yaml`

```yaml
prompts:
  cot:
    vqa_system: |
      TASK: Answer the question by selecting exactly one answer.
      
      RULES:
      1. Observation: Only describe features that distinguish answers
      2. Analysis: Compare candidates using probability distribution
      3. Conclusion: Output ONE answer from the allowed list

    vqa_user: |
      Question: {question}
      Allowed: {allowed_answers}
      Primary: {primary_answer}
      Distribution: {answer_distribution}
      
      Observation: [视觉特征描述]
      Analysis: [基于分布的推理]
      Conclusion: [最终答案]
```

### 优缺点

**优点**：
- ✅ 简单直接，易于理解
- ✅ 完全可控，无需数据
- ✅ 调试方便

**缺点**：
- ❌ 固定不变，无法优化
- ❌ 质量受限，依赖人工设计
- ❌ 适应性差

### 适用场景

- ✅ 快速原型开发
- ✅ 简单任务
- ✅ 资源受限（无数据、无算力）
- ❌ 高质量需求

---

## DSPy方式

### 原理

使用DSPy框架，通过优化算法自动生成和调优Prompt。

### 核心流程

```
DSPy流程：
1. 定义Signature（输入输出规范）
   ↓
2. 准备训练数据（示例）
   ↓
3. 选择优化算法（MIPROv2、Few-Shot等）
   ↓
4. 运行优化（自动调优Prompt）
   ↓
5. 输出优化后的Prompt
```

### 两种DSPy方法对比

| 方法 | 说明 | 复杂度 | 质量 | 推荐 |
|-----|------|-------|------|------|
| **dspy_fewshot** | Few-Shot示例 | 中 | 高 | ✅ 推荐 |
| **dspy** | MIPROv2优化 | 高 | 最高 | 推荐 |

### 优缺点

**优点**：
- ✅ 自动优化，无需人工调参
- ✅ 适应性强，根据数据自动调整
- ✅ 高质量，通过算法优化达到最优

**缺点**：
- ❌ 复杂度高，需要理解DSPy
- ❌ 资源消耗大，需要数据和算力
- ❌ 不稳定，优化结果可能不收敛

### 适用场景

- ✅ 高质量需求（生产环境）
- ✅ 复杂推理任务
- ✅ 有充足的数据和算力
- ❌ 资源受限
- ❌ 简单任务

---

## 任务导向方式

### 原理

基于真实标签数据，针对特定任务定制Prompt。

### 核心思想

```
任务导向流程：
1. 收集真实标签数据（硬标签、软标签、CoT）
   ↓
2. 分析任务特征（答案分布、置信度、质量）
   ↓
3. 提取常见模式（高频答案、典型错误）
   ↓
4. 定制任务特定Prompt
   ↓
5. 验证和迭代
```

### 为什么推荐任务导向？

| 维度 | 普通方式 | DSPy方式 | 任务导向方式 |
|------|---------|---------|------------|
| **数据依赖** | 无 | 高 | 中等 |
| **实现复杂度** | 低 | 高 | 中 |
| **质量** | 中等 | 高 | 高 |
| **可控性** | 高 | 低 | 中 |
| **稳定性** | 高 | 中 | 高 |
| **适应性** | 低 | 高 | 高 |

### 数据结构

#### VQA标签数据

```python
{
    'image_id': 123,
    'question': 'How many people?',
    'hard_label': {
        'answer': 'one',
        'confidence': 0.85
    },
    'soft_label': {
        'answer_distribution': {
            'one': 0.5234,
            'two': 0.4766,
            ...
        },
        'primary_answer': 'one',
        'allowed_answers': ['one', 'two', ...]
    },
    'cot_reasoning': {
        'structured_reasoning': {
            'observation': '...',
            'analysis': '...',
            'conclusion': 'one'
        }
    }
}
```

---

## 策略选择建议

### 推荐使用顺序

1. **首选**: `dspy_fewshot` - 自动选择高质量示例，效果最好
2. **备选**: `real_labels` - 简单快速，质量稳定
3. **进阶**: `dspy` - 需要优化prompt结构时使用
4. **备用**: `pattern_based` - 无标注数据时使用

### 不同场景推荐

| 场景 | 推荐策略 | 原因 |
|-----|---------|------|
| **生产环境** | dspy_fewshot 或 real_labels | 高质量、稳定 |
| **快速原型** | real_labels 或 模板 | 简单快速 |
| **研究实验** | dspy | 自动优化 |
| **无标注数据** | pattern_based | 唯一选择 |

---

## VQA任务详细说明

### 核心特点

**任务目标**：回答视觉问题，输出单个答案

**关键信息**：
- 问题类型：数字、颜色、二元、位置、物体、动作等
- 答案空间：开放域（任何单词都可能是答案）
- 软标签：概率分布（多答案候选）
- 硬标签：正确答案 + 置信度

### 软标签利用

**如何利用软标签**:

```python
# 1. 提取允许答案列表（过滤后的有效答案）
allowed_answers = soft_label['allowed_answers']

# 2. 格式化概率分布（用于指导推理）
distribution_str = format_distribution(soft_label['answer_distribution'])

# 3. 填充到Prompt
prompt = template.format(
    question=question,
    allowed_answers=', '.join(allowed_answers),
    primary_answer=hard_label['answer'],
    answer_distribution=distribution_str
)
```

### VQA Prompt示例

```
System:
TASK: Answer by selecting ONE answer from the allowed list.

User:
Question: How many people are wearing headphones?
Allowed: one, two, three, zero, four
Primary: one
Distribution: one:0.52, two:0.32, three:0.10

Observation: One person in the foreground wears headphones.
Analysis: The primary answer 'one' has highest probability (0.52).
Conclusion: one
```

---

## 输出格式优化

### YAML换行处理

所有策略生成的YAML文件都经过优化处理：

```yaml
# ✅ 正确的换行处理（使用 | 块样式）
prompts:
  cot:
    vqa_system: |
      Line 1
      Line 2
      Line 3
      
      Multiple paragraphs
      are preserved

# ❌ 错误的处理方式（使用转义符）
vqa_system: "Line 1\\nLine 2\\nLine 3"
```

---

## 执行方法

### 方式一：使用工具模块（推荐）

```bash
# 运行所有工具
python -m tools all

# 单独生成Prompt
python -m tools prompt_generator --strategy real_labels

# 查看帮助
python -m tools --help
```

### 方式二：使用配置文件

```bash
# 编辑配置
vim configs/tools.yaml

# 运行
python -m tools all
```

### 方式三：命令行参数

```bash
# 自定义参数
python -m tools prompt_generator \
    --strategy dspy_fewshot \
    --source_dir outputs/merged \
    --num_samples 50 \
    --output outputs/prompts/vqa_en.yaml
```

### 执行时间对比

| 策略 | 准备时间 | 执行时间 | 总时间 |
|-----|---------|---------|--------|
| **real_labels** | 30分钟（数据） | 1-2分钟 | 31-32分钟 |
| **dspy_fewshot** | 30分钟（数据） | 3-5分钟 | 33-35分钟 |
| **dspy** | 30分钟（数据） | 60-180分钟 | 90-210分钟 |
| **pattern_based** | 0分钟 | 1-2分钟 | 1-2分钟 |

---

## 最佳实践

### 生产环境推荐

```bash
# 1. 生成足够的标签数据
python scripts/run_full_pipeline.py --max_samples 1000

# 2. 使用dspy_fewshot生成Prompt（推荐）
python -m tools prompt_generator --strategy dspy_fewshot --num_samples 100

# 3. 验证生成的Prompt
cat outputs/prompts/vqa_en.yaml

# 4. 运行主流程测试
python scripts/run_full_pipeline.py --max_samples 10
```

### 快速原型推荐

```bash
# 使用real_labels策略（快速）
python -m tools prompt_generator --strategy real_labels --num_samples 50
```

### 参数调优建议

#### VQA参数

```yaml
# configs/tools.yaml
prompt_generation:
  num_samples: 100  # 推荐100-500
  
  dspy_fewshot:
    min_confidence: 0.7  # 推荐0.7-0.9
    top_k: 10            # 推荐5-15
```

### 常见问题与解决

#### 问题1：VQA Prompt质量低

**症状**：模型输出包含噪音、幻觉

**解决**：
```bash
# 1. 检查数据质量
ls -la outputs/merged/

# 2. 增加训练数据
python -m tools prompt_generator --num_samples 200

# 3. 使用dspy_fewshot策略
python -m tools prompt_generator --strategy dspy_fewshot
```

#### 问题2：生成的YAML格式有问题

**症状**：换行符显示为\n

**解决**：
- 所有策略已自动处理换行
- 如仍有问题，重新生成即可

---

## 总结

### 四种策略对比总结

| 策略 | 质量 | 复杂度 | 时间 | 推荐指数 |
|-----|------|--------|------|----------|
| **real_labels** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **dspy_fewshot** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **dspy** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **pattern_based** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 关键要点

1. **推荐策略**: 优先使用 `dspy_fewshot` 或 `real_labels`
2. **数据质量**: Prompt质量取决于真实标签数据质量
3. **软标签利用**: 充分利用软标签的概率分布信息
4. **格式优化**: 自动处理YAML换行和转义

---

## 📁 相关文件

### 工具模块

| 文件 | 说明 |
|------|------|
| [tools/prompt/generator.py](../tools/prompt/generator.py) | Prompt生成器（4种策略） |
| [tools/candidate/closure.py](../tools/candidate/closure.py) | 候选集封闭 |
| [tools/README.md](../tools/README.md) | 工具模块使用说明 |

### 配置文件

| 文件 | 说明 |
|------|------|
| [configs/tools.yaml](../configs/tools.yaml) | 工具配置文件 |
| [configs/prompts_en.yaml](../configs/prompts_en.yaml) | Prompt模板（英文） |
| [configs/default.yaml](../configs/default.yaml) | 主配置文件 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/prompts/vqa_en.yaml` | 生成的Prompt |
| `outputs/candidate_sets/closure_data.json` | 生成的候选集 |

---

**文档版本**: v3.0
**最后更新**: 2026-07-24
**维护者**: AI Assistant
**整合说明**: 整合了工具模块策略和Prompt生成完整指南