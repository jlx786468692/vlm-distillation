# Prompt生成完整指南

**重要说明**: 本文档专注于VQA（Visual Question Answering）任务的Prompt生成策略。项目已不再支持Detection、Captioning等其他任务。

## 📋 目录

1. [概述](#概述)
2. [三种Prompt生成方式](#三种prompt生成方式)
3. [普通方式](#普通方式)
4. [DSPy方式](#dspy方式)
5. [任务导向方式](#任务导向方式)
6. [VQA任务Prompt生成](#vqa任务prompt生成)
7. [Detect任务Prompt生成](#detect任务prompt生成)
8. [VQA与Detect对比](#vqa与detect对比)
9. [执行方法](#执行方法)
10. [最佳实践](#最佳实践)

---

## 概述

### 本文档内容

本文档整合了所有Prompt生成相关的内容，包括：
- ✅ 三种Prompt生成方式的详细说明和对比
- ✅ VQA和Detect任务的Prompt生成策略
- ✅ 软标签和硬标签的利用方法
- ✅ 具体执行方法和最佳实践
- ✅ Detect任务的最新支持更新

### 适用人群

- 📌 **算法工程师**：需要生成高质量Prompt用于模型蒸馏
- 📌 **研究人员**：需要理解不同Prompt生成方式的优劣
- 📌 **项目维护者**：需要维护和优化Prompt模板

---

## 三种Prompt生成方式

### 对比总览

本项目支持三种Prompt生成方式：

| 方式 | 特点 | 适用场景 | 复杂度 | 推荐指数 |
|------|------|---------|--------|----------|
| **普通方式** | 基于模板，静态固定 | 快速原型、简单任务 | ⭐ 低 | ⭐⭐ |
| **DSPy方式** | 基于优化，自动调整 | 高质量需求、复杂推理 | ⭐⭐⭐ 高 | ⭐⭐⭐ |
| **任务导向方式** | 基于真实数据，任务定制 | 生产环境、高质量蒸馏 | ⭐⭐ 中 | ⭐⭐⭐⭐⭐ |

### 核心差异

```
普通方式：模板 → 填充变量 → 输出Prompt
         ↓
      固定不变，无需数据

DSPy方式：数据 → 优化算法 → 自动生成Prompt
         ↓
      自动调优，需要数据

任务导向方式：真实数据 → 任务分析 → 定制Prompt
             ↓
          高质量，任务特化
```

### 方式选择决策树

```
需求分析：
├─ 是否有真实标签数据？
│  ├─ 否 → 普通方式
│  └─ 是 → 继续判断
│
├─ 是否有充足算力（GPU + 时间）？
│  ├─ 否 → 任务导向方式
│  └─ 是 → 继续判断
│
├─ 是否需要最高质量？
│  ├─ 否 → 任务导向方式
│  └─ 是 → DSPy方式（可选）或 任务导向方式（推荐）
```

---

## 普通方式

### 原理

基于预定义的模板，通过变量填充生成Prompt。

### 配置文件

**位置**: [configs/prompts_en.yaml](../configs/prompts_en.yaml)

```yaml
prompts:
  standard:
    vqa: |
      Look at the image and answer: {question}
      Rules:
      1. ONE word only. For numbers, use English words (e.g., "one", "two", "three", NOT "1", "2", "3")
      2. No explanation
      Answer:

    detection: |
      Detect all objects. Output JSON only.
      Format: {'objects': [{'category': 'name', 'bbox': [x1,y1,x2,y2], 'confidence': 0.9}]}
      JSON:

  cot:
    vqa_system: |
      TASK: Answer the question by selecting exactly one answer from the given allowed answer list.
      
      STRICT MANDATORY RULES:
      1. Observation Writing Rule
        Only write visual features that directly help distinguish different candidate answers.
      
      2. Analysis Writing Rule
        Reason based on probability distribution, start reasoning with the highest probability primary answer.
      
      3. Output Format Absolute Restrictions
        - FORBID ALL MARKDOWN SYMBOLS
        - The output must contain only three plain text paragraphs
        - No extra line breaks, decorative symbols

    vqa_user: |
      Question: {question}
      ALLOWED ANSWERS: {allowed_answers}
      PRIMARY ANSWER: {primary_answer}
      PROBABILITY DISTRIBUTION: {answer_distribution}
      
      Observation: Focus on answer-relevant features
      Analysis: Use probability distribution to guide reasoning
      Conclusion: Final Answer: {primary_answer}
```

### 代码实现

**位置**: [src/models/teacher_model.py](../src/models/teacher_model.py)

```python
def _construct_prompt(self, question: str, task: str) -> str:
    """
    Construct task-specific prompt from configuration file.
    """
    # 加载配置
    prompts_config = self.config.get("prompts_config")
    
    # 选择模板
    if task == "vqa":
        template = prompts_config['cot']['vqa_user']
    elif task == "detection":
        template = prompts_config['cot']['detection_user']
    
    # 填充变量
    prompt = template.format(
        question=question,
        allowed_answers=allowed_answers,
        primary_answer=primary_answer,
        answer_distribution=distribution_str
    )
    
    return prompt
```

### 优缺点

**优点**：
- ✅ 简单直接：易于理解和实现
- ✅ 可控性强：完全控制Prompt内容
- ✅ 无需数据：不需要训练数据
- ✅ 调试方便：问题容易定位

**缺点**：
- ❌ 固定不变：无法自动优化
- ❌ 质量受限：依赖人工设计
- ❌ 适应性差：难以适应不同数据分布

### 适用场景

- ✅ 快速原型开发
- ✅ 简单任务（答案类型单一）
- ✅ 资源受限（无数据、无算力）
- ❌ 高质量需求
- ❌ 复杂推理任务

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
3. 选择优化算法（MIPROv2、BootstrapFewShot等）
   ↓
4. 运行优化（自动调优Prompt）
   ↓
5. 输出优化后的Prompt
```

### Signature定义

**位置**: [scripts/dspy_prompt_optimizer_v2.py](../scripts/dspy_prompt_optimizer_v2.py)

```python
class VQACoTSignatureV2(Signature):
    """
    VQA Chain-of-Thought推理签名 - 答案导向版本
    
    输入：
    - 图像路径
    - 问题
    - 允许的答案列表（从软标签提取）
    - 主要答案（硬标签）
    - 答案概率分布（软标签）
    
    输出：
    - 观察：聚焦于答案相关的视觉特征
    - 分析：基于概率分布的答案选择推理
    - 结论：最终答案
    """
    image_path: str = dspy.InputField(desc="Path to the image file")
    question: str = dspy.InputField(desc="Question about the image")
    allowed_answers: str = dspy.InputField(desc="Allowed answers")
    primary_answer: str = dspy.InputField(desc="Primary answer from hard label")
    answer_distribution: str = dspy.InputField(desc="Probability distribution")
    
    observation: str = dspy.OutputField(desc="Visual features")
    analysis: str = dspy.OutputField(desc="Reasoning")
    conclusion: str = dspy.OutputField(desc="Final answer")


class DetectionCoTSignatureV2(Signature):
    """
    Detection Chain-of-Thought推理签名
    
    输入：图像路径
    输出：扫描、对象、验证三个步骤
    """
    image_path: str = dspy.InputField(desc="Path to the image file")
    
    scanning: str = dspy.OutputField(desc="Quick scan for objects")
    objects: str = dspy.OutputField(desc="List of detected objects")
    verification: str = dspy.OutputField(desc="Verification of completeness")
```

### 执行命令

```bash
# 安装DSPy
pip install dspy-ai

# 运行VQA优化
python scripts/dspy_prompt_optimizer_v2.py --task vqa --num_samples 100

# 运行Detect优化
python scripts/dspy_prompt_optimizer_v2.py --task detection --num_samples 100

# 运行所有任务优化
python scripts/dspy_prompt_optimizer_v2.py --task all --num_samples 100
```

### 优缺点

**优点**：
- ✅ 自动优化：无需人工调参
- ✅ 适应性强：根据数据分布自动调整
- ✅ 高质量：通过算法优化达到最优
- ✅ 可解释：优化过程可追踪

**缺点**：
- ❌ 复杂度高：需要理解DSPy框架
- ❌ 资源消耗大：需要大量数据和算力
- ❌ 不稳定：优化结果可能不收敛
- ❌ 黑盒：优化过程不透明

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

### 为什么选择任务导向？

#### 对比表

| 维度 | 普通方式 | DSPy方式 | 任务导向方式（本项目） |
|------|---------|---------|---------------------|
| **数据依赖** | 无 | 高 | 中等 |
| **实现复杂度** | 低 | 高 | 中 |
| **质量** | 中等 | 高 | 高 |
| **可控性** | 高 | 低 | 中 |
| **稳定性** | 高 | 中 | 高 |
| **适应性** | 低 | 高 | 高 |

#### 选择原因

```
问题分析：
- VQA和Detect任务差异大
- 需要利用真实标签信息（软标签、硬标签）
- 需要任务特化（不同任务不同策略）
- 需要稳定性（生产环境）

结论：
- 普通方式：质量不够，无法利用真实数据
- DSPy方式：复杂度高，不稳定，资源消耗大
- 任务导向方式：平衡了质量、复杂度、稳定性 ✅
```

### 数据结构

#### VQA标签数据

```python
{
    'image_id': 123,
    'image_path': 'data/coco/val2014/...',
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

#### Detect标签数据

```python
{
    'image_id': 123,
    'image_path': 'data/coco/val2014/...',
    'hard_label': {
        'objects': [
            {
                'category': 'person',
                'confidence': 0.95,
                'bbox': [x1, y1, x2, y2]
            }
        ],
        'num_objects': 1
    },
    'soft_label': {
        'object_soft_labels': [
            {
                'category': 'person',
                'confidence': 0.95,
                'bbox': [x1, y1, x2, y2],
                'category_distribution': {
                    'person': 0.85,
                    'man': 0.10,
                    'woman': 0.05
                },
                'bbox_soft_label': {
                    'teacher_bbox': [x1, y1, x2, y2],
                    'confidence': 0.95,
                    'teacher_weight': 0.7
                }
            }
        ]
    }
}
```

### 执行命令

```bash
# 基于真实标签生成VQA Prompt
python scripts/generate_prompt_from_real_labels.py --task vqa --num_samples 100

# 基于真实标签生成Detect Prompt
python scripts/generate_prompt_from_real_labels.py --task detection --num_samples 100
```

### 优缺点

**优点**：
- ✅ 高质量：基于真实数据定制
- ✅ 任务特化：针对VQA/Detect优化
- ✅ 利用软标签：充分利用分布信息
- ✅ 可控可调：可根据需求调整
- ✅ 稳定可靠：生产环境验证

**缺点**：
- ❌ 需要数据：依赖真实标签
- ❌ 迭代成本：需要多次验证
- ❌ 人工参与：仍需人工审查

### 适用场景

- ✅ 生产环境（知识蒸馏）
- ✅ 高质量需求
- ✅ 有真实标签数据
- ✅ 需要任务特化

---

## VQA任务Prompt生成

### 核心特点

**任务目标**：回答视觉问题，输出单个答案

**关键信息**：
- 问题类型：数字、颜色、二元、位置、物体、动作等
- 答案空间：开放域（任何单词都可能是答案）
- 软标签：概率分布（多答案候选）
- 硬标签：正确答案 + 置信度

### Prompt设计策略

#### System Prompt（任务定义）

```yaml
vqa_system: |
  TASK: Answer the question by selecting ONE answer from the allowed list.
  
  RULES:
  1. Observation: Only describe features that distinguish answers
  2. Analysis: Compare candidates using probability distribution
  3. Conclusion: Output ONE answer from the allowed list
```

**关键设计**：
- ✅ 明确任务：从允许列表中选择答案
- ✅ 利用软标签：基于概率分布推理
- ✅ 简洁约束：避免冗余描述

#### User Prompt（具体问题）

```yaml
vqa_user: |
  Question: {question}
  Allowed: {allowed_answers}
  Primary: {primary_answer}
  Distribution: {answer_distribution}
  
  Observation: [视觉特征描述]
  Analysis: [基于分布的推理]
  Conclusion: [最终答案]
```

**关键设计**：
- ✅ 提供答案列表（从软标签提取）
- ✅ 提供概率分布（指导推理）
- ✅ 提供主要答案（硬标签参考）

### 软标签利用

**来源**: VQA软标签生成器

```python
soft_label = {
    'answer_distribution': {
        'one': 0.5234,    # 从logits提取，经过过滤
        'two': 0.4766,
        'three': 0.0001,
        ...
    },
    'primary_answer': 'one',          # 硬标签
    'allowed_answers': ['one', 'two', ...]  # 过滤后的有效答案
}
```

**如何利用**：

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
Distribution: one:0.52, two:0.32, three:0.10, zero:0.05, four:0.01

Observation: One person in the foreground wears headphones. No others visible.
Analysis: The primary answer 'one' has highest probability (0.52). Visual count confirms one person.
Conclusion: one
```

---

## Detect任务Prompt生成

### 核心特点

**任务目标**：检测图像中的所有对象，输出类别 + 边界框

**关键信息**：
- 类别空间：固定（COCO 80类 + background）
- 输出格式：结构化JSON
- 硬标签：检测到的对象列表
- 软标签：类别分布 + bbox软化

### Prompt设计策略

#### System Prompt（任务定义）

```yaml
detection_system: |
  TASK: Find objects through three-step process.
  
  OUTPUT FORMAT (exactly 3 paragraphs):
  Scanning: What you see overall
  Objects: List of detected items
  Verification: Confirm detection complete
  
  HARD CONSTRAINTS:
  1. Reasoning only - NO JSON output in text
  2. No coordinates in reasoning (handled separately)
  3. Plain sentences, no meta-formatting
```

**关键设计**：
- ✅ 三步骤流程：扫描 → 检测 → 验证
- ✅ 约束输出格式（纯文本推理）
- ✅ 避免在文本中输出JSON

#### User Prompt（具体问题）

```yaml
detection_user: |
  Find objects in this image:
  
  Scanning: [整体扫描]
  Objects: [检测到的对象列表]
  Verification: [验证完整性]
```

**关键设计**：
- ✅ 简洁直接（不需要复杂的约束）
- ✅ 三步骤引导（结构化推理）

### 软标签利用

**来源**: Detect软标签生成器

```python
soft_label = {
    'object_soft_labels': [
        {
            'category': 'person',
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.85,
            'category_distribution': {
                'person': 0.70,
                'people': 0.30
            },
            'bbox_soft_label': {
                'teacher_bbox': [x1, y1, x2, y2],
                'soft_bbox': [x1', y1', x2', y2'],
                'teacher_weight': 0.7
            }
        },
        ...
    ],
    'num_objects': 3
}
```

**如何利用**：

```python
# Detect任务的Prompt生成相对简单
# 主要关注：
# 1. 对象列表（硬标签）
# 2. 置信度信息（可选）

objects = hard_label['objects']
num_objects = len(objects)

# Prompt中不直接使用软标签
# 软标签主要用于：
# 1. 类别分布（category_distribution）
# 2. 边界框软化（bbox_soft_label）
```

### Detect Prompt示例

```
System:
TASK: Find objects through three-step process.

OUTPUT FORMAT:
Scanning: What you see overall
Objects: List of detected items
Verification: Confirm detection complete

User:
Find objects in this image:

Scanning: The image shows a street scene with several people and vehicles.
Objects: 3 persons detected - one in foreground left, two in background. 1 car on the right side.
Verification: All significant objects detected. No additional items visible.
```

---

## VQA与Detect对比

### 核心差异对比表

| 维度 | VQA任务 | Detect任务 |
|------|---------|-----------|
| **任务类型** | 视觉问答（单答案） | 目标检测（多对象） |
| **输出格式** | 单词/数字 | 结构化JSON |
| **Prompt复杂度** | 高（需要软标签信息） | 低（简单引导） |
| **软标签利用** | 高（概率分布） | 中（类别分布） |
| **硬标签利用** | 主要答案 | 对象列表 |
| **约束重点** | 答案选择、避免幻觉 | 结构化输出、完整性 |

### Prompt生成流程对比

#### VQA流程

```
真实标签数据（硬标签 + 软标签 + CoT）
    ↓
提取关键信息：
  - 问题文本
  - 允许答案列表（从软标签过滤）
  - 概率分布（软标签）
  - 主要答案（硬标签）
    ↓
填充Prompt模板：
  {question}
  {allowed_answers}
  {answer_distribution}
  {primary_answer}
    ↓
输出：答案导向的Prompt
```

#### Detect流程

```
真实标签数据（硬标签 + 软标签）
    ↓
提取关键信息：
  - 对象列表（硬标签）
  - 类别分布（软标签）
  - 边界框信息（硬标签）
    ↓
填充Prompt模板：
  Find objects in this image:
  Scanning:
  Objects:
  Verification:
    ↓
输出：检测导向的Prompt
```

### 软标签利用差异

#### VQA软标签利用（高）

```python
# Prompt中直接使用软标签信息
User Prompt:
  Allowed: {allowed_answers}              # 从软标签提取
  Distribution: {answer_distribution}      # 软标签概率分布
  Primary: {primary_answer}               # 硬标签答案
```

#### Detect软标签利用（中）

```python
# Prompt中不直接使用软标签
# 主要用于后处理：
# 1. 类别分布分析
# 2. 边界框软化
```

---

## 执行方法

### 普通方式执行

```bash
# 直接使用模板（已在teacher_model.py中集成）
# 无需单独执行，推理时自动加载

# 修改模板
vim configs/prompts_en.yaml

# 重新运行蒸馏
python scripts/run_distillation.py --task vqa --num_samples 100
```

### DSPy方式执行

```bash
# 安装依赖
pip install dspy-ai

# 运行VQA优化
python scripts/dspy_prompt_optimizer_v2.py \
    --task vqa \
    --num_train 50 \
    --num_test 20

# 运行Detect优化
python scripts/dspy_prompt_optimizer_v2.py \
    --task detection \
    --num_train 50 \
    --num_test 20

# 运行所有任务优化
python scripts/dspy_prompt_optimizer_v2.py \
    --task all \
    --num_train 50 \
    --num_test 20
```

### 任务导向方式执行（推荐）

```bash
# Step 1: 生成真实标签数据
python scripts/run_distillation.py \
    --task vqa \
    --task detection \
    --num_samples 1000

# Step 2: 基于真实标签生成Prompt
python scripts/generate_prompt_from_real_labels.py \
    --task vqa \
    --num_samples 500

python scripts/generate_prompt_from_real_labels.py \
    --task detection \
    --num_samples 500

# Step 3: 查看生成的Prompt
ls -lh configs/generated_prompts/

# Step 4: 应用到配置
cp configs/generated_prompts/vqa_real_labels_*.yaml configs/prompts_en.yaml
```

### 执行时间对比

| 方式 | 准备时间 | 执行时间 | 总时间 |
|------|---------|---------|--------|
| **普通方式** | 0分钟 | 0分钟 | ~0分钟 |
| **DSPy方式** | 30分钟（数据） | 60-180分钟 | 90-210分钟 |
| **任务导向** | 30分钟（数据） | 5-10分钟 | 35-40分钟 |

---

## 最佳实践

### 推荐方案

**生产环境**：
- ✅ 使用**任务导向方式**
- ✅ VQA任务需要500+样本
- ✅ Detect任务需要200+样本
- ✅ 定期迭代优化

**快速原型**：
- ✅ 使用**普通方式**
- ✅ 快速验证想法
- ✅ 后期迁移到任务导向

**研究实验**：
- ✅ 可尝试**DSPy方式**
- ✅ 需要充足资源
- ✅ 关注优化稳定性

### VQA任务最佳实践

```bash
# 推荐：任务导向方式
# 1. 生成足够的数据（至少500-1000样本）
python scripts/run_distillation.py --task vqa --num_samples 1000

# 2. 基于真实数据生成Prompt
python scripts/generate_prompt_from_real_labels.py --task vqa --num_samples 500

# 3. 验证Prompt质量
python scripts/test_soft_label_fix.py

# 4. 迭代优化（如果质量不满意）
# - 调整过滤参数
# - 增加训练数据
# - 重新生成Prompt
```

### Detect任务最佳实践

```bash
# 推荐：任务导向方式（Detect相对简单）
# 1. 生成数据（200-500样本即可）
python scripts/run_distillation.py --task detection --num_samples 500

# 2. 基于真实数据生成Prompt
python scripts/generate_prompt_from_real_labels.py --task detection --num_samples 200

# 3. 验证检测质量
python scripts/run_distillation.py --task detection --num_samples 50 --validate
```

### 参数调优建议

#### VQA参数

```yaml
# configs/default.yaml
distillation:
  soft_labels:
    temperature: 4          # 控制分布平滑度（2-8）
    top_k_logits: 50        # Top-K兜底数量（30-100）
    
vqa_token_filter:
  config_file: "configs/vqa_token_filter.yaml"
```

#### Detect参数

```yaml
# configs/default.yaml
distillation:
  soft_labels:
    temperature: 4
    
detect:
  categories:               # COCO 80类 + background
    - person
    - car
    ...
    - background
```

### 常见问题与解决

#### 问题1：VQA Prompt质量低

**症状**：模型输出包含噪音、幻觉

**解决**：
```bash
# 1. 检查软标签质量
python scripts/diagnose_soft_labels.py

# 2. 调整过滤参数
vim configs/vqa_token_filter.yaml  # 增加黑名单

# 3. 增加训练数据
python scripts/run_distillation.py --task vqa --num_samples 2000

# 4. 重新生成Prompt
python scripts/generate_prompt_from_real_labels.py --task vqa --num_samples 1000
```

#### 问题2：Detect Prompt不完整

**症状**：模型漏检对象

**解决**：
```bash
# 1. 检查硬标签质量
python scripts/validate_detection_labels.py

# 2. 调整置信度阈值
vim configs/default.yaml  # 降低 confidence_threshold

# 3. 增加Few-shot示例
vim configs/prompts_en.yaml  # 增加检测示例
```

---

## 总结

### 三种方式对比总结

| 方式 | 质量 | 复杂度 | 时间 | 推荐指数 |
|------|------|--------|------|----------|
| **普通方式** | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **DSPy方式** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **任务导向方式** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### 关键要点

**VQA任务**：
1. 复杂度高：需要引导模型基于分布推理
2. 高度依赖软标签：概率分布、允许答案列表
3. 需要强约束：避免幻觉、限定答案范围

**Detect任务**：
1. 复杂度低：任务目标明确
2. 中度依赖软标签：主要用于后处理
3. 约束较少：输出格式固定

---

## 📁 相关文件

### 脚本文件

| 文件 | 说明 |
|------|------|
| [scripts/generate_prompt_from_real_labels.py](../scripts/generate_prompt_from_real_labels.py) | 任务导向Prompt生成脚本 |
| [scripts/dspy_prompt_optimizer_v2.py](../scripts/dspy_prompt_optimizer_v2.py) | DSPy Prompt优化脚本 |
| [src/models/teacher_model.py](../src/models/teacher_model.py) | Teacher模型（Prompt加载） |

### 配置文件

| 文件 | 说明 |
|------|------|
| [configs/prompts_en.yaml](../configs/prompts_en.yaml) | Prompt配置文件 |
| [configs/default.yaml](../configs/default.yaml) | 默认配置文件 |
| [configs/vqa_token_filter.yaml](../configs/vqa_token_filter.yaml) | VQA Token过滤器配置 |

### 生成输出

| 文件 | 说明 |
|------|------|
| `configs/generated_prompts/vqa_real_labels_*.yaml` | VQA生成的Prompt |
| `configs/generated_prompts/detection_real_labels_*.yaml` | Detect生成的Prompt |
| `configs/generated_prompts/dspy_v2_*_prompts_*.yaml` | DSPy优化后的Prompt |

---

**文档版本**: v2.0
**最后更新**: 2026-07-24
**维护者**: AI Assistant
**整合说明**: 整合了Prompt生成完整指南和Detect任务支持更新