# DSPy Prompt自动生成工具 - 完整指南

**版本**: v1.0  
**更新时间**: 2026-07-21  
**作者**: Claude Code Assistant

---

## 📋 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [方案对比](#方案对比)
4. [详细使用指南](#详细使用指南)
5. [Prompt设计原则](#prompt设计原则)
6. [高级用法](#高级用法)
7. [模型问题诊断](#模型问题诊断)
8. [故障排查](#故障排查)
9. [最佳实践](#最佳实践)
10. [附录](#附录)

---

## 概述

### 项目背景

在VQA和Detection任务的CoT（Chain-of-Thought）标签生成中，prompt的质量直接影响生成效果。手动优化prompt费时费力，且难以找到最优模板。

本工具提供**两种自动化方案**来生成和优化prompt：

1. **简化版（推荐）**：基于数据模式分析的模板化方案
2. **DSPy版（进阶）**：使用DSPy框架的完整自动化方案

### 核心功能

- ✅ 自动分析COCO数据模式（对象、颜色、场景等）
- ✅ 自动生成优化的prompt模板
- ✅ 自动更新配置文件
- ✅ 支持VQA和Detection任务
- ✅ 提供质量评估和测试工具

### 创建的文件

#### 核心脚本

| 文件 | 功能 | 需要模型 | 推荐度 |
|------|------|---------|--------|
| [scripts/simple_prompt_generator.py](../scripts/simple_prompt_generator.py) | 简化版生成器 | ❌ 不需要 | ⭐⭐⭐⭐⭐ |
| [scripts/dspy_prompt_optimizer.py](../scripts/dspy_prompt_optimizer.py) | DSPy完整版 | ✅ 需要 | ⭐⭐⭐ |
| [scripts/quick_start.sh](../scripts/quick_start.sh) | 一键启动脚本 | ❌ 不需要 | ⭐⭐⭐⭐ |
| [scripts/test_generated_prompts.py](../scripts/test_generated_prompts.py) | Prompt测试工具 | ✅ 需要 | ⭐⭐⭐ |
| [scripts/diagnose_model.py](../scripts/diagnose_model.py) | 模型诊断工具 | ❌ 不需要 | ⭐⭐⭐ |
| [scripts/fix_model_config.sh](../scripts/fix_model_config.sh) | 配置修复脚本 | ❌ 不需要 | ⭐⭐⭐ |

---

## 快速开始

### 方案一：一键生成（最简单）

```bash
# 1. 准备COCO数据（首次使用，约1GB）
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d data/coco/

# 2. 运行快速启动脚本
chmod +x scripts/quick_start.sh
./scripts/quick_start.sh

# 3. 查看生成的prompt
cat configs/prompts_en.yaml
```

### 方案二：分步执行

```bash
# Step 1: 分析数据模式
python scripts/simple_prompt_generator.py --mode analyze --num_samples 1000

# Step 2: 生成Prompt
python scripts/simple_prompt_generator.py --mode generate --task vqa

# Step 3: 测试效果（可选，需要模型）
python scripts/test_generated_prompts.py \
    --image data/coco/val2014/COCO_val2014_000000000042.jpg \
    --task vqa \
    --question "Is there a person?"
```

### 执行时间估算

| 步骤 | 时间 | 说明 |
|------|------|------|
| 数据准备 | 5-10分钟 | COCO数据下载（约1GB） |
| 数据分析 | 1-3分钟 | 取决于样本数 |
| Prompt生成 | 几秒 | 基于模板生成 |
| **总计** | **约10分钟** | 首次使用 |

---

## 方案对比

### 特性对比

| 特性 | 简化版本 | DSPy版本 |
|------|---------|---------|
| **自动化程度** | 中（需要模式分析） | 高（完全自动） |
| **稳定性** | 高（基于模板） | 较低（依赖框架） |
| **灵活性** | 中（模板固定） | 高（可自定义优化） |
| **学习曲线** | 平缓（模板易懂） | 较陡（需要学习DSPy） |
| **是否需要模型** | ❌ 不需要 | ✅ 需要 |
| **执行速度** | 快（几秒） | 慢（几分钟到几小时） |
| **推荐场景** | 快速迭代开发 | 大规模自动化 |

### 推荐选择

**推荐使用简化版本**，理由：
1. ✅ **无需下载40GB模型**
2. ✅ **执行速度快**（几秒完成）
3. ✅ **更稳定**（不依赖复杂框架）
4. ✅ **易于理解和调试**

DSPy版本适合：
- 需要完全自动化的场景
- 有大量训练数据
- 对prompt质量有极高要求

---

## 详细使用指南

### 1. 数据准备

#### 下载COCO数据集

```bash
# 创建数据目录
mkdir -p data/coco

# 下载图像（约1GB）
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/coco/

# 下载标注（约200MB）
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d data/coco/
```

#### 验证数据完整性

```bash
# 检查图像数量
ls data/coco/val2014/ | wc -l  # 应该约5000张

# 检查标注文件
ls -lh data/coco/annotations/captions_val2014.json  # 应该约200MB
```

### 2. 数据分析

分析COCO数据模式，提取常见对象、颜色、场景等特征：

```bash
python scripts/simple_prompt_generator.py \
    --mode analyze \
    --coco_root data/coco/val2014 \
    --annotation data/coco/annotations/captions_val2014.json \
    --num_samples 1000
```

**参数说明**：
- `--mode analyze`: 分析模式
- `--coco_root`: COCO图像目录
- `--annotation`: COCO标注文件
- `--num_samples`: 分析样本数（建议1000-5000）

**输出示例**：
```
分析结果:
------------------------------------------------------------
常见对象: ['person', 'car', 'chair', 'dog', 'cat', 'bicycle', 'cup', 'bottle', 'table', 'bird']
常见颜色: ['white', 'black', 'red', 'blue', 'green']
常见场景: ['indoor', 'outdoor', 'street', 'room', 'kitchen']
常见动作: ['standing', 'sitting', 'walking', 'holding', 'wearing']

✓ 分析结果已保存到: configs/generated_prompts/pattern_analysis.json
```

### 3. Prompt生成

基于分析结果生成优化的prompt：

```bash
python scripts/simple_prompt_generator.py \
    --mode generate \
    --task vqa \
    --num_samples 1000
```

**参数说明**：
- `--mode generate`: 生成模式
- `--task vqa`: 任务类型（vqa/detection/all）
- `--num_samples`: 分析样本数

**生成文件**：
- `configs/generated_prompts/optimized_vqa_prompts_YYYYMMDD_HHMMSS.yaml`
- 自动更新 `configs/prompts_en.yaml`
- 自动备份原配置

### 4. Prompt测试（可选）

测试生成的prompt效果：

```bash
python scripts/test_generated_prompts.py \
    --image data/coco/val2014/COCO_val2014_000000000042.jpg \
    --task vqa \
    --question "Is there a person in the image?" \
    --answers "yes,no" \
    --primary "yes"
```

**注意**：测试需要加载模型。

---

## Prompt设计原则

### 1. VQA CoT Prompt结构

#### System Prompt（系统规则）

包含以下要素：

```yaml
vqa_system: |
  # 1. 任务定义
  TASK: Answer visual questions through structured three-step reasoning.

  # 2. 输出格式
  OUTPUT FORMAT (exactly 3 paragraphs):
  Observation: Describe what you see (objects, colors, counts, positions)
  Analysis: Connect observations to the question
  Conclusion: State the answer clearly

  # 3. 硬性约束
  HARD CONSTRAINTS:
  1. ANSWER FORMAT: Final answer MUST be ONE word or number from allowed answers
  2. NO SPECULATION: Use only what you actually see
  3. FORBIDDEN WORDS: appear, seem, look like, suggest
  4. NO META-CONTENT: No JSON, no braces, no quotes

  # 4. 数据驱动内容
  COMMON OBJECTS: person, car, chair, dog, cat, bicycle...
  COMMON SCENES: indoor, outdoor, street, room, kitchen
  COMMON COLORS: white, black, red, blue, green

  # 5. 质量标准
  QUALITY STANDARD:
  Observation: Pure visual facts (what, where, how many)
  Analysis: Logical reasoning connecting observation to question
  Conclusion: Clear, concise answer matching allowed answers

  # 6. 具体示例
  EXAMPLES:
  Question: Is there a dog?
  Observation: I see a living room with an animal on the floor.
  Analysis: The animal has four legs and dog features.
  Conclusion: Final Answer: yes
```

#### User Prompt（具体问题）

```yaml
vqa_user: |
  Question: {question}
  Allowed answers: {allowed_answers}
  Required answer: {primary_answer}

  Write exactly three paragraphs:

  Observation:
  [Describe what you see]

  Analysis:
  [Connect to the question]

  Conclusion:
  Final Answer: {primary_answer}
```

### 2. 关键优化点

#### ✅ 好的Prompt特征

1. **明确的输出格式**
   ```yaml
   OUTPUT FORMAT (exactly 3 paragraphs):
   Observation: [纯视觉描述]
   Analysis: [逻辑推理]
   Conclusion: [清晰答案]
   ```

2. **硬性约束**
   ```yaml
   HARD CONSTRAINTS:
   1. ANSWER FORMAT: Final answer MUST be ONE word
   2. NO SPECULATION: Use only what you actually see
   3. FORBIDDEN WORDS: appear, seem, look like, suggest
   4. NO META-CONTENT: No JSON, no braces, no quotes
   ```

3. **具体示例**
   ```yaml
   EXAMPLES:
   Question: Is there a dog?
   Observation: I see a living room with an animal.
   Analysis: It has dog features.
   Conclusion: Final Answer: yes
   ```

#### ❌ 避免的问题

1. **过于模糊的指令**
   ```yaml
   # 不好
   vqa_user: "Answer the question: {question}"

   # 好
   vqa_user: |
     Question: {question}
     Allowed answers: {allowed_answers}

     Write exactly three paragraphs:
     Observation: [what you see]
     Analysis: [logical connection]
     Conclusion: Final Answer: {primary_answer}
   ```

2. **缺少示例**
   ```yaml
   # 不好 - 没有示例
   system: "Answer the question step by step."

   # 好 - 包含具体示例
   system: |
     TASK: Answer visual questions.

     EXAMPLES:
     Question: Is there a dog?
     Observation: I see an animal with fur.
     Analysis: It has dog features.
     Conclusion: Final Answer: yes
   ```

3. **缺少约束**
   ```yaml
   # 不好 - 模型可能输出JSON
   vqa_system: "Analyze the image."

   # 好 - 明确禁止非期望格式
   vqa_system: |
     NO META-CONTENT: No JSON, no braces, no quotes.
     Output plain text only.
   ```

---

## 高级用法

### 1. 自定义优化目标

修改 `simple_prompt_generator.py` 添加特定约束：

```python
def generate_vqa_system_prompt(self, patterns: Dict) -> str:
    """自定义系统prompt生成逻辑"""

    # 基础prompt
    base_prompt = self._get_base_prompt()

    # 根据任务特点添加特定约束
    custom_constraints = """
    ADDITIONAL CONSTRAINTS FOR YOUR TASK:
    5. Always mention object counts if relevant
    6. Describe spatial relationships (left, right, center)
    7. Include confidence indicators for uncertain observations
    """

    return base_prompt + custom_constraints
```

### 2. 使用真实数据

加载真实的hard_label和soft_label数据：

```python
# 在 simple_prompt_generator.py 中添加

def load_real_training_data(self):
    """加载真实的训练数据"""
    import json

    # 加载hard_label数据
    hard_labels = json.load(open('data/hard_labels.json'))

    # 加载soft_label数据
    soft_labels = json.load(open('data/soft_labels.json'))

    # 提取问题和答案模式
    question_answer_pairs = []

    for item in hard_labels['vqa']:
        question = item['question']
        answer = item['answer']
        allowed_answers = list(soft_labels['vqa'][item['id']]['answer_distribution'].keys())

        question_answer_pairs.append({
            'question': question,
            'answer': answer,
            'allowed_answers': allowed_answers
        })

    return question_answer_pairs
```

### 3. 迭代优化流程

```bash
# 1. 分析数据
python scripts/simple_prompt_generator.py --mode analyze

# 2. 生成初始prompt
python scripts/simple_prompt_generator.py --mode generate

# 3. 在少量数据上测试
python scripts/simple_prompt_generator.py --mode test --test_image <图像路径>

# 4. 根据测试结果调整prompt模板（手动编辑yaml）

# 5. 重新生成并测试（循环迭代）
```

### 4. 增加分析样本数

更准确的数据模式分析：

```bash
# 默认1000样本
python scripts/simple_prompt_generator.py --mode analyze --num_samples 1000

# 增加到5000样本（更准确）
python scripts/simple_prompt_generator.py --mode analyze --num_samples 5000

# 使用全部数据（最准确，但耗时）
python scripts/simple_prompt_generator.py --mode analyze --num_samples 5000
```

---

## 模型问题诊断

### 问题现象

运行 `dspy_prompt_optimizer.py` 时报错：

```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

### 问题原因

模型文件是**Git LFS占位符**，实际文件没有下载。

检查模型状态：

```bash
# 检查tokenizer.json大小
ls -lh models/Qwen2.5-VL-72B-Instruct-AWQ/tokenizer.json
# 应该约11MB，如果只有133字节说明未下载

# 查看内容
cat models/Qwen2.5-VL-72B-Instruct-AWQ/tokenizer.json
# 如果看到 "version https://git-lfs.github.com" 说明是占位符
```

### 你的模型状态

| 模型 | 状态 | tokenizer大小 | 能否使用 |
|------|------|-------------|---------|
| 7B模型 | ✅ 已下载 | 6.8MB | ✅ 可以使用 |
| 72B模型 | ❌ Git LFS占位符 | 133字节 | ❌ 无法使用 |

### 为什么之前能运行？

你的 `teacher_model.py` 默认使用：

```python
self.model_name = model_name or self.config.get("teacher.model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
```

**默认使用7B模型（已下载）**，而不是72B模型（未下载）。

### 解决方案

#### 方案1：使用简化版（推荐，无需模型）

```bash
# 不需要下载模型，直接生成prompt
python scripts/simple_prompt_generator.py --mode generate --task vqa
```

#### 方案2：使用已下载的7B模型

修改配置文件：

```bash
# 自动修复
chmod +x scripts/fix_model_config.sh
./scripts/fix_model_config.sh

# 或手动修改 configs/model_config.yaml
# 将: name: "models/Qwen2.5-VL-72B-Instruct-AWQ"
# 改为: name: "models/Qwen2.5-VL-7B-Instruct"
```

#### 方案3：下载72B模型

```bash
# 使用Git LFS下载
cd models/Qwen2.5-VL-72B-Instruct-AWQ
git lfs pull  # 下载约40GB

# 或使用HuggingFace CLI
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --local-dir models/Qwen2.5-VL-72B-Instruct-AWQ
```

### 运行诊断

```bash
python scripts/diagnose_model.py --model models/Qwen2.5-VL-72B-Instruct-AWQ
```

---

## 故障排查

### 问题1：COCO数据未找到

**症状**：
```
⚠ COCO数据集未找到
```

**解决方案**：
```bash
# 下载COCO数据
mkdir -p data/coco
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d data/coco/
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d data/coco/
```

### 问题2：Prompt效果不好

**症状**：
- 生成的CoT格式不正确
- 模型输出JSON而不是纯文本
- 答案不准确

**解决方案**：
```bash
# 1. 增加分析样本数
python scripts/simple_prompt_generator.py --num_samples 5000

# 2. 手动编辑yaml添加更多示例
vim configs/prompts_en.yaml

# 3. 使用真实hard_label数据
# 修改 simple_prompt_generator.py 的 load_real_training_data 方法

# 4. 测试并迭代
python scripts/simple_prompt_generator.py --mode test --test_image <图像>
```

### 问题3：DSPy导入失败

**症状**：
```
⚠ DSPy未安装
```

**解决方案**：
```bash
# 方案A：安装DSPy
pip install dspy-ai

# 方案B：使用简化版本（推荐）
python scripts/simple_prompt_generator.py --mode generate
```

### 问题4：配置文件未更新

**症状**：
- 生成的prompt没有应用到配置文件

**解决方案**：
```bash
# 检查文件权限
ls -l configs/prompts_en.yaml

# 手动复制
cp configs/generated_prompts/optimized_vqa_prompts_*.yaml configs/prompts_en.yaml
```

### 问题5：模型加载失败

**症状**：
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**解决方案**：
```bash
# 运行诊断
python scripts/diagnose_model.py --model models/Qwen2.5-VL-72B-Instruct-AWQ

# 使用简化版（无需模型）
python scripts/simple_prompt_generator.py --mode generate

# 或使用7B模型
./scripts/fix_model_config.sh
```

---

## 最佳实践

### 1. 开发流程

```bash
# 1. 从小数据开始测试
python scripts/simple_prompt_generator.py --num_samples 100

# 2. 验证流程正常
cat configs/prompts_en.yaml

# 3. 增加样本数优化
python scripts/simple_prompt_generator.py --num_samples 1000

# 4. 手动调整prompt
vim configs/prompts_en.yaml

# 5. 测试效果
python scripts/test_generated_prompts.py --image <测试图像>
```

### 2. 迭代优化策略

1. **生成**：自动生成初始prompt
2. **测试**：在少量数据上测试效果
3. **分析**：找出问题（格式、内容、约束）
4. **调整**：手动编辑yaml文件
5. **重新生成**：如果需要，重新生成

### 3. 质量监控

使用quality_score监控生成质量：

```python
# 在测试输出中查看
quality_score: 0.85 / 1.00

# 评分标准：
# - 长度要求：0.25分
# - 格式要求：0.25分
# - 禁止词汇：0.25分
# - 答案匹配：0.25分
```

### 4. 配置管理

```bash
# 每次生成都会自动备份
ls configs/prompts_en.yaml.backup_*

# 恢复到之前版本
cp configs/prompts_en.yaml.backup_20260721_143022 configs/prompts_en.yaml
```

### 5. 性能优化

| 场景 | 推荐样本数 | 执行时间 |
|------|----------|---------|
| 快速测试 | 100 | 几秒 |
| 正常使用 | 1000 | 1-2分钟 |
| 高质量生成 | 5000 | 5-10分钟 |
| 最佳效果 | 10000+ | 10-20分钟 |

---

## 附录

### 生成的Prompt示例

#### VQA System Prompt

```yaml
TASK: Answer visual questions through structured three-step reasoning.

OUTPUT FORMAT (exactly 3 paragraphs):
Observation: Describe what you see (objects, colors, counts, positions)
Analysis: Connect observations to the question
Conclusion: State the answer clearly

HARD CONSTRAINTS:
1. ANSWER FORMAT: Final answer MUST be ONE word or number from allowed answers
2. NO SPECULATION: Use only what you actually see in the image
3. FORBIDDEN WORDS: appear, seem, look like, suggest, possible, probably, might
4. NO META-CONTENT: No JSON, no braces, no quotes, no markdown
5. BE SPECIFIC: Use concrete descriptions, not vague phrases

COMMON OBJECTS (for reference):
person, car, chair, dog, cat, bicycle, cup, bottle, table, bird

COMMON SCENES:
indoor, outdoor, street, room, kitchen

COMMON COLORS:
white, black, red, blue, green

QUALITY STANDARD:
Observation: Pure visual facts (what, where, how many)
Analysis: Logical reasoning connecting observation to question
Conclusion: Clear, concise answer matching allowed answers

EXAMPLES:

Example 1 - Object Detection:
Question: Is there a dog in the image?
Allowed answers: yes, no
Observation: I see a living room with a couch, a coffee table, and an animal sitting on the floor.
Analysis: The animal has four legs, fur, and distinctive dog features like floppy ears and a wagging tail.
Conclusion: Final Answer: yes

Example 2 - Color Recognition:
Question: What color is the car?
Allowed answers: red, blue, green, black, white, other
Observation: There is a vehicle parked on the street. The vehicle has a glossy finish.
Analysis: Looking at the car's paint, it has a bright, warm tone. The color is clearly red.
Conclusion: Final Answer: red
```

#### VQA User Prompt

```yaml
Question: {question}
Allowed answers: {allowed_answers}
Required answer: {primary_answer}

Write exactly three paragraphs:

Observation:
[Describe what you see in the image]

Analysis:
[Connect your observations to the question]

Conclusion:
Final Answer: {primary_answer}
```

### 参考资料

- [DSPy官方文档](https://dspy.ai/)
- [COCO数据集官网](https://cocodataset.org/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Qwen模型文档](https://huggingface.co/Qwen)

### 相关文档

- [模型下载问题修复](模型下载问题修复.md)
- [模型状态详解](模型状态详解.md)
- [CoT删除reasoning_steps冗余字段](CoT删除reasoning_steps冗余字段.md)
- [JSON输出格式_最终版](JSON输出格式_最终版.md)

---

## 总结

### 推荐工作流程

1. **立即开始**：使用简化版生成prompt（无需模型）
   ```bash
   ./scripts/quick_start.sh
   ```

2. **验证效果**：检查生成的prompt
   ```bash
   cat configs/prompts_en.yaml
   ```

3. **运行标签生成**：使用新prompt生成CoT标签
   ```bash
   python src/distillation/distiller.py
   ```

### 核心优势

- ✅ **无需手动编写**：自动生成优化的prompt
- ✅ **数据驱动**：基于真实COCO数据分析
- ✅ **快速迭代**：几分钟完成生成和测试
- ✅ **无缝集成**：自动更新配置文件
- ✅ **稳定可靠**：基于模板，不依赖复杂框架

---

**文档版本**: v1.0  
**最后更新**: 2026-07-21

如有问题，请查看故障排查章节或运行诊断工具：
```bash
python scripts/diagnose_model.py
```