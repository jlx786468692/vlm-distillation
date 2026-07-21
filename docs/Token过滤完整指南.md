# Token过滤完整指南

## 📋 目录

1. [策略概述](#策略概述)
2. [三层防护逻辑](#三层防护逻辑)
3. [设计思路](#设计思路)
4. [完整处理流程](#完整处理流程)
5. [核心实现步骤](#核心实现步骤)
6. [代码实现详解](#代码实现详解)
7. [等价Token合并](#等价token合并)
8. [配置参数说明](#配置参数说明)
9. [效果示例](#效果示例)
10. [问题诊断与修复](#问题诊断与修复)
11. [测试验证](#测试验证)
12. [性能影响](#性能影响)
13. [相关文件](#相关文件)

---

## 策略概述

本项目采用**黑名单优先、硬标签保护、Top-K兜底**的三层防护策略，用于VQA软标签生成中的噪音token剔除。

> **核心原则**：拦截绝对不可能是单字答案的Token，确保正确答案永不丢失，保留合理的候选集。

### 核心目标

1. **精准过滤**：拦截BPE碎片、标点符号、特殊Token等噪音
2. **安全兜底**：硬标签保护确保正确答案永不丢失
3. **多样性保障**：Top-K兜底防止过度过滤
4. **概率合并**：避免同一答案的多个表示分散概率

### 三层防护优先级

```
┌─────────────────────────────────────────────────────────┐
│  第一层：黑名单（核心防线）                              │
│    ├─ BPE子词碎片：'Ġt', 'Ġa', 'ur', 'qu', 'ing'...   │
│    ├─ 标点符号：'.', ',', '!', '?'...                  │
│    ├─ 特殊Token：'<pad>', '</s>', '<unk>'...          │
│    ├─ 纯空格/空字符串：'', '   '                        │
│    └─ 纯符号组合：'@#', '$%'...                        │
│    【目标】拦截绝对不可能是单字答案的Token               │
├─────────────────────────────────────────────────────────┤
│  第二层：硬标签保护（安全网）                            │
│    └─ 强制保留hard_label_id                             │
│    【目标】确保正确答案永不丢失                          │
├─────────────────────────────────────────────────────────┤
│  第三层：Top-K兜底（多样性保障）                         │
│    └─ 如果过滤后<10个token，从Top-K补充                │
│    【目标】防止过度过滤，保留候选集                      │
└─────────────────────────────────────────────────────────┘
```

### 为什么采用黑名单而不是白名单？

| 对比项 | 白名单策略 | 黑名单策略（本项目） |
|--------|-----------|---------------------|
| **核心思路** | 只保留预定义的有效答案 | 拦截明确的噪音Token |
| **适用场景** | 答案类型明确且有限（数字、颜色、二元） | VQA答案多样，难以预定义所有有效答案 |
| **优点** | 过滤精准，噪音少 | 不遗漏未预定义的有效答案 |
| **缺点** | 可能遗漏未知类型的有效答案 | 可能保留少量噪音 |
| **本项目选择** | ❌ 不适用 | ✅ 适用（答案类型多样） |

**关键原因**：
- VQA答案类型多样：数字、颜色、位置、大小、物体、动作等
- 无法预定义所有可能的有效答案
- 采用黑名单策略，只拦截明确的噪音，保留更广泛的候选集

---

## 三层防护逻辑

> **修复日期**: 2026-07-17  
> **修复原因**: 根据业界标准的VQA过滤实践（Blacklist + Hard Label Protection + Top-K Fallback），完善了代码实现

### 三层防护架构

```
┌─────────────────────────────────────────────────────────┐
│  第一层：黑名单（核心防线）                              │
│  ├─ BPE子词碎片（'Ġt', 'Ġa', 'ur', 'qu', 'ing'...）   │
│  ├─ 标点符号（'.', ',', '!', '?'...）                  │
│  ├─ 特殊Token（'<pad>', '</s>', '<unk>'...）          │
│  ├─ 纯空格/空字符串                                     │
│  └─ 纯符号组合（'@#', '$%'...）                        │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  第二层：硬标签保护（安全网）                            │
│  └─ 强制保留hard_label_id，确保正确答案永不丢失         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  第三层：Top-K兜底（多样性保障）                         │
│  └─ 如果过滤后<10个token，从Top-K补充                   │
└─────────────────────────────────────────────────────────┘
```

### 第一层：黑名单（核心防线）

**目标**：拦截绝对不可能是单字答案的Token

**拦截对象**（VQA中99%的噪音来源）：

#### 1. BPE子词碎片（最主要）

```python
# 这些在Qwen/BERT词表中数量巨大（约占词表30%~50%）
# 但永远不可能单独作为VQA答案
'Ġt', 'Ġa', 'ur', 'qu', 'ing', 'er', 'th', 'Ġthe', ...
```

**关键特征**：
- 以 `Ġ` 开头（BPE子词标记）
- 1-2字母的碎片（如 'ur', 'qu', 'te'）
- 纯子词片段（如 'ing', 'er', 'th'）

#### 2. 标点符号与特殊字符

```python
'.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']'
'{', '}', '-', '_', '+', '=', '*', '&', '^', '%', '$', '#'
'@', '!', '~', '`', '|', '\\', '/', '<', '>'
```

#### 3. 模型特殊Token

```python
'<pad>', '</s>', '<unk>', '<image>', '<bos>', '<eos>',
'<s>', '</s>', '
```

#### 4. 纯空格/空字符串

```python
'', ' ', '  ', '   ', '\t', '\n', '\r'
```

#### 5. 数字以外的纯符号组合

```python
'@#', '$%', '^^', '***', '---', '___', '...', '!!!'
```

### 第二层：硬标签保护（安全网）

**目标**：确保正确答案永不丢失

**实现方式**：

```python
# 在Token ID层级强制保护（而不是字符串层级）
hard_label_token_ids = set()
if primary_answer_lower:
    # 将主答案编码为token ID
    encoded_ids = self.teacher.tokenizer.encode(primary_answer_lower, add_special_tokens=False)
    hard_label_token_ids = set(encoded_ids)

# 无论黑名单如何，强制将hard_label_id加入放行列表
for i, token_id in enumerate(first_token_indices):
    if token_id.item() in hard_label_token_ids:
        valid_token_mask[i] = True  # 强制保留
        continue
    
    # 然后应用黑名单过滤
    if self.token_filter.is_valid_token(token_str, question):
        valid_token_mask[i] = True
```

**保护原因**：
- 极少数情况下，正确答案的Token可能因为词表构造原因被黑名单误伤
- 必须兜底保护，确保训练数据质量

### 第三层：Top-K兜底（多样性保障）

**目标**：防止过滤后分布过于稀疏

**触发条件**：
- 黑名单过滤后剩余Token少于N个（当前N=10）

**实现逻辑**：

```python
min_valid_tokens = 10
num_valid = valid_token_mask.sum().item()

if num_valid < min_valid_tokens and num_valid > 0:
    # 从原始Top-K中补充未被黑名单拦截的token
    top_k_fallback = min(self.top_k * 2, len(first_token_indices))
    top_k_indices = torch.topk(token_probs_raw, top_k_fallback).indices
    
    # 补充逻辑（使用较宽松的过滤策略）
    for idx in top_k_indices:
        if not valid_token_mask[idx]:
            token_str = self.teacher.tokenizer.decode([token_id.item()]).strip()
            
            # 不使用上下文感知，避免过度过滤
            if self.token_filter.is_valid_token(token_str, None):
                valid_token_mask[idx] = True
                
                # 检查是否达到最小数量
                if valid_token_mask.sum().item() >= min_valid_tokens:
                    break
```

**为什么需要Top-K补充**：
- VQA中正确答案可能排在20名开外（尤其是Teacher不确定时）
- 需要保留一定的候选集，避免过度过滤

### 关键改进点对比

| 改进点 | 修复前 | 修复后 |
|--------|--------|--------|
| **BPE碎片过滤** | ❌ 缺失 | ✅ 新增'Ġ'开头检测 |
| **硬标签保护** | ❌ 字符串层级 | ✅ Token ID层级 |
| **Top-K兜底** | ❌ 仅在"全部过滤"时触发 | ✅ 少于10个时补充 |
| **特殊Token黑名单** | ❌ 不完整 | ✅ 完整黑名单 |

---

## 设计思路

### 问题分析

在VQA软标签生成过程中，Teacher模型输出的logits包含大量噪音token：

```
问题: "How many people are wearing headphones?"
原始logits分布:
  ✓ one: 0.0815     ← 有效答案（数字）
  ✓ two: 0.0738     ← 有效答案（数字）
  ✗ yes: 0.0361     ← 噪音（二元答案）
  ✗ no: 0.0339      ← 噪音（二元答案）
  ✗ Ġt: 0.0073      ← 噪音（BPE碎片）
  ✗ Ġa: 0.0070      ← 噪音（BPE碎片）
  ✗ .: 0.0069       ← 噪音（标点符号）
  ✗ <pad>: 0.0050   ← 噪音（特殊Token）
```

**问题**：噪音token会分散有效答案的概率，影响软标签质量。

**主要噪音来源**（约占词表30%-50%）：
1. **BPE子词碎片**：'Ġt', 'Ġa', 'ur', 'qu', 'ing', 'er', 'th'（最主要）
2. **标点符号**：'.', ',', '!', '?', '\n'
3. **特殊Token**：'<pad>', '</s>', '<unk>', '<image>'
4. **纯空格/空字符串**：'', '   '
5. **纯符号组合**：'@#', '$%', '^^'

### 解决方案

采用**三层防护机制**：

#### 第一层：黑名单（核心防线）

**策略**：拦截绝对不可能是单字答案的Token

```python
# 在Token ID层级过滤
for i, token_id in enumerate(first_token_indices):
    token_str = tokenizer.decode([token_id.item()]).strip()
    
    # 检查是否在黑名单中
    if token_str in special_token_blacklist:  # 特殊Token
        continue
    if token_str in punctuation_blacklist:    # 标点符号
        continue
    if token_str.startswith('Ġ'):              # BPE碎片
        continue
    if token_str in whitespace_blacklist:     # 空格
        continue
    
    # 保留未在黑名单中的token
    valid_token_mask[i] = True
```

**为什么这样设计**：
- ✅ BPE碎片永远不可能单独作为VQA答案
- ✅ 标点符号和特殊Token不可能是有效答案
- ✅ 过滤明确的噪音，不遗漏未知类型的有效答案

#### 第二层：硬标签保护（安全网）

**策略**：强制保留hard_label对应的Token ID

```python
# 获取hard_label的token ID
hard_label_token_ids = set()
if primary_answer:
    encoded_ids = tokenizer.encode(primary_answer, add_special_tokens=False)
    hard_label_token_ids = set(encoded_ids)

# 强制保留（即使被黑名单误伤）
for i, token_id in enumerate(first_token_indices):
    if token_id.item() in hard_label_token_ids:
        valid_token_mask[i] = True  # 强制保留
        continue
    
    # 然后应用黑名单过滤
    # ...
```

**为什么需要硬标签保护**：
- 🔧 极少数情况下，正确答案的Token可能因为词表构造原因被黑名单误伤
- 🔧 必须兜底保护，确保训练数据质量
- 🔧 在Token ID层级保护，避免编码问题

#### 第三层：Top-K兜底（多样性保障）

**策略**：如果过滤后Token少于10个，从Top-K补充

```python
min_valid_tokens = 10
num_valid = valid_token_mask.sum().item()

if num_valid < min_valid_tokens and num_valid > 0:
    # 从Top-K补充
    top_k_indices = torch.topk(token_probs, top_k_fallback).indices
    
    # 补充逻辑（使用宽松过滤）
    for idx in top_k_indices:
        if not valid_token_mask[idx]:
            token_str = tokenizer.decode([token_id.item()]).strip()
            
            # 不使用上下文感知，避免过度过滤
            if is_valid_token(token_str, None):
                valid_token_mask[idx] = True
                
                if valid_token_mask.sum().item() >= min_valid_tokens:
                    break
```

**为什么需要Top-K补充**：
- 🎯 VQA中正确答案可能排在20名开外（尤其是Teacher不确定时）
- 🎯 需要保留一定的候选集，避免过度过滤
- 🎯 使用宽松过滤策略（不使用上下文感知）

### 为什么采用三层防护？

| 层级 | 作用 | 必要性 |
|------|------|---------|
| **第一层：黑名单** | 拦截明确噪音 | ✅ 必要：BPE碎片、标点符号等绝不可能是答案 |
| **第二层：硬标签保护** | 保护正确答案 | ✅ 必要：防止误过滤，确保数据质量 |
| **第三层：Top-K兜底** | 多样性保障 | ✅ 必要：防止过度过滤，保留候选集 |

### 设计优势

| 优势 | 说明 |
|------|------|
| **噪音过滤能力强** | BPE碎片、标点符号等明确噪音被拦截 |
| **正确答案不丢失** | 硬标签保护确保正确答案永远保留 |
| **适应性广** | 不依赖问题类型，适用于所有VQA问题 |
| **实现简单** | 维护一个黑名单即可，无需维护多个白名单 |
| **兜底保障完善** | Top-K兜底防止过度过滤 |

---

## 完整处理流程

```
┌──────────────────────────────────────────────────────────────────┐
│                    Token过滤完整流程                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: 模型推理                                                   │
│   - 输入: 图像 + 问题                                              │
│   - 输出: 答案 + logits (top_k_indices, top_k_values)             │
│   - 位置: teacher_model.py:inference_vqa()                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 2: 提取第一个Token的Logits                                    │
│   - 输入: logits_data                                              │
│   - 处理: 取答案第一个token的logits                                │
│   - 输出: first_token_indices, first_token_logits                 │
│   - 位置: soft_label_gen.py:593-617                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 3: 温度缩放                                                   │
│   - 公式: scaled_logits = logits / temperature                    │
│   - 作用: 控制概率分布的平滑程度                                    │
│   - 参数: temperature = 4 (来自配置)                               │
│   - 位置: soft_label_gen.py:620-622                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 4: 三层防护过滤                                               │
│                                                                    │
│   4.1 第一层：黑名单过滤                                            │
│       - BPE子词碎片（'Ġt', 'Ġa', 'ur'...）                        │
│       - 标点符号、特殊Token、纯空格                                 │
│                                                                    │
│   4.2 第二层：硬标签保护                                            │
│       - 强制保留hard_label_id                                       │
│       - 确保正确答案永不丢失                                        │
│                                                                    │
│   4.3 第三层：Top-K兜底                                             │
│       - 如果过滤后<10个token，从Top-K补充                          │
│                                                                    │
│   位置: soft_label_gen.py:624-692                                 │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 有效token数量?   │
                    └─────────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
          >= 10  │                         │ < 10
                 ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ Step 5a: 使用过滤结果      │  │ Step 5b: Top-K补充        │
│                           │  │                           │
│ - 应用mask到logits        │  │ - 从Top-100中补充         │
│   scaled_logits[~mask]    │  │ - 使用宽松过滤策略        │
│     = -1e9                │  │ - 补充到至少10个token     │
│                           │  │                           │
│ - 计算softmax             │  │ 位置: soft_label_gen.py:  │
│   token_probs = softmax(  │  │       676-691             │
│     scaled_logits_filtered│  │                           │
│   )                       │  │                           │
└──────────────────────────┘  └──────────────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 6: 解码并过滤低概率Token                                      │
│   - 解码token ID到字符串                                           │
│   - 过滤概率 < 0.001 的token                                       │
│   - 过滤特殊token: <s>, </s>, <pad>, ...                          │
│   - 位置: soft_label_gen.py:693-727                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 7: 等价Token合并                                              │
│   - 示例: '1' + 'one' = 'one'                                      │
│   - 示例: 'gray' + 'grey' = 'gray'                                 │
│   - 方法: merge_equivalent_tokens(distribution)                    │
│   - 位置: soft_label_gen.py:729-745                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 8: 归一化概率分布                                              │
│   - 确保概率和为 1.0                                                │
│   - distribution = {k: v/total for k, v in distribution.items()}  │
│   - 位置: soft_label_gen.py:747-752                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 9: 字符串层级二次过滤（安全网）                                 │
│   - 使用VQATokenFilter再次确认                                     │
│   - 确保万无一失                                                   │
│   - 位置: soft_label_gen.py:754-766                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        最终答案分布
```

---

## 核心实现步骤

### Step 1: 模型推理获取Logits

**位置**: `src/models/teacher_model.py:137-171`

```python
def inference_vqa(self, image, question, return_logits=True):
    """
    执行VQA推理
    
    Args:
        image: 图像路径
        question: 问题文本
        return_logits: 是否返回logits
    
    Returns:
        {
            'answer': 'one',
            'confidence': 0.85,
            'logits': {
                'top_k_indices': tensor([100, 101, ...]),  # token IDs
                'top_k_values': tensor([3.2, 3.0, ...])    # logits值
            }
        }
    """
    # 构建prompt
    prompt = self._construct_prompt(question, task="vqa")
    
    # 准备输入
    inputs = self._prepare_inputs(image, prompt)
    
    # 生成（return_logits=True时使用temperature=0确保确定性）
    outputs = self._generate(inputs, return_logits=return_logits)
    
    # 处理输出
    result = self._process_vqa_outputs(outputs, return_logits)
    
    return result
```

**关键点**：
- `return_logits=True` 时，强制使用 `temperature=0`（贪婪解码）
- 确保生成答案和logits top-1一致

### Step 2: 提取第一个Token的Logits

**位置**: `src/distillation/soft_label_gen.py:593-617`

```python
# 提取logits
if 'top_k_indices' in logits_data and 'top_k_values' in logits_data:
    token_indices = logits_data['top_k_indices']
    token_logits = logits_data['top_k_values']

    # 处理不同维度
    if token_indices.dim() == 1:
        # 已经是 [top_k] 形状
        first_token_indices = token_indices
        first_token_logits = token_logits
    elif token_indices.dim() == 2:
        # [num_tokens, top_k] 形状，取第一个token
        first_token_indices = token_indices[0]
        first_token_logits = token_logits[0]
    else:
        # [batch, num_tokens, top_k] 形状
        first_token_indices = token_indices[0, 0]
        first_token_logits = token_logits[0, 0]
```

**为什么取第一个token**：
- VQA答案通常是单个token（如 "one", "green", "yes"）
- 第一个token的logits包含了答案的概率分布

### Step 3: 温度缩放

**位置**: `src/distillation/soft_label_gen.py:620-622`

```python
# 温度缩放
scaled_logits = first_token_logits / self.temperature
```

**温度参数作用**：
- `temperature` 越大，概率分布越平滑
- `temperature = 4`：减小峰值，增加低概率答案的权重
- 标准知识蒸馏公式：`soft_probs = softmax(logits / T)`

### Step 4: 三层防护过滤

**位置**: `src/distillation/soft_label_gen.py:624-692`

#### 4.1 第一层：黑名单过滤

```python
# 初始化有效token掩码
valid_token_mask = torch.zeros_like(scaled_logits, dtype=torch.bool)

# 遍历所有token
for i, token_id in enumerate(first_token_indices):
    # 解码token ID到字符串
    token_str = self.teacher.tokenizer.decode([token_id.item()]).strip()
    
    # 使用VQATokenFilter判断是否有效（包含黑名单检测）
    if self.token_filter and self.token_filter.is_valid_token(token_str, question):
        valid_token_mask[i] = True
```

#### 4.2 第二层：硬标签保护

```python
# 获取hard_label对应的token ID
hard_label_token_ids = set()
if primary_answer_lower:
    try:
        encoded_ids = self.teacher.tokenizer.encode(primary_answer_lower, add_special_tokens=False)
        hard_label_token_ids = set(encoded_ids)
    except Exception as e:
        self.logger.warning(f"[Hard Label Protection] Failed: {e}")

# 强制保留hard_label_id
for i, token_id in enumerate(first_token_indices):
    if token_id.item() in hard_label_token_ids:
        valid_token_mask[i] = True
        continue
    
    # 然后应用黑名单过滤
    # ...
```

#### 4.3 第三层：Top-K兜底

```python
# 检查有效token数量
num_valid = valid_token_mask.sum().item()
min_valid_tokens = 10

if num_valid < min_valid_tokens and num_valid > 0:
    # 从Top-K补充
    self.logger.info(f"[Top-K Fallback] Only {num_valid} tokens, supplementing...")
    
    # 计算原始概率
    token_probs_raw = torch.softmax(scaled_logits, dim=-1)
    
    # 取Top-100作为候选集
    top_k_fallback = min(self.top_k * 2, len(first_token_indices))
    top_k_indices = torch.topk(token_probs_raw, top_k_fallback).indices
    
    # 补充逻辑（宽松过滤）
    for idx in top_k_indices:
        if not valid_token_mask[idx]:
            token_str = self.teacher.tokenizer.decode([first_token_indices[idx].item()]).strip()
            
            # 使用宽松过滤（不使用上下文感知）
            if self.token_filter.is_valid_token(token_str, None):
                valid_token_mask[idx] = True
                
                if valid_token_mask.sum().item() >= min_valid_tokens:
                    break
```

### Step 5: 应用Mask到Logits

**位置**: `src/distillation/soft_label_gen.py:693-727`

```python
# 将无效token的logits设为-1e9
scaled_logits_filtered = scaled_logits.clone()
scaled_logits_filtered[~valid_token_mask] = -1e9

# 计算softmax得到概率
token_probs = torch.softmax(scaled_logits_filtered, dim=-1)
```

**为什么设为 -1e9**：
- Softmax公式：`softmax(x_i) = exp(x_i) / sum(exp(x_j))`
- 当 `x_i = -1e9` 时，`exp(-1e9) ≈ 0`
- 无效token的概率接近0，自然被剔除

---

## 代码实现详解

### 1. VQATokenFilter类

**文件**: `src/utils/vqa_token_filter.py`

**核心属性**：

```python
class VQATokenFilter:
    def __init__(self):
        # ===== 第一层：黑名单 =====
        # 特殊Token黑名单
        self.special_token_blacklist = {
            '<pad>', '</s>', '<unk>', '<image>', '<bos>', '<eos>', ...
        }
        
        # 标点符号黑名单
        self.punctuation_blacklist = {
            '.', ',', '!', '?', ';', ':', '"', "'", ...
        }
        
        # 纯空格/空字符串黑名单
        self.whitespace_blacklist = {'', ' ', '  ', '   ', '\t', '\n', '\r'}
        
        # ===== 有效答案集 =====
        # 数字答案集
        self.number_answers = {
            'zero', 'one', 'two', ..., '0', '1', '2', ...
        }
        
        # 颜色答案集
        self.color_answers = {
            'red', 'blue', 'green', 'gray', 'grey', ...
        }
        
        # 二元答案集
        self.binary_answers = {'yes', 'no', 'maybe'}
        
        # 等价token映射
        self.equivalent_tokens = {
            '1': 'one',
            '2': 'two',
            'gray': 'gray',
            'grey': 'gray',  # 英式拼写 -> 美式
            ...
        }
```

**核心方法**：

| 方法 | 功能 | 使用场景 |
|------|------|---------|
| `is_valid_token(token, question)` | 判断token是否有效（包含黑名单检测） | 第一层过滤 |
| `get_canonical_token(token)` | 获取标准形式 | 等价token合并 |
| `merge_equivalent_tokens(distribution)` | 合并等价token | 后处理 |
| `filter_distribution(distribution, ...)` | 过滤分布 | 二次过滤 |

### 2. SoftLabelGenerator类

**文件**: `src/distillation/soft_label_gen.py`

**核心流程**：

```python
class SoftLabelGenerator:
    def generate_vqa_soft_labels(self, image_path, question, ...):
        """
        生成VQA软标签
        
        流程：
        1. 从hard_label获取logits
        2. 调用_process_vqa_logits处理（三层防护）
        3. 返回答案分布
        """
        if hard_label_result and 'logits' in hard_label_result:
            distribution = self._process_vqa_logits(
                logits_data=hard_label_result['logits'],
                question=question,
                primary_answer=hard_label_result['answer']
            )
        
        return {
            'answer_distribution': distribution,
            'primary_answer': primary_answer,
            ...
        }
    
    def _process_vqa_logits(self, logits_data, ...):
        """
        处理VQA logits（核心实现）
        
        流程：
        1. 温度缩放
        2. 第一层：黑名单过滤
        3. 第二层：硬标签保护
        4. 第三层：Top-K兜底（如需要）
        5. 等价token合并
        6. 归一化
        """
        # ... 实现见上文 ...
```

---

## 等价Token合并

### 为什么需要合并？

同一答案可能有多个表示：

```
原始分布（未合并）:
  'one': 0.30
  '1': 0.20      ← 同一个答案
  'gray': 0.15
  'grey': 0.10   ← 同一个答案

合并后:
  'one': 0.50    ← 合并了 'one' + '1'
  'gray': 0.25   ← 合并了 'gray' + 'grey'
```

### 合并规则

**文件**: `src/utils/vqa_token_filter.py:41-150`

```python
self.equivalent_tokens = {
    # 数字 <-> 英文单词
    '1': 'one',
    '2': 'two',
    '3': 'three',
    ...
    
    # 颜色大小写 + 拼写变体
    'green': 'green',
    'Green': 'green',
    'GREEN': 'green',
    
    'gray': 'gray',
    'grey': 'gray',  # 英式 -> 美式
    'Gray': 'gray',
    'Grey': 'gray',
    
    # 二元答案大小写
    'yes': 'yes',
    'Yes': 'yes',
    'YES': 'yes',
    ...
}
```

### 合并实现

```python
def merge_equivalent_tokens(self, distribution):
    """
    合并等价token的概率
    
    Args:
        distribution: {'1': 0.3, 'one': 0.2, ...}
    
    Returns:
        merged: {'one': 0.5, ...}
    """
    merged = {}
    
    for token, prob in distribution.items():
        # 获取标准形式
        canonical = self.get_canonical_token(token)
        
        # 合并概率
        if canonical in merged:
            merged[canonical] += prob
        else:
            merged[canonical] = prob
    
    # 归一化
    total = sum(merged.values())
    if total > 0:
        merged = {k: v/total for k, v in merged.items()}
    
    return merged
```

---

## 配置参数说明

### 配置文件

**文件**: `configs/default.yaml`

```yaml
distillation:
  soft_labels:
    enabled: true
    temperature: 4        # 温度缩放参数
    top_k_logits: 50      # Top-K兜底数量
    save_format: "json"
```

### 参数详解

| 参数 | 默认值 | 作用 | 调整建议 |
|------|--------|------|---------|
| `temperature` | 4 | 温度缩放参数，控制分布平滑度 | - 增大：分布更平滑，低概率答案权重增加<br>- 减小：分布更尖锐，主答案权重增加 |
| `top_k_logits` | 50 | Top-K兜底数量 | - 增大：兜底保留更多token<br>- 减小：兜底保留更少token |

**重要**：
- `top_k_logits` 是兜底参数，不是默认使用
- 三层防护是主策略，top_k只是备选

### 推荐配置值

| 场景 | temperature | top_k_logits | 说明 |
|------|-------------|--------------|------|
| **快速测试** | 4 | 20-30 | 减少计算开销 |
| **生产环境** | 4 | 50 | 默认推荐值 |
| **高质量蒸馏** | 4-8 | 100-200 | 更多候选答案，计算开销增加 |

---

## 效果示例

### 示例1：数字问题（三层防护生效）

```
问题: "How many people are wearing headphones?"

原始Logits分布:
  'one': 0.0815     ✓ 有效（数字答案）
  'two': 0.0738     ✓ 有效（数字答案）
  'yes': 0.0361     ✗ 噪音（二元答案）
  'no': 0.0339      ✗ 噪音（二元答案）
  'Ġt': 0.0073      ✗ 噪音（BPE碎片）
  'Ġa': 0.0070      ✗ 噪音（BPE碎片）
  '.': 0.0069       ✗ 噪音（标点符号）
  '<pad>': 0.0050   ✗ 噪音（特殊Token）

第一层：黑名单过滤:
  过滤掉: Ġt, Ġa, ., <pad>（明确噪音）
  保留: one, two, yes, no（通过黑名单）

第二层：硬标签保护:
  主答案 'one' 强制保留（即使被黑名单误伤）

第三层：Top-K兜底:
  有效token数 = 4 >= 10? 否
  → 从Top-100补充到至少10个token

最终分布:
  'one': 0.5234     (合并了 'one' + '1')
  'two': 0.4766     (合并了 'two' + '2')
  + 8个补充的低概率token（多样性保障）
```

### 示例2：颜色问题（硬标签保护生效）

```
问题: "What color is the car?"
主答案: 'gray'（硬标签）

原始Logits分布:
  'gray': 0.35      ✓ 有效（颜色答案）
  'blue': 0.30      ✓ 有效（颜色答案）
  'Grey': 0.15      ✓ 有效（大小写）
  'one': 0.05       ✗ 噪音（数字答案）
  'Ġt': 0.03        ✗ 噪音（BPE碎片）
  '.': 0.02         ✗ 噪音（标点符号）

第一层：黑名单过滤:
  过滤掉: Ġt, .（明确噪音）
  保留: gray, blue, Grey, one

第二层：硬标签保护:
  'gray' 强制保留（即使被黑名单误伤）
  注意：硬标签保护在Token ID层级，'gray'和'Grey'可能对应不同token

第三层：Top-K兜底:
  有效token数 = 4 >= 10? 否
  → 从Top-100补充到至少10个token

等价Token合并:
  'gray' + 'Grey' = 'gray'

最终分布:
  'gray': 0.4375    (合并了 'gray' + 'Grey')
  'blue': 0.3750
  'one': 0.0625     (通过黑名单，但概率低)
  + 7个补充的低概率token
```

### 示例3：硬标签保护生效（关键示例）

```
问题: "How many apples are there?"
主答案: 'two'（硬标签）

假设场景：极端情况下，'two'被黑名单误伤

原始Logits分布:
  'two': 0.08       ✓ 有效（数字答案）
  'one': 0.07       ✓ 有效（数字答案）
  'Ġt': 0.05        ✗ 噪音（BPE碎片）

错误场景（无硬标签保护）:
  如果'two'被黑名单误伤 → 过滤掉'two'
  → 错误！正确答案丢失

正确场景（有硬标签保护）:
  第一层：黑名单过滤
    过滤掉: Ġt
    假设误伤: 'two'（假设被误判为噪音）

  第二层：硬标签保护（关键）
    'two'的token ID在hard_label_token_ids中
    → 强制保留，覆盖黑名单判断
    
  最终分布:
    'two': 0.53      （保留）
    'one': 0.47

✅ 硬标签保护确保正确答案永不丢失
```

---

## 问题诊断与修复

### 问题1：噪音Token未被过滤

**问题描述**：

在计数题的分布中，发现以下噪音token未被过滤：

```json
{
  "one": 0.3308,
  "two": 0.2694,
  "yes": 0.1192,    // ❌ 应该被过滤（二元答案）
  "no": 0.0717,     // ❌ 应该被过滤（二元答案）
  "three": 0.0683,
  "zero": 0.0587,
  "four": 0.0359,
  "five": 0.0232,
  "tw": 0.0122,     // ❌ 应该被过滤（碎片token）
  "six": 0.0106
}
```

**问题原因**：

1. **"tw" 未在黑名单中**
   - 原因: `tw` 碎片token未添加到 `short_fragments` 黑名单
   - 解决: 已添加到配置文件

2. **"yes/no" 未被任务适配过滤**
   - 可能原因：
     - 任务适配过滤未执行
     - 配置文件路径错误
     - 任务类型推断错误

**修复方案**：

#### 修复1：添加碎片Token到黑名单

```yaml
# configs/vqa_token_filter.yaml
"short_fragments": ["a", "i", "u", "e", "tw", ...]
```

#### 修复2：检查VQATokenFilter初始化

```python
# ❌ 错误：初始化失败
self.token_filter = VQATokenFilter()

# ✅ 正确：带异常处理
try:
    self.token_filter = VQATokenFilter()
    self.logger.info("✓ VQA Token过滤器初始化成功")
except Exception as e:
    self.logger.warning(f"VQA Token过滤器初始化失败: {e}")
    self.token_filter = None
```

#### 修复3：检查配置文件路径

```yaml
# configs/default.yaml
vqa_token_filter:
  config_file: "configs/vqa_token_filter.yaml"  # 确保路径正确
```

**验证测试**：

```bash
# 测试计数题过滤
python scripts/test_counting_filter.py

# 测试配置加载
python scripts/test_config_loading.py

# 运行实际数据处理
python scripts/run_pipeline.py --max_samples 10 --steps distillation
```

---

### 问题2：Top-K配置未生效

**问题描述**：

日志显示：
```
[Blacklist Filter] 389/500 tokens passed blacklist filter
```

但配置文件中 `top_k_logits` 设置为 50：
```yaml
distillation:
  soft_labels:
    top_k_logits: 50  # 配置值为 50
```

预期应该是：`389/50 tokens passed blacklist filter`

**问题原因**：

在 `src/models/teacher_model.py` 中：

```python
# 第1591行 - 错误的代码
top_k = self.config.get('distillation.soft_labels.top_k', 500)  # ❌ 默认值是 500
```

问题：
1. **配置键错误**: 使用了 `'distillation.soft_labels.top_k'` 而不是 `'distillation.soft_labels.top_k_logits'`
2. **默认值错误**: 默认值硬编码为 500，而不是从配置文件读取

**影响分析**：

| 影响类型 | 说明 |
|---------|------|
| **内存使用** | 处理 500 个 tokens 而不是 50 个，内存占用增加 10 倍 |
| **计算时间** | Blacklist filter 检查 500 个 tokens，耗时增加 10 倍 |
| **日志可读性** | 显示 "500 tokens" 让用户困惑配置是否生效 |
| **数据质量** | 无影响（最终保留数量由 `max_answers` 控制） |

**修复方案**：

```python
# 修复前
top_k = self.config.get('distillation.soft_labels.top_k', 500)

# 修复后
top_k = self.config.get('distillation.soft_labels.top_k_logits', 50)  # ✓ 使用正确的配置键和默认值
```

**验证测试**：

```bash
# 测试配置读取
python scripts/test_top_k_config.py

# 运行实际数据处理
python scripts/run_pipeline.py --steps distillation --max_samples 10
```

预期日志：
```
[Blacklist Filter] 38/50 tokens passed blacklist filter  # ✓ 现在是 50 个 tokens
```

**性能对比**：

| 指标 | 修复前 (top_k=500) | 修复后 (top_k=50) | 提升 |
|------|-------------------|------------------|------|
| 内存使用 | ~2KB/sample | ~0.2KB/sample | 10倍 |
| Blacklist检查 | 500 tokens | 50 tokens | 10倍 |

---

### 问题3：Prompt模板大括号冲突

**问题描述**：

JSON 格式示例中的大括号 `{}` 被 Python 的 `.format()` 误解析为格式化占位符。

**修复方案**：

将 `.format()` 改为 `.replace()` 方法：

```python
# 修复前（有问题）
prompt = cot_template.format(**format_kwargs)

# 修复后（正确）
prompt = cot_template
if '{question}' in prompt:
    prompt = prompt.replace('{question}', question)
if '{primary_answer}' in prompt and primary_answer:
    prompt = prompt.replace('{primary_answer}', primary_answer)
if '{allowed_answers}' in prompt and allowed_answers:
    prompt = prompt.replace('{allowed_answers}', answers_str)
```

**位置**: `src/models/teacher_model.py:430-442`

---

### 问题4：CoT输出过于简洁

**问题描述**：

`Observation` 输出过于简洁（如只输出 "White"）。

**问题原因**：

Prompt 模板中 `Observation:` 后没有指导内容，模型不知道应该写什么。

**修复方案**：

添加明确的写作指导：

```yaml
# 修复前
Observation:
Analysis:
Conclusion:

# 修复后
Observation: Describe what you see in the image - the main objects, their colors, positions, quantities, and key visual features. Be specific and factual.
Analysis: Analyze how these observations relate to the question. Compare visual evidence and rule out uncertain details.
Conclusion: Summarize the reasoning and provide the Final Answer matching the reference answer exactly.
```

**位置**: `configs/prompts_en.yaml:82-84`

---

## 测试验证

### 测试脚本

#### 1. 三层防护验证

```bash
python scripts/test_three_layer_filter.py
```

验证内容：
- ✅ 第一层：黑名单过滤（BPE碎片、标点符号等）
- ✅ 第二层：硬标签保护
- ✅ 第三层：Top-K兜底

#### 2. 黑名单策略对比

```bash
python scripts/test_blacklist_strategy.py
```

#### 3. 等价Token合并测试

```bash
python scripts/test_merge_quick.py
```

#### 4. 计数题过滤测试

```bash
python scripts/test_counting_filter.py
```

预期输出：
```
✓ 'tw' 被正确识别为无效token（噪音）
✓ 'yes' 在 binary 白名单中: True
✓ 'no' 在 binary 白名单中: True
✓ 'yes' 在 count 白名单中: False
✓ 'no' 在 count 白名单中: False
```

#### 5. 配置加载测试

```bash
python scripts/test_config_loading.py
```

预期输出：
```
Step 1: 黑名单过滤
  ✓ 'tw' 已被黑名单过滤

Step 2: 任务适配过滤
  ✓ 'yes' 已被过滤
  ✓ 'no' 已被过滤

最终分布：
  one: 0.3308
  two: 0.2694
  three: 0.0683
  ...
```

### 运行完整测试

```bash
# 验证三层防护逻辑
python scripts/test_three_layer_filter.py

# 可视化处理流程
python scripts/visualize_logits_comparison.py

# 测试等价token合并
python scripts/test_merge_quick.py

# 验证黑名单策略对比
python scripts/test_blacklist_strategy.py
```

---

## 性能影响

### 存储空间

- **无明显变化**：过滤后的分布更稀疏，可能略有减少

### 推理速度

- **略微增加**：需要解码token ID进行字符串判断（约增加5-10%）

### 训练效果

- **预期显著提升**：噪音Token被有效过滤，软标签质量提升

---

## 相关文件

### 核心实现文件

| 文件 | 功能 | 关键行号 |
|------|------|---------|
| [src/utils/vqa_token_filter.py](../src/utils/vqa_token_filter.py) | Token过滤器实现（三层防护） | 20-48, 279-353 |
| [src/distillation/soft_label_gen.py](../src/distillation/soft_label_gen.py) | 软标签生成器（三层防护） | 620-692 |
| [src/models/teacher_model.py](../src/models/teacher_model.py) | Teacher模型推理 | 137-637 |
| [configs/default.yaml](../configs/default.yaml) | 配置文件 | 91-96 |
| [configs/vqa_token_filter.yaml](../configs/vqa_token_filter.yaml) | Token过滤配置 | - |

### 测试和可视化脚本

| 文件 | 功能 |
|------|------|
| [scripts/test_three_layer_filter.py](../scripts/test_three_layer_filter.py) | 三层防护验证测试 |
| [scripts/test_blacklist_strategy.py](../scripts/test_blacklist_strategy.py) | 黑名单策略对比测试 |
| [scripts/visualize_logits_comparison.py](../scripts/visualize_logits_comparison.py) | Logits处理可视化 |
| [scripts/test_merge_quick.py](../scripts/test_merge_quick.py) | 等价token合并测试 |
| [scripts/test_counting_filter.py](../scripts/test_counting_filter.py) | 计数题过滤测试 |

---

## 使用方法

### 运行测试

```bash
# 验证三层防护逻辑
python scripts/test_three_layer_filter.py

# 验证黑名单策略对比
python scripts/test_blacklist_strategy.py

# 可视化处理流程
python scripts/visualize_logits_comparison.py

# 测试等价token合并
python scripts/test_merge_quick.py

# 测试计数题过滤
python scripts/test_counting_filter.py

# 测试配置加载
python scripts/test_config_loading.py
```

### 调整配置

修改 `configs/default.yaml`：

```yaml
distillation:
  soft_labels:
    temperature: 4      # 温度缩放参数（建议范围：2-8）
    top_k_logits: 50    # Top-K兜底数量（建议范围：30-100）
```

---

## 总结

### 核心要点

1. **三层防护**：黑名单（核心防线）+ 硬标签保护（安全网）+ Top-K兜底（多样性保障）
2. **黑名单优先**：拦截明确噪音（BPE碎片、标点、特殊Token），不遗漏有效答案
3. **硬标签保护**：Token ID层级保护，确保正确答案永不丢失
4. **Top-K兜底**：多样性保障，防止过度过滤
5. **概率集中**：等价token合并，避免概率分散

### 设计优势

| 优势 | 说明 |
|------|------|
| **黑名单完善** | BPE碎片、特殊Token、标点符号完整过滤（约占词表30%-50%） |
| **硬标签保护** | Token ID层级保护，确保正确答案永不丢失 |
| **Top-K兜底** | 多样性保障，防止过度过滤 |
| **适应性广** | 不依赖问题类型，适用于所有VQA问题 |
| **实现简单** | 维护一个黑名单即可，无需维护多个白名单 |
| **配置灵活** | 通过配置文件调整参数 |

---

## 更新日志

### v4.0 (2026-07-21)
- 合并 `Token过滤策略说明.md`、`VQA_Token过滤问题诊断.md`、`Top-K配置问题修复.md`
- 整合为完整的 Token 过滤指南
- 添加问题诊断与修复章节

### v3.1 (2026-07-21)
- 修复 `.format()` 大括号冲突问题，改用 `.replace()` 方法
- 优化 VQA CoT prompt 模板，增强 Observation 输出质量
- 添加 CoT 解析调试日志，便于诊断模型输出问题
- 排除 `allowed_answers` 字段，不保存到 JSON 输出

### v3.0 (2026-07-17)
- 从白名单策略改为黑名单+硬标签保护+Top-K兜底的三层防护策略

---

**文档版本**: v4.0（完整合并版）
**最后更新**: 2026-07-21
**维护者**: VLM Distillation Team