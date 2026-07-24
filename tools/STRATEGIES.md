# 🔧 工具模块 - 完整配置说明

## Prompt生成策略详解

### 策略1: real_labels（默认，推荐）

**适用场景**: 有高质量真实标签数据

**特点**:
- 直接使用真实标签数据
- 速度快，质量高
- 简单可靠

**使用**:
```bash
python -m tools prompt_generator --strategy real_labels
```

---

### 策略2: dspy_fewshot（推荐）

**适用场景**: 有真实标签数据，需要few-shot示例

**特点**:
- 自动选择高质量示例（置信度>0.7）
- 构建结构化few-shot prompt
- 包含完整的CoT推理示例

**核心逻辑**:
1. 从真实数据中筛选高质量样本
2. 选择标准：高置信度 + 完整CoT + 清晰分布
3. 构建包含多个示例的prompt

**使用**:
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

**特点**:
- 使用DSPy的MIPROv2方法
- 自动优化prompt指令和结构
- 需要更多时间和资源

**使用**:
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

**特点**:
- 基于数据模式分析
- 不需要真实标签
- 质量相对较低

---

## 输出格式优化

所有策略生成的YAML文件都经过优化处理：

### 换行处理
```yaml
prompts:
  cot:
    vqa_system: |
      Line 1
      Line 2
      Line 3
      
      Multiple paragraphs
      are preserved
```

### 不使用转义符
```yaml
# ❌ 旧方式（不使用）
vqa_system: "Line 1\\nLine 2\\nLine 3"

# ✅ 新方式（使用）
vqa_system: |
  Line 1
  Line 2
  Line 3
```

---

## 推荐使用顺序

1. **首选**: `dspy_fewshot` - 自动选择高质量示例，效果最好
2. **备选**: `real_labels` - 简单快速，质量稳定
3. **进阶**: `dspy` - 需要优化prompt结构时使用
4. **备用**: `pattern_based` - 无标注数据时使用

---

## 示例对比

### real_labels 输出示例
```
简单的prompt模板，包含基本规则和少量示例
```

### dspy_fewshot 输出示例
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

### dspy 输出示例
```
经过MIPROv2优化的prompt结构
可能包含自动生成的指令优化
```

---

**总结**: 推荐使用 `dspy_fewshot` 或 `real_labels` 策略，两者都能提供高质量的prompt。