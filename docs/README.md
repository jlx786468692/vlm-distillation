# VLM数据蒸馏系统文档

> **项目版本**: 1.1.0 | **更新日期**: 2026-07-24

---

## 📚 文档导航

| 文档 | 说明 | 推荐阅读 |
|------|------|----------|
| **[数据蒸馏流程指南](数据蒸馏流程指南.md)** | 整体架构、流程设计、核心概念 | ⭐ **必读** |
| **[环境依赖说明](环境依赖说明.md)** | 安装步骤、依赖包、系统要求 | 新手入门 |
| **[Prompt生成统一指南](Prompt生成统一指南.md)** | Prompt模板设计与优化 | 标签生成 |
| **[Token过滤完整指南](Token过滤完整指南.md)** | Token过滤机制详解 | 软标签优化 |
| **[三种标签生成和训练指南](三种标签生成和训练指南.md)** | 硬标签/软标签/CoT生成与训练 | 标签详解 |
| **[数据清洗系统详细报告](数据清洗系统详细报告.md)** | 异常检测、质量评分、清洗规则 | 数据清洗 |
| **[数据质量校验系统详解](数据质量校验系统详解.md)** | KL散度、ECE、幻觉检测、质量判定 | 质量评估 |
| **[VQA数据蒸馏方案对比分析](VQA数据蒸馏方案对比分析.md)** | 与Qwen-VL官方方案对比 | 优化参考 |

---

## 🚀 快速入门

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/vlm-distillation.git
cd vlm-distillation

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或: venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 验证安装
python -c "from src import ConfigManager; print('OK')"
```

### 2. 数据准备

```bash
# 下载数据（或手动下载COCO val2014）
python scripts/download_coco.py --split val2014 --output ./data/coco

# 数据结构
data/coco/
├── val2014/                   # 验证集图像
│   └── COCO_val2014_*.jpg
└── annotations/
    └── v2_OpenEnded_*.json
```

### 3. 配置说明

主配置文件：`configs/default.yaml`

```yaml
# 核心配置
teacher:
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  device: "cuda"
  
data:
  max_samples: 100   # 测试用少量样本
  
distillation:
  tasks: ["vqa"]
  hard_labels.confidence_threshold: 0.4
  soft_labels.temperature: 4
  
cleaning:
  min_quality_score: 35.0
  min_confidence: 0.3
```

### 4. 第一次运行

```bash
# 最小测试: 处理10张图像
python scripts/run_full_pipeline.py --samples 10

# 完整流程: 处理100张
python scripts/run_full_pipeline.py --samples 100

# 自定义参数
python scripts/run_full_pipeline.py \
    --samples 1000 \
    --min-quality 40 \
    --min-confidence 0.5
```

### 5. 输出结构

```
outputs/
├── merged/                    # 蒸馏数据（唯一输出文件夹）
│   └── {image_id}.json        # 每张图片一个文件
│
├── cleaned/                   # 清洗后数据
│   ├── cleaned/               # 高质量数据
│   ├── removed/               # 低质量数据
│   └── cleaning_report.json
│
├── visualizations/            # 可视化图表
│   ├── score_comparison.png
│   ├── sample_count.png
│   └── pipeline_timing.png
│
├── validation_initial.json    # 初始验证报告
├── validation_final.json      # 最终验证报告
└── pipeline_report.json       # 流程总报告
```

---

## 📋 常用操作

### 分步骤运行

```bash
# Step 1: 仅蒸馏
python scripts/run_distillation.py --samples 100

# Step 2: 仅清洗
python scripts/clean_data.py --input outputs/merged --min-quality 40

# Step 3: 仅可视化
python scripts/run_full_pipeline.py --steps visualization \
    --input-dir outputs/cleaned/cleaned \
    --before-dir outputs/merged

# Step 4: 质量校验
python scripts/run_full_pipeline.py --steps quality_validation \
    --input-dir outputs/cleaned/cleaned
```

### Python API调用

```python
from scripts.run_full_pipeline import FullPipelineRunner

# 初始化
runner = FullPipelineRunner('configs/default.yaml')

# 运行完整流程
result = runner.run_full_pipeline(
    max_samples=100,
    tasks=['vqa'],
    min_quality=40.0,
    min_confidence=0.5
)

# 查看结果
print(f"成功: {result['success']}")
print(f"步骤: {result['steps_completed']}")
print(f"总耗时: {result['total_duration_seconds']}秒")
```

### 断点续传

```bash
# 中途中断后继续
python scripts/run_distillation.py --samples 5000 \
    --resume outputs/checkpoint_latest.json
```

---

## 📊 核心流程

### 六步骤流程概览

| 步骤 | 名称 | 输入 | 输出 |
|------|------|------|------|
| **1** | 数据蒸馏 | COCO数据集 | merged/*.json |
| **2** | 初始验证 | merged/*.json | validation_initial.json |
| **3** | 数据清洗 | merged/*.json | cleaned/*.json |
| **4** | 最终验证 | cleaned/*.json | validation_final.json |
| **5** | 质量校验 | cleaned/*.json | quality_validation.json |
| **6** | 可视化 | 所有数据 | PNG图表 |

### 三重标签生成

| 标签类型 | 内容 | 用途 |
|----------|------|------|
| **硬标签** | 最终预测结果 + 置信度 | 确定性答案 |
| **软标签** | 温度缩放概率分布 | 知识蒸馏 |
| **思维链** | 结构化推理过程 | 推理能力迁移 |

---

## 📖 详细文档索引

### 按功能模块

| 功能 | 文档 | 核心内容 |
|------|------|----------|
| **整体架构** | [数据蒸馏流程指南](数据蒸馏流程指南.md) | 流程设计、核心模块、配置参数 |
| **Prompt设计** | [Prompt生成统一指南](Prompt生成统一指南.md) | 模板类型、优化策略、配置方法 |
| **Token过滤** | [Token过滤完整指南](Token过滤完整指南.md) | 四层过滤、黑名单、任务适配 |
| **标签生成** | [三种标签生成和训练指南](三种标签生成和训练指南.md) | 硬/软标签/CoT生成与训练 |
| **数据清洗** | [数据清洗系统详细报告](数据清洗系统详细报告.md) | 异常检测、质量评分、清洗规则 |
| **质量校验** | [数据质量校验系统详解](数据质量校验系统详解.md) | KL散度、ECE、幻觉检测 |
| **方案对比** | [VQA数据蒸馏方案对比分析](VQA数据蒸馏方案对比分析.md) | Qwen-VL官方方案对比 |

### 按任务类型

| 任务 | 推荐文档 |
|------|----------|
| **初次使用** | 环境依赖说明 → 数据蒸馏流程指南 |
| **标签生成** | 三种标签生成和训练指南 → Prompt生成统一指南 |
| **数据清洗** | 数据清洗系统详细报告 |
| **质量评估** | 数据质量校验系统详解 |
| **优化参考** | VQA数据蒸馏方案对比分析 |

---

## ⚙️ 命令行参数

### 完整参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/default.yaml` |
| `--samples` | 最大处理样本数 | 配置文件值 |
| `--tasks` | 任务列表 | `[vqa]` |
| `--steps` | 运行步骤 | `[distillation, cleaning]` |
| `--min-quality` | 清洗最小质量分数 | `35.0` |
| `--min-confidence` | 清洗最小置信度 | `0.3` |
| `--input-dir` | 输入数据目录 | `None` |
| `--before-dir` | 清洗前数据目录 | `None` |
| `--output-dir` | 输出目录 | `./outputs` |
| `--dry-run` | 测试配置 | `False` |
| `--resume` | 断点续传 | `None` |

---

## 🔧 配置参数速查

### 蒸馏配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `distillation.tasks` | `['vqa']` | 任务类型 |
| `distillation.hard_labels.confidence_threshold` | `0.4` | 置信度阈值 |
| `distillation.soft_labels.temperature` | `4` | 温度参数 |
| `distillation.soft_labels.top_k_logits` | `50` | Top-K logits |

### 清洗配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cleaning.min_confidence` | `0.3` | 最低置信度 |
| `cleaning.min_quality_score` | `35.0` | 最低质量分数 |
| `cleaning.auto_remove_invalid` | `true` | 自动移除无效数据 |
| `cleaning.save_removed_data` | `true` | 保存被移除数据 |

### 可视化配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `visualization.output_dir` | `./outputs/visualizations` | 输出目录 |
| `visualization.plot_format` | `png` | 图像格式 |
| `visualization.dpi` | `150` | 图像分辨率 |

---

## ❓ 常见问题

### Q1: 输出文件夹变化

```
旧版本 (已移除):
outputs/hard_labels/
outputs/soft_labels/
outputs/cot_reasoning/

新版本:
outputs/merged/  (唯一输出文件夹)
```

### Q2: 如何选择合适的置信度阈值？

- **高精度场景**: 0.7+ (严格过滤)
- **平衡场景**: 0.4-0.6 (推荐)
- **数据量优先**: 0.3-0.4 (宽松)

### Q3: 温度参数如何选择？

| Temperature | 效果 | 适用场景 |
|------------|------|----------|
| `T=1` | 分布尖锐 | 高置信度样本 |
| `T=2-4` | 分布平滑 | 推荐默认值 |
| `T=5+` | 分布过于平滑 | 不推荐 |

---

## 📞 获取帮助

- **文档问题**: 查看 [docs/](./) 目录下的详细文档
- **配置问题**: 参考 [configs/default.yaml](../configs/default.yaml)
- **代码问题**: 查看 [src/](../src/) 源码注释

---

**文档版本**: 1.1.0 | **最后更新**: 2026-07-24