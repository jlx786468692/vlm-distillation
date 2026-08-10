# 智能驾驶数据筛选工具

## 📋 功能介绍

从COCO数据集中筛选出与智能驾驶相关的数据，包括：
- 交通工具（汽车、卡车、公交车、自行车等）
- 道路设施（交通灯、停止标志、道路、人行横道等）
- 交通参与者（行人、骑行者等）

## 🎯 核心特性

### 多维度综合打分

采用三维评分机制，确保筛选质量：

1. **类别得分（40%）**：基于检测对象类别
   - 核心类别（car, truck, bus, traffic light等）权重更高
   - 对象数量越多，得分越高

2. **文本语义得分（30%）**：基于Captions和VQA问题
   - 检测驾驶相关关键词
   - 支持模糊匹配和短语匹配

3. **场景特征得分（30%）**：基于图像元数据
   - 宽高比（道路场景多为宽屏）
   - 对象数量（复杂场景）

### COCO格式兼容

输出的标注文件完全兼容COCO格式，可直接用于训练：
- `instances_val2014.json` - 检测标注
- `captions_val2014.json` - 描述标注
- `v2_mscoco_val2014_questions.json` - VQA问题
- `v2_mscoco_val2014_annotations.json` - VQA答案
- `person_keypoints_val2014.json` - 关键点标注

## 🚀 快速开始

### 1. 基本使用

```bash
# 使用默认配置（处理val2014数据集）
python tools/driving_filter/run_filter.py

# 使用自定义配置文件
python tools/driving_filter/run_filter.py --config configs/driving_filter.yaml
```

### 2. 自定义参数

```bash
# 处理训练集
python tools/driving_filter/run_filter.py --split train2014

# 调整筛选阈值（更严格）
python tools/driving_filter/run_filter.py --threshold 0.6

# 调整筛选阈值（更宽松）
python tools/driving_filter/run_filter.py --threshold 0.4

# 指定输出目录
python tools/driving_filter/run_filter.py --output ./data/my_driving_data

# 复制图像文件（而非符号链接）
python tools/driving_filter/run_filter.py --copy-images
```

### 3. 组合使用

```bash
# 完整示例：处理训练集，阈值0.6，复制图像
python tools/driving_filter/run_filter.py \
  --split train2014 \
  --threshold 0.6 \
  --output ./data/driving_train \
  --copy-images
```

## 📁 输出结构

```
data/filter_coco/
├── annotations/
│   ├── instances_val2014.json              # 检测标注
│   ├── captions_val2014.json               # 描述标注
│   ├── v2_mscoco_val2014_questions.json    # VQA问题
│   ├── v2_mscoco_val2014_annotations.json  # VQA答案
│   └── person_keypoints_val2014.json       # 关键点标注
├── images/
│   └── val2014/                            # 图像文件（符号链接或复制）
│       ├── 000000397133.jpg
│       └── ...
└── metadata/
    ├── filter_statistics.json              # 筛选统计信息
    └── filter_scores.json                  # 每张图像的得分详情
```

## ⚙️ 配置文件说明

配置文件位于 `configs/driving_filter.yaml`，主要配置项：

### 类别配置

```yaml
categories:
  driving_categories:
    - 1   # person
    - 3   # car ⭐
    - 6   # bus ⭐
    - 8   # truck ⭐
    - 10  # traffic light ⭐

  weights:
    3: 2.0   # car权重2.0
    6: 2.0   # bus权重2.0
    'default': 1.0
```

### 评分配置

```yaml
scoring:
  category_weight: 0.4    # 类别权重
  text_weight: 0.3        # 文本权重
  scene_weight: 0.3       # 场景权重
  score_threshold: 0.5    # 筛选阈值
```

### 关键词配置

关键词库位于 `configs/driving_keywords.txt`，可自定义添加或删除关键词。

## 📊 统计信息

运行后会生成详细的统计报告：

### 基本统计
- 总图像数
- 筛选图像数
- 保留率

### 得分分布
- 平均分、中位数、标准差
- 最小值、最大值
- 25/75分位数

### 类别分布
- 各类别出现频次
- 高频类别统计

### 质量分布
- 高质量样本（≥0.8）
- 中等质量样本（[0.5, 0.8)）
- 低质量样本（<0.5）

## 🔧 高级用法

### Python API

```python
from tools.driving_filter import DrivingDataFilter
from src.data.coco_loader import COCODataLoader

# 初始化
coco_loader = COCODataLoader()
coco_loader.initialize(split="val2014")

filter_engine = DrivingDataFilter(config_path="configs/driving_filter.yaml")

# 执行筛选
filtered_img_ids, scores = filter_engine.run(coco_loader, output_dir="./data/filter_coco")

# 查看统计信息
stats = filter_engine.get_statistics()
print(stats)
```

### 自定义关键词

```python
from tools.driving_filter import KeywordMatcher

# 自定义关键词集合
custom_keywords = {
    'car', 'truck', 'bus',
    'traffic light', 'stop sign'
}

# 创建匹配器
matcher = KeywordMatcher(keywords=custom_keywords)
```

## ⚠️ 注意事项

### 数据准备

确保COCO数据集已正确下载：

```
data/coco/
├── annotations/
│   ├── instances_val2014.json
│   ├── captions_val2014.json
│   ├── v2_mscoco_val2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│   └── person_keypoints_val2014.json
└── val2014/
    ├── 000000397133.jpg
    └── ...
```

### Windows符号链接

Windows系统创建符号链接需要管理员权限。如果失败，工具会自动回退到复制文件。

建议使用 `--copy-images` 参数，或以管理员身份运行。

### 性能优化

- 大数据集建议关闭进度条：`processing.show_progress: false`
- Windows下建议关闭多进程：`processing.use_multiprocessing: false`

## 📈 效果示例

以val2014数据集为例（5000张图像）：

```
总图像数: 5000
筛选图像数: 2345
保留率: 46.9%

得分分布:
  平均分: 0.67
  中位数: 0.71
  标准差: 0.15

质量分布:
  高质量 (≥0.8): 892 (38.0%)
  中等质量 ([0.5, 0.8)): 1453 (61.8%)
  低质量 (<0.5): 0 (0.0%)

类别分布:
  car: 1892
  person: 1234
  truck: 456
  bus: 234
  traffic light: 312
```

## 🔄 后续扩展

1. **场景分类**：细分城市道路、高速公路、停车场等
2. **天气识别**：识别晴天、雨天、雾天等
3. **时间识别**：识别白天、夜晚、黄昏等
4. **难度分级**：根据对象数量和遮挡情况标注难度
5. **主动学习**：使用模型预测筛选质量，迭代优化

## 📝 更新日志

- **v1.0.0** (2026-08-07)
  - 初始版本发布
  - 支持多维度综合打分
  - COCO格式兼容输出
  - 详细统计报告

## 📧 反馈与支持

如有问题或建议，请提交Issue或联系开发团队。