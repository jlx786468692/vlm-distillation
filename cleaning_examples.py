"""
数据清洗使用示例
================

演示如何使用DataCleaner清洗蒸馏数据。
"""

# ============================================================
# 示例1: 基础清洗流程
# ============================================================

"""
基础使用 - 清洗生成的蒸馏数据:
```python
from src import ConfigManager
from src.cleaning import DataCleaner

# 1. 初始化配置
config = ConfigManager('configs/default.yaml')

# 2. 创建清洗器
cleaner = DataCleaner(config)

# 3. 执行清洗
report = cleaner.clean_directory(
    data_dir='./outputs/merged',
    output_dir='./outputs/cleaned'
)

# 4. 查看结果
print(f"清洗完成:")
print(f"  输入: {report['summary']['total_input']} 条")
print(f"  清洗后: {report['summary']['cleaned_count']} 条")
print(f"  移除: {report['summary']['removed_count']} 条")
print(f"  移除率: {report['summary']['removal_rate']:.1%}")
```
"""

# ============================================================
# 示例2: 自定义清洗参数
# ============================================================

"""
自定义清洗阈值:
```python
from src import ConfigManager
from src.cleaning import DataCleaner

# 严格清洗 - 只保留高质量数据
config = ConfigManager()
config.set('cleaning.min_quality_score', 50.0)      # 提高质量阈值
config.set('cleaning.min_confidence', 0.7)          # 提高置信度阈值
config.set('cleaning.auto_remove_invalid', True)    # 自动移除无效数据

cleaner = DataCleaner(config)
report = cleaner.clean_directory('./outputs/merged')

print(f"严格清洗后保留: {report['summary']['cleaned_count']} 条高质量数据")

# 宽松清洗 - 最大化数据保留
config.set('cleaning.min_quality_score', 20.0)      # 降低质量阈值
config.set('cleaning.min_confidence', 0.3)          # 降低置信度阈值
config.set('cleaning.auto_remove_invalid', False)   # 仅标记不移除

cleaner = DataCleaner(config)
report = cleaner.clean_directory('./outputs/merged')

print(f"宽松清洗后保留: {report['summary']['cleaned_count']} 条")
```
"""

# ============================================================
# 示例3: 分析清洗报告
# ============================================================

"""
深入分析清洗报告:
```python
import json
from src import ConfigManager
from src.cleaning import DataCleaner

config = ConfigManager()
cleaner = DataCleaner(config)

# 执行清洗
report = cleaner.clean_directory('./outputs/merged')

# 保存报告
with open('./outputs/cleaned/cleaning_report_detailed.json', 'w') as f:
    json.dump(report, f, indent=2)

# 分析异常统计
anomalies = report['anomaly_statistics']
print("\\n异常统计:")
for anomaly_type, count in anomalies.items():
    if count > 0:
        print(f"  {anomaly_type}: {count} 个")

# 分析质量分布
quality = report['quality_statistics']
print("\\n质量统计:")
print(f"  平均质量: {quality['average_quality_score']:.1f}")
print(f"  最高质量: {quality['max_quality_score']:.1f}")
print(f"  最低质量: {quality['min_quality_score']:.1f}")

print("\\n质量分布:")
dist = quality['quality_distribution']
print(f"  高质量(≥70): {dist['high_quality']} 个")
print(f"  中等质量(50-70): {dist['medium_quality']} 个")
print(f"  低质量(<50): {dist['low_quality']} 个")

# 查看建议
print("\\n优化建议:")
for rec in report['recommendations']:
    print(f"  {rec}")
```
"""

# ============================================================
# 示例4: 处理被移除的数据
# ============================================================

"""
分析被移除的数据:
```python
import json
from pathlib import Path

# 加载被移除的数据
removed_dir = Path('./outputs/cleaned/removed')
removed_files = list(removed_dir.glob('*.json'))

print(f"\\n分析 {len(removed_files)} 个被移除的样本:")

# 统计移除原因
reasons_count = {}
for removed_file in removed_files[:50]:  # 分析前50个
    with open(removed_file, 'r') as f:
        data = json.load(f)

    reasons = data.get('_removal_reasons', [])
    for reason in reasons:
        reasons_count[reason] = reasons_count.get(reason, 0) + 1

print("\\n移除原因统计:")
for reason, count in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {reason}: {count} 次")

# 查看具体样本
print("\\n示例样本:")
sample_file = removed_files[0]
with open(sample_file, 'r') as f:
    sample = json.load(f)

print(f"  image_id: {sample.get('image_id')}")
print(f"  quality_score: {sample.get('_quality_score', 0):.1f}")
print(f"  reasons: {sample.get('_removal_reasons', [])}")
```
"""

# ============================================================
# 示例5: 批量清洗与验证
# ============================================================

"""
结合清洗与验证:
```python
from src import ConfigManager
from src.cleaning import DataCleaner

config = ConfigManager()
cleaner = DataCleaner(config)

# Step 1: 清洗数据
report = cleaner.clean_directory(
    data_dir='./outputs/merged',
    output_dir='./outputs/cleaned'
)

# Step 2: 验证清洗效果
if report['summary']['removal_rate'] > 0.3:
    print("\\n⚠️ 警告: 移除率超过30%，请检查:")
    print("  1. 清洗阈值是否过于严格")
    print("  2. 原始数据生成质量")
    print("  3. Prompt模板是否合理")

    # 建议降低阈值
    print("\\n建议配置:")
    print("  min_quality_score: 25.0")
    print("  min_confidence: 0.4")

elif report['summary']['removal_rate'] < 0.05:
    print("\\n✓ 清洗效果良好，移除率合理")
    print(f"  数据质量分布正常")

else:
    print("\\n✓ 清洗效果适中")
    print(f"  保留 {report['summary']['cleaned_count']} 条数据用于训练")
```
"""

# ============================================================
# 示例6: 分阶段清洗策略
# ============================================================

"""
分阶段清洗 - 逐步优化:
```python
from src import ConfigManager
from src.cleaning import DataCleaner

# 阶段1: 初步清洗（移除明显异常）
config = ConfigManager()
config.set('cleaning.auto_remove_invalid', True)      # 移除无效答案
config.set('cleaning.min_quality_score', 10.0)        # 极低阈值
config.set('cleaning.auto_repair_bbox', True)         # 修复bbox

cleaner1 = DataCleaner(config)
report1 = cleaner1.clean_directory(
    './outputs/merged',
    './outputs/cleaned_stage1'
)

print(f"阶段1: 移除明显异常 {report1['summary']['removed_count']} 条")

# 阶段2: 精细清洗（提高阈值）
config.set('cleaning.min_quality_score', 30.0)        # 中等阈值
config.set('cleaning.min_confidence', 0.5)            # 中等置信度

cleaner2 = DataCleaner(config)
report2 = cleaner2.clean_directory(
    './outputs/cleaned_stage1/cleaned',
    './outputs/cleaned_stage2'
)

print(f"阶段2: 精细清洗移除 {report2['summary']['removed_count']} 条")

# 阶段3: 最终清洗（严格筛选）
config.set('cleaning.min_quality_score', 50.0)        # 高阈值
config.set('cleaning.min_cot_quality', 0.7)           # 高CoT质量

cleaner3 = DataCleaner(config)
report3 = cleaner3.clean_directory(
    './outputs/cleaned_stage2/cleaned',
    './outputs/final'
)

print(f"阶段3: 最终保留 {report3['summary']['cleaned_count']} 条高质量数据")
print(f"\\n总移除率: {(report1['summary']['removed_count'] + report2['summary']['removed_count'] + report3['summary']['removed_count']) / report1['summary']['total_input'] * 100:.1f}%")
```
"""

# ============================================================
# 示例7: 按任务清洗
# ============================================================

"""
按任务类型清洗:
```python
import json
from pathlib import Path
from src.cleaning import DataCleaner
from src import ConfigManager

config = ConfigManager()
cleaner = DataCleaner(config)

# 加载清洗后的数据
cleaned_dir = Path('./outputs/cleaned/cleaned')
all_data = []

for json_file in cleaned_dir.glob('*.json'):
    with open(json_file, 'r') as f:
        all_data.append(json.load(f))

# 按任务统计质量
task_quality = {
    'vqa': [],
    'captioning': [],
    'detection': []
}

for data in all_data:
    tasks = data.get('tasks', {})
    quality_score = data.get('quality_score', 0)

    for task_name in tasks.keys():
        if task_name in task_quality:
            task_quality[task_name].append(quality_score)

# 打印各任务质量
print("\\n各任务质量统计:")
for task_name, scores in task_quality.items():
    if scores:
        avg_quality = sum(scores) / len(scores)
        print(f"  {task_name}:")
        print(f"    数量: {len(scores)}")
        print(f"    平均质量: {avg_quality:.1f}")
        print(f"    最高质量: {max(scores):.1f}")
        print(f"    最低质量: {min(scores):.1f}")
```
"""

# ============================================================
# 示例8: 清洗前后对比
# ============================================================

"""
对比清洗前后数据质量:
```python
from src import ConfigManager
from src.cleaning import DataCleaner

config = ConfigManager()
cleaner = DataCleaner(config)

# 清洗前统计
before_stats = {
    'total': 5000,
    'avg_confidence': 0.65,
    'invalid_count': 150
}

# 执行清洗
report = cleaner.clean_directory('./outputs/merged')

# 清洗后统计
after_stats = {
    'total': report['summary']['cleaned_count'],
    'avg_quality': report['quality_statistics']['average_quality_score'],
    'removal_rate': report['summary']['removal_rate']
}

# 对比展示
print("\\n清洗前后对比:")
print(f"\\n数据量:")
print(f"  清洗前: {before_stats['total']}")
print(f"  清洗后: {after_stats['total']}")
print(f"  移除率: {after_stats['removal_rate']:.1%}")

print(f"\\n数据质量:")
print(f"  清洗前平均置信度: {before_stats['avg_confidence']:.2f}")
print(f"  清洗后平均质量: {after_stats['avg_quality']:.1f}/100")

print(f"\\n改进效果:")
quality_improvement = (after_stats['avg_quality'] / 100 - before_stats['avg_confidence']) / before_stats['avg_confidence'] * 100
print(f"  质量提升: {quality_improvement:.1f}%")
```
"""

# ============================================================
# 示例9: 导出清洗摘要
# ============================================================

"""
生成清洗摘要报告:
```python
from src import ConfigManager
from src.cleaning import DataCleaner
from datetime import datetime

config = ConfigManager()
cleaner = DataCleaner(config)

# 执行清洗
report = cleaner.clean_directory('./outputs/merged')

# 生成摘要报告
summary_report = {
    'title': 'VLM Distillation Data Cleaning Summary',
    'date': datetime.now().strftime('%Y-%m-%d'),
    'input': report['summary']['total_input'],
    'output': report['summary']['cleaned_count'],
    'removed': report['summary']['removed_count'],
    'removal_rate': f"{report['summary']['removal_rate']*100:.1f}%",
    'quality_improvement': {
        'before': 'Mixed quality (65% avg confidence)',
        'after': f"{report['quality_statistics']['average_quality_score']:.1f}/100 quality score",
        'improvement': f"+{(report['quality_statistics']['average_quality_score'] - 65)/65*100:.1f}%"
    },
    'recommendations': report['recommendations'],
}

# 打印摘要
print("="*60)
print(summary_report['title'])
print("="*60)
print(f"日期: {summary_report['date']}")
print(f"\\n数据清洗结果:")
print(f"  输入数据: {summary_report['input']} 条")
print(f"  输出数据: {summary_report['output']} 条")
print(f"  移除数据: {summary_report['removed']} 条")
print(f"  移除率: {summary_report['removal_rate']}")

print(f"\\n质量改进:")
print(f"  清洗前: {summary_report['quality_improvement']['before']}")
print(f"  清洗后: {summary_report['quality_improvement']['after']}")
print(f"  提升: {summary_report['quality_improvement']['improvement']}")

print("="*60)
```
"""

# ============================================================
# 示例10: 命令行清洗
# ============================================================

"""
命令行清洗示例:

```bash
# 基础清洗
python scripts/clean_data.py --input ./outputs/merged

# 详细输出
python scripts/clean_data.py --input ./outputs/merged --verbose

# 严格清洗
python scripts/clean_data.py \
    --input ./outputs/merged \
    --min-confidence 0.7 \
    --min-quality 50

# 宽松清洗
python scripts/clean_data.py \
    --input ./outputs/merged \
    --min-confidence 0.3 \
    --min-quality 20

# 自定义输出
python scripts/clean_data.py \
    --input ./outputs/merged \
    --output ./outputs/final_cleaned \
    --report final_cleaning_report.json

# 测试配置（不保存文件）
python scripts/clean_data.py --input ./outputs/merged --dry-run

# 保留无效数据
python scripts/clean_data.py --input ./outputs/merged --keep-invalid

# 禁用自动修复
python scripts/clean_data.py --input ./outputs/merged --no-repair

# 禁用去重
python scripts/clean_data.py --input ./outputs/merged --no-deduplicate
```
"""

print("\n" + "="*60)
print("Data Cleaning Examples")
print("="*60)
print("\nSee examples above for various cleaning patterns.")
print("\nKey usage patterns:")
print("  1. Basic cleaning: cleaner.clean_directory()")
print("  2. Custom thresholds: config.set('cleaning.min_quality_score', X)")
print("  3. Analyze report: report['summary'], report['recommendations']")
print("  4. Staged cleaning: Multiple passes with increasing thresholds")
print("  5. Command-line: python scripts/clean_data.py with various flags")
print("\nFor detailed documentation, see docs/软件详细设计报告.md")
print("="*60)
