"""
验证对比脚本
============

对比清洗前后的数据质量变化。

Usage:
    python scripts/compare_validation.py \
        --before outputs/validation_initial.json \
        --after outputs/validation_final.json
"""

import argparse
import json
import sys
from pathlib import Path


def compare_validation_reports(before_path: str, after_path: str):
    """对比清洗前后验证结果"""

    # 加载报告
    with open(before_path, 'r') as f:
        before = json.load(f)

    with open(after_path, 'r') as f:
        after = json.load(f)

    # 打印对比
    print("="*60)
    print("数据清洗前后对比")
    print("="*60)

    print("\n数据量对比：")
    print(f"  清洗前总文件: {before['total_files']}")
    print(f"  清洗后总文件: {after['total_files']}")
    print(f"  移除文件数: {before['total_files'] - after['total_files']}")
    print(f"  移除率: {(before['total_files'] - after['total_files'])/before['total_files']*100:.1f}%")

    print("\n数据质量对比：")
    before_valid_rate = before['valid_files']/before['total_files']*100 if before['total_files']>0 else 0
    after_valid_rate = after['valid_files']/after['total_files']*100 if after['total_files']>0 else 0

    print(f"  清洗前有效文件: {before['valid_files']}")
    print(f"  清洗后有效文件: {after['valid_files']}")
    print(f"  清洗前有效率: {before_valid_rate:.1f}%")
    print(f"  清洗后有效率: {after_valid_rate:.1f}%")
    print(f"  质量提升: {after_valid_rate - before_valid_rate:.1f}%")

    print("\n无效数据对比：")
    print(f"  清洗前无效文件: {before['invalid_files']}")
    print(f"  清洗后无效文件: {after['invalid_files']}")
    print(f"  清除无效数据: {before['invalid_files'] - after['invalid_files']}")

    # 判断清洗效果
    print("\n清洗效果评估：")
    if after_valid_rate >= 95:
        print("  ✅ 清洗效果很好，数据质量达标（有效率≥95%）")
        print("  建议：可以直接用于训练")
    elif after_valid_rate >= 90:
        print("  ✅ 清洗效果良好，数据质量较好（有效率≥90%）")
        print("  建议：可用于训练")
    elif after_valid_rate >= 80:
        print("  ⚠️ 清洗效果一般，数据质量中等（有效率≥80%）")
        print("  建议：可以使用，但可能需要进一步优化")
    else:
        print("  ❌ 清洗效果不佳，数据质量较低（有效率<80%）")
        print("  建议：需要调整清洗参数或重新清洗")

    print("\n"+ "="*60)

    return {
        'before': before,
        'after': after,
        'improvement': after_valid_rate - before_valid_rate,
        'removal_rate': (before['total_files'] - after['total_files'])/before['total_files']*100,
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Compare validation reports before and after cleaning"
    )

    parser.add_argument(
        '--before',
        type=str,
        required=True,
        help='Path to validation report before cleaning'
    )

    parser.add_argument(
        '--after',
        type=str,
        required=True,
        help='Path to validation report after cleaning'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for comparison report'
    )

    args = parser.parse_args()

    # 对比
    result = compare_validation_reports(args.before, args.after)

    # 保存对比报告
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n对比报告已保存: {args.output}")


if __name__ == "__main__":
    sys.exit(main())