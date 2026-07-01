"""
数据清洗脚本
============

执行数据清洗的命令行工具。
"""

import argparse
import sys
import json
from pathlib import Path

# 兼容两种导入方式：安装后和未安装
try:
    # 安装后的导入方式
    from src import ConfigManager, setup_logger, DataCleaner
except ImportError:
    # 未安装时的导入方式（开发模式）
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src import ConfigManager, setup_logger, DataCleaner


def main():
    """Main entry point for data cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean VLM distillation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cleaning
  python scripts/clean_data.py --input ./outputs/merged

  # Custom thresholds
  python scripts/clean_data.py --input ./outputs/merged --min-confidence 0.6 --min-quality 40

  # Keep invalid data instead of removing
  python scripts/clean_data.py --input ./outputs/merged --keep-invalid

  # Output to custom directory
  python scripts/clean_data.py --input ./outputs/merged --output ./outputs/final

Cleaning rules:
  - Remove data with quality score < min-quality threshold
  - Remove data with invalid answers ('unknown', 'N/A', empty)
  - Remove data with empty results
  - Remove data with format errors
  - Remove data with multiple low confidence tasks (>=2)
  - Repair anomalous bounding boxes (if enabled)
  - Deduplicate similar answers (if enabled)
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='./outputs/merged',
        help='Input data directory (default: ./outputs/merged)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for cleaned data (default: <input_dir>/../cleaned)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold for anomaly detection (default: 0.5)'
    )

    parser.add_argument(
        '--min-quality',
        type=float,
        default=30.0,
        help='Minimum quality score threshold (0-100, default: 30.0)'
    )

    parser.add_argument(
        '--min-cot-quality',
        type=float,
        default=0.5,
        help='Minimum CoT logical flow score (default: 0.5)'
    )

    parser.add_argument(
        '--min-answer-length',
        type=int,
        default=3,
        help='Minimum answer length in characters (default: 3)'
    )

    parser.add_argument(
        '--max-answer-length',
        type=int,
        default=100,
        help='Maximum answer length in characters (default: 100)'
    )

    parser.add_argument(
        '--keep-invalid',
        action='store_true',
        help='Keep invalid data instead of removing (mark only)'
    )

    parser.add_argument(
        '--no-repair',
        action='store_true',
        help='Disable automatic bbox repair'
    )

    parser.add_argument(
        '--no-deduplicate',
        action='store_true',
        help='Disable answer deduplication'
    )

    parser.add_argument(
        '--save-removed',
        action='store_true',
        default=True,
        help='Save removed data to separate directory (default: True)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default='cleaning_report.json',
        help='Output cleaning report file (default: cleaning_report.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed cleaning information'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without actually saving files (for testing)'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        name="clean_data",
        level="DEBUG" if args.verbose else "INFO",
        console_output=True
    )

    logger.info("="*60)
    logger.info("VLM Distillation Data Cleaning")
    logger.info("="*60)

    # Load configuration
    logger.info(f"\nLoading configuration from: {args.config}")
    config = ConfigManager(args.config)

    # Override config with command-line arguments
    logger.info("\nConfiguring cleaning parameters:")
    config.set('cleaning.min_confidence', args.min_confidence)
    config.set('cleaning.min_quality_score', args.min_quality)
    config.set('cleaning.min_cot_quality', args.min_cot_quality)
    config.set('cleaning.min_answer_length', args.min_answer_length)
    config.set('cleaning.max_answer_length', args.max_answer_length)
    config.set('cleaning.auto_remove_invalid', not args.keep_invalid)
    config.set('cleaning.auto_repair_bbox', not args.no_repair)
    config.set('cleaning.deduplicate_answers', not args.no_deduplicate)
    config.set('cleaning.save_removed_data', args.save_removed)

    # Log configuration
    logger.info(f"  min_confidence:      {args.min_confidence}")
    logger.info(f"  min_quality_score:   {args.min_quality}")
    logger.info(f"  min_cot_quality:     {args.min_cot_quality}")
    logger.info(f"  auto_remove_invalid: {not args.keep_invalid}")
    logger.info(f"  auto_repair_bbox:    {not args.no_repair}")
    logger.info(f"  deduplicate_answers: {not args.no_deduplicate}")

    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    # Check for data files
    json_files = list(input_dir.glob("*.json"))
    valid_files = [
        f for f in json_files
        if not f.name.startswith('checkpoint')
        and not f.name.startswith('merged_summary')
        and not f.name.startswith('cleaning_report')
    ]

    if not valid_files:
        logger.error(f"No valid data files found in {args.input}")
        logger.error("Expected JSON files from distillation output")
        sys.exit(1)

    logger.info(f"\nFound {len(valid_files)} data files to clean")

    # Create cleaner
    cleaner = DataCleaner(config)
    logger.info(f"\nDataCleaner initialized: {cleaner}")

    # Dry run check
    if args.dry_run:
        logger.info("\n" + "="*60)
        logger.info("DRY RUN - Configuration validated, no files will be saved")
        logger.info("="*60)
        return 0

    # Execute cleaning
    logger.info("\n" + "-"*60)
    logger.info("Starting data cleaning...")
    logger.info("-"*60)

    try:
        report = cleaner.clean_directory(args.input, args.output)

        # Save detailed report
        output_dir = Path(args.output or str(Path(args.input).parent / "cleaned"))
        report_file = output_dir / args.report

        logger.info(f"\nSaving detailed report to: {report_file}")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Display summary
        logger.info("\n" + "="*60)
        logger.info("Cleaning Summary")
        logger.info("="*60)

        summary = report['summary']
        logger.info(f"\nStatistics:")
        logger.info(f"  Total input:       {summary['total_input']}")
        logger.info(f"  Cleaned data:      {summary['cleaned_count']}")
        logger.info(f"  Removed data:      {summary['removed_count']}")
        logger.info(f"  Removal rate:      {summary['removal_rate']*100:.1f}%")
        logger.info(f"  Duplicates:        {summary['duplicate_count']}")
        logger.info(f"  Repaired:          {summary['repaired_count']}")

        # Quality statistics
        quality = report['quality_statistics']
        logger.info(f"\nQuality Statistics:")
        logger.info(f"  Average quality:   {quality['average_quality_score']:.1f}")
        logger.info(f"  Median quality:    {quality['median_quality_score']:.1f}")
        logger.info(f"  Min quality:       {quality['min_quality_score']:.1f}")
        logger.info(f"  Max quality:       {quality['max_quality_score']:.1f}")

        logger.info(f"\nQuality Distribution:")
        dist = quality['quality_distribution']
        logger.info(f"  High quality (≥70): {dist['high_quality']}")
        logger.info(f"  Medium (50-70):     {dist['medium_quality']}")
        logger.info(f"  Low quality (<50):  {dist['low_quality']}")

        # Anomaly statistics
        logger.info(f"\nAnomaly Statistics:")
        anomalies = report['anomaly_statistics']
        for anomaly_type, count in anomalies.items():
            if count > 0:
                logger.info(f"  {anomaly_type}: {count}")

        # Recommendations
        logger.info(f"\nRecommendations:")
        recommendations = report['recommendations']
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\n{i}. {rec}")

        # Output locations
        logger.info(f"\nOutput Files:")
        logger.info(f"  Cleaned data:  {output_dir / 'cleaned'}/")
        logger.info(f"  Removed data:  {output_dir / 'removed'}/ (if any)")
        logger.info(f"  Cleaning report: {report_file}")

        logger.info("\n" + "="*60)
        logger.info("✓ Data cleaning completed successfully!")
        logger.info("="*60)

        # Return status based on cleaning quality
        removal_rate = summary['removal_rate']
        if removal_rate > 0.3:  # More than 30% removed
            logger.warning(
                "\n⚠ Warning: High removal rate detected.\n"
                "Consider adjusting cleaning thresholds or improving data generation quality."
            )
            return 1

        return 0

    except Exception as e:
        logger.error(f"\nCleaning failed: {e}")
        logger.error("Please check the error and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
