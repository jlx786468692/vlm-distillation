"""
Main Distillation Runner
========================

Entry point for running the complete distillation pipeline.
"""

import argparse
import sys
from pathlib import Path

# 兼容两种导入方式：安装后和未安装
try:
    # 安装后的导入方式
    from src import ConfigManager, TeacherModel, Distiller, COCODataLoader, setup_logger
    from src.distillation import HardLabelGenerator, SoftLabelGenerator, CoTGenerator
except ImportError:
    # 未安装时的导入方式（开发模式）
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src import ConfigManager, TeacherModel, Distiller, COCODataLoader, setup_logger
    from src.distillation import HardLabelGenerator, SoftLabelGenerator, CoTGenerator


def main():
    """Main entry point for distillation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run VLM data distillation pipeline"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (overrides config)'
    )

    parser.add_argument(
        '--task',
        type=str,
        nargs='+',
        choices=['vqa', 'captioning', 'detection'],
        default=['vqa', 'captioning', 'detection'],
        help='Tasks to run (overrides config)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate results after distillation'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations of results'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run setup without actual processing'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        name="run_distillation",
        level="INFO",
        log_file="./logs/distillation.log",
        console_output=True
    )

    logger.info("="*60)
    logger.info("VLM Data Distillation Pipeline!!!")
    logger.info("="*60)

    # Load configuration
    logger.info(f"\nLoading configuration from: {args.config}")
    config = ConfigManager(args.config)

    # Override config with command-line arguments
    if args.samples:
        logger.info(f"Overriding max_samples: {args.samples}")
        config.set('data.max_samples', args.samples)

    if args.task:
        logger.info(f"Overriding tasks: {args.task}")
        config.set('distillation.tasks', args.task)

    if args.output_dir:
        logger.info(f"Overriding output_dir: {args.output_dir}")
        config.set('output.root_dir', args.output_dir)

    # Validate configuration
    if not config.validate():
        logger.error("Configuration validation failed!")
        sys.exit(1)

    logger.info("Configuration loaded successfully")

    # Initialize components
    logger.info("\n" + "-"*60)
    logger.info("Initializing components...")
    logger.info("-"*60)

    # Initialize data loader
    logger.info("\n1. Initializing COCO dataset loader...")
    try:
        coco_loader = COCODataLoader(config)
        coco_loader.initialize(config.get('data.val_split', 'val2017'))
        summary = coco_loader.get_annotation_summary()
        logger.info(f"Dataset loaded: {summary}")
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        sys.exit(1)

    # Initialize teacher model
    logger.info("\n2. Loading teacher model...")
    try:
        teacher = TeacherModel(config)
        model_info = teacher.get_model_info()
        logger.info(f"Teacher model loaded: {model_info}")
    except Exception as e:
        logger.error(f"Failed to load teacher model: {e}")
        logger.error("Make sure you have GPU access and model is available")
        sys.exit(1)

    # Initialize distiller
    logger.info("\n3. Creating distiller...")
    distiller = Distiller(
        teacher_model=teacher,
        config=config,
        data_manager=None  # Will be auto-created
    )

    logger.info(f"Distiller initialized: {distiller}")

    # Dry run check
    if args.dry_run:
        logger.info("\n" + "="*60)
        logger.info("DRY RUN - Setup complete, no processing will be done")
        logger.info("="*60)

        status = distiller.get_processing_status()
        logger.info(f"\nProcessing status: {status}")

        return 0

    # Run distillation
    logger.info("\n" + "="*60)
    logger.info("Starting distillation process...")
    logger.info("="*60)

    try:
        results = distiller.run_distillation(
            max_samples=args.samples,
            checkpoint_path=args.resume
        )

        logger.info("\n" + "="*60)
        logger.info("Distillation completed!")
        logger.info("="*60)

        logger.info(f"\nResults summary:")
        logger.info(f"- Processed images: {results['processed_count']}")
        logger.info(f"- Failed images: {results['failed_count']}")
        logger.info(f"- Statistics: {results['statistics']}")

        # Validate results if requested
        if args.validate:
            logger.info("\n" + "-"*60)
            logger.info("Validating results...")
            logger.info("-"*60)

            validation_report = distiller.validate_results()
            logger.info(f"Validation report: {validation_report}")

            if not validation_report['valid']:
                logger.warning("Validation found issues!")
                for error in validation_report['errors']:
                    logger.error(f"  - {error}")

        # Create visualizations if requested
        if args.visualize:
            logger.info("\n" + "-"*60)
            logger.info("Creating visualizations...")
            logger.info("-"*60)

            from src.utils.visualization import visualize_results
            viz_results = visualize_results(
                results['merged_data_path'],
                num_samples=5
            )
            logger.info(f"Visualizations created: {viz_results}")

        logger.info("\n" + "="*60)
        logger.info("Pipeline execution completed successfully!")
        logger.info("="*60)

        return 0

    except Exception as e:
        logger.error(f"\nDistillation failed: {e}")
        logger.error("Check logs for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
