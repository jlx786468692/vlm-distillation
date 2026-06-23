"""
Data Validation Script
======================

Validates generated distillation data.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from src import ConfigManager, setup_logger


def validate_json_schema(data: Dict) -> bool:
    """
    Validate JSON schema of distillation data.

    Args:
        data: Data dictionary to validate

    Returns:
        True if valid
    """
    required_keys = ['image_id', 'tasks']

    for key in required_keys:
        if key not in data:
            return False

    return True


def validate_task_data(task_data: Dict) -> Dict[str, Any]:
    """
    Validate task-specific data.

    Args:
        task_data: Task data dictionary

    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'issues': [],
    }

    # Check for required components
    if 'hard_label' not in task_data and 'soft_label' not in task_data:
        report['issues'].append('No labels found')
        report['valid'] = False

    # Check hard label structure
    if 'hard_label' in task_data:
        hard_label = task_data['hard_label']
        if 'image_id' not in hard_label:
            report['issues'].append('Hard label missing image_id')
            report['valid'] = False

    # Check soft label structure
    if 'soft_label' in task_data:
        soft_label = task_data['soft_label']
        if 'temperature' not in soft_label:
            report['issues'].append('Soft label missing temperature')
            report['valid'] = False

    # Check CoT structure
    if 'cot_reasoning' in task_data:
        cot = task_data['cot_reasoning']
        if 'raw_reasoning' not in cot:
            report['issues'].append('CoT missing raw_reasoning')
            report['valid'] = False

    return report


def validate_directory(data_dir: str) -> Dict[str, Any]:
    """
    Validate all data files in directory.

    Args:
        data_dir: Directory containing data files

    Returns:
        Overall validation report
    """
    data_path = Path(data_dir)

    report = {
        'valid': True,
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'issues': [],
        'file_reports': [],
    }

    # Find all JSON files
    json_files = list(data_path.glob("*.json"))

    report['total_files'] = len(json_files)

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate schema
            if not validate_json_schema(data):
                report['invalid_files'] += 1
                report['file_reports'].append({
                    'file': str(json_file),
                    'valid': False,
                    'issue': 'Invalid schema',
                })
                continue

            # Validate each task
            tasks = data.get('tasks', {})
            all_tasks_valid = True
            task_issues = []

            for task_name, task_data in tasks.items():
                task_report = validate_task_data(task_data)
                if not task_report['valid']:
                    all_tasks_valid = False
                    task_issues.extend(task_report['issues'])

            if all_tasks_valid:
                report['valid_files'] += 1
                report['file_reports'].append({
                    'file': str(json_file),
                    'valid': True,
                })
            else:
                report['invalid_files'] += 1
                report['file_reports'].append({
                    'file': str(json_file),
                    'valid': False,
                    'issues': task_issues,
                })

        except Exception as e:
            report['invalid_files'] += 1
            report['issues'].append(f"Error processing {json_file}: {e}")

    # Overall validity
    report['valid'] = report['invalid_files'] == 0

    return report


def generate_validation_report(report: Dict) -> str:
    """
    Generate human-readable validation report.

    Args:
        report: Validation report dictionary

    Returns:
        Formatted report string
    """
    lines = []

    lines.append("="*60)
    lines.append("Data Validation Report")
    lines.append("="*60)

    lines.append(f"\nSummary:")
    lines.append(f"  Total files: {report['total_files']}")
    lines.append(f"  Valid files: {report['valid_files']}")
    lines.append(f"  Invalid files: {report['invalid_files']}")
    lines.append(f"  Overall status: {'VALID' if report['valid'] else 'INVALID'}")

    if report['issues']:
        lines.append(f"\nOverall Issues:")
        for issue in report['issues']:
            lines.append(f"  - {issue}")

    if report['invalid_files'] > 0:
        lines.append(f"\nInvalid Files:")
        for file_report in report['file_reports']:
            if not file_report['valid']:
                lines.append(f"  File: {file_report['file']}")
                if 'issue' in file_report:
                    lines.append(f"    Issue: {file_report['issue']}")
                if 'issues' in file_report:
                    for issue in file_report['issues']:
                        lines.append(f"    - {issue}")

    lines.append("\n" + "="*60)

    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate distillation data"
    )

    parser.add_argument(
        '--input',
        type=str,
        default='./outputs/merged',
        help='Input directory with data files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for validation report'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed validation information'
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger(
        name="validate_data",
        level="INFO",
        console_output=True
    )

    logger.info("="*60)
    logger.info("Data Validation")
    logger.info("="*60)

    logger.info(f"\nInput directory: {args.input}")

    # Validate directory
    logger.info("\nValidating data files...")
    report = validate_directory(args.input)

    # Generate report
    report_text = generate_validation_report(report)
    print(report_text)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nDetailed report saved to: {args.output}")

    # Show detailed report if verbose
    if args.verbose:
        logger.info("\nDetailed file reports:")
        for file_report in report['file_reports']:
            status = "VALID" if file_report['valid'] else "INVALID"
            logger.info(f"  {file_report['file']}: {status}")
            if 'issues' in file_report:
                for issue in file_report['issues']:
                    logger.info(f"    - {issue}")

    # Return status
    if report['valid']:
        logger.info("\n✓ All data files are valid!")
        return 0
    else:
        logger.warning(f"\n✗ Found {report['invalid_files']} invalid files!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
