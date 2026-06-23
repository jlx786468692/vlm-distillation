"""
Quick Start Examples
====================

Example scripts showing how to use the VLM distillation pipeline.
"""

# Example 1: Basic Distillation Pipeline
"""
Basic usage of the distillation pipeline:

```python
from src import ConfigManager, TeacherModel, Distiller

# Load configuration
config = ConfigManager('configs/default.yaml')

# Initialize components
teacher = TeacherModel(config)
distiller = Distiller(teacher, config)

# Run distillation
results = distiller.run_distillation(max_samples=100)

print(f"Processed {results['processed_count']} images")
print(f"Results saved to: {results['merged_data_path']}")
```
"""

# Example 2: Single Task Distillation
"""
Run distillation for specific task only:

```python
from src import ConfigManager, TeacherModel, Distiller

config = ConfigManager()
teacher = TeacherModel(config)

# Only process VQA task
distiller = Distiller(teacher, config)
config.set('distillation.tasks', ['vqa'])

results = distiller.run_distillation(max_samples=50)
```
"""

# Example 3: Resume from Checkpoint
"""
Resume processing from previous checkpoint:

```python
from src import ConfigManager, TeacherModel, Distiller

config = ConfigManager()
teacher = TeacherModel(config)
distiller = Distiller(teacher, config)

# Resume from checkpoint
results = distiller.run_distillation(
    max_samples=500,
    checkpoint_path='./outputs/checkpoint_latest.json'
)
```
"""

# Example 4: Generate Only Hard Labels
"""
Generate only hard labels (skip soft labels and CoT):

```python
from src import ConfigManager, TeacherModel, Distiller

config = ConfigManager()
teacher = TeacherModel(config)

# Disable soft labels and CoT
config.set('distillation.soft_labels.enabled', False)
config.set('distillation.cot.enabled', False)

distiller = Distiller(teacher, config)
results = distiller.run_distillation()
```
"""

# Example 5: Validate Results
"""
Validate generated distillation results:

```python
from scripts.validate_data import validate_directory, generate_validation_report

# Validate merged results
report = validate_directory('./outputs/merged')

# Print report
print(generate_validation_report(report))

if report['valid']:
    print("✓ All data is valid!")
else:
    print(f"✗ Found {report['invalid_files']} invalid files")
```
"""

# Example 6: Load and Use Distilled Data
"""
Load distilled data for training student model:

```python
import json
from pathlib import Path

# Load distilled results
merged_dir = Path('./outputs/merged')
results = []

for json_file in merged_dir.glob('*.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
        results.append(data)

# Access data
for result in results[:5]:
    print(f"Image ID: {result['image_id']}")

    # VQA data
    if 'vqa' in result['tasks']:
        vqa = result['tasks']['vqa']
        print(f"  VQA Answer: {vqa['hard_label']['answer']}")
        print(f"  VQA CoT: {vqa['cot_reasoning']['raw_reasoning'][:100]}")

    # Captioning data
    if 'captioning' in result['tasks']:
        captioning = result['tasks']['captioning']
        print(f"  Caption: {captioning['hard_label']['captions'][0]}")

    # Detection data
    if 'detection' in result['tasks']:
        detection = result['tasks']['detection']
        print(f"  Objects detected: {len(detection['hard_label']['objects'])}")
```
"""

# Example 7: Custom CoT Prompt
"""
Customize Chain-of-Thought prompts:

```python
from src import ConfigManager, TeacherModel
from src.distillation import CoTGenerator

config = ConfigManager()
teacher = TeacherModel(config)
cot_gen = CoTGenerator(teacher, config)

# Generate CoT with default prompt
cot_result = cot_gen.generate_vqa_cot(
    'image.jpg',
    'What objects are in the image?',
    'img_001'
)

# Custom reasoning structure
print(cot_result['structured_reasoning'])
```
"""

# Example 8: Command-line Usage
"""
Run from command line:

```bash
# Basic run
python scripts/run_distillation.py

# With options
python scripts/run_distillation.py \
    --samples 100 \
    --task vqa captioning \
    --output-dir ./outputs_test \
    --validate \
    --visualize

# Resume from checkpoint
python scripts/run_distillation.py \
    --resume ./outputs/checkpoint_latest.json

# Dry run (setup only, no processing)
python scripts/run_distillation.py --dry-run

# Download COCO dataset
python scripts/download_coco.py \
    --split val2017 \
    --output ./data/coco

# Validate generated data
python scripts/validate_data.py \
    --input ./outputs/merged \
    --verbose
```
"""

# Example 9: Student Model Placeholder
"""
Initialize student model for future training:

```python
from src import StudentModel

# Student model placeholder (will be configured for actual model later)
student = StudentModel()

# Check if loaded
if student.is_loaded():
    print(f"Student model loaded: {student.model_name}")
else:
    print("Student model is placeholder for future training")

# Prepare for training (when model is loaded)
if student.is_loaded():
    student.prepare_for_training()
```
"""

# Example 10: Processing Statistics
"""
Monitor processing statistics:

```python
from src import Distiller, ConfigManager, TeacherModel

config = ConfigManager()
teacher = TeacherModel(config)
distiller = Distiller(teacher, config)

# Get current status
status = distiller.get_processing_status()

print(f"Progress: {status['progress_percent']:.1f}%")
print(f"Processed: {status['processed_images']}/{status['total_images']}")
print(f"Elapsed time: {status['elapsed_time']}")

# Run distillation and get final stats
results = distiller.run_distillation()
print(f"Final statistics: {results['statistics']}")
```
"""

print("\n" + "="*60)
print("VLM Distillation Examples")
print("="*60)
print("\nSee examples above for various usage patterns.")
print("\nKey usage patterns:")
print("  1. Basic pipeline: ConfigManager + TeacherModel + Distiller")
print("  2. Task selection: config.set('distillation.tasks', [...])")
print("  3. Checkpointing: Use --resume flag or checkpoint_path parameter")
print("  4. Validation: validate_directory() and validate_results()")
print("  5. Command-line: run_distillation.py with various flags")
print("\nFor detailed documentation, see README.md")
print("="*60)
