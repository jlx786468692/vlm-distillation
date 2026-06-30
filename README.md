# VLM Data Distillation Pipeline

A comprehensive data distillation pipeline for Vision-Language Models (VLMs), enabling knowledge transfer from large teacher models to smaller student models through hard labels, soft labels, and Chain-of-Thought (CoT) reasoning.

## 🎯 Overview

This project provides a complete pipeline for distilling knowledge from **Qwen2.5-VL-7B-Instruct** (teacher model) on the **COCO dataset**, generating rich training data for smaller VLM student models. The generated data includes:

- **Hard Labels**: Final predictions with confidence scores
- **Soft Labels**: Probability distributions with temperature scaling
- **Chain-of-Thought (CoT)**: Step-by-step reasoning processes

### Supported Tasks

- **Visual Question Answering (VQA)**: Answer questions about images
- **Image Captioning**: Generate descriptive captions
- **Object Detection**: Detect and localize objects with bounding boxes

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/vlm-distillation.git
cd vlm-distillation

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Additional Setup

For Qwen-VL models, you may need to install additional utilities:

```bash
pip install qwen-vl-utils
```

## 🔧 Configuration

Configuration files are located in the `configs/` directory:

- **default.yaml**: Main configuration
- **model_config.yaml**: Model-specific settings
- **distillation.yaml**: Distillation parameters

### Quick Configuration Example

```yaml
# configs/default.yaml
teacher:
  model_name: "Qwen/Qwen2.5-VL-7B-Instruct"
  device: "cuda"
  precision: "bf16"

data:
  coco_root: "./data/coco"
  batch_size: 8
  max_samples: 5000  # Set to null for full dataset

distillation:
  tasks:
    - "vqa"
    - "captioning"
    - "detection"

  hard_labels:
    enabled: true
    confidence_threshold: 0.7

  soft_labels:
    enabled: true
    temperature: 2.0
    top_k_logits: 100

  cot:
    enabled: true
    max_length: 512

output:
  root_dir: "./outputs"
  merge_outputs: true
```

## 🚀 Quick Start

### 1. Download COCO Dataset

```bash
python scripts/download_coco.py --split train2017 --output ./data/coco
```

Or manually download from [COCO Dataset](https://cocodataset.org/#download):

- Images: train2017, val2017
- Annotations: captions, instances, questions

### 2. Run Distillation

```bash
# Basic usage
python scripts/run_distillation.py

# With custom config
python scripts/run_distillation.py --config configs/default.yaml

# Specific task only
python scripts/run_distillation.py --task vqa --samples 1000

# Resume from checkpoint
python scripts/run_distillation.py --resume outputs/checkpoint_500.json
```

### 3. Validate Generated Data

```bash
python scripts/validate_data.py --input outputs/merged/
```

### 4. Clean Generated Data ✨ New

Clean the distilled data to improve quality before student model training:

```bash
# Basic cleaning
python scripts/clean_data.py --input outputs/merged

# Custom thresholds
python scripts/clean_data.py \
    --input outputs/merged \
    --min-confidence 0.6 \
    --min-quality 40

# Preview without saving (dry run)
python scripts/clean_data.py --input outputs/merged --dry-run

# Keep invalid data (mark only, don't remove)
python scripts/clean_data.py --input outputs/merged --keep-invalid
```

The cleaning process performs:
- **Anomaly Detection**: Identifies low confidence, invalid answers, empty results, etc.
- **Quality Scoring**: Computes comprehensive quality scores (0-100)
- **Data Filtering**: Removes low-quality data based on thresholds
- **Data Repair**: Fixes anomalous bounding boxes automatically
- **Deduplication**: Marks duplicate answers for review
- **Report Generation**: Provides detailed cleaning statistics and recommendations

Cleaned data is saved to `outputs/cleaned/` with:
- `cleaned/`: High-quality data ready for training
- `removed/`: Low-quality data for analysis
- `cleaning_report.json`: Comprehensive cleaning report

## 📊 Output Format

Generated data is saved in JSON format. Here's an example:

```json
{
  "image_id": "COCO_val2014_000000123456",
  "image_path": "val2014/COCO_val2014_000000123456.jpg",
  "tasks": {
    "vqa": {
      "hard_label": {
        "question": "What is the person doing?",
        "answer": "riding a bike",
        "confidence": 0.95
      },
      "soft_label": {
        "answer_distribution": {
          "riding a bike": 0.95,
          "standing": 0.03,
          "walking": 0.02
        },
        "temperature": 2.0
      },
      "cot_reasoning": "First, I identify the person in the center of the image. Next, I observe they are on a bicycle. The motion blur suggests movement. Therefore, the person is riding a bike."
    },
    "captioning": {
      "hard_label": {
        "caption": "A person riding a bicycle down a city street...",
        "confidence": 0.92
      },
      "soft_label": {
        "caption_variations": [
          {"caption": "A person riding a bicycle...", "score": 0.92},
          {"caption": "A cyclist on a city street...", "score": 0.88}
        ]
      },
      "cot_reasoning": "I start by identifying the main subject..."
    },
    "detection": {
      "hard_label": {
        "objects": [
          {"class": "person", "bbox": [100, 150, 200, 300], "confidence": 0.98},
          {"class": "bicycle", "bbox": [90, 180, 220, 320], "confidence": 0.96}
        ]
      },
      "soft_label": {
        "object_distributions": [...]
      },
      "cot_reasoning": "Scanning the image systematically..."
    }
  },
  "metadata": {
    "teacher_model": "Qwen2.5-VL-7B-Instruct",
    "timestamp": "2026-06-17T10:30:00Z",
    "processing_time_ms": 245
  }
}
```

## 🏗️ Project Structure

```
vlm-distillation/
├── configs/              # Configuration files
│   ├── default.yaml      # Main config (includes cleaning params)
│   ├── model_config.yaml
│   └── distillation.yaml
│
├── src/                  # Source code
│   ├── data/            # Data loading and processing
│   │   ├── coco_loader.py
│   │   ├── image_processor.py
│   │   └── data_manager.py
│   │
│   ├── models/          # Model interfaces
│   │   ├── teacher_model.py
│   │   ├── student_model.py
│   │   └── model_utils.py
│   │
│   ├── distillation/    # Core distillation logic
│   │   ├── hard_label_gen.py
│   │   ├── soft_label_gen.py
│   │   ├── cot_generator.py
│   │   └── distiller.py
│   │
│   ├── cleaning/        # Data cleaning ✨ New
│   │   ├── data_cleaner.py
│   │   └── __init__.py
│   │
│   ├── utils/           # Utilities
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── visualization.py
│   │
│   └── export/          # Data export
│       └── json_exporter.py
│
├── scripts/             # Executable scripts
│   ├── run_distillation.py
│   ├── clean_data.py     ✨ New
│   ├── download_coco.py
│   └── validate_data.py
│
├── outputs/             # Generated outputs
│   ├── merged/          # Raw distilled data
│   ├── cleaned/         ✨ New - Cleaned data
│   │   ├── cleaned/     # High-quality data
│   │   ├── removed/     # Low-quality data
│   │   └── cleaning_report.json
│   └── archive/
│
├── tests/               # Unit tests
│   └── test_distillation.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## 🎓 Usage Examples

### Python API Usage

```python
from src import ConfigManager, TeacherModel, Distiller, COCODataLoader

# Load configuration
config = ConfigManager()

# Initialize teacher model
teacher = TeacherModel(config)

# Load COCO dataset
data_loader = COCODataLoader(config)
images = data_loader.load_split("val2017", max_samples=100)

# Run distillation
distiller = Distiller(teacher, config)
results = distiller.process_batch(images)

# Save results
from src import JSONExporter
exporter = JSONExporter(config)
exporter.save_batch(results, output_dir="./outputs/merged")
```

### Customize Distillation

```python
# Only generate hard labels
distiller = Distiller(
    teacher,
    tasks=["vqa"],
    enable_hard_labels=True,
    enable_soft_labels=False,
    enable_cot=False
)

# Custom CoT generation
cot_generator = CoTGenerator(teacher)
cot_generator.set_prompt_template(
    vqa="Analyze this image systematically. Question: {question}\n"
         "Step 1: Identify visual elements.\n"
         "Step 2: Analyze context.\n"
         "Step 3: Provide reasoning.\n"
         "Answer:"
)
```

## 📈 Performance Optimization

### Memory Management

```yaml
# In configs/default.yaml
teacher:
  use_gradient_checkpointing: false
  load_in_8bit: true  # Enable 8-bit quantization for memory savings

data:
  batch_size: 4  # Reduce batch size for memory constraints
```

### Multi-GPU Support

```python
# Use multiple GPUs for parallel processing
teacher = TeacherModel(
    config,
    device_map="balanced",  # Distribute across GPUs
    num_gpus=4
)
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Integration test on small sample:

```bash
python scripts/run_distillation.py --samples 10 --validate
```

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{vlm_distillation_2026,
  title={VLM Data Distillation Pipeline},
  author={VLM-Distillation Team},
  year={2026},
  url={https://github.com/yourusername/vlm-distillation}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions and support:
- Email: team@example.com
- Issues: [GitHub Issues](https://github.com/yourusername/vlm-distillation/issues)

## 🙏 Acknowledgments

- [Qwen Team](https://github.com/QwenLM) for the Qwen2.5-VL model
- [COCO Dataset](https://cocodataset.org/) for the comprehensive annotations
- [Hugging Face](https://huggingface.co/) for Transformers library

---

Made with ❤️ by VLM-Distillation Team
