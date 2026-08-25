"""
完整数据管道脚本（瘦身版）
=========================

脚本只做三件事：
  1. 设置 multiprocessing spawn 模式 + vLLM 日志级别
  2. 解析 --config / --steps 命令行参数
  3. 调用 src/pipeline/runner.PipelineRunner().run_full_pipeline(steps)

实际编排逻辑全部位于 src/pipeline/runner.py。

六步流水线（configs/default.yaml 中 pipeline.default_steps）：
  1. distillation          —— 数据蒸馏
  2. cleaning              —— 数据清洗
  3. prepare_training_data —— 生成训练 jsonl
  4. training              —— 学生模型训练
  5. evaluation            —— 学生模型评估（蒸馏质量/驾驶场景适配/部署效率）
  6. visualization         —— 可视化

另可选 quality_validation（不列入默认六步）。

Usage:
    # 运行默认六步
    python scripts/run_full_pipeline.py

    # 指定配置文件
    python scripts/run_full_pipeline.py --config configs/custom.yaml

    # 只跑部分步骤
    python scripts/run_full_pipeline.py --steps prepare_training_data training evaluation
"""

# 🔧 vLLM 多 GPU 推理（tensor_parallel）需要 spawn 模式，必须先于任何导入设置
import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # 已设置过

import os
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')

import argparse
import sys
from pathlib import Path

# 兼容两种导入方式
try:
    from src.pipeline import PipelineRunner
except ImportError:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.pipeline import PipelineRunner


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="Run the full VLM data distillation pipeline (6 steps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)',
    )

    parser.add_argument(
        '--steps',
        nargs='+',
        default=None,
        choices=[
            'distillation', 'cleaning', 'prepare_training_data',
            'training', 'evaluation', 'visualization', 'quality_validation',
        ],
        help='Steps to run (overrides pipeline.default_steps in config)',
    )

    args = parser.parse_args()

    # 编排逻辑全部在 src 中
    runner = PipelineRunner(config_path=args.config)

    # 步骤优先级：命令行 > 配置文件
    steps = args.steps
    if steps is None:
        steps = runner.config.get('pipeline.default_steps', runner.DEFAULT_STEPS)

    result = runner.run_full_pipeline(steps=steps)

    # 退出码
    sys.exit(0 if result.get('success') else 1)


if __name__ == '__main__':
    main()
