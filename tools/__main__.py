#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块 - 统一执行入口
======================

使用方式：
    python -m tools                              # 显示帮助
    python -m tools prompt_generator             # 生成Prompt
    python -m tools prompt_optimizer             # 优化Prompt
    python -m tools candidate_closure            # 生成候选集
    python -m tools all                          # 运行所有工具
"""

import sys
import argparse
from pathlib import Path
import yaml
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_config(config_path: str = "configs/tools.yaml") -> dict:
    """
    加载配置

    优先级：
    1. configs/tools.yaml（工具独立配置）
    2. configs/default.yaml（主配置）
    """
    path = Path(config_path)

    # 如果是默认路径，检查tools.yaml是否存在
    if config_path == "configs/tools.yaml":
        tools_config_path = Path("configs/tools.yaml")
        default_config_path = Path("configs/default.yaml")

        # 优先使用tools.yaml
        if tools_config_path.exists():
            with open(tools_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        # 如果tools.yaml不存在，使用default.yaml
        elif default_config_path.exists():
            with open(default_config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}

    # 如果指定了其他配置文件，直接加载
    elif path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    return {}


def run_prompt_generator(args):
    """运行Prompt生成"""
    from tools.prompt.generator import PromptGenerator

    print("\n" + "="*60)
    print("🚀 Prompt Generator")
    print("="*60)

    config = load_config(args.config)
    generator = PromptGenerator(config.get('prompt_generation', {}))
    prompts = generator.generate()

    if not prompts:
        print("❌ Prompt生成失败")
        return

    # 保存
    output_path = Path(config.get('prompt_generation', {}).get('output_file', 'outputs/prompts/vqa_en.yaml'))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(prompts, f, default_flow_style=False, allow_unicode=True)

    print(f"✓ Prompt已生成: {output_path}")


def run_prompt_optimizer(args):
    """运行Prompt优化"""
    print("\n" + "="*60)
    print("🔧 Prompt Optimizer")
    print("="*60)

    print("⚠ Prompt优化功能尚未实现")
    print("  请使用 prompt_generator 生成Prompt")


def run_candidate_closure(args):
    """运行候选集封闭"""
    from tools.candidate.closure import CandidateClosure

    print("\n" + "="*60)
    print("🎯 Candidate Closure")
    print("="*60)

    config = load_config(args.config)
    closure = CandidateClosure(config.get('candidate_closure', {}))
    data = closure.generate()

    if not data:
        print("❌ 候选集生成失败")
        return

    # 保存
    output_path = Path(config.get('candidate_closure', {}).get('output_file', 'outputs/candidate_sets/closure_data.json'))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ 候选集已生成: {output_path}")


def run_all(args):
    """运行所有工具"""
    print("\n" + "="*60)
    print("🚀 Running All Tools")
    print("="*60)

    run_prompt_generator(args)
    run_candidate_closure(args)

    print("\n" + "="*60)
    print("✓ All tools completed!")
    print("="*60)

    # 显示生成的文件
    print("\n📦 生成的文件:")
    print("-"*60)

    prompt_file = Path("outputs/prompts/vqa_en.yaml")
    if prompt_file.exists():
        print(f"  ✓ Prompt: {prompt_file}")

    closure_file = Path("outputs/candidate_sets/closure_data.json")
    if closure_file.exists():
        print(f"  ✓ 候选集: {closure_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="工具模块 - Prompt生成和候选集封闭",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m tools all                      # 运行所有工具
  python -m tools prompt_generator         # 只生成Prompt
  python -m tools candidate_closure        # 只生成候选集
  python -m tools all --force              # 强制重新生成
        """
    )

    parser.add_argument(
        'task',
        choices=['prompt_generator', 'prompt_optimizer', 'candidate_closure', 'all'],
        help='要运行的工具任务'
    )

    parser.add_argument(
        '--config',
        default='configs/tools.yaml',
        help='配置文件路径（默认：configs/tools.yaml）'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新生成'
    )

    args = parser.parse_args()

    # 执行任务
    if args.task == 'prompt_generator':
        run_prompt_generator(args)
    elif args.task == 'prompt_optimizer':
        run_prompt_optimizer(args)
    elif args.task == 'candidate_closure':
        run_candidate_closure(args)
    elif args.task == 'all':
        run_all(args)


if __name__ == "__main__":
    main()