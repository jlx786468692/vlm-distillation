#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具模块 - 统一执行入口
======================

使用方式：
    python -m tools                              # 显示帮助
    python -m tools prompt_generator             # 生成Prompt
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
    """运行候选集封闭（千问开源流水线三阶段）"""
    from pathlib import Path
    import subprocess
    import sys

    print("\n" + "="*60)
    print("🎯 Candidate Closure (千问开源流水线三阶段)")
    print("="*60)
    print("\n阶段1：构建全局超大候选池 C_all")
    print("阶段2：COCO图像分场景")
    print("阶段3：分场景过滤生成局部小闭合集")

    # 检查必需的COCO标注文件
    annotations_dir = Path("data/coco/annotations")

    required_files = {
        'VQA标注': annotations_dir / "v2_mscoco_train2014_annotations.json",
        'COCO Captions': annotations_dir / "captions_train2014.json",
        'COCO Instances': annotations_dir / "instances_train2014.json"
    }

    missing_files = []
    for name, file_path in required_files.items():
        if not file_path.exists():
            missing_files.append((name, file_path))

    if missing_files:
        print("\n❌ 缺少必需的COCO标注文件：")
        for name, file_path in missing_files:
            print(f"  {name}: {file_path}")

        print("\n请下载COCO和VQA数据集：")
        print("\n1. COCO数据集：")
        print("   访问: https://cocodataset.org/#download")
        print("   下载: 2014 Train/Val annotations")
        print("   解压到: data/coco/annotations/")

        print("\n2. VQA v2数据集：")
        print("   访问: https://visualqa.org/download.html")
        print("   下载: Trainable data: v2_Annotations_Train_mscoco.zip")
        print("   解压到: data/coco/annotations/")

        print("\n或者使用已有的merged数据（简化方案）：")
        print("   python tools/candidate/generate_vqa_vocab.py --source merged")

        return

    # 检查输出文件
    global_pool_file = Path("data/global_candidate_pool.json")
    scene_mapping_file = Path("data/imgid2scene.json")
    scene_candidates_file = Path("data/scene_candidates.json")

    # 阶段1：构建全局超大候选池
    if not global_pool_file.exists():
        print("\n" + "="*60)
        print("阶段1：构建全局超大候选池")
        print("="*60)

        try:
            # 从配置读取是否使用教师模型
            config = load_config(args.config)
            closure_config = config.get('candidate_closure', {})
            use_teacher_model = closure_config.get('use_teacher_model', False)
            teacher_model_path = closure_config.get('teacher_model', 'models/Qwen2.5-VL-32B-Instruct-AWQ')
            max_expand_samples = closure_config.get('max_expand_samples', 1000)

            # 构建命令
            cmd = [sys.executable, "tools/candidate/stage1_build_global_pool.py"]

            if use_teacher_model:
                cmd.extend([
                    "--use-teacher-model",
                    "--teacher-model", teacher_model_path,
                    "--max-expand-samples", str(max_expand_samples)
                ])
                print(f"使用教师模型扩充: {teacher_model_path}")
                print(f"扩充样本数: {max_expand_samples}")

            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent.parent,
                check=True
            )
            print("✓ 阶段1完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 阶段1失败: {e}")
            return
    else:
        print(f"\n✓ 全局候选池已存在: {global_pool_file}")

    # 阶段2：COCO图像分场景
    if not scene_mapping_file.exists():
        print("\n" + "="*60)
        print("阶段2：COCO图像分场景")
        print("="*60)

        try:
            result = subprocess.run(
                [sys.executable, "tools/candidate/stage2_scene_mapping.py"],
                cwd=Path(__file__).parent.parent,
                check=True
            )
            print("✓ 阶段2完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 阶段2失败: {e}")
            return
    else:
        print(f"✓ 场景映射已存在: {scene_mapping_file}")

    # 阶段3：分场景过滤生成局部小闭合集
    if not scene_candidates_file.exists():
        print("\n" + "="*60)
        print("阶段3：分场景过滤生成局部小闭合集")
        print("="*60)

        try:
            result = subprocess.run(
                [sys.executable, "tools/candidate/stage3_scene_closure.py"],
                cwd=Path(__file__).parent.parent,
                check=True
            )
            print("✓ 阶段3完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ 阶段3失败: {e}")
            return
    else:
        print(f"✓ 分场景候选集已存在: {scene_candidates_file}")

    # 显示结果
    print("\n" + "="*60)
    print("✓ 千问开源流水线三阶段完成")
    print("="*60)

    print("\n输出文件：")
    print(f"  1. 全局候选池: {global_pool_file}")
    print(f"  2. 场景映射: {scene_mapping_file}")
    print(f"  3. 分场景候选集: {scene_candidates_file}")

    # 显示统计信息
    if scene_candidates_file.exists():
        import json
        with open(scene_candidates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print("\n场景候选集统计：")
        total_candidates = 0
        for scene, scene_data in data.get('scenes', {}).items():
            count = scene_data.get('count', 0)
            total_candidates += count
            print(f"  {scene:15s}: {count:4d} 个候选")

        print(f"\n总候选数: {total_candidates}")
        print(f"平均候选数: {total_candidates / len(data.get('scenes', {})):.0f}")

    print("\n💡 提示：")
    print("  - 候选集会在软标签生成时自动调用")
    print("  - 数据来源：VQA标注 + COCO Caption + 教师模型（可选）")
    print("  - 场景划分：COCO原生12大场景")
    print("  - 三层过滤：物体硬匹配 + 频次过滤 + 语义过滤")


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