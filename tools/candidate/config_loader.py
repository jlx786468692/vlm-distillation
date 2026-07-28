"""
候选集封闭配置加载
==================

从configs/tools.yaml读取配置，包括教师模型Prompt。
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_candidate_closure_config(config_file: str = 'configs/tools.yaml') -> Dict[str, Any]:
    """
    加载候选集封闭配置

    Returns:
        配置字典
    """
    config_path = Path(config_file)

    if not config_path.exists():
        print(f"⚠️  配置文件不存在: {config_path}")
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config.get('candidate_closure', {})


def get_teacher_prompt_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取教师模型Prompt配置

    Args:
        config: 配置字典（如果为None，从文件读取）

    Returns:
        Prompt配置字典
    """
    if config is None:
        config = load_candidate_closure_config()

    return config.get('teacher_prompt', {})


def get_teacher_generation_params(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    获取教师模型生成参数

    Args:
        config: 配置字典（如果为None，从文件读取）

    Returns:
        生成参数字典
    """
    if config is None:
        config = load_candidate_closure_config()

    return config.get('generation_params', {})


def build_prompt(answer: str, prompt_config: Dict[str, Any] = None) -> str:
    """
    构建Prompt

    Args:
        answer: 答案
        prompt_config: Prompt配置

    Returns:
        构建好的Prompt
    """
    if prompt_config is None:
        prompt_config = get_teacher_prompt_config()

    template = prompt_config.get('template', '')

    if not template:
        # 默认prompt
        template = """请列出答案"{answer}"的所有可能的同义答案、单复数变体、近似表达。

要求：
1. 输出所有简短答案、同义词、单复数变体
2. 用逗号分隔
3. 不要有多余文字
4. 包含原文答案

示例：
输入：dog
输出：dog, dogs, puppy, puppies, canine, hound

输入答案：{answer}
输出："""

    return template.format(answer=answer)


# 示例使用
if __name__ == "__main__":
    print("="*60)
    print("候选集封闭配置示例")
    print("="*60)

    # 加载配置
    config = load_candidate_closure_config()

    print("\n配置项：")
    for key, value in config.items():
        if key in ['teacher_prompt', 'generation_params']:
            print(f"  {key}: ...（详细配置）")
        else:
            print(f"  {key}: {value}")

    # Prompt配置
    prompt_config = get_teacher_prompt_config(config)

    print("\nPrompt模板：")
    print("-"*60)
    template = prompt_config.get('template', '')
    print(template.format(answer="<ANSWER>"))

    # 生成参数
    gen_params = get_teacher_generation_params(config)

    print("\n生成参数：")
    for key, value in gen_params.items():
        print(f"  {key}: {value}")

    # 构建示例prompt
    print("\n示例Prompt（答案='dog'）：")
    print("-"*60)
    prompt = build_prompt('dog', prompt_config)
    print(prompt)