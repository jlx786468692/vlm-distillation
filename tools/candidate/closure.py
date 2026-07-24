#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
候选集封闭生成器
==============

支持策略：
- frequency_based: 基于频率统计
- semantic_clustering: 基于语义聚类（预留）

使用方式：
    python -m tools.candidate.closure
    python -m tools.candidate.closure --strategy frequency_based --max_candidates 100
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import Counter

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


class CandidateClosure:
    """候选集封闭生成器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config
        self.source_dir = Path(config.get('source_dir', 'data/coco/annotations'))
        self.strategy = config.get('strategy', 'frequency_based')
        self.min_frequency = config.get('min_frequency', 5)
        self.max_candidates = config.get('max_candidates', 100)
        self.logger = get_logger()

    def generate(self) -> Dict[str, Any]:
        """
        生成候选集

        Returns:
            候选集数据
        """
        self.logger.info(f"🎯 使用 {self.strategy} 策略生成候选集")
        self.logger.info(f"  数据源: {self.source_dir}")

        if self.strategy == 'frequency_based':
            return self._generate_frequency_based()
        elif self.strategy == 'semantic_clustering':
            return self._generate_semantic_clustering()
        else:
            raise ValueError(f"未知策略: {self.strategy}")

    def _generate_frequency_based(self) -> Dict[str, Any]:
        """基于频率生成候选集"""
        # 加载VQA标注
        answer_counter = self._load_answers()

        if not answer_counter:
            self.logger.warning("没有找到答案数据，使用默认候选集")
            return self._get_default_candidates()

        # 过滤和选择
        filtered = [
            (ans, freq) for ans, freq in answer_counter.items()
            if freq >= self.min_frequency
        ]

        # 排序并选择Top-K
        top_candidates = sorted(filtered, key=lambda x: x[1], reverse=True)[:self.max_candidates]
        candidate_list = [ans for ans, freq in top_candidates]

        # 计算覆盖率
        total = sum(answer_counter.values())
        covered = sum(freq for ans, freq in top_candidates)
        coverage = covered / total if total > 0 else 0

        self.logger.info(f"✓ 候选集生成完成")
        self.logger.info(f"  总答案数: {len(answer_counter)}")
        self.logger.info(f"  过滤后: {len(filtered)}")
        self.logger.info(f"  最终候选数: {len(candidate_list)}")
        self.logger.info(f"  覆盖率: {coverage:.2%}")

        return {
            'strategy': 'frequency_based',
            'candidates': candidate_list,
            'frequency_map': {ans: freq for ans, freq in top_candidates},
            'metadata': {
                'total_answers': len(answer_counter),
                'filtered': len(filtered),
                'final_count': len(candidate_list),
                'min_frequency': self.min_frequency,
                'max_candidates': self.max_candidates,
                'coverage': coverage,
                'generated_at': datetime.now().isoformat()
            }
        }

    def _generate_semantic_clustering(self) -> Dict[str, Any]:
        """基于语义聚类生成候选集"""
        self.logger.warning("语义聚类策略尚未实现，使用频率策略")
        return self._generate_frequency_based()

    def _load_answers(self) -> Counter:
        """加载答案"""
        answer_counter = Counter()

        # 尝试加载VQA标注
        vqa_file = self.source_dir / "vqa_val2014.json"

        if vqa_file.exists():
            self.logger.info(f"加载VQA标注: {vqa_file}")
            with open(vqa_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for ann in data.get('annotations', []):
                if 'answers' in ann:
                    for ans_obj in ann['answers']:
                        answer = ans_obj.get('answer', '').lower().strip()
                        if answer:
                            answer_counter[answer] += 1
                elif 'answer' in ann:
                    answer = ann['answer'].lower().strip()
                    if answer:
                        answer_counter[answer] += 1

            self.logger.info(f"✓ 加载 {sum(answer_counter.values())} 个答案")

        # 如果没有VQA标注，尝试从merged数据中提取
        if not answer_counter:
            merged_dir = Path("outputs/merged")
            if merged_dir.exists():
                self.logger.info("从merged数据中提取答案")
                json_files = list(merged_dir.glob("COCO_val2014_*.json"))

                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        if 'tasks' in data and 'vqa' in data['tasks']:
                            hard_label = data['tasks']['vqa'].get('hard_label', {})
                            answer = hard_label.get('answer', '')
                            if answer:
                                answer_counter[answer.lower().strip()] += 1
                    except Exception:
                        continue

                self.logger.info(f"✓ 从merged数据中提取 {sum(answer_counter.values())} 个答案")

        return answer_counter

    def _get_default_candidates(self) -> Dict[str, Any]:
        """获取默认候选集"""
        default_candidates = [
            "yes", "no", "1", "2", "3", "4", "5",
            "red", "blue", "green", "black", "white", "yellow",
            "right", "left", "sitting", "standing", "walking",
            "man", "woman", "dog", "cat", "car", "tree"
        ]

        self.logger.info("使用默认候选集")

        return {
            'strategy': 'default',
            'candidates': default_candidates,
            'metadata': {
                'source': 'default',
                'final_count': len(default_candidates),
                'generated_at': datetime.now().isoformat()
            }
        }


def main():
    """独立执行入口"""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="候选集封闭")
    parser.add_argument('--config', default='configs/tools.yaml')
    parser.add_argument('--strategy', choices=['frequency_based', 'semantic_clustering'], default='frequency_based')
    parser.add_argument('--source_dir', default='data/coco/annotations')
    parser.add_argument('--min_frequency', type=int, default=5)
    parser.add_argument('--max_candidates', type=int, default=100)
    parser.add_argument('--output', default='outputs/candidate_sets/closure_data.json')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 合并参数
    closure_config = config.get('candidate_closure', {})
    if args.strategy:
        closure_config['strategy'] = args.strategy
    if args.source_dir:
        closure_config['source_dir'] = args.source_dir
    if args.min_frequency:
        closure_config['min_frequency'] = args.min_frequency
    if args.max_candidates:
        closure_config['max_candidates'] = args.max_candidates

    # 生成
    closure = CandidateClosure(closure_config)
    data = closure.generate()

    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ 候选集已保存: {output_path}")


if __name__ == "__main__":
    main()