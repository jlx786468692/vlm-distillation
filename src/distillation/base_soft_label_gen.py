"""
Base Soft Label Generator
=========================

软标签生成器的基类，提供公共功能：
- 温度缩放
- Top-K概率提取
- 数据序列化
- 保存和验证
- 统计信息

子类需要实现：
- generate_*_soft_labels(): 特定任务的软标签生成
- _process_*_logits(): 特定任务的logits处理
"""

import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..models.teacher_model import TeacherModel
from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class BaseSoftLabelGenerator:
    """
    软标签生成器基类

    提供所有软标签生成任务的公共功能：
    - 温度缩放（Temperature Scaling）
    - Top-K概率提取
    - 数据序列化
    - 文件保存
    - 数据验证
    - 统计信息
    """

    def __init__(
        self,
        teacher_model: TeacherModel,
        config: Optional[ConfigManager] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        """
        初始化软标签生成器基类

        Args:
            teacher_model: Teacher模型实例
            config: 配置管理器
            temperature: 温度参数（覆盖配置）
            top_k: Top-K参数（覆盖配置）
        """
        self.teacher = teacher_model
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # 温度参数：控制概率分布的平滑程度
        # temperature越高，分布越平滑（更接近均匀分布）
        # temperature越低，分布越尖锐（更接近hard label）
        self.temperature = temperature or self.config.get(
            "distillation.soft_labels.temperature", 2.0
        )

        # Top-K参数：保留概率最高的K个候选
        self.top_k = top_k or self.config.get(
            "distillation.soft_labels.top_k", 50
        )

        self.logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(temp={self.temperature}, top_k={self.top_k})"
        )

    # ==================
    # 公共方法：温度缩放和概率提取
    # ==================

    def _apply_temperature(
        self,
        logits: torch.Tensor
    ) -> torch.Tensor:
        """
        应用温度缩放到logits

        标准公式：
            scaled_logits = logits / temperature
            scaled_probs = softmax(scaled_logits, dim=-1)

        Args:
            logits: Logits张量（不是概率）

        Returns:
            温度缩放后的概率分布
        """
        # 应用温度缩放
        scaled_logits = logits / self.temperature

        # 计算softmax得到概率
        scaled_probs = torch.softmax(scaled_logits, dim=-1)

        return scaled_probs

    def _get_top_k_probabilities(
        self,
        probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        提取Top-K概率

        Args:
            probs: 概率张量

        Returns:
            包含top-k索引和值的字典
        """
        # 如果需要，展平张量
        if probs.dim() > 1:
            probs_flat = probs.view(-1)
        else:
            probs_flat = probs

        # 获取top-k
        top_k = min(self.top_k, probs_flat.size(0))
        top_values, top_indices = torch.topk(probs_flat, top_k)

        return {
            'indices': top_indices,
            'values': top_values,
        }

    # ==================
    # 公共方法：数据序列化
    # ==================

    def _make_serializable(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将张量转换为可序列化格式

        Args:
            data: 可能包含张量的字典

        Returns:
            JSON可序列化的字典
        """
        serializable = {}

        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                serializable[key] = value.tolist()
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._make_serializable(value)
            elif isinstance(value, list):
                serializable[key] = [
                    self._make_serializable(v) if isinstance(v, dict) else
                    v.tolist() if isinstance(v, (torch.Tensor, np.ndarray)) else v
                    for v in value
                ]
            else:
                serializable[key] = value

        return serializable

    # ==================
    # 公共方法：文件保存
    # ==================

    def save_soft_labels(
        self,
        soft_labels: Dict[str, Any],
        output_path: str
    ) -> bool:
        """
        保存软标签到文件

        Args:
            soft_labels: 软标签数据
            output_path: 保存路径

        Returns:
            是否成功
        """
        try:
            # 转换张量为可序列化格式
            serializable = self._make_serializable(soft_labels)

            # 创建目录
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # 保存JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Soft labels saved to {path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save soft labels: {e}")
            return False

    # ==================
    # 公共方法：数据验证
    # ==================

    def validate_soft_labels(
        self,
        soft_labels: Dict[str, Any],
        required_keys: Optional[List[str]] = None
    ) -> bool:
        """
        验证软标签结构

        Args:
            soft_labels: 软标签字典
            required_keys: 必需的键列表（子类可覆盖）

        Returns:
            是否有效
        """
        # 默认必需键（子类可扩展）
        if required_keys is None:
            required_keys = ['temperature', 'timestamp']

        for key in required_keys:
            if key not in soft_labels:
                self.logger.warning(f"Missing key in soft labels: {key}")
                return False

        return True

    # ==================
    # 公共方法：统计信息
    # ==================

    def get_statistics(
        self,
        soft_labels_list: List[Dict]
    ) -> Dict[str, Any]:
        """
        计算软标签统计信息

        Args:
            soft_labels_list: 软标签列表

        Returns:
            统计字典
        """
        stats = {
            'total_count': len(soft_labels_list),
            'by_task': {},
            'average_temperature': self.temperature,
            'total_probabilities': 0,
        }

        for label in soft_labels_list:
            task = label.get('task', 'unknown')
            if task not in stats['by_task']:
                stats['by_task'][task] = 0
            stats['by_task'][task] += 1

            # 统计概率数量（VQA）
            if 'answer_distribution' in label:
                stats['total_probabilities'] += len(label['answer_distribution'])

            # 统计概率数量（Detection）
            if 'category_distribution' in label:
                stats['total_probabilities'] += len(label['category_distribution'])

        return stats

    # ==================
    # 公共方法：相似类别查找（子类可覆盖）
    # ==================

    def _get_similar_categories(
        self,
        category: str,
        category_list: Optional[List[str]] = None
    ) -> List[str]:
        """
        获取相似类别（子类可覆盖）

        Args:
            category: 类别名称
            category_list: 类别列表（可选）

        Returns:
            相似类别列表
        """
        # 基类提供简单实现
        # 子类可以覆盖此方法提供更复杂的相似度计算
        return [category]

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}"
            f"(teacher={self.teacher.model_name}, "
            f"temp={self.temperature}, "
            f"top_k={self.top_k})"
        )