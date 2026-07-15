"""
JSON Exporter (重构版)
======================

每个图片单独保存为JSON文件，文件名=image_id
不再使用batch文件和archive归档
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class JSONExporter:
    """
    导出蒸馏结果为JSON格式（重构版）

    特点：
    - 每个图片单独保存一个JSON文件
    - 文件名使用image_id（如：391895.json）
    - 直接保存到merged目录，无需后续合并
    - 不使用batch文件和archive归档
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None
    ):
        """
        Initialize JSON Exporter.

        Args:
            config: Configuration manager
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # Output settings
        self.output_dir = Path(self.config.get("output.root_dir", "./outputs"))
        self.merged_dir = Path(self.config.get("output.merged_dir", "./outputs/merged"))

        # Ensure output directories exist
        self._ensure_output_dirs()

    def _ensure_output_dirs(self) -> None:
        """Create output directories."""
        dirs = [
            self.output_dir,
            self.merged_dir,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directories created: {self.merged_dir}")

    def save_image_result(
        self,
        image_result: Dict[str, Any],
        image_id: Optional[str] = None
    ) -> bool:
        """
        保存单个图片的蒸馏结果

        Args:
            image_result: 图片结果字典，包含:
                - image_id: 图片ID
                - image_path: 图片路径（用于提取文件名）
                - tasks: 各任务结果（vqa, detection等）
                - metadata: 元数据
            image_id: 可选，如果提供则使用此ID作为文件名

        Returns:
            True if successful
        """
        # 获取image_id
        if image_id is None:
            image_id = image_result.get('image_id')

        if not image_id:
            self.logger.error("无法保存：缺少image_id")
            return False

        # 验证结果结构
        if not self._validate_result(image_result):
            self.logger.warning(f"图片 {image_id} 结果结构无效，跳过保存")
            return False

        # 关键改进：从image_path提取原始文件名
        # 例如：/data/coco/val2014/COCO_val2014_000000391895.jpg -> COCO_val2014_000000391895
        image_path = image_result.get('image_path', '')

        if image_path:
            # 提取文件名（去掉路径和扩展名）
            from pathlib import Path as PathLib
            filename = PathLib(image_path).stem  # 去掉.jpg扩展名
            json_filename = f"{filename}.json"
        else:
            # 如果没有image_path，使用image_id
            json_filename = f"{image_id}.json"

        # 构建文件路径：merged/{filename}.json
        output_path = self.merged_dir / json_filename

        try:
            # 添加保存时间戳
            if 'metadata' not in image_result:
                image_result['metadata'] = {}

            image_result['metadata']['save_timestamp'] = datetime.now().isoformat()

            # 🔧 关键修复：转换Tensor为可序列化格式
            serializable_result = self._make_serializable(image_result)

            # 保存JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"图片结果已保存: {output_path.name} (image_id={image_id})")
            return True

        except Exception as e:
            self.logger.error(f"保存图片 {image_id} 失败: {e}")
            return False

    def save_batch_results(
        self,
        batch_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        保存批处理中的所有图片结果

        新逻辑：遍历batch中的每个图片，单独保存

        Args:
            batch_results: 批处理结果，包含:
                - batch_id: 批次ID
                - images: 图片结果列表

        Returns:
            保存统计信息
        """
        images = batch_results.get('images', [])

        if not images:
            self.logger.warning("批处理结果中无图片数据")
            return {
                'total': 0,
                'saved': 0,
                'failed': 0
            }

        saved_count = 0
        failed_count = 0

        for image_result in images:
            image_id = image_result.get('image_id')

            if self.save_image_result(image_result, image_id):
                saved_count += 1
            else:
                failed_count += 1
                self.logger.warning(f"图片 {image_id} 保存失败")

        total_count = len(images)

        self.logger.info(
            f"批处理保存完成: 总计 {total_count}, "
            f"成功 {saved_count}, 失败 {failed_count}"
        )

        return {
            'total': total_count,
            'saved': saved_count,
            'failed': failed_count,
            'batch_id': batch_results.get('batch_id')
        }

    def _validate_result(
        self,
        result: Dict[str, Any]
    ) -> bool:
        """
        验证结果结构

        Args:
            result: Result dictionary

        Returns:
            True if valid
        """
        # 必需的顶级字段
        required_keys = ['image_id']

        for key in required_keys:
            if key not in result:
                self.logger.warning(f"缺少必需字段: {key}")
                return False

        # 验证tasks结构（如果存在）
        if 'tasks' in result:
            if not isinstance(result['tasks'], dict):
                self.logger.warning("tasks字段应为字典类型")
                return False

            for task_name, task_data in result['tasks'].items():
                if not isinstance(task_data, dict):
                    self.logger.warning(f"任务 {task_name} 数据应为字典类型")
                    return False

        return True

    def load_image_result(
        self,
        image_id: str,
        image_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        加载单个图片的结果

        Args:
            image_id: 图片ID
            image_path: 可选，图片路径（用于提取文件名）

        Returns:
            图片结果字典，如果不存在返回None
        """
        # 确定JSON文件名
        if image_path:
            from pathlib import Path as PathLib
            filename = PathLib(image_path).stem
            json_filename = f"{filename}.json"
        else:
            # 尝试两种格式：
            # 1. 直接使用image_id（如：391895.json）
            # 2. 假设是COCO格式（如：COCO_val2014_000000391895.json）
            json_filename = f"{image_id}.json"

        file_path = self.merged_dir / json_filename

        # 如果直接文件不存在，尝试查找匹配的文件
        if not file_path.exists():
            # 尝试查找包含该image_id的文件
            possible_files = list(self.merged_dir.glob(f"*{image_id}*.json"))
            if possible_files:
                file_path = possible_files[0]
            else:
                self.logger.warning(f"图片 {image_id} 的结果文件不存在")
                return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载图片 {image_id} 结果失败: {e}")
            return None

    def get_saved_image_ids(self) -> list:
        """
        获取已保存的所有图片ID列表

        Returns:
            图片ID列表
        """
        json_files = list(self.merged_dir.glob("*.json"))

        # 提取image_id（文件名）
        image_ids = [f.stem for f in json_files if f.stem != "merged_summary"]

        return sorted(image_ids)

    def generate_summary(self) -> Dict[str, Any]:
        """
        生成merged目录的摘要文件

        Returns:
            摘要信息
        """
        image_ids = self.get_saved_image_ids()

        summary = {
            'total_images': len(image_ids),
            'image_ids': image_ids[:100],  # 只记录前100个
            'output_dir': str(self.merged_dir),
            'tasks': self.config.get("distillation.tasks", []),
            'generated_timestamp': datetime.now().isoformat(),
        }

        # 保存摘要文件
        summary_path = self.merged_dir / "merged_summary.json"

        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"摘要文件已生成: {summary_path}")
            self.logger.info(f"总计已保存 {len(image_ids)} 个图片结果")

        except Exception as e:
            self.logger.error(f"生成摘要文件失败: {e}")

        return summary

    def _make_serializable(self, data: Any) -> Any:
        """
        递归转换数据为JSON可序列化格式。

        🔧 关键：将Tensor转换为list，避免JSON序列化错误
        🔧 过滤：移除top_k_indices和top_k_values，减少存储大小

        Args:
            data: 待转换的数据（可能包含Tensor）

        Returns:
            可序列化的数据
        """
        import torch

        if isinstance(data, torch.Tensor):
            # Tensor转换为list
            return data.tolist()

        elif isinstance(data, dict):
            # 🔧 过滤掉中间计算数据，只保留最终结果，减少JSON存储大小
            # topk数据：用于内部概率计算
            # logits相关：模型输出的原始logits数据
            # token_ids：答案token序列，用于置信度计算
            excluded_keys = {
                'top_k_indices', 'top_k_values', 'top_k', 'vocab_size',  # topk数据
                'logits', 'answer_token_ids',  # logits相关数据
            }
            return {
                k: self._make_serializable(v)
                for k, v in data.items()
                if k not in excluded_keys
            }

        elif isinstance(data, list):
            # 递归处理列表
            return [self._make_serializable(v) for v in data]

        elif isinstance(data, tuple):
            # 元组转换为列表
            return [self._make_serializable(v) for v in data]

        else:
            # 其他类型直接返回（int, float, str, bool, None等）
            return data

    def cleanup(self) -> None:
        """
        清理临时文件（如果需要）

        新逻辑：不再需要清理batch文件和archive
        """
        self.logger.info("无需清理临时文件（已使用单图片保存模式）")


# 向后兼容的方法（已弃用）
    def save_result(self, result: Dict[str, Any], output_path: str, validate: bool = True) -> bool:
        """已弃用：请使用 save_image_result"""
        self.logger.warning("save_result 已弃用，建议使用 save_image_result")
        return self.save_image_result(result)

    def save_batch(self, batch_results: Dict[str, Any], output_path: str) -> bool:
        """已弃用：请使用 save_batch_results"""
        self.logger.warning("save_batch 已弃用，建议使用 save_batch_results")
        stats = self.save_batch_results(batch_results)
        return stats['saved'] > 0

    def merge_all_results(self, batch_files=None) -> Dict[str, Any]:
        """已弃用：不再需要合并步骤"""
        self.logger.info("merge_all_results 已弃用：数据已直接保存为单图片文件")
        self.logger.info(f"请直接使用 {self.merged_dir} 目录中的数据")

        # 生成摘要
        return self.generate_summary()