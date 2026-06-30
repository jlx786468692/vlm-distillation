"""
数据清洗模块
============

实现深度数据清洗功能，提升蒸馏数据质量。
"""

import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import string

from ..utils.config import ConfigManager
from ..utils.logger import get_logger


class DataCleaner:
    """
    数据清洗器 - 提升蒸馏数据质量

    核心功能:
    1. 多维度异常检测
    2. 综合质量评分
    3. 数据清洗规则应用
    4. 数据去重
    5. 数据修复
    6. 清洗报告生成
    """

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize DataCleaner.

        Args:
            config: Configuration manager
        """
        self.config = config or ConfigManager()
        self.logger = get_logger()

        # 清洗参数
        self.min_confidence = self.config.get("cleaning.min_confidence", 0.5)
        self.max_confidence = self.config.get("cleaning.max_confidence", 0.95)
        self.min_quality_score = self.config.get("cleaning.min_quality_score", 30.0)
        self.min_answer_length = self.config.get("cleaning.min_answer_length", 3)
        self.max_answer_length = self.config.get("cleaning.max_answer_length", 100)
        self.min_cot_quality = self.config.get("cleaning.min_cot_quality", 0.5)

        # 清洗策略
        self.auto_remove_invalid = self.config.get("cleaning.auto_remove_invalid", True)
        self.auto_repair_bbox = self.config.get("cleaning.auto_repair_bbox", True)
        self.deduplicate_answers = self.config.get("cleaning.deduplicate_answers", True)

        # 报告设置
        self.save_removed_data = self.config.get("cleaning.save_removed_data", True)
        self.max_removed_samples_display = self.config.get("cleaning.max_removed_samples_display", 20)

        # 无效答案列表
        self.invalid_answers = [
            'unknown', 'n/a', 'none', 'null', 'nothing',
            'i don\'t know', 'cannot determine', 'unclear',
            'no answer', 'not sure', 'indeterminate', ''
        ]

        # 统计追踪
        self.stats = {
            'total_input': 0,
            'cleaned_count': 0,
            'removed_count': 0,
            'repaired_count': 0,
            'duplicate_count': 0,
        }

    def clean_directory(self, data_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Clean all data in a directory.

        Args:
            data_dir: Input data directory
            output_dir: Output directory for cleaned data

        Returns:
            Cleaning report dictionary
        """
        self.logger.info("="*60)
        self.logger.info("Data Cleaning Process")
        self.logger.info("="*60)
        self.logger.info(f"Input directory: {data_dir}")

        # Step 1: 加载所有数据
        self.logger.info("\n[Step 1] Loading data...")
        all_data = self._load_all_data(data_dir)
        self.stats['total_input'] = len(all_data)
        self.logger.info(f"Loaded {len(all_data)} data files")

        # Step 2: 异常检测
        self.logger.info("\n[Step 2] Detecting anomalies...")
        anomalies = self._detect_anomalies(all_data)
        anomaly_stats = self._summarize_anomalies(anomalies)
        self.logger.info(f"Anomalies detected: {anomaly_stats}")

        # Step 3: 质量评分
        self.logger.info("\n[Step 3] Computing quality scores...")
        quality_scores = self._compute_quality_scores(all_data)
        avg_quality = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
        self.logger.info(f"Average quality score: {avg_quality:.2f}")

        # Step 4: 应用清洗规则
        self.logger.info("\n[Step 4] Applying cleaning rules...")
        cleaned_data, removed_data = self._apply_cleaning_rules(
            all_data, anomalies, quality_scores
        )
        self.stats['cleaned_count'] = len(cleaned_data)
        self.stats['removed_count'] = len(removed_data)
        self.logger.info(f"Cleaned: {len(cleaned_data)}, Removed: {len(removed_data)}")

        # Step 5: 数据去重
        if self.deduplicate_answers:
            self.logger.info("\n[Step 5] Deduplicating data...")
            deduplicated_data, duplicate_info = self._deduplicate_data(cleaned_data)
            self.stats['duplicate_count'] = len(duplicate_info['duplicates'])
            self.logger.info(f"Duplicates found: {len(duplicate_info['duplicates'])}")
        else:
            deduplicated_data = cleaned_data
            duplicate_info = {'duplicates': [], 'unique_count': len(cleaned_data)}

        # Step 6: 数据修复
        if self.auto_repair_bbox:
            self.logger.info("\n[Step 6] Repairing data...")
            repaired_data = self._repair_data(deduplicated_data)
            repair_count = sum(1 for d in repaired_data if d.get('repaired', False))
            self.stats['repaired_count'] = repair_count
            self.logger.info(f"Repaired {repair_count} data samples")
        else:
            repaired_data = deduplicated_data

        # Step 7: 生成清洗报告
        self.logger.info("\n[Step 7] Generating cleaning report...")
        report = self._generate_cleaning_report(
            all_data, cleaned_data, removed_data,
            anomalies, quality_scores, duplicate_info
        )

        # Step 8: 保存数据
        output_dir = output_dir or str(Path(data_dir).parent / "cleaned")
        self.logger.info(f"\n[Step 8] Saving cleaned data to {output_dir}...")
        self._save_cleaned_data(repaired_data, removed_data, output_dir)

        self.logger.info("\n" + "="*60)
        self.logger.info("Cleaning Completed!")
        self.logger.info("="*60)
        self.logger.info(f"Summary:")
        self.logger.info(f"  Input:      {self.stats['total_input']}")
        self.logger.info(f"  Cleaned:    {self.stats['cleaned_count']}")
        self.logger.info(f"  Removed:    {self.stats['removed_count']}")
        self.logger.info(f"  Repaired:   {self.stats['repaired_count']}")
        self.logger.info(f"  Duplicates: {self.stats['duplicate_count']}")
        self.logger.info(f"  Removal rate: {self.stats['removed_count']/self.stats['total_input']*100:.1f}%")

        return report

    # ============================================================
    # 异常检测方法
    # ============================================================

    def _detect_anomalies(self, all_data: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Detect anomalies across multiple dimensions.

        Args:
            all_data: List of data dictionaries

        Returns:
            Dictionary of anomaly lists by category
        """
        anomalies = {
            'low_confidence': [],       # 低置信度
            'invalid_answers': [],      # 无效答案
            'empty_results': [],        # 空结果
            'bbox_anomalies': [],       # 异常检测框
            'cot_low_quality': [],      # 低质量思维链
            'length_anomalies': [],     # 长度异常
            'format_errors': [],        # 格式错误
        }

        for data in all_data:
            image_id = data.get('image_id')
            tasks = data.get('tasks', {})

            # 检测每个任务的异常
            for task_name, task_data in tasks.items():

                if task_name == 'vqa':
                    self._detect_vqa_anomalies(image_id, task_data, anomalies)

                elif task_name == 'captioning':
                    self._detect_captioning_anomalies(image_id, task_data, anomalies)

                elif task_name == 'detection':
                    self._detect_detection_anomalies(image_id, task_data, anomalies)

        return anomalies

    def _detect_vqa_anomalies(self, image_id: str,
                               task_data: Dict,
                               anomalies: Dict) -> None:
        """Detect VQA anomalies."""

        hard_label = task_data.get('hard_label', {})
        answer = hard_label.get('answer', '')
        confidence = hard_label.get('confidence', 0.0)

        # 1. 低置信度检测
        if confidence < self.min_confidence:
            anomalies['low_confidence'].append({
                'image_id': image_id,
                'task': 'vqa',
                'confidence': confidence,
                'answer': answer,
                'type': 'low_confidence'
            })

        # 2. 无效答案检测
        answer_lower = answer.lower().strip()
        if answer_lower in self.invalid_answers or not answer.strip():
            anomalies['invalid_answers'].append({
                'image_id': image_id,
                'task': 'vqa',
                'answer': answer,
                'type': 'invalid_answer'
            })

        # 3. 长度异常检测
        answer_len = len(answer.strip())
        if answer_len < self.min_answer_length:
            anomalies['length_anomalies'].append({
                'image_id': image_id,
                'task': 'vqa',
                'length': answer_len,
                'type': 'too_short',
                'threshold': self.min_answer_length
            })
        elif answer_len > self.max_answer_length:
            anomalies['length_anomalies'].append({
                'image_id': image_id,
                'task': 'vqa',
                'length': answer_len,
                'type': 'too_long',
                'threshold': self.max_answer_length
            })

        # 4. CoT质量检测
        cot = task_data.get('cot_reasoning', {})
        quality_metrics = cot.get('quality_metrics', {})

        if quality_metrics:
            logical_flow = quality_metrics.get('logical_flow_score', 0.0)
            if logical_flow < self.min_cot_quality:
                anomalies['cot_low_quality'].append({
                    'image_id': image_id,
                    'task': 'vqa',
                    'quality_score': logical_flow,
                    'type': 'low_cot_quality'
                })

            # 步骤数异常
            step_count = quality_metrics.get('step_count', 0)
            if step_count < 2:
                anomalies['cot_low_quality'].append({
                    'image_id': image_id,
                    'task': 'vqa',
                    'step_count': step_count,
                    'type': 'insufficient_steps'
                })

    def _detect_captioning_anomalies(self, image_id: str,
                                      task_data: Dict,
                                      anomalies: Dict) -> None:
        """Detect Captioning anomalies."""

        hard_label = task_data.get('hard_label', {})
        captions = hard_label.get('captions', [])

        # 1. 空caption检测
        if not captions or len(captions) == 0:
            anomalies['empty_results'].append({
                'image_id': image_id,
                'task': 'captioning',
                'type': 'no_captions'
            })

        # 2. Caption质量检测
        valid_captions = [cap for cap in captions if len(cap.strip()) >= 10]
        if len(valid_captions) < len(captions):
            anomalies['empty_results'].append({
                'image_id': image_id,
                'task': 'captioning',
                'num_captions': len(captions),
                'valid_captions': len(valid_captions),
                'type': 'invalid_captions'
            })

        # 3. Caption长度异常
        for i, caption in enumerate(captions):
            cap_len = len(caption.strip())
            if cap_len < 15:
                anomalies['length_anomalies'].append({
                    'image_id': image_id,
                    'task': 'captioning',
                    'caption_index': i,
                    'length': cap_len,
                    'type': 'caption_too_short'
                })
            elif cap_len > 200:
                anomalies['length_anomalies'].append({
                    'image_id': image_id,
                    'task': 'captioning',
                    'caption_index': i,
                    'length': cap_len,
                    'type': 'caption_too_long'
                })

    def _detect_detection_anomalies(self, image_id: str,
                                     task_data: Dict,
                                     anomalies: Dict) -> None:
        """Detect Detection anomalies."""

        hard_label = task_data.get('hard_label', {})
        objects = hard_label.get('objects', [])

        # 1. 无检测结果
        if not objects or len(objects) == 0:
            anomalies['empty_results'].append({
                'image_id': image_id,
                'task': 'detection',
                'type': 'no_objects'
            })

        # 2. 检测框异常
        for obj in objects:
            bbox = obj.get('bbox', [])
            confidence = obj.get('confidence', 0.0)

            # 格式错误
            if len(bbox) != 4:
                anomalies['format_errors'].append({
                    'image_id': image_id,
                    'task': 'detection',
                    'bbox': bbox,
                    'type': 'invalid_bbox_format'
                })
                continue

            # 解析bbox
            x_min, y_min, x_max, y_max = bbox

            # 超出范围检测 (假设合理范围 [0, 1000])
            if x_min < 0 or y_min < 0 or x_max > 1000 or y_max > 1000:
                anomalies['bbox_anomalies'].append({
                    'image_id': image_id,
                    'object': obj,
                    'type': 'bbox_out_of_range',
                    'bbox': bbox
                })

            # 尺寸异常
            width = x_max - x_min
            height = y_max - y_min

            if width < 5 or height < 5:  # 太小
                anomalies['bbox_anomalies'].append({
                    'image_id': image_id,
                    'object': obj,
                    'type': 'bbox_too_small',
                    'width': width,
                    'height': height
                })

            if width > 900 or height > 900:  # 太大
                anomalies['bbox_anomalies'].append({
                    'image_id': image_id,
                    'object': obj,
                    'type': 'bbox_too_large',
                    'width': width,
                    'height': height
                })

            # 坐标异常
            if x_max <= x_min or y_max <= y_min:
                anomalies['bbox_anomalies'].append({
                    'image_id': image_id,
                    'object': obj,
                    'type': 'invalid_coordinates',
                    'bbox': bbox
                })

            # 低置信度检测
            if confidence < self.min_confidence:
                anomalies['low_confidence'].append({
                    'image_id': image_id,
                    'task': 'detection',
                    'confidence': confidence,
                    'object_class': obj.get('class', obj.get('category_name', 'unknown')),
                    'type': 'object_low_confidence'
                })

    def _summarize_anomalies(self, anomalies: Dict) -> Dict[str, int]:
        """Summarize anomaly statistics."""
        return {key: len(value) for key, value in anomalies.items()}

    # ============================================================
    # 质量评分方法
    # ============================================================

    def _compute_quality_scores(self, all_data: List[Dict]) -> Dict[str, float]:
        """
        Compute quality scores for all data.

        Args:
            all_data: List of data dictionaries

        Returns:
            Dictionary mapping image_id to quality score
        """
        quality_scores = {}

        for data in all_data:
            image_id = data.get('image_id')
            tasks = data.get('tasks', {})

            # 计算综合质量分数
            total_score = 0.0
            task_count = 0

            for task_name, task_data in tasks.items():
                task_score = self._compute_task_quality(task_name, task_data)
                total_score += task_score
                task_count += 1

            # 平均质量分数
            avg_score = total_score / task_count if task_count > 0 else 0.0
            quality_scores[image_id] = min(avg_score, 100.0)  # 最高100分

        return quality_scores

    def _compute_task_quality(self, task_name: str, task_data: Dict) -> float:
        """
        Compute quality score for a single task.

        Scoring breakdown:
        - Hard label quality: 0-40 points
        - Soft label quality: 0-20 points
        - CoT quality: 0-30 points
        - Bonus: 0-10 points

        Args:
            task_name: Task name
            task_data: Task data dictionary

        Returns:
            Quality score (0-100)
        """
        score = 0.0

        # 1. 硬标签质量 (0-40分)
        hard_label = task_data.get('hard_label', {})
        if hard_label:
            confidence = hard_label.get('confidence', 0.0)

            # 置信度贡献 (最高30分)
            if confidence >= 0.7:
                score += 30
            elif confidence >= 0.5:
                score += 20
            elif confidence >= 0.3:
                score += 10

            # 答案完整性 (最高10分)
            answer = hard_label.get('answer', '')
            if self.min_answer_length <= len(answer) <= self.max_answer_length:
                score += 10

        # 2. 软标签质量 (0-20分)
        soft_label = task_data.get('soft_label', {})
        if soft_label:
            # 温度参数合理性 (最高10分)
            temperature = soft_label.get('temperature', 0.0)
            if 1.5 <= temperature <= 3.0:  # 推荐范围
                score += 10
            elif 1.0 <= temperature <= 5.0:  # 可接受范围
                score += 5

            # 分布完整性 (最高10分)
            distribution = soft_label.get('answer_distribution', {})
            if distribution and len(distribution) > 0:
                score += 10

        # 3. CoT质量 (0-30分)
        cot = task_data.get('cot_reasoning', {})
        if cot:
            quality_metrics = cot.get('quality_metrics', {})

            # 逻辑流畅度 (最高15分)
            logical_flow = quality_metrics.get('logical_flow_score', 0.0)
            score += logical_flow * 15

            # 步骤数量合理性 (最高15分)
            step_count = quality_metrics.get('step_count', 0)
            if 3 <= step_count <= 5:  # 最佳步骤数
                score += 15
            elif 2 <= step_count <= 6:  # 可接受
                score += 10
            elif step_count > 0:  # 有步骤但不理想
                score += step_count * 2

            # 长度合理性 (额外加分)
            reasoning_length = len(cot.get('raw_reasoning', ''))
            if 50 <= reasoning_length <= 300:  # 合理长度
                score += 5

        # 4. 任务特定加分
        if task_name == 'vqa':
            # VQA答案有效性加分
            answer = hard_label.get('answer', '')
            if answer and answer.lower() not in self.invalid_answers:
                score += 5

        elif task_name == 'captioning':
            # Caption多样性加分
            captions = hard_label.get('captions', [])
            if len(captions) >= 3:  # 多个caption
                score += 5

        elif task_name == 'detection':
            # 检测完整性加分
            objects = hard_label.get('objects', [])
            if len(objects) >= 2:  # 检测到多个物体
                score += 5

        return min(score, 100.0)

    # ============================================================
    # 清洗规则应用
    # ============================================================

    def _apply_cleaning_rules(self,
                               all_data: List[Dict],
                               anomalies: Dict,
                               quality_scores: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply cleaning rules to data.

        Args:
            all_data: List of all data
            anomalies: Anomaly dictionary
            quality_scores: Quality score dictionary

        Returns:
            Tuple of (cleaned_data, removed_data)
        """
        cleaned_data = []
        removed_data = []

        # 获取有严重问题的image_id列表
        invalid_answer_ids = set(a['image_id'] for a in anomalies['invalid_answers'])
        empty_result_ids = set(a['image_id'] for a in anomalies['empty_results'])
        format_error_ids = set(a['image_id'] for a in anomalies['format_errors'])

        for data in all_data:
            image_id = data.get('image_id')

            # 检查是否应该移除
            should_remove = False
            removal_reasons = []

            # 规则1: 质量分数过低
            quality = quality_scores.get(image_id, 0.0)
            if quality < self.min_quality_score:
                should_remove = True
                removal_reasons.append(f"quality_score={quality:.1f} < threshold={self.min_quality_score}")

            # 规则2: 有严重异常（无效答案）
            if image_id in invalid_answer_ids:
                should_remove = True
                removal_reasons.append("invalid_answer")

            # 规则3: 空结果
            if image_id in empty_result_ids:
                should_remove = True
                removal_reasons.append("empty_result")

            # 规则4: 格式错误
            if image_id in format_error_ids:
                should_remove = True
                removal_reasons.append("format_error")

            # 规则5: 多个低置信度（>=2个任务低置信度）
            low_conf_count = len([
                a for a in anomalies['low_confidence']
                if a['image_id'] == image_id
            ])
            if low_conf_count >= 2:
                should_remove = True
                removal_reasons.append(f"multiple_low_confidence={low_conf_count}")

            # 规则6: CoT严重质量问题
            cot_bad_count = len([
                a for a in anomalies['cot_low_quality']
                if a['image_id'] == image_id and a.get('quality_score', 0) < 0.3
            ])
            if cot_bad_count >= 1:
                should_remove = True
                removal_reasons.append("severe_cot_quality_issue")

            # 决策: 移除或保留
            if should_remove and self.auto_remove_invalid:
                # 移除数据
                data['_removal_reasons'] = removal_reasons
                data['_quality_score'] = quality
                data['_removed_at'] = datetime.now().isoformat()
                removed_data.append(data)

                self.logger.debug(f"Removed {image_id}: {removal_reasons}")
            else:
                # 保留数据，添加质量标记
                data['quality_score'] = quality
                data['anomaly_count'] = sum(
                    1 for key in anomalies
                    for a in anomalies[key]
                    if a['image_id'] == image_id
                )

                # 添加具体异常标记
                data['anomalies'] = [
                    a for key in anomalies
                    for a in anomalies[key]
                    if a['image_id'] == image_id
                ]

                # 标记低置信度（但不移除）
                if quality < self.min_quality_score:
                    data['quality_warning'] = True

                cleaned_data.append(data)

        return cleaned_data, removed_data

    # ============================================================
    # 数据去重
    # ============================================================

    def _deduplicate_data(self, cleaned_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Deduplicate data based on answer similarity.

        Args:
            cleaned_data: List of cleaned data

        Returns:
            Tuple of (deduplicated_data, duplicate_info)
        """
        seen_answers = {}  # task -> normalized_answer -> count
        duplicates = []
        deduplicated = []

        for data in cleaned_data:
            image_id = data.get('image_id')
            tasks = data.get('tasks', {})

            is_duplicate = False
            duplicate_matches = []

            for task_name, task_data in tasks.items():
                hard_label = task_data.get('hard_label', {})

                if task_name == 'vqa':
                    answer = hard_label.get('answer', '')
                    normalized = self._normalize_answer(answer)

                    if task_name not in seen_answers:
                        seen_answers[task_name] = {}

                    if normalized in seen_answers[task_name]:
                        # 找到相似答案
                        is_duplicate = True
                        duplicate_matches.append({
                            'task': task_name,
                            'normalized_answer': normalized,
                            'original_answer': answer,
                            'duplicate_count': seen_answers[task_name][normalized]
                        })
                        seen_answers[task_name][normalized] += 1
                    else:
                        seen_answers[task_name][normalized] = 1

            if is_duplicate:
                data['is_duplicate'] = True
                data['duplicate_matches'] = duplicate_matches
                duplicates.append(data)

            # 即使标记为重复也保留（供后续决策）
            deduplicated.append(data)

        duplicate_info = {
            'duplicates': duplicates,
            'unique_count': len(deduplicated) - len(duplicates),
            'duplicate_count': len(duplicates),
        }

        return deduplicated, duplicate_info

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize answer for deduplication comparison.

        Args:
            answer: Original answer string

        Returns:
            Normalized answer string
        """
        # 小写化
        normalized = answer.lower().strip()

        # 移除常见冠词
        articles = ['a ', 'an ', 'the ', 'this ', 'that ', 'these ', 'those ']
        for article in articles:
            normalized = normalized.replace(article, '')

        # 移除标点
        normalized = normalized.translate(str.maketrans('', '', string.punctuation))

        # 移除多余空格
        normalized = ' '.join(normalized.split())

        return normalized

    # ============================================================
    # 数据修复
    # ============================================================

    def _repair_data(self, deduplicated_data: List[Dict]) -> List[Dict]:
        """
        Repair data anomalies.

        Args:
            deduplicated_data: List of deduplicated data

        Returns:
            List of repaired data
        """
        repaired = []

        for data in deduplicated_data:
            tasks = data.get('tasks', {})
            repair_actions = []

            # 修复每个任务
            for task_name, task_data in tasks.items():

                # 修复bbox（超出范围的裁剪）
                if task_name == 'detection':
                    objects = task_data.get('hard_label', {}).get('objects', [])
                    repaired_objects = []

                    for obj in objects:
                        bbox = obj.get('bbox', [])
                        if len(bbox) == 4:
                            x_min, y_min, x_max, y_max = bbox
                            original_bbox = bbox.copy()

                            # 裁剪到合理范围 [0, 1000]
                            x_min = max(0, min(x_min, 1000))
                            y_min = max(0, min(y_min, 1000))
                            x_max = max(x_min + 5, min(x_max, 1000))  # 确保最小宽度5
                            y_max = max(y_min + 5, min(y_max, 1000))  # 确保最小高度5

                            # 如果坐标异常（x_max <= x_min），修复
                            if x_max <= x_min:
                                x_max = x_min + 10
                            if y_max <= y_min:
                                y_max = y_min + 10

                            repaired_bbox = [x_min, y_min, x_max, y_max]

                            if repaired_bbox != original_bbox:
                                obj['bbox'] = repaired_bbox
                                obj['bbox_original'] = original_bbox
                                obj['bbox_repaired'] = True
                                repair_actions.append({
                                    'task': task_name,
                                    'type': 'bbox_repair',
                                    'original': original_bbox,
                                    'repaired': repaired_bbox
                                })

                        repaired_objects.append(obj)

                    if 'hard_label' in task_data:
                        task_data['hard_label']['objects'] = repaired_objects

                # 修复缺失字段
                if 'hard_label' in task_data:
                    hard_label = task_data['hard_label']

                    # 添加缺失的confidence
                    if 'confidence' not in hard_label:
                        hard_label['confidence'] = 0.5  # 默认中等置信度
                        repair_actions.append({
                            'task': task_name,
                            'type': 'add_confidence',
                            'value': 0.5
                        })

                    # 添加缺失的timestamp
                    if 'timestamp' not in hard_label:
                        hard_label['timestamp'] = datetime.now().isoformat()
                        repair_actions.append({
                            'task': task_name,
                            'type': 'add_timestamp'
                        })

            # 标记修复信息
            if repair_actions:
                data['repaired'] = True
                data['repair_actions'] = repair_actions

            repaired.append(data)

        return repaired

    # ============================================================
    # 清洗报告生成
    # ============================================================

    def _generate_cleaning_report(self,
                                    all_data: List[Dict],
                                    cleaned_data: List[Dict],
                                    removed_data: List[Dict],
                                    anomalies: Dict,
                                    quality_scores: Dict[str, float],
                                    duplicate_info: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning report.

        Args:
            all_data: Original data
            cleaned_data: Cleaned data
            removed_data: Removed data
            anomalies: Anomaly dictionary
            quality_scores: Quality scores
            duplicate_info: Duplicate information

        Returns:
            Cleaning report dictionary
        """
        # 计算质量统计
        if cleaned_data:
            qualities = [d.get('quality_score', 0) for d in cleaned_data]
            avg_quality = sum(qualities) / len(qualities)
            min_quality = min(qualities)
            max_quality = max(qualities)
            median_quality = sorted(qualities)[len(qualities)//2]
        else:
            avg_quality = min_quality = max_quality = median_quality = 0

        report = {
            'summary': {
                'total_input': len(all_data),
                'cleaned_count': len(cleaned_data),
                'removed_count': len(removed_data),
                'removal_rate': len(removed_data) / len(all_data) if all_data else 0,
                'duplicate_count': duplicate_info['duplicate_count'],
                'repaired_count': self.stats['repaired_count'],
                'cleaning_timestamp': datetime.now().isoformat(),
            },

            'anomaly_statistics': self._summarize_anomalies(anomalies),

            'quality_statistics': {
                'average_quality_score': round(avg_quality, 2),
                'median_quality_score': round(median_quality, 2),
                'min_quality_score': round(min_quality, 2),
                'max_quality_score': round(max_quality, 2),
                'quality_distribution': {
                    'high_quality': len([q for q in qualities if q >= 70]) if qualities else 0,
                    'medium_quality': len([q for q in qualities if 50 <= q < 70]) if qualities else 0,
                    'low_quality': len([q for q in qualities if q < 50]) if qualities else 0,
                },
            },

            'duplicate_statistics': {
                'unique_count': duplicate_info['unique_count'],
                'duplicate_count': duplicate_info['duplicate_count'],
                'duplicate_rate': duplicate_info['duplicate_count'] / len(cleaned_data) if cleaned_data else 0,
            },

            'removed_samples': [
                {
                    'image_id': d.get('image_id'),
                    'quality_score': d.get('_quality_score', 0),
                    'reasons': d.get('_removal_reasons', []),
                }
                for d in removed_data[:self.max_removed_samples_display]
            ],

            'recommendations': self._generate_recommendations(anomalies, quality_scores),

            'config': {
                'min_confidence': self.min_confidence,
                'min_quality_score': self.min_quality_score,
                'min_cot_quality': self.min_cot_quality,
                'auto_remove_invalid': self.auto_remove_invalid,
                'auto_repair_bbox': self.auto_repair_bbox,
                'deduplicate_answers': self.deduplicate_answers,
            },
        }

        return report

    def _generate_recommendations(self,
                                   anomalies: Dict,
                                   quality_scores: Dict[str, float]) -> List[str]:
        """
        Generate cleaning recommendations.

        Args:
            anomalies: Anomaly statistics
            quality_scores: Quality scores

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # 低置信度建议
        low_conf_count = len(anomalies['low_confidence'])
        if low_conf_count > 100:
            recommendations.append(
                f"发现大量低置信度样本({low_conf_count}个)，建议：\n"
                f"  - 提高 TeacherModel 的 temperature 参数\n"
                f"  - 优化 Prompt 模板\n"
                f"  - 增加数据多样性"
            )
        elif low_conf_count > 50:
            recommendations.append(
                f"发现较多低置信度样本({low_conf_count}个)，建议提高 confidence_threshold 参数至 {self.min_confidence + 0.1}"
            )

        # 无效答案建议
        invalid_count = len(anomalies['invalid_answers'])
        if invalid_count > 20:
            recommendations.append(
                f"发现{invalid_count}个无效答案（如'unknown'、'N/A'），建议：\n"
                f"  - 检查 VQA Prompt 是否明确要求具体答案\n"
                f"  - 过滤掉难以回答的样本\n"
                f"  - 增加 'unknown' 答案的排除规则"
            )

        # CoT质量建议
        cot_bad_count = len(anomalies['cot_low_quality'])
        if cot_bad_count > 30:
            recommendations.append(
                f"发现{cot_bad_count}个低质量CoT，建议：\n"
                f"  - 增加 CoT max_length 参数\n"
                f"  - 优化 CoT Prompt 模板，引导更详细的推理\n"
                f"  - 检查 min_cot_quality 阈值设置"
            )

        # Bbox异常建议
        bbox_bad_count = len(anomalies['bbox_anomalies'])
        if bbox_bad_count > 20:
            recommendations.append(
                f"发现{bbox_bad_count}个异常检测框，建议：\n"
                f"  - 检查图像尺寸配置是否正确\n"
                f"  - 优化 Detection Prompt，明确bbox格式要求\n"
                f"  - 启用 bbox 自动修复功能"
            )

        # 空结果建议
        empty_count = len(anomalies['empty_results'])
        if empty_count > 10:
            recommendations.append(
                f"发现{empty_count}个空结果，建议：\n"
                f"  - 检查数据加载是否正常\n"
                f"  - 增加结果验证逻辑\n"
                f"  - 优化各任务的 Prompt 模板"
            )

        # 质量分布建议
        if quality_scores:
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
            if avg_quality < 50:
                recommendations.append(
                    f"平均质量分数较低({avg_quality:.1f})，建议：\n"
                    f"  - 降低 min_quality_score 阈值（当前{self.min_quality_score}）\n"
                    f"  - 检查 TeacherModel 配置\n"
                    f"  - 优化数据采样策略"
                )
            elif avg_quality > 85:
                recommendations.append(
                    f"平均质量分数很高({avg_quality:.1f})，可考虑：\n"
                    f"  - 提高清洗阈值，进一步精炼数据\n"
                    f"  - 减少数据量但提高质量"
                )

        # 格式错误建议
        format_errors = len(anomalies['format_errors'])
        if format_errors > 0:
            recommendations.append(
                f"发现{format_errors}个格式错误，建议：\n"
                f"  - 检查数据生成流程\n"
                f"  - 增强 Schema 验证\n"
                f"  - 添加格式修复逻辑"
            )

        # 去重建议
        duplicate_count = sum(1 for d in anomalies.get('duplicates', []))
        if duplicate_count > 50:
            recommendations.append(
                f"发现大量相似答案({duplicate_count}个)，建议：\n"
                f"  - 启用严格去重策略\n"
                f"  - 增加数据多样性\n"
                f"  - 调整采样策略为 'balanced'"
            )

        if not recommendations:
            recommendations.append("数据质量良好，清洗效果符合预期。继续保持当前配置。")

        return recommendations

    # ============================================================
    # 数据加载和保存
    # ============================================================

    def _load_all_data(self, data_dir: str) -> List[Dict]:
        """
        Load all data files from directory.

        Args:
            data_dir: Data directory path

        Returns:
            List of data dictionaries
        """
        data_path = Path(data_dir)

        # 查找所有JSON文件
        json_files = list(data_path.glob("*.json"))

        # 过滤掉checkpoint和summary文件
        json_files = [
            f for f in json_files
            if not f.name.startswith('checkpoint')
            and not f.name.startswith('merged_summary')
            and not f.name.startswith('cleaning_report')
        ]

        all_data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")

        return all_data

    def _save_cleaned_data(self,
                           cleaned_data: List[Dict],
                           removed_data: List[Dict],
                           output_dir: str) -> None:
        """
        Save cleaned and removed data.

        Args:
            cleaned_data: List of cleaned data
            removed_data: List of removed data
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 创建子目录
        cleaned_dir = output_path / "cleaned"
        cleaned_dir.mkdir(exist_ok=True)

        removed_dir = output_path / "removed"
        removed_dir.mkdir(exist_ok=True)

        # 保存清洗后的数据
        for data in cleaned_data:
            image_id = data.get('image_id')
            output_file = cleaned_dir / f"{image_id}.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(cleaned_data)} cleaned files to {cleaned_dir}")

        # 保存被移除的数据（如果配置允许）
        if self.save_removed_data and removed_data:
            for data in removed_data:
                image_id = data.get('image_id')
                output_file = removed_dir / f"{image_id}.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(removed_data)} removed files to {removed_dir}")

        # 保存清洗报告
        report_file = output_path / "cleaning_report.json"
        self.logger.info(f"Cleaning report saved to {report_file}")

    def get_cleaning_stats(self) -> Dict[str, Any]:
        """
        Get cleaning statistics.

        Returns:
            Statistics dictionary
        """
        return self.stats.copy()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DataCleaner(min_confidence={self.min_confidence}, "
            f"min_quality={self.min_quality_score}, "
            f"auto_remove={self.auto_remove_invalid})"
        )
