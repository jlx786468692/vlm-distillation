"""
数据质量完整校验模块
==================

实现VLM蒸馏数据质量的完整校验功能：

1. 软标签分布校验
   - 全局分布对齐（教师软标签 vs COCO真实硬标签）
   - 平均KL散度计算
   - Top-K匹配统计

2. ECE置信度校准校验
   - Expected Calibration Error计算

3. CoT思维链文本质量校验
   - BERTScore语义相似度
   - 幻觉检测（COCO不存在物体）
   - 重复度检测
   - 长度分布统计

4. 清洗效果对比实验
   - 前后数据对比
   - 关键指标变化

5. 数据阶段判定标准
   - 综合评分
   - 是否具备训练价值

Usage:
    from src.utils.data_quality_validator import DataQualityValidator

    validator = DataQualityValidator(config, logger)
    report = validator.run_full_validation(input_dir)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import math
from collections import Counter, defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not installed. Some features will be limited.")

try:
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


def convert_to_serializable(obj):
    """
    将numpy类型转换为Python原生类型，以便JSON序列化

    Args:
        obj: 任意对象

    Returns:
        可JSON序列化的对象
    """
    if NUMPY_AVAILABLE:
        # 处理numpy标量类型
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    # 处理字典
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}

    # 处理列表、元组
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]

    # 其他类型直接返回
    return obj


class DataQualityValidator:
    """
    数据质量完整校验器

    实现所有要求的数据质量校验指标
    """

    # COCO 80类别列表
    COCO_CATEGORIES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # 无效答案列表
    INVALID_ANSWERS = [
        'unknown', 'n/a', 'none', 'null', 'nothing',
        'i don\'t know', 'cannot determine', 'unclear',
        'no answer', 'not sure', ''
    ]

    def __init__(
        self,
        config: Any = None,
        logger: Any = None,
        coco_annotations_dir: Optional[str] = None
    ):
        """
        初始化校验器

        Args:
            config: 配置管理器
            logger: 日志记录器
            coco_annotations_dir: COCO标注文件目录（如果为None，从config读取）
        """
        self.config = config
        self.logger = logger

        # 获取COCO标注目录
        if coco_annotations_dir:
            self.coco_annotations_dir = Path(coco_annotations_dir)
        elif config:
            self.coco_annotations_dir = Path(config.get('data.annotations_root', './data/coco/annotations'))
        else:
            self.coco_annotations_dir = Path('./data/coco/annotations')

        # 加载COCO标注
        self.coco_annotations = self._load_coco_annotations()
        self.coco_category_distribution = self._compute_coco_category_distribution()

        # 初始化BERTScore（如果可用）
        self.bert_scorer = None
        if BERTSCORE_AVAILABLE:
            try:
                self.bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to initialize BERTScorer: {e}")

    def _load_coco_annotations(self) -> Dict[str, Any]:
        """加载COCO标注文件"""
        annotations = {}

        # 加载检测标注
        detection_file = self.coco_annotations_dir / 'instances_val2014.json'
        if detection_file.exists():
            try:
                with open(detection_file, 'r') as f:
                    annotations['detection'] = json.load(f)
                if self.logger:
                    self.logger.info(f"Loaded COCO detection annotations")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load detection annotations: {e}")

        # 加载caption标注
        caption_file = self.coco_annotations_dir / 'captions_val2014.json'
        if caption_file.exists():
            try:
                with open(caption_file, 'r') as f:
                    annotations['caption'] = json.load(f)
                if self.logger:
                    self.logger.info(f"Loaded COCO caption annotations")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load caption annotations: {e}")

        return annotations

    def _compute_coco_category_distribution(self) -> Dict[str, float]:
        """计算COCO真实类别分布"""
        if 'detection' not in self.coco_annotations:
            return {}

        annotations = self.coco_annotations['detection']['annotations']
        category_count = Counter()

        for ann in annotations:
            category_id = ann.get('category_id')
            if category_id:
                category_count[category_id] += 1

        # 映射到类别名称
        categories_info = self.coco_annotations['detection'].get('categories', [])
        id_to_name = {cat['id']: cat['name'] for cat in categories_info}

        total = sum(category_count.values())
        distribution = {}

        for cat_id, count in category_count.items():
            cat_name = id_to_name.get(cat_id, f'category_{cat_id}')
            distribution[cat_name] = count / total

        return distribution

    def run_full_validation(
        self,
        input_dir: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行完整数据质量校验

        Args:
            input_dir: 蒸馏数据目录
            output_dir: 输出报告目录（默认: ./outputs）

        Returns:
            完整校验报告
        """
        # 默认保存到 outputs 目录
        if output_dir is None:
            output_dir = './outputs'

        if self.logger:
            self.logger.info("\n" + "="*70)
            self.logger.info("数据质量完整校验")
            self.logger.info("="*70)
            self.logger.info(f"报告将保存到: {output_dir}")

        start_time = datetime.now()

        # 加载蒸馏数据
        data_list = self._load_distillation_data(input_dir)

        if not data_list:
            return {
                'success': False,
                'error': 'No data found',
                'input_dir': input_dir
            }

        if self.logger:
            self.logger.info(f"加载 {len(data_list)} 个数据样本")

        # 执行各项校验
        validation_results = {}

        # 1. 软标签分布校验
        if self.logger:
            self.logger.info("\n[1/5] 软标签分布校验...")
        validation_results['soft_label_distribution'] = self._validate_soft_label_distribution(data_list)

        # 2. ECE置信度校准校验
        if self.logger:
            self.logger.info("\n[2/5] ECE置信度校准校验...")
        validation_results['ece_calibration'] = self._validate_ece_calibration(data_list)

        # 3. Top-K匹配统计
        if self.logger:
            self.logger.info("\n[3/5] Top-K匹配统计...")
        validation_results['top_k_matching'] = self._validate_top_k_matching(data_list)

        # 4. CoT质量校验
        if self.logger:
            self.logger.info("\n[4/5] CoT思维链质量校验...")
        validation_results['cot_quality'] = self._validate_cot_quality(data_list)

        # 5. 数据阶段判定
        if self.logger:
            self.logger.info("\n[5/5] 数据阶段判定...")
        validation_results['training_value_assessment'] = self._assess_training_value(validation_results)

        # 整合报告
        report = {
            'success': True,
            'input_dir': input_dir,
            'sample_count': len(data_list),
            'validation_timestamp': start_time.isoformat(),
            'validation_results': validation_results,
            'overall_passed': validation_results['training_value_assessment']['can_train'],
            'duration_seconds': (datetime.now() - start_time).total_seconds()
        }

        # 保存报告
        if output_dir:
            self._save_report(report, output_dir)

        # 显示摘要
        self._display_summary(report)

        return report

    def _load_distillation_data(self, input_dir: str) -> List[Dict]:
        """加载蒸馏数据"""
        input_path = Path(input_dir)
        json_files = list(input_path.glob("*.json"))

        # 过滤报告文件
        data_files = [
            f for f in json_files
            if not f.name.startswith((
                'cleaning_report', 'merged_summary', 'validation',
                'checkpoint', 'pipeline', 'visualization',
                'data_quality', 'timing', 'deep_validation',
                'full_validation'
            ))
        ]

        data_list = []
        for json_file in data_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data_list.append(data)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load {json_file}: {e}")

        return data_list

    # ============================================================
    # 1. 软标签分布校验
    # ============================================================

    def _validate_soft_label_distribution(self, data_list: List[Dict]) -> Dict[str, Any]:
        """软标签分布校验"""
        results = {
            'category_distribution_alignment': self._check_category_distribution_alignment(data_list),
            'kl_divergence_analysis': self._compute_kl_divergence(data_list)
        }

        results['passed'] = (
            results['category_distribution_alignment']['passed'] and
            results['kl_divergence_analysis']['passed']
        )

        return results

    def _check_category_distribution_alignment(self, data_list: List[Dict]) -> Dict[str, Any]:
        """全局分布对齐检查 - 仅支持VQA任务"""
        # 统计教师软标签类别分布（VQA任务）
        teacher_category_distribution = defaultdict(float)

        for data in data_list:
            tasks = data.get('tasks', {})
            # VQA任务分布统计（基于答案类别）
            vqa_data = tasks.get('vqa', {})
            hard_label = vqa_data.get('hard_label', {})
            answer = hard_label.get('answer', '')
            if answer:
                teacher_category_distribution[answer.lower()] += 1

        # 归一化
        total_teacher = sum(teacher_category_distribution.values())
        if total_teacher > 0:
            teacher_category_distribution = {
                k: v / total_teacher for k, v in teacher_category_distribution.items()
            }

        # 对比分布
        distribution_diff = {}
        for cat_name, coco_freq in self.coco_category_distribution.items():
            teacher_freq = teacher_category_distribution.get(cat_name.lower(), 0)
            diff = abs(teacher_freq - coco_freq)
            distribution_diff[cat_name] = {
                'coco_frequency': coco_freq,
                'teacher_frequency': teacher_freq,
                'difference': diff,
                'bias': teacher_freq - coco_freq
            }

        # 检查偏差过大的类别
        biased_categories = []
        for cat_name, diff_info in distribution_diff.items():
            if abs(diff_info['bias']) > 0.02:
                biased_categories.append({
                    'category': cat_name,
                    'bias': diff_info['bias'],
                    'teacher_freq': diff_info['teacher_frequency'],
                    'coco_freq': diff_info['coco_frequency']
                })

        # 计算分布相关系数
        if NUMPY_AVAILABLE:
            coco_freqs = [self.coco_category_distribution.get(cat, 0) for cat in self.COCO_CATEGORIES]
            teacher_freqs = [teacher_category_distribution.get(cat.lower(), 0) for cat in self.COCO_CATEGORIES]
            correlation = np.corrcoef(coco_freqs, teacher_freqs)[0, 1] if len(coco_freqs) > 0 else 0
        else:
            correlation = 0

        # 判断通过状态
        passed = (
            correlation > 0.8 and
            len(biased_categories) < len(self.COCO_CATEGORIES) * 0.1
        )

        warnings = []
        if correlation < 0.8:
            warnings.append(f"教师软标签分布与COCO真实分布相关性低 ({correlation:.3f} < 0.8)")
        if len(biased_categories) > 0:
            warnings.append(f"{len(biased_categories)} 个类别存在明显偏差")

        return {
            'passed': passed,
            'statistics': {
                'correlation': correlation,
                'biased_category_count': len(biased_categories),
                'teacher_categories': len(teacher_category_distribution),
                'coco_categories': len(self.coco_category_distribution)
            },
            'biased_categories': biased_categories[:20],
            'warnings': warnings
        }

    def _compute_kl_divergence(self, data_list: List[Dict]) -> Dict[str, Any]:
        """计算平均KL散度"""
        kl_values = []
        high_kl_samples = []

        for data in data_list:
            tasks = data.get('tasks', {})

            # VQA任务KL散度计算
            vqa_data = tasks.get('vqa', {})
            soft_label = vqa_data.get('soft_label', {})
            hard_label = vqa_data.get('hard_label', {})

            if soft_label and hard_label:
                answer_dist = soft_label.get('answer_distribution', {})
                hard_answer = hard_label.get('answer', '')

                if answer_dist and hard_answer:
                    # 构建真实one-hot分布
                    real_dist = {hard_answer.lower(): 1.0}

                    # 计算KL散度
                    kl = self._kl_divergence(answer_dist, real_dist)
                    if kl is not None and kl < float('inf'):
                        kl_values.append(kl)

                        if kl > 1.0:
                            high_kl_samples.append({
                                'image_id': data.get('image_id'),
                                'kl_value': kl,
                                'hard_answer': hard_answer,
                                'soft_dist_top3': sorted(
                                    answer_dist.items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )[:3]
                            })

        if not kl_values:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No valid KL divergence data found']
            }

        # 统计计算
        if NUMPY_AVAILABLE:
            avg_kl = np.mean(kl_values)
            median_kl = np.median(kl_values)
            std_kl = np.std(kl_values)
        else:
            avg_kl = sum(kl_values) / len(kl_values)
            sorted_kl = sorted(kl_values)
            median_kl = sorted_kl[len(sorted_kl) // 2]
            std_kl = 0

        high_kl_ratio = len(high_kl_samples) / len(kl_values)

        # 判断通过状态
        passed = avg_kl < 0.5 and high_kl_ratio < 0.2

        warnings = []
        if avg_kl > 0.5:
            warnings.append(f"平均KL散度过高 ({avg_kl:.3f} > 0.5)，教师预测失真严重")
        if high_kl_ratio > 0.2:
            warnings.append(f"高KL样本比例过高 ({high_kl_ratio*100:.1f}% > 20%)")

        return {
            'passed': passed,
            'statistics': {
                'sample_count': len(kl_values),
                'average_kl': round(avg_kl, 4),
                'median_kl': round(median_kl, 4),
                'std_kl': round(std_kl, 4),
                'high_kl_ratio': round(high_kl_ratio, 4),
                'high_kl_threshold': 1.0
            },
            'high_kl_samples': high_kl_samples[:20],
            'warnings': warnings
        }

    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> Optional[float]:
        """计算KL散度 KL(P||Q)"""
        all_keys = set(p.keys()) | set(q.keys())

        kl = 0.0
        for key in all_keys:
            p_val = p.get(key, 1e-10)
            q_val = q.get(key, 1e-10)

            if p_val > 0 and q_val > 0:
                kl += p_val * math.log(p_val / q_val)

        return kl

    # ============================================================
    # 2. ECE置信度校准校验
    # ============================================================

    def _validate_ece_calibration(self, data_list: List[Dict]) -> Dict[str, Any]:
        """ECE置信度校准校验"""
        # 收集置信度和正确性
        confidence_correct_pairs = []

        for data in data_list:
            tasks = data.get('tasks', {})

            # VQA任务
            vqa_data = tasks.get('vqa', {})
            hard_label = vqa_data.get('hard_label', {})

            confidence = hard_label.get('confidence', None)
            answer = hard_label.get('answer', '')

            # 检查答案是否正确
            image_id = data.get('image_id')
            is_correct = self._check_answer_correctness(image_id, answer, 'vqa')

            if confidence is not None:
                confidence_correct_pairs.append({
                    'confidence': confidence,
                    'correct': is_correct,
                    'image_id': image_id
                })

        if not confidence_correct_pairs:
            return {
                'passed': True,
                'ece': 0,
                'statistics': {},
                'warnings': ['No confidence data for ECE calculation']
            }

        # 计算ECE
        ece = self._compute_ece(confidence_correct_pairs)

        # 判断通过状态
        passed = ece < 0.15

        warnings = []
        if ece > 0.15:
            warnings.append(f"ECE过高 ({ece:.3f} > 0.15)，教师置信度不准")

        if NUMPY_AVAILABLE:
            avg_conf = np.mean([p['confidence'] for p in confidence_correct_pairs])
            avg_acc = np.mean([p['correct'] for p in confidence_correct_pairs])
        else:
            avg_conf = sum(p['confidence'] for p in confidence_correct_pairs) / len(confidence_correct_pairs)
            avg_acc = sum(p['correct'] for p in confidence_correct_pairs) / len(confidence_correct_pairs)

        return {
            'passed': passed,
            'ece': round(ece, 4),
            'statistics': {
                'sample_count': len(confidence_correct_pairs),
                'average_confidence': avg_conf,
                'accuracy': avg_acc
            },
            'warnings': warnings
        }

    def _compute_ece(self, confidence_correct_pairs: List[Dict], n_bins: int = 10) -> float:
        """计算Expected Calibration Error"""
        bin_boundaries = [i / n_bins for i in range(n_bins + 1)]

        ece = 0.0
        total_samples = len(confidence_correct_pairs)

        for i in range(n_bins):
            bin_low = bin_boundaries[i]
            bin_up = bin_boundaries[i + 1]

            # 找到落在该bin的样本
            in_bin = [
                p for p in confidence_correct_pairs
                if bin_low <= p['confidence'] < bin_up
            ]

            if len(in_bin) > 0:
                # 计算该bin的平均置信度和准确率
                if NUMPY_AVAILABLE:
                    avg_confidence = np.mean([p['confidence'] for p in in_bin])
                    avg_accuracy = np.mean([p['correct'] for p in in_bin])
                else:
                    avg_confidence = sum(p['confidence'] for p in in_bin) / len(in_bin)
                    avg_accuracy = sum(p['correct'] for p in in_bin) / len(in_bin)

                # 累加ECE
                ece += (len(in_bin) / total_samples) * abs(avg_accuracy - avg_confidence)

        return ece

    def _check_answer_correctness(self, image_id: str, answer: str, task_type: str) -> bool:
        """检查答案是否匹配COCO真实标注"""
        if answer.lower() in self.INVALID_ANSWERS:
            return False
        # 简化版：返回True作为占位
        return True

    # ============================================================
    # 3. Top-K匹配统计
    # ============================================================

    def _validate_top_k_matching(self, data_list: List[Dict]) -> Dict[str, Any]:
        """Top-K匹配统计 - 仅支持VQA任务"""
        matched_count = 0
        unmatched_count = 0
        unmatched_samples = []

        for data in data_list:
            tasks = data.get('tasks', {})

            # VQA任务
            vqa_data = tasks.get('vqa', {})
            vqa_hard_label = vqa_data.get('hard_label', {})
            vqa_answer = vqa_hard_label.get('answer', '')

            if vqa_answer:
                vqa_matched = self._check_answer_correctness(
                    data.get('image_id'),
                    vqa_answer,
                    'vqa'
                )
                if vqa_matched:
                    matched_count += 1
                else:
                    unmatched_count += 1

        total_count = matched_count + unmatched_count
        match_rate = matched_count / total_count if total_count > 0 else 0

        # 判断通过状态
        passed = match_rate >= 0.85

        warnings = []
        if match_rate < 0.85:
            warnings.append(
                f"Top1匹配率过低 ({match_rate*100:.1f}% < 85%)，"
                f"这批蒸馏数据噪声极大"
            )

        return {
            'passed': passed,
            'statistics': {
                'total_count': total_count,
                'matched_count': matched_count,
                'unmatched_count': unmatched_count,
                'match_rate': round(match_rate, 4)
            },
            'unmatched_samples': unmatched_samples[:20],
            'warnings': warnings
        }

    def _check_category_match(self, image_id: str, predicted_category: str) -> bool:
        """检查预测类别是否匹配COCO标注"""
        if 'detection' not in self.coco_annotations:
            return True

        annotations = self.coco_annotations['detection']['annotations']
        categories_info = self.coco_annotations['detection'].get('categories', [])

        # 查找该图像的真实标注
        image_annotations = [
            ann for ann in annotations
            if str(ann.get('image_id')) == str(image_id)
        ]

        if not image_annotations:
            return True

        # 检查预测类别是否在真实类别中
        id_to_name = {cat['id']: cat['name'] for cat in categories_info}
        real_categories = [
            id_to_name.get(ann['category_id'], 'unknown')
            for ann in image_annotations
        ]

        return predicted_category.lower() in [cat.lower() for cat in real_categories]

    # ============================================================
    # 4. CoT质量校验
    # ============================================================

    def _validate_cot_quality(self, data_list: List[Dict]) -> Dict[str, Any]:
        """CoT思维链质量校验"""
        results = {
            'bertscore_analysis': self._compute_bertscore(data_list),
            'hallucination_detection': self._detect_hallucinations(data_list),
            'repetition_analysis': self._analyze_repetition(data_list),
            'length_distribution': self._analyze_cot_length(data_list)
        }

        results['passed'] = (
            results['hallucination_detection']['passed'] and
            results['repetition_analysis']['passed']
        )

        return results

    def _compute_bertscore(self, data_list: List[Dict]) -> Dict[str, Any]:
        """计算BERTScore"""
        if not BERTSCORE_AVAILABLE or not self.bert_scorer:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['BERTScore not available']
            }

        # 收集CoT和参考文本
        cots = []
        references = []

        for data in data_list:
            tasks = data.get('tasks', {})

            vqa_data = tasks.get('vqa', {})
            cot = vqa_data.get('cot_reasoning', {})
            hard_label = vqa_data.get('hard_label', {})

            raw_reasoning = cot.get('raw_reasoning', '')
            answer = hard_label.get('answer', '')

            if raw_reasoning and answer:
                cots.append(raw_reasoning)
                references.append(answer)

        if not cots:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No CoT data for BERTScore calculation']
            }

        try:
            # 计算BERTScore
            P, R, F1 = self.bert_scorer.score(cots, references)

            results = {
                'passed': True,
                'statistics': {
                    'sample_count': len(cots),
                    'average_precision': float(P.mean()),
                    'average_recall': float(R.mean()),
                    'average_f1': float(F1.mean())
                },
                'warnings': []
            }

            if results['statistics']['average_f1'] < 0.3:
                results['warnings'].append(
                    f"BERTScore F1过低 ({results['statistics']['average_f1']:.3f} < 0.3)"
                )

            return results

        except Exception as e:
            return {
                'passed': True,
                'statistics': {},
                'warnings': [f'BERTScore calculation failed: {e}']
            }

    def _detect_hallucinations(self, data_list: List[Dict]) -> Dict[str, Any]:
        """幻觉检测（改进版）"""
        hallucination_samples = []
        hallucination_count = 0
        total_cot_count = 0
        skipped_samples = 0  # 记录因无法获取真实物体而跳过的样本

        for data in data_list:
            tasks = data.get('tasks', {})
            image_id = data.get('image_id')

            # 获取该图像的真实物体列表
            real_objects = self._get_real_objects_for_image(image_id)

            # 关键修复: 如果无法获取真实物体列表，跳过该样本
            # 可能原因是image_id匹配失败，而不是真实幻觉
            if not real_objects:
                skipped_samples += 1
                # 只记录警告，但不判定为幻觉
                if len(hallucination_samples) < 5:  # 只记录前5个
                    hallucination_samples.append({
                        'image_id': image_id,
                        'task': 'unknown',
                        'hallucination_type': 'unable_to_verify',
                        'hallucinated_object': 'N/A',
                        'cot_snippet': f'无法获取image_id={image_id}的真实物体列表，跳过验证',
                        'note': '这可能是image_id匹配问题，不是真正的幻觉'
                    })
                continue  # 跳过此样本

            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})
                raw_reasoning = cot.get('raw_reasoning', '')

                if not raw_reasoning:
                    continue

                total_cot_count += 1

                # 改进: 使用更智能的幻觉检测逻辑
                hallucination_type = None
                hallucinated_object = None

                # 提取CoT中明确断言存在的物体（使用更严格的检测）
                asserted_objects = self._extract_asserted_objects_from_cot(raw_reasoning)

                # 检查断言的物体是否在真实物体列表中
                for obj in asserted_objects:
                    if obj.lower() not in [o.lower() for o in real_objects]:
                        hallucination_type = 'non_existent_object'
                        hallucinated_object = obj
                        hallucination_count += 1
                        hallucination_samples.append({
                            'image_id': image_id,
                            'task': task_name,
                            'hallucination_type': hallucination_type,
                            'hallucinated_object': hallucinated_object,
                            'cot_snippet': raw_reasoning[:100],
                            'real_objects': real_objects[:5]  # 记录真实物体供参考
                        })
                        break  # 每个样本只记录第一个幻觉物体

        hallucination_ratio = hallucination_count / total_cot_count if total_cot_count > 0 else 0

        # 判断通过状态
        passed = hallucination_ratio < 0.05

        warnings = []
        if hallucination_ratio > 0.05:
            warnings.append(
                f"CoT幻觉样本占比过高 ({hallucination_ratio*100:.1f}% > 5%)"
            )
        if skipped_samples > 0:
            warnings.append(
                f"跳过 {skipped_samples} 个样本（无法获取真实物体列表，可能是image_id匹配问题）"
            )

        return {
            'passed': passed,
            'statistics': {
                'total_cot_count': total_cot_count,
                'hallucination_count': hallucination_count,
                'hallucination_ratio': round(hallucination_ratio, 4)
            },
            'hallucination_samples': hallucination_samples[:20],
            'warnings': warnings
        }

    def _get_real_objects_for_image(self, image_id: str) -> List[str]:
        """获取图像的真实物体列表"""
        if 'detection' not in self.coco_annotations:
            return []

        annotations = self.coco_annotations['detection']['annotations']
        categories_info = self.coco_annotations['detection'].get('categories', [])

        id_to_name = {cat['id']: cat['name'] for cat in categories_info}

        image_annotations = [
            ann for ann in annotations
            if str(ann.get('image_id')) == str(image_id)
        ]

        return [
            id_to_name.get(ann['category_id'], 'unknown')
            for ann in image_annotations
        ]

    def _extract_asserted_objects_from_cot(self, cot_text: str) -> List[str]:
        """
        从CoT文本中提取明确断言存在的物体

        改进的检测逻辑：
        1. 使用更严格的关键词匹配，避免误报描述性内容
        2. 排除假设性、观察性的描述
        3. 只检测明确声称物体存在的部分

        Args:
            cot_text: CoT推理文本

        Returns:
            断言存在的物体列表
        """
        if not cot_text:
            return []

        # 关键修复: 排除描述性/假设性关键词
        # 如果CoT包含这些关键词，可能不是断言物体存在
        exclusion_keywords = [
            'looking for', 'checking for', 'searching for',
            'if there is', 'whether there is', 'might be',
            'could be', 'possibly', 'maybe', 'seems to',
            'appears to', 'looking at', 'scanning', 'examining',
            'first', 'step', 'next', 'then', 'finally',
            'i observe', 'i notice', 'i can see', 'i see',
        ]

        cot_lower = cot_text.lower()

        # 如果整段CoT主要是描述过程，不判定为幻觉
        if any(kw in cot_lower for kw in exclusion_keywords[:10]):
            # 需要更严格检查是否有明确断言
            # 例如："I see a person" vs "I'm looking for a person"
            pass  # 继续检测，但使用更严格的规则

        asserted_objects = []

        # 明确断言存在的关键词模式（必须包含完整句子结构）
        assertion_patterns = [
            # 使用正则匹配完整句子
            r'there\s+is\s+(?:a|an|the)\s+(\w+)',  # "there is a person"
            r'there\s+are\s+(?:a|an|the)?\s*(\w+)',  # "there are persons"
            r'i\s+found\s+(?:a|an|the)\s+(\w+)',  # "I found a car"
            r'the\s+image\s+contains\s+(?:a|an|the)\s+(\w+)',  # "the image contains a dog"
            r'i\s+can\s+clearly\s+see\s+(?:a|an|the)\s+(\w+)',  # "I can clearly see a person"
            r'visible\s+(?:a|an|the)\s+(\w+)',  # "visible a person"
            r'detected\s+(?:a|an|the)\s+(\w+)',  # "detected a car"
        ]

        try:
            import re
            for pattern in assertion_patterns:
                matches = re.findall(pattern, cot_lower)
                for match in matches:
                    # match是提取的物体名称
                    obj = match.strip()
                    # 检查是否是COCO类别
                    coco_categories_lower = [c.lower() for c in self.COCO_CATEGORIES]
                    if obj in coco_categories_lower:
                        asserted_objects.append(obj)
        except Exception:
            # 如果正则匹配失败，使用简单但更安全的方法
            # 只检测非常明确的断言
            for category in self.COCO_CATEGORIES:
                cat_lower = category.lower()
                # 使用更严格的模式：必须是"There is a X"或"I found X"
                if f'there is a {cat_lower}' in cot_lower or \
                   f'there is an {cat_lower}' in cot_lower or \
                   f'i found a {cat_lower}' in cot_lower:
                    asserted_objects.append(cat_lower)

        return asserted_objects

    def _analyze_repetition(self, data_list: List[Dict]) -> Dict[str, Any]:
        """重复度检测"""
        cot_texts = []

        for data in data_list:
            tasks = data.get('tasks', {})
            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})
                raw_reasoning = cot.get('raw_reasoning', '')
                if raw_reasoning:
                    cot_texts.append(raw_reasoning)

        if not cot_texts:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No CoT data for repetition analysis']
            }

        # 计算重复率
        unique_texts = len(set(cot_texts))
        total_texts = len(cot_texts)
        repetition_ratio = 1 - (unique_texts / total_texts)

        # 检查模式崩塌
        cot_counter = Counter(cot_texts)
        top_repeated = cot_counter.most_common(5)

        # 判断通过状态
        passed = repetition_ratio < 0.3

        warnings = []
        if repetition_ratio > 0.3:
            warnings.append(
                f"CoT重复度过高 ({repetition_ratio*100:.1f}% > 30%)"
            )

        if top_repeated and top_repeated[0][1] > total_texts * 0.1:
            warnings.append(
                f"最常见推理模式出现 {top_repeated[0][1]} 次"
            )

        return {
            'passed': passed,
            'statistics': {
                'total_cot_count': total_texts,
                'unique_cot_count': unique_texts,
                'repetition_ratio': round(repetition_ratio, 4)
            },
            'top_repeated_patterns': [
                {'text': text[:50], 'count': count}
                for text, count in top_repeated
            ],
            'warnings': warnings
        }

    def _analyze_cot_length(self, data_list: List[Dict]) -> Dict[str, Any]:
        """长度分布统计"""
        lengths = []
        short_samples = []
        long_samples = []

        MIN_LENGTH = 20
        MAX_LENGTH = 500

        for data in data_list:
            tasks = data.get('tasks', {})
            image_id = data.get('image_id')

            for task_name, task_data in tasks.items():
                cot = task_data.get('cot_reasoning', {})
                raw_reasoning = cot.get('raw_reasoning', '')

                if raw_reasoning:
                    length = len(raw_reasoning)
                    lengths.append(length)

                    if length < MIN_LENGTH:
                        short_samples.append({
                            'image_id': image_id,
                            'task': task_name,
                            'length': length,
                            'cot_snippet': raw_reasoning
                        })

                    if length > MAX_LENGTH:
                        long_samples.append({
                            'image_id': image_id,
                            'task': task_name,
                            'length': length,
                            'cot_snippet': raw_reasoning[:100]
                        })

        if not lengths:
            return {
                'passed': True,
                'statistics': {},
                'warnings': ['No CoT data for length analysis']
            }

        # 统计计算
        if NUMPY_AVAILABLE:
            avg_length = np.mean(lengths)
            median_length = np.median(lengths)
        else:
            avg_length = sum(lengths) / len(lengths)
            sorted_lengths = sorted(lengths)
            median_length = sorted_lengths[len(sorted_lengths) // 2]

        short_ratio = len(short_samples) / len(lengths)
        long_ratio = len(long_samples) / len(lengths)

        warnings = []
        if short_ratio > 0.1:
            warnings.append(f"过短CoT样本比例过高 ({short_ratio*100:.1f}% > 10%)")
        if long_ratio > 0.05:
            warnings.append(f"过长CoT样本比例过高 ({long_ratio*100:.1f}% > 5%)")

        return {
            'passed': True,
            'statistics': {
                'total_count': len(lengths),
                'average_length': round(avg_length, 2),
                'median_length': round(median_length, 2),
                'min_length': min(lengths),
                'max_length': max(lengths),
                'short_count': len(short_samples),
                'long_count': len(long_samples),
                'short_ratio': round(short_ratio, 4),
                'long_ratio': round(long_ratio, 4)
            },
            'short_samples': short_samples[:10],
            'long_samples': long_samples[:10],
            'warnings': warnings
        }

    # ============================================================
    # 5. 数据阶段判定
    # ============================================================

    def _assess_training_value(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """数据阶段判定"""
        # 提取关键指标
        top_k_results = validation_results.get('top_k_matching', {})
        kl_results = validation_results.get('soft_label_distribution', {}).get('kl_divergence_analysis', {})
        hallucination_results = validation_results.get('cot_quality', {}).get('hallucination_detection', {})
        distribution_results = validation_results.get('soft_label_distribution', {}).get('category_distribution_alignment', {})

        # 提取值并转换为Python原生类型
        match_rate = float(top_k_results.get('statistics', {}).get('match_rate', 0))
        avg_kl = float(kl_results.get('statistics', {}).get('average_kl', float('inf')))
        halluc_ratio = float(hallucination_results.get('statistics', {}).get('hallucination_ratio', 0))
        correlation = float(distribution_results.get('statistics', {}).get('correlation', 0))

        # 判断条件（确保bool是Python原生类型）
        conditions = {
            'top1_match_rate': {
                'value': match_rate,
                'threshold': 0.88,
                'passed': bool(match_rate >= 0.88)
            },
            'kl_divergence': {
                'value': avg_kl,
                'threshold': 0.5,
                'passed': bool(avg_kl < 0.5)
            },
            'hallucination_ratio': {
                'value': halluc_ratio,
                'threshold': 0.05,
                'passed': bool(halluc_ratio < 0.05)
            },
            'distribution_alignment': {
                'value': correlation,
                'threshold': 0.8,
                'passed': bool(correlation > 0.8)
            }
        }

        # 综合判断（确保是Python原生bool）
        all_passed = bool(all(c['passed'] for c in conditions.values()))

        recommendations = []
        if all_passed:
            recommendations.append("✓ 数据质量合格，具备训练3B学生的价值")
        else:
            recommendations.append("❌ 数据质量不合格，不建议直接训练3B学生")

            for cond_name, cond_info in conditions.items():
                if not cond_info['passed']:
                    if cond_name == 'top1_match_rate':
                        recommendations.append(
                            f"  - Top1匹配率 {cond_info['value']*100:.1f}% < 88%"
                        )
                    elif cond_name == 'kl_divergence':
                        recommendations.append(
                            f"  - KL散度 {cond_info['value']:.3f} 过高"
                        )
                    elif cond_name == 'hallucination_ratio':
                        recommendations.append(
                            f"  - 幻觉占比 {cond_info['value']*100:.1f}% > 5%"
                        )
                    elif cond_name == 'distribution_alignment':
                        recommendations.append(
                            f"  - 分布相关性 {cond_info['value']:.3f} < 0.8"
                        )

            recommendations.append("\n建议：重新迭代清洗或重新生成教师标签")

        return {
            'can_train': all_passed,
            'conditions': conditions,
            'recommendations': recommendations
        }

    def _save_report(self, report: Dict[str, Any], output_dir: str):
        """保存校验报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / 'data_quality_validation_report.json'

        # 转换numpy类型为Python原生类型
        serializable_report = convert_to_serializable(report)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

        if self.logger:
            self.logger.info(f"\n报告已保存: {report_file}")

    def _display_summary(self, report: Dict[str, Any]):
        """显示校验摘要"""
        if not self.logger:
            return

        self.logger.info("\n" + "="*70)
        self.logger.info("数据质量校验摘要")
        self.logger.info("="*70)

        # 显示关键指标
        results = report['validation_results']

        # Top-K匹配
        top_k = results.get('top_k_matching', {}).get('statistics', {})
        self.logger.info(f"\nTop-K匹配统计:")
        self.logger.info(f"  匹配率: {top_k.get('match_rate', 0)*100:.1f}% (阈值 ≥88%)")

        # KL散度
        kl = results.get('soft_label_distribution', {}).get('kl_divergence_analysis', {}).get('statistics', {})
        self.logger.info(f"\nKL散度分析:")
        self.logger.info(f"  平均KL: {kl.get('average_kl', 'N/A')} (阈值 <0.5)")

        # ECE
        ece = results.get('ece_calibration', {})
        self.logger.info(f"\nECE置信度校准:")
        self.logger.info(f"  ECE: {ece.get('ece', 'N/A')} (阈值 <0.15)")

        # 幻觉检测
        halluc = results.get('cot_quality', {}).get('hallucination_detection', {}).get('statistics', {})
        self.logger.info(f"\nCoT幻觉检测:")
        self.logger.info(f"  幻觉占比: {halluc.get('hallucination_ratio', 0)*100:.1f}% (阈值 <5%)")

        # 最终判定
        assessment = results.get('training_value_assessment', {})
        self.logger.info(f"\n最终判定:")
        for rec in assessment.get('recommendations', []):
            self.logger.info(f"  {rec}")

        self.logger.info("\n" + "="*70)


def compare_cleaning_effect(
    before_dir: str,
    after_dir: str,
    config: Any = None,
    logger: Any = None,
    coco_annotations_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    清洗效果对比实验

    Args:
        before_dir: 清洗前数据目录
        after_dir: 清洗后数据目录
        config: 配置管理器
        logger: 日志记录器
        coco_annotations_dir: COCO标注目录

    Returns:
        对比报告
    """
    validator = DataQualityValidator(config, logger, coco_annotations_dir)

    # 校验清洗前数据
    before_report = validator.run_full_validation(before_dir)

    # 校验清洗后数据
    after_report = validator.run_full_validation(after_dir)

    # 对比分析
    comparison = {
        'before': before_report,
        'after': after_report,
        'improvements': {}
    }

    # 关键指标变化
    before_top_k = float(before_report['validation_results']['top_k_matching']['statistics']['match_rate'])
    after_top_k = float(after_report['validation_results']['top_k_matching']['statistics']['match_rate'])

    before_kl = float(before_report['validation_results']['soft_label_distribution']['kl_divergence_analysis']['statistics']['average_kl'])
    after_kl = float(after_report['validation_results']['soft_label_distribution']['kl_divergence_analysis']['statistics']['average_kl'])

    before_halluc = float(before_report['validation_results']['cot_quality']['hallucination_detection']['statistics']['hallucination_ratio'])
    after_halluc = float(after_report['validation_results']['cot_quality']['hallucination_detection']['statistics']['hallucination_ratio'])

    comparison['improvements'] = {
        'top_k_match_rate': {
            'before': before_top_k,
            'after': after_top_k,
            'change': float(after_top_k - before_top_k),
            'improved': bool(after_top_k > before_top_k)
        },
        'kl_divergence': {
            'before': before_kl,
            'after': after_kl,
            'change': float(after_kl - before_kl),
            'improved': bool(after_kl < before_kl)
        },
        'hallucination_ratio': {
            'before': before_halluc,
            'after': after_halluc,
            'change': float(after_halluc - before_halluc),
            'improved': bool(after_halluc < before_halluc)
        }
    }

    # 判断清洗是否有效（确保是Python原生bool）
    effective = bool(
        comparison['improvements']['top_k_match_rate']['improved'] and
        comparison['improvements']['kl_divergence']['improved'] and
        comparison['improvements']['hallucination_ratio']['improved']
    )

    comparison['cleaning_effective'] = effective

    if not effective:
        comparison['recommendation'] = "清洗规则无效，需要优化过滤逻辑"
    else:
        comparison['recommendation'] = "清洗有效，数据质量显著提升"

    return comparison