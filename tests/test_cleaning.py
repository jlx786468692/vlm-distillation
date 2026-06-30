"""
数据清洗单元测试
================

测试DataCleaner的各个核心功能。
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from src.cleaning import DataCleaner
from src import ConfigManager


class TestDataCleanerInit(unittest.TestCase):
    """测试DataCleaner初始化"""

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = ConfigManager()
        config.set('cleaning.min_confidence', 0.6)
        config.set('cleaning.min_quality_score', 40.0)

        cleaner = DataCleaner(config)

        self.assertEqual(cleaner.min_confidence, 0.6)
        self.assertEqual(cleaner.min_quality_score, 40.0)

    def test_init_without_config(self):
        """测试无配置初始化（使用默认值）"""
        cleaner = DataCleaner()

        self.assertEqual(cleaner.min_confidence, 0.5)  # 默认值
        self.assertEqual(cleaner.min_quality_score, 30.0)

    def test_invalid_answers_list(self):
        """测试无效答案列表"""
        cleaner = DataCleaner()

        self.assertIn('unknown', cleaner.invalid_answers)
        self.assertIn('n/a', cleaner.invalid_answers)
        self.assertIn('', cleaner.invalid_answers)


class TestAnomalyDetection(unittest.TestCase):
    """测试异常检测功能"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.cleaner = DataCleaner(self.config)

    def test_detect_low_confidence(self):
        """测试低置信度检测"""
        data = [{
            'image_id': 'test_001',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'test answer',
                        'confidence': 0.3  # 低置信度
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        self.assertEqual(len(anomalies['low_confidence']), 1)
        self.assertEqual(anomalies['low_confidence'][0]['image_id'], 'test_001')

    def test_detect_invalid_answer(self):
        """测试无效答案检测"""
        data = [{
            'image_id': 'test_002',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'unknown',  # 无效答案
                        'confidence': 0.8
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        self.assertEqual(len(anomalies['invalid_answers']), 1)
        self.assertEqual(anomalies['invalid_answers'][0]['answer'], 'unknown')

    def test_detect_empty_caption(self):
        """测试空caption检测"""
        data = [{
            'image_id': 'test_003',
            'tasks': {
                'captioning': {
                    'hard_label': {
                        'captions': []  # 空caption
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        self.assertEqual(len(anomalies['empty_results']), 1)

    def test_detect_bbox_anomaly(self):
        """测试bbox异常检测"""
        data = [{
            'image_id': 'test_004',
            'tasks': {
                'detection': {
                    'hard_label': {
                        'objects': [
                            {'bbox': [-10, 20, 30, 40], 'confidence': 0.9}  # 超出范围
                        ]
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        self.assertEqual(len(anomalies['bbox_anomalies']), 1)

    def test_detect_length_anomaly(self):
        """测试长度异常检测"""
        data = [{
            'image_id': 'test_005',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'a',  # 过短（<3字符）
                        'confidence': 0.9
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        self.assertEqual(len(anomalies['length_anomalies']), 1)

    def test_detect_format_error(self):
        """测试格式错误检测"""
        data = [{
            'image_id': 'test_006',
            'tasks': {
                'detection': {
                    'hard_label': {
                        'objects': [
                            {'bbox': [10, 20], 'confidence': 0.9}  # bbox格式错误（只有2个值）
                        ]
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        self.assertEqual(len(anomalies['format_errors']), 1)

    def test_detect_multiple_anomalies(self):
        """测试多异常同时检测"""
        data = [{
            'image_id': 'test_007',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'unknown',  # 无效答案
                        'confidence': 0.3     # 低置信度
                    }
                },
                'detection': {
                    'hard_label': {
                        'objects': []
                    }
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        # 应检测到多个异常
        total_anomalies = sum(len(v) for v in anomalies.values())
        self.assertGreater(total_anomalies, 1)


class TestQualityScoring(unittest.TestCase):
    """测试质量评分功能"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.cleaner = DataCleaner(self.config)

    def test_compute_high_quality_score(self):
        """测试高质量数据评分"""
        data = [{
            'image_id': 'test_high',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'a kitchen with white cabinets',
                        'confidence': 0.95
                    },
                    'soft_label': {
                        'temperature': 2.0,
                        'answer_distribution': {'kitchen': 0.95, 'bathroom': 0.05}
                    },
                    'cot_reasoning': {
                        'quality_metrics': {
                            'logical_flow_score': 0.8,
                            'step_count': 4
                        },
                        'raw_reasoning': 'First, I observe. Next, I analyze. Then, I conclude.'
                    }
                }
            }
        }]

        scores = self.cleaner._compute_quality_scores(data)

        # 高质量数据应得高分（>70）
        self.assertGreater(scores['test_high'], 70)

    def test_compute_low_quality_score(self):
        """测试低质量数据评分"""
        data = [{
            'image_id': 'test_low',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'unknown',
                        'confidence': 0.3
                    }
                }
            }
        }]

        scores = self.cleaner._compute_quality_scores(data)

        # 低质量数据应得低分（<30）
        self.assertLess(scores['test_low'], 30)

    def test_compute_medium_quality_score(self):
        """测试中等质量数据评分"""
        data = [{
            'image_id': 'test_medium',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'kitchen',
                        'confidence': 0.6
                    },
                    'soft_label': {
                        'temperature': 2.5,
                        'answer_distribution': {'kitchen': 0.6, 'bathroom': 0.3}
                    }
                }
            }
        }]

        scores = self.cleaner._compute_quality_scores(data)

        # 中等质量应在50-70范围
        self.assertGreater(scores['test_medium'], 40)
        self.assertLess(scores['test_medium'], 70)


class TestCleaningRules(unittest.TestCase):
    """测试清洗规则应用"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.config.set('cleaning.auto_remove_invalid', True)
        self.cleaner = DataCleaner(self.config)

    def test_remove_low_quality(self):
        """测试移除低质量数据"""
        all_data = [{
            'image_id': 'low_quality',
            'quality_score': 25.0,  # < min_quality_score
            'tasks': {'vqa': {}}
        }, {
            'image_id': 'high_quality',
            'quality_score': 75.0,
            'tasks': {'vqa': {}}
        }]

        anomalies = {'low_confidence': [], 'invalid_answers': [], 'empty_results': [],
                     'bbox_anomalies': [], 'cot_low_quality': [], 'length_anomalies': [],
                     'format_errors': []}

        scores = {'low_quality': 25.0, 'high_quality': 75.0}

        cleaned, removed = self.cleaner._apply_cleaning_rules(all_data, anomalies, scores)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0]['image_id'], 'low_quality')

    def test_remove_invalid_answer(self):
        """测试移除无效答案数据"""
        all_data = [{
            'image_id': 'invalid',
            'tasks': {
                'vqa': {
                    'hard_label': {'answer': 'unknown'}
                }
            }
        }, {
            'image_id': 'valid',
            'tasks': {
                'vqa': {
                    'hard_label': {'answer': 'kitchen'}
                }
            }
        }]

        anomalies = {
            'invalid_answers': [{'image_id': 'invalid', 'answer': 'unknown'}],
            'low_confidence': [], 'empty_results': [], 'bbox_anomalies': [],
            'cot_low_quality': [], 'length_anomalies': [], 'format_errors': []
        }

        scores = {'invalid': 10.0, 'valid': 70.0}

        cleaned, removed = self.cleaner._apply_cleaning_rules(all_data, anomalies, scores)

        self.assertEqual(len(removed), 1)

    def test_keep_marked_invalid(self):
        """测试保留标记但不移除（keep-invalid模式）"""
        self.config.set('cleaning.auto_remove_invalid', False)
        cleaner = DataCleaner(self.config)

        all_data = [{
            'image_id': 'test',
            'quality_score': 20.0,
            'tasks': {}
        }]

        anomalies = {'invalid_answers': [{'image_id': 'test'}],
                     'low_confidence': [], 'empty_results': [], 'bbox_anomalies': [],
                     'cot_low_quality': [], 'length_anomalies': [], 'format_errors': []}

        scores = {'test': 20.0}

        cleaned, removed = cleaner._apply_cleaning_rules(all_data, anomalies, scores)

        # 不自动移除，应保留在cleaned中（带标记）
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(removed), 0)
        self.assertTrue(cleaned[0].get('quality_warning', False))


class TestDeduplication(unittest.TestCase):
    """测试数据去重"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.config.set('cleaning.deduplicate_answers', True)
        self.cleaner = DataCleaner(self.config)

    def test_normalize_answer(self):
        """测试答案标准化"""
        # 测试移除冠词
        result = self.cleaner._normalize_answer("A kitchen")
        self.assertEqual(result, "kitchen")

        # 测试移除标点
        result = self.cleaner._normalize_answer("kitchen!")
        self.assertNotIn('!', result)

        # 测试小写化
        result = self.cleaner._normalize_answer("KITCHEN")
        self.assertEqual(result, "kitchen")

    def test_detect_duplicates(self):
        """测试检测重复答案"""
        data = [{
            'image_id': 'dup_1',
            'tasks': {
                'vqa': {
                    'hard_label': {'answer': 'a kitchen'}
                }
            }
        }, {
            'image_id': 'dup_2',
            'tasks': {
                'vqa': {
                    'hard_label': {'answer': 'the kitchen'}  # 与dup_1相似
                }
            }
        }]

        deduplicated, info = self.cleaner._deduplicate_data(data)

        # 应检测到重复
        self.assertEqual(info['duplicate_count'], 1)


class TestDataRepair(unittest.TestCase):
    """测试数据修复"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.config.set('cleaning.auto_repair_bbox', True)
        self.cleaner = DataCleaner(self.config)

    def test_repair_bbox_out_of_range(self):
        """测试修复超出范围的bbox"""
        data = [{
            'image_id': 'repair_test',
            'tasks': {
                'detection': {
                    'hard_label': {
                        'objects': [
                            {'bbox': [-10, -20, 1500, 2000]}  # 超出范围
                        ]
                    }
                }
            }
        }]

        repaired = self.cleaner._repair_data(data)

        # bbox应被裁剪到合理范围
        bbox = repaired[0]['tasks']['detection']['hard_label']['objects'][0]['bbox']

        self.assertGreaterEqual(bbox[0], 0)  # x_min >= 0
        self.assertGreaterEqual(bbox[1], 0)  # y_min >= 0
        self.assertLessEqual(bbox[2], 1000)  # x_max <= 1000
        self.assertLessEqual(bbox[3], 1000)  # y_max <= 1000

    def test_repair_invalid_coordinates(self):
        """测试修复无效坐标"""
        data = [{
            'image_id': 'coord_test',
            'tasks': {
                'detection': {
                    'hard_label': {
                        'objects': [
                            {'bbox': [100, 200, 50, 80]}  # x_max < x_min
                        ]
                    }
                }
            }
        }]

        repaired = self.cleaner._repair_data(data)

        bbox = repaired[0]['tasks']['detection']['hard_label']['objects'][0]['bbox']

        # 修复后 x_max > x_min
        self.assertGreater(bbox[2], bbox[0])

    def test_add_missing_confidence(self):
        """测试补全缺失置信度"""
        data = [{
            'image_id': 'missing_conf',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'test'
                        # confidence缺失
                    }
                }
            }
        }]

        repaired = self.cleaner._repair_data(data)

        # 应添加默认置信度
        confidence = repaired[0]['tasks']['vqa']['hard_label'].get('confidence')
        self.assertIsNotNone(confidence)
        self.assertEqual(confidence, 0.5)


class TestCleaningReport(unittest.TestCase):
    """测试清洗报告生成"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.cleaner = DataCleaner(self.config)

    def test_generate_report_structure(self):
        """测试报告结构"""
        all_data = [{'image_id': 'test', 'tasks': {}}]
        cleaned_data = [{'image_id': 'test', 'tasks': {}}]
        removed_data = []
        anomalies = {'low_confidence': [], 'invalid_answers': [], 'empty_results': [],
                     'bbox_anomalies': [], 'cot_low_quality': [], 'length_anomalies': [],
                     'format_errors': []}
        quality_scores = {'test': 75.0}
        duplicate_info = {'duplicates': [], 'unique_count': 1}

        report = self.cleaner._generate_cleaning_report(
            all_data, cleaned_data, removed_data,
            anomalies, quality_scores, duplicate_info
        )

        # 验证报告结构
        self.assertIn('summary', report)
        self.assertIn('anomaly_statistics', report)
        self.assertIn('quality_statistics', report)
        self.assertIn('recommendations', report)

    def test_report_summary(self):
        """测试报告摘要"""
        all_data = [{'image_id': i, 'tasks': {}} for i in range(100)]
        cleaned_data = [{'image_id': i, 'tasks': {}} for i in range(90)]
        removed_data = [{'image_id': i, 'tasks': {}} for i in range(90, 100)]
        anomalies = {'low_confidence': [], 'invalid_answers': [], 'empty_results': [],
                     'bbox_anomalies': [], 'cot_low_quality': [], 'length_anomalies': [],
                     'format_errors': []}
        quality_scores = {str(i): 75.0 for i in range(100)}
        duplicate_info = {'duplicates': [], 'unique_count': 90}

        report = self.cleaner._generate_cleaning_report(
            all_data, cleaned_data, removed_data,
            anomalies, quality_scores, duplicate_info
        )

        summary = report['summary']

        self.assertEqual(summary['total_input'], 100)
        self.assertEqual(summary['cleaned_count'], 90)
        self.assertEqual(summary['removed_count'], 10)
        self.assertEqual(summary['removal_rate'], 0.1)

    def test_generate_recommendations(self):
        """测试生成建议"""
        anomalies = {
            'low_confidence': [{'image_id': i} for i in range(150)],  # 大量低置信度
            'invalid_answers': [{'image_id': i} for i in range(30)],
            'cot_low_quality': [{'image_id': i} for i in range(50)],
            'empty_results': [], 'bbox_anomalies': [], 'length_anomalies': [], 'format_errors': []
        }

        quality_scores = {str(i): 60.0 for i in range(200)}

        recommendations = self.cleaner._generate_recommendations(anomalies, quality_scores)

        # 应生成建议
        self.assertGreater(len(recommendations), 0)

        # 建议应包含关键词
        rec_text = ' '.join(recommendations)
        self.assertIn('低置信度', rec_text)
        self.assertIn('无效答案', rec_text)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()

        # 创建测试数据
        test_data = {
            'image_id': 'test_001',
            'tasks': {
                'vqa': {
                    'hard_label': {
                        'answer': 'kitchen',
                        'confidence': 0.8,
                        'question': 'what is this?'
                    }
                }
            }
        }

        with open(self.input_dir / "test_001.json", 'w') as f:
            json.dump(test_data, f)

        self.config = ConfigManager()
        self.cleaner = DataCleaner(self.config)

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir)

    def test_full_cleaning_flow(self):
        """测试完整清洗流程"""
        report = self.cleaner.clean_directory(
            str(self.input_dir),
            str(self.output_dir)
        )

        # 验证清洗完成
        self.assertIn('summary', report)
        self.assertGreater(report['summary']['total_input'], 0)

        # 验证输出文件创建
        cleaned_dir = self.output_dir / "cleaned"
        self.assertTrue(cleaned_dir.exists())


class TestEdgeCases(unittest.TestCase):
    """测试边缘情况"""

    def setUp(self):
        """设置测试环境"""
        self.config = ConfigManager()
        self.cleaner = DataCleaner(self.config)

    def test_empty_data(self):
        """测试空数据"""
        anomalies = self.cleaner._detect_anomalies([])

        # 所有异常类型应为空列表
        for key, value in anomalies.items():
            self.assertEqual(len(value), 0)

    def test_missing_tasks(self):
        """测试缺失tasks字段"""
        data = [{'image_id': 'test'}]

        anomalies = self.cleaner._detect_anomalies(data)

        # 应优雅处理缺失字段
        for key, value in anomalies.items():
            self.assertEqual(len(value), 0)

    def test_partial_data(self):
        """测试部分数据缺失"""
        data = [{
            'image_id': 'partial',
            'tasks': {
                'vqa': {
                    'hard_label': {}  # 空硬标签
                }
            }
        }]

        anomalies = self.cleaner._detect_anomalies(data)

        # 应检测到空结果或格式错误
        total_anomalies = sum(len(v) for v in anomalies.values())
        self.assertGreater(total_anomalies, 0)


def run_tests():
    """运行所有测试"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
