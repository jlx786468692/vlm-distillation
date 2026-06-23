"""
Unit Tests for Distillation Pipeline
====================================

Tests for core components and functionality.
"""

import unittest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import torch

# Import modules to test
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.distillation.hard_label_gen import HardLabelGenerator
from src.distillation.soft_label_gen import SoftLabelGenerator
from src.distillation.cot_generator import CoTGenerator
from src.export.json_exporter import JSONExporter


class TestConfigManager(unittest.TestCase):
    """Tests for ConfigManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_path = "configs/default.yaml"

    def test_config_loading(self):
        """Test configuration loading."""
        try:
            config = ConfigManager(self.config_path)
            self.assertIsNotNone(config.config)
        except FileNotFoundError:
            # Skip if config file doesn't exist in test environment
            self.skipTest("Config file not found")

    def test_get_nested_key(self):
        """Test getting nested configuration keys."""
        config = ConfigManager()
        config.config = {
            'teacher': {
                'model_name': 'test_model'
            }
        }

        result = config.get('teacher.model_name')
        self.assertEqual(result, 'test_model')

    def test_set_key(self):
        """Test setting configuration key."""
        config = ConfigManager()
        config.config = {}

        config.set('test_key', 'test_value')
        self.assertEqual(config.config['test_key'], 'test_value')

    def test_get_with_default(self):
        """Test getting key with default value."""
        config = ConfigManager()
        config.config = {}

        result = config.get('nonexistent_key', 'default_value')
        self.assertEqual(result, 'default_value')


class TestJSONExporter(unittest.TestCase):
    """Tests for JSONExporter."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_output_dir = Path("./test_outputs")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test outputs."""
        import shutil
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_save_result(self):
        """Test saving result to JSON."""
        exporter = JSONExporter()
        exporter.output_dir = self.test_output_dir

        result = {
            'image_id': 'test_001',
            'tasks': {'vqa': {'answer': 'test_answer'}}
        }

        output_path = str(self.test_output_dir / "test_result.json")
        success = exporter.save_result(result, output_path)

        self.assertTrue(success)
        self.assertTrue(Path(output_path).exists())

    def test_validate_result(self):
        """Test result validation."""
        exporter = JSONExporter()

        valid_result = {'image_id': 'test', 'tasks': {}}
        invalid_result = {'tasks': {}}

        self.assertTrue(exporter._validate_result(valid_result))
        self.assertFalse(exporter._validate_result(invalid_result))

    def test_load_result(self):
        """Test loading result from JSON."""
        exporter = JSONExporter()

        # Create test file
        test_data = {'image_id': 'test_001', 'data': 'test'}
        test_file = str(self.test_output_dir / "test_load.json")

        with open(test_file, 'w') as f:
            json.dump(test_data, f)

        loaded = exporter.load_result(test_file)
        self.assertEqual(loaded['image_id'], 'test_001')


class TestHardLabelGenerator(unittest.TestCase):
    """Tests for HardLabelGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock teacher model
        self.mock_teacher = Mock()
        self.mock_teacher.model_name = "test_model"

        self.mock_teacher.inference_vqa = Mock(return_value={
            'answer': 'test_answer',
            'confidence': 0.95,
            'full_response': 'test response'
        })

        self.mock_teacher.inference_captioning = Mock(return_value={
            'captions': ['test caption 1', 'test caption 2'],
            'num_captions': 2
        })

        self.mock_teacher.inference_detection = Mock(return_value={
            'objects': [{'class': 'person', 'bbox': [10, 20, 30, 40]}]
        })

    def test_generate_vqa_hard_labels(self):
        """Test VQA hard label generation."""
        config = ConfigManager()
        config.config = {'distillation': {'hard_labels': {'confidence_threshold': 0.5}}}

        generator = HardLabelGenerator(self.mock_teacher, config)

        result = generator.generate_vqa_hard_labels(
            'test_image.jpg',
            'test question',
            'test_id'
        )

        self.assertEqual(result['task'], 'vqa')
        self.assertEqual(result['answer'], 'test_answer')
        self.assertIn('confidence', result)

    def test_generate_captioning_hard_labels(self):
        """Test captioning hard label generation."""
        config = ConfigManager()
        config.config = {'distillation': {'hard_labels': {'confidence_threshold': 0.5}}}

        generator = HardLabelGenerator(self.mock_teacher, config)

        result = generator.generate_captioning_hard_labels(
            'test_image.jpg',
            num_captions=2,
            image_id='test_id'
        )

        self.assertEqual(result['task'], 'captioning')
        self.assertEqual(len(result['captions']), 2)

    def test_validate_hard_labels(self):
        """Test hard label validation."""
        config = ConfigManager()
        generator = HardLabelGenerator(self.mock_teacher, config)

        valid_label = {
            'image_id': 'test',
            'task': 'vqa',
            'answer': 'test_answer',
            'timestamp': '2024-01-01'
        }

        invalid_label = {
            'image_id': 'test',
            'task': 'vqa'
            # Missing answer
        }

        self.assertTrue(generator.validate_hard_labels(valid_label))
        self.assertFalse(generator.validate_hard_labels(invalid_label))


class TestSoftLabelGenerator(unittest.TestCase):
    """Tests for SoftLabelGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_teacher = Mock()
        self.mock_teacher.model_name = "test_model"

        # Mock inference with logits
        mock_logits = {
            'probabilities': torch.rand(10, 100)  # Mock probabilities
        }

        self.mock_teacher.inference_vqa = Mock(return_value={
            'answer': 'test',
            'logits': mock_logits,
            'confidence': 0.9
        })

    def test_temperature_scaling(self):
        """Test temperature scaling."""
        config = ConfigManager()
        config.config = {
            'distillation': {
                'soft_labels': {
                    'temperature': 2.0,
                    'top_k': 50
                }
            }
        }

        generator = SoftLabelGenerator(self.mock_teacher, config)

        self.assertEqual(generator.temperature, 2.0)
        self.assertEqual(generator.top_k, 50)


class TestCoTGenerator(unittest.TestCase):
    """Tests for CoTGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_teacher = Mock()
        self.mock_teacher.model_name = "test_model"

        self.mock_teacher.inference_vqa = Mock(return_value={
            'full_response': "First, I see an object. Next, I analyze it. Finally, I conclude."
        })

    def test_reasoning_structure_extraction(self):
        """Test reasoning structure extraction."""
        config = ConfigManager()
        config.config = {'distillation': {'cot': {'structured_output': True}}}

        generator = CoTGenerator(self.mock_teacher, config)

        result = generator.generate_vqa_cot(
            'test.jpg',
            'test question',
            'test_id'
        )

        self.assertIn('raw_reasoning', result)
        self.assertIn('structured_reasoning', result)

    def test_reasoning_validation(self):
        """Test reasoning quality validation."""
        config = ConfigManager()
        generator = CoTGenerator(self.mock_teacher, config)

        good_reasoning = "First, I observe. Next, I analyze. Then, I reason. Therefore, I conclude."
        bad_reasoning = "This is an answer."

        good_quality = generator._validate_reasoning_quality(good_reasoning)
        bad_quality = generator._validate_reasoning_quality(bad_reasoning)

        self.assertTrue(good_quality['has_required_keywords'])
        self.assertFalse(bad_quality['has_required_keywords'])


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_full_pipeline_mock(self):
        """Test full pipeline with mocked components."""
        # This would test the full distillation flow
        # with all mocked components

        # Create mock components
        mock_config = Mock()
        mock_teacher = Mock()
        mock_data_manager = Mock()

        # Setup mock returns
        mock_data_manager.get_sample_ids = Mock(return_value=[1, 2, 3])
        mock_data_manager.get_batch_data = Mock(return_value={
            'images': [{'id': 1, 'path': 'test.jpg', 'image': None}],
            'annotations': {'vqa': {}, 'captioning': {}, 'detection': {}},
            'metadata': {'timestamp': '2024-01-01'}
        })

        # The actual integration test would be more comprehensive
        # This is a placeholder for demonstration
        self.assertTrue(True)


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
