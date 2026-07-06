"""
Configuration Manager
=====================

Handles loading and managing YAML configuration files for the distillation pipeline.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    Manages configuration loading and access for the VLM distillation pipeline.

    Supports loading from multiple YAML files and merging configurations.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Path to main configuration file (default.yaml).
                        If None, uses default path.
        """
        self.config_root = Path(__file__).parent.parent.parent / "configs"
        self.config_path = config_path or str(self.config_root / "default.yaml")
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load and merge configuration files."""
        # Load main config
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Load prompts config if specified or exists
        prompts_config_path = self.config.get('prompts_config')
        if prompts_config_path:
            prompts_path = Path(prompts_config_path)
        else:
            prompts_path = self.config_root / "prompts.yaml"

        if prompts_path.exists():
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_config = yaml.safe_load(f)
                # 直接合并，prompts.yaml 已经有正确的结构
                if 'prompts' in prompts_config:
                    self.config['prompts'] = prompts_config['prompts']
                else:
                    # 如果没有 'prompts' 键，直接合并到 prompts section
                    self._merge_config(self.config, prompts_config, "prompts")

        # Load model config if exists
        model_config_path = self.config_root / "model_config.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = yaml.safe_load(f)
                self._merge_config(self.config, model_config, "model")

        # Load distillation config if exists
        distill_config_path = self.config_root / "distillation.yaml"
        if distill_config_path.exists():
            with open(distill_config_path, 'r', encoding='utf-8') as f:
                distill_config = yaml.safe_load(f)
                self._merge_config(self.config, distill_config, "distillation")

    def _merge_config(self, base: Dict, new: Dict, section: str) -> None:
        """
        Merge new configuration into base configuration.

        Args:
            base: Base configuration dictionary
            new: New configuration to merge
            section: Section name to merge under
        """
        if section not in base:
            base[section] = {}

        for key, value in new.items():
            if isinstance(value, dict) and key in base[section] and isinstance(base[section][key], dict):
                # Deep merge for dictionaries
                base[section][key].update(value)
            else:
                base[section][key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports nested keys with dot notation).

        Args:
            key: Configuration key (e.g., "teacher.model_name" or "data.batch_size")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration. If None, saves to original path.
        """
        save_path = path or self.config_path

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def validate(self) -> bool:
        """
        Validate configuration for required keys and proper types.

        Returns:
            True if configuration is valid, False otherwise
        """
        required_keys = [
            "teacher.model_name",
            "data.coco_root",
            "output.root_dir",
        ]

        for key in required_keys:
            if self.get(key) is None:
                return False

        return True

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., "teacher", "data", "distillation")

        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(config_path='{self.config_path}', keys={len(self.config)})"


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)