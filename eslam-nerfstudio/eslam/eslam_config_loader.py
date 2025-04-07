"""
ESLAM Config Loader

This module provides functionality to load and parse ESLAM configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dataclasses import asdict

from .eslam_config import (
    ESLAMCameraConfig,
    ESLAMTrackingConfig,
    ESLAMMappingConfig,
    ESLAMRenderingConfig,
    ESLAMModelConfig
)

class ESLAMConfigLoader:
    """Loader for ESLAM configuration files."""

    def __init__(self, config_path: str):
        """
        Initialize the config loader.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.base_config_dir = self.config_path.parent
        self.config = {}

    def load(self) -> Dict[str, Any]:
        """
        Load and parse the configuration file.

        Returns:
            Dictionary containing the parsed configuration
        """
        # Load the main config file
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Handle inheritance
        if 'inherit_from' in self.config:
            base_config_path = self.base_config_dir / self.config['inherit_from']
            base_loader = ESLAMConfigLoader(str(base_config_path))
            base_config = base_loader.load()
            
            # Update with current config, preserving base values where not overridden
            for key, value in base_config.items():
                if key not in self.config:
                    self.config[key] = value

        return self.config

    def get_camera_config(self) -> ESLAMCameraConfig:
        """Get camera configuration."""
        if 'cam' not in self.config:
            return ESLAMCameraConfig()
        return ESLAMCameraConfig(**self.config['cam'])

    def get_tracking_config(self) -> ESLAMTrackingConfig:
        """Get tracking configuration."""
        if 'tracking' not in self.config:
            return ESLAMTrackingConfig()
        return ESLAMTrackingConfig(**self.config['tracking'])

    def get_mapping_config(self) -> ESLAMMappingConfig:
        """Get mapping configuration."""
        if 'mapping' not in self.config:
            return ESLAMMappingConfig()
        return ESLAMMappingConfig(**self.config['mapping'])

    def get_rendering_config(self) -> ESLAMRenderingConfig:
        """Get rendering configuration."""
        if 'rendering' not in self.config:
            return ESLAMRenderingConfig()
        return ESLAMRenderingConfig(**self.config['rendering'])

    def get_model_config(self) -> ESLAMModelConfig:
        """Get model configuration."""
        if 'model' not in self.config:
            return ESLAMModelConfig()
        return ESLAMModelConfig(**self.config['model'])

    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary."""
        return {
            'camera': self.get_camera_config(),
            'tracking': self.get_tracking_config(),
            'mapping': self.get_mapping_config(),
            'rendering': self.get_rendering_config(),
            'model': self.get_model_config()
        }

    def save_config(self, output_path: str) -> None:
        """
        Save the current configuration to a YAML file.

        Args:
            output_path: Path where to save the configuration
        """
        config_dict = {}
        for key, config in self.get_all_configs().items():
            config_dict[key] = asdict(config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False) 