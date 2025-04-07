"""
ESLAM DataManager

This module implements the data manager for ESLAM, handling RGB-D data processing.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional

import torch
import numpy as np
from pathlib import Path

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.images import BasicImages

from .eslam_config import ESLAMCameraConfig, ESLAMTrackingConfig

@dataclass
class ESLAMDataManagerConfig(VanillaDataManagerConfig):
    """ESLAM DataManager Config

    Add your custom datamanager config parameters here.
    """
    # "这个配置类是用来配置 ESLAMDataManager 的，当需要创建实例时，就使用 ESLAMDataManager 类" 
    _target: Type = field(default_factory=lambda: ESLAMDataManager)
    
    # Camera and tracking configurations
    camera: ESLAMCameraConfig = field(default_factory=ESLAMCameraConfig)
    tracking: ESLAMTrackingConfig = field(default_factory=ESLAMTrackingConfig)

class ESLAMDataManager(VanillaDataManager):
    """ESLAM DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """
    config: ESLAMDataManagerConfig
    camera: ESLAMCameraConfig
    tracking: ESLAMTrackingConfig

    def __init__(
        self,
        config: ESLAMDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )
        self.camera = config.camera
        self.tracking = config.tracking

    def setup_train(self):
        """Setup the datamanager for training."""
        super().setup_train()
        # Additional setup for ESLAM training
        self._setup_depth_processing()

    def _setup_depth_processing(self):
        """Setup depth processing parameters."""
        if self.train_dataset is not None:
            # Create masks for valid pixels
            self.valid_pixels_mask = torch.ones((self.camera.H, self.camera.W), dtype=torch.bool, device=self.device)
            if self.camera.crop_edge > 0:
                self.valid_pixels_mask[:self.camera.crop_edge, :] = False
                self.valid_pixels_mask[-self.camera.crop_edge:, :] = False
                self.valid_pixels_mask[:, :self.camera.crop_edge] = False
                self.valid_pixels_mask[:, -self.camera.crop_edge:] = False
            
            if self.tracking.ignore_edge_W > 0 or self.tracking.ignore_edge_H > 0:
                self.valid_pixels_mask[:self.tracking.ignore_edge_H, :] = False
                self.valid_pixels_mask[-self.tracking.ignore_edge_H:, :] = False
                self.valid_pixels_mask[:, :self.tracking.ignore_edge_W] = False
                self.valid_pixels_mask[:, -self.tracking.ignore_edge_W:] = False

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        
        # Process depth data
        if "depth" in image_batch:
            image_batch["depth"] = image_batch["depth"] / self.camera.png_depth_scale
        
        # Apply valid pixels mask
        if hasattr(self, 'valid_pixels_mask'):
            image_batch["mask"] = self.valid_pixels_mask.unsqueeze(0)
        
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        return ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for training."""
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        """Returns the number of rays per batch for evaluation."""
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        """Returns the path to the data."""
        return self.config.dataparser.data

    def get_train_dataset(self) -> InputDataset:
        """Returns the training dataset."""
        return self.train_dataset

    def get_eval_dataset(self) -> InputDataset:
        """Returns the evaluation dataset."""
        return self.eval_dataset
