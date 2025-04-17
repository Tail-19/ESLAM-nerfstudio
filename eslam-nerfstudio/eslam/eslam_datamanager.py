"""
ESLAM DataManager

This module implements the data manager for ESLAM, handling RGB-D data processing.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union, Optional

import torch
import numpy as np
from pathlib import Path
from eslam.image_encoder import BaseImageEncoder
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
class ESLAMDataManagerConfig(ParallelDataManagerConfig):
    
    _target: Type = field(default_factory=lambda: ESLAMDataManager)
    

class ESLAMDataManager(ParallelDataManager, Generic[TDataset]):
    config: ESLAMDataManagerConfig

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

    def custom_ray_processor(
        self, ray_bundle: RayBundle, batch: Dict
    ) -> Tuple[RayBundle, Dict]:
        """An API to add latents, metadata, or other further customization to the RayBundle dataloading process that is parallelized."""
        # ray_indices = batch["indices"]
        # batch["clip"], clip_scale = self.clip_interpolator(ray_indices)
        # batch["dino"] = self.dino_dataloader(ray_indices)
        # ray_bundle.metadata["clip_scales"] = clip_scale

        # Assume all cameras have the same focal length and image dimensions.
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
