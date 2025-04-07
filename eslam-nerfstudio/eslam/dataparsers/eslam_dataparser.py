"""
ESLAM Data Parser

This module implements the data parser for ESLAM, handling RGB-D data loading and processing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2
import glob
import os
import re

from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.images import BasicImages

@dataclass
class ESLAMDataParserConfig(DataParserConfig):
    """ESLAM Data Parser Config

    Args:
        data: Path to data directory
        scale_factor: Scaling factor to apply to the camera poses
        depth_unit_scale_factor: Scaling factor to apply to the depth values
        crop_edge: Number of pixels to crop from the edges
    """

    _target: Type = field(default_factory=lambda: ESLAMDataParser)
    data: Path = Path("data/")
    scale_factor: float = 1.0
    depth_unit_scale_factor: float = 1000.0  # Convert depth from mm to meters
    crop_edge: int = 0

@dataclass
class ESLAMDataParser(DataParser):
    """ESLAM Data Parser

    Args:
        config: ESLAMDataParserConfig object
    """

    config: ESLAMDataParserConfig

    def __init__(self, config: ESLAMDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.depth_unit_scale_factor: float = config.depth_unit_scale_factor
        self.crop_edge: int = config.crop_edge

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Generate dataparser outputs for a split.

        Args:
            split: Split to generate outputs for

        Returns:
            DataparserOutputs containing the outputs
        """
        # Load camera parameters
        camera_params = self._load_camera_params()
        
        # Load images and depths
        image_filenames, depth_filenames = self._get_image_depth_filenames()
        
        # Load camera poses
        poses = self._load_camera_poses()
        
        # Create cameras
        cameras = self._create_cameras(poses, camera_params)
        
        # Create scene box
        scene_box = self._create_scene_box(poses)
        
        # Create metadata
        metadata = {
            "depth_filenames": depth_filenames,
            "depth_unit_scale_factor": self.depth_unit_scale_factor,
            "crop_edge": self.crop_edge,
        }

        return DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
        )

    def _load_camera_params(self) -> Dict:
        """Load camera parameters from config or calibration file.

        Returns:
            Dictionary containing camera parameters
        """
        # Default camera parameters (can be overridden by config)
        return {
            "fx": 600.0,
            "fy": 600.0,
            "cx": 599.5,
            "cy": 339.5,
            "width": 1200,
            "height": 680,
        }

    def _get_image_depth_filenames(self) -> Tuple[List[Path], List[Path]]:
        """Get lists of image and depth filenames.

        Returns:
            Tuple of (image_filenames, depth_filenames)
        """
        # Get all image and depth files
        image_paths = sorted(
            glob.glob(f'{self.data}/rgb/*.png'),
            key=self._sort_key
        )
        depth_paths = sorted(
            glob.glob(f'{self.data}/depth/*.png'),
            key=self._sort_key
        )
        
        # Convert to Path objects
        image_filenames = [Path(p) for p in image_paths]
        depth_filenames = [Path(p) for p in depth_paths]
        
        return image_filenames, depth_filenames

    def _sort_key(self, filepath: str) -> int:
        """Sort key function for file paths.
        
        Args:
            filepath: Path to file
            
        Returns:
            Integer for sorting
        """
        base_name = os.path.basename(filepath)
        num = re.findall(r'\d+', base_name)
        if num:
            return int(num[0])
        else:
            return base_name

    def _load_camera_poses(self) -> torch.Tensor:
        """Load camera poses from file.

        Returns:
            Tensor of camera poses
        """
        poses = []
        with open(f'{self.data}/traj.txt', "r") as f:
            lines = f.readlines()
        
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # Convert from ESLAM to nerfstudio coordinate system
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
            
        return torch.stack(poses)

    def _create_cameras(self, poses: torch.Tensor, camera_params: Dict) -> Cameras:
        """Create Cameras object from poses and parameters.

        Args:
            poses: Tensor of camera poses
            camera_params: Dictionary of camera parameters

        Returns:
            Cameras object
        """
        # Convert poses to nerfstudio format
        poses = poses[:, :3, :4]
        
        return Cameras(
            camera_to_worlds=poses,
            fx=camera_params["fx"],
            fy=camera_params["fy"],
            cx=camera_params["cx"],
            cy=camera_params["cy"],
            width=camera_params["width"],
            height=camera_params["height"],
            camera_type=CameraType.PERSPECTIVE,
        )

    def _create_scene_box(self, poses: torch.Tensor) -> SceneBox:
        """Create scene box from camera poses.

        Args:
            poses: Tensor of camera poses

        Returns:
            SceneBox object
        """
        # Get camera positions
        camera_positions = poses[:, :3, 3]
        
        # Calculate scene bounds
        min_bounds = camera_positions.min(dim=0)[0]
        max_bounds = camera_positions.max(dim=0)[0]
        
        # Add some padding
        padding = 1.0
        min_bounds -= padding
        max_bounds += padding
        
        return SceneBox(aabb=torch.stack([min_bounds, max_bounds])) 