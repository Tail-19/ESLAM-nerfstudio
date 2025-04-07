"""
ESLAM Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

from eslam.eslam_datamanager import (
    ESLAMDataManagerConfig,
)
from eslam.eslam_model import ESLAMModelConfig
from eslam.eslam_pipeline import (
    ESLAMPipelineConfig,
)
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
import yaml
from pathlib import Path

@dataclass
class ESLAMCameraConfig:
    """ESLAM Camera Configuration"""
    H: int = 680
    W: int = 1200
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 599.5
    cy: float = 339.5
    png_depth_scale: float = 1000.0
    crop_edge: int = 0

@dataclass
class ESLAMTrackingConfig:
    """ESLAM Tracking Configuration"""
    ignore_edge_W: int = 75
    ignore_edge_H: int = 75
    vis_freq: int = 400
    vis_inside_freq: int = 400
    const_speed_assumption: bool = True
    no_vis_on_first_frame: bool = True
    gt_camera: bool = False
    lr_T: float = 0.001
    lr_R: float = 0.001
    pixels: int = 2000
    iters: int = 8
    w_sdf_fs: float = 10
    w_sdf_center: float = 200
    w_sdf_tail: float = 50
    w_depth: float = 1
    w_color: float = 5

@dataclass
class ESLAMMappingConfig:
    """ESLAM Mapping Configuration"""
    every_frame: int = 4
    joint_opt: bool = True
    joint_opt_cam_lr: float = 0.001
    no_vis_on_first_frame: bool = True
    no_mesh_on_first_frame: bool = True
    no_log_on_first_frame: bool = True
    vis_freq: int = 400
    vis_inside_freq: int = 400
    mesh_freq: int = 4000
    ckpt_freq: int = 2000
    keyframe_every: int = 4
    mapping_window_size: int = 20
    keyframe_selection_method: str = 'overlap'
    lr_first_factor: float = 5
    lr_factor: float = 1
    pixels: int = 4000
    iters_first: int = 1000
    iters: int = 15
    w_sdf_fs: float = 5
    w_sdf_center: float = 200
    w_sdf_tail: float = 10
    w_depth: float = 0.1
    w_color: float = 5
    bound: Optional[List[List[float]]] = None
    marching_cubes_bound: Optional[List[List[float]]] = None

@dataclass
class ESLAMRenderingConfig:
    """ESLAM Rendering Configuration"""
    n_stratified: int = 32
    n_importance: int = 8
    perturb: bool = True
    learnable_beta: bool = True

@dataclass
class ESLAMModelConfig(NerfactoModelConfig):
    """ESLAM Model Configuration"""
    c_dim: int = 32
    truncation: float = 0.06
    planes_res: Dict[str, float] = field(default_factory=lambda: {
        "coarse": 0.24,
        "fine": 0.06,
        "bound_dividable": 0.24
    })
    c_planes_res: Dict[str, float] = field(default_factory=lambda: {
        "coarse": 0.24,
        "fine": 0.03
    })

@dataclass
class ESLAMConfig:
    """Main ESLAM configuration class."""
    camera: ESLAMCameraConfig = field(default_factory=ESLAMCameraConfig)
    tracking: ESLAMTrackingConfig = field(default_factory=ESLAMTrackingConfig)
    mapping: ESLAMMappingConfig = field(default_factory=ESLAMMappingConfig)
    rendering: ESLAMRenderingConfig = field(default_factory=ESLAMRenderingConfig)
    model: ESLAMModelConfig = field(default_factory=ESLAMModelConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "ESLAMConfig":
        """Load configuration from a YAML file."""
        from .eslam_config_loader import ESLAMConfigLoader
        loader = ESLAMConfigLoader(config_path)
        loader.load()
        return cls(**loader.get_all_configs())
    
class ESLAMConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.base_config_dir = self.config_path.parent
        self.config = {}

    def load(self) -> Dict[str, Any]:
        # 加载主配置文件
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 处理继承
        if 'inherit_from' in self.config:
            base_config_path = self.base_config_dir / self.config['inherit_from']
            base_loader = ESLAMConfigLoader(str(base_config_path))
            base_config = base_loader.load()
            
            # 更新配置，保留基础值
            for key, value in base_config.items():
                if key not in self.config:
                    self.config[key] = value

        return self.config

    def get_all_configs(self) -> Dict[str, Any]:
        return {
            "camera": self.config.get("cam", ESLAMCameraConfig()),
            "tracking": self.config.get("tracking", ESLAMTrackingConfig()),
            "mapping": self.config.get("mapping", ESLAMMappingConfig()),
            "rendering": self.config.get("rendering", ESLAMRenderingConfig()),
            "model": self.config.get("model", ESLAMModelConfig()),
        }


eslam = MethodSpecification(
    config=TrainerConfig(
        method_name="eslam",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=30000,
        mixed_precision=True,
        pipeline=ESLAMPipelineConfig(
            datamanager=ESLAMDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=ESLAMModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                average_init_density=0.01,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=50000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=5000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="ESLAM implementation in nerfstudio framework.",
)

