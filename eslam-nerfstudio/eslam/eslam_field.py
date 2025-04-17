"""
ESLAM Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.nerfacto_field import NerfactoField  # for subclassing NerfactoField
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.field_utils import FieldHeadNames
from nerfstudio.data.scene_box import SceneBox

class ESLAMNerfField(NerfactoField):
    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        cfg: dict,
    ) -> None:
        super().__init__(aabb=aabb, num_images=num_images)
        self.init_planes(cfg)
        self.extract_camera_params(cfg)
        self.get_mlp(cfg)

        
    def get_output(self, ray_samples: RaySamples):
        
        pos_normalized = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        
        outputs = {}
        outputs[FieldHeadNames.RGB] = self.get_raw_rgb(pos_normalized)
        outputs[FieldHeadNames.SDF] = self.get_raw_sdf(pos_normalized)
        return outputs
        
    def get_raw_sdf(self, pos_normalized):
        feat = self.sample_plane_feature(pos_normalized, self.sdf_planes_xy, self.sdf_planes_xz, self.sdf_planes_yz)
        sdf = self.sdf_mlp(feat)
        return sdf
    
    def get_raw_rgb(self, pos_normalized):
        feat = self.sample_plane_feature(pos_normalized, self.rgb_planes_xy, self.rgb_planes_xz, self.rgb_planes_yz)
        rgb = self.rgb_mlp(feat)
        return rgb
        
    def sample_plane_feature(self, pos_normalized, planes_xy, planes_xz, planes_yz):
        """
        Sample feature from planes
        Args:
            pos_normalized (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        vgrid = pos_normalized[None, :, None]

        feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i], vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i], vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i], vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            feat.append(xy + xz + yz)
        feat = torch.cat(feat, dim=-1)

        return feat

    def get_mlp(self, cfg):
            ## layers for SDF decoder
        c_dim = cfg['model']['c_dim']
        hidden_size = cfg['model']['hidden_size'] #16
        n_blocks = cfg['model']['n_blocks'] #2
        
        self.sdf_mlp = tcnn.Network(
            n_input_dims=2 * c_dim,  # 输入维度
            n_output_dims=1,         # 输出维度（SDF值）
            network_config={
                "otype": "FullyFusedMLP",  # 使用完全融合的MLP
                "activation": "ReLU",       # 激活函数
                "output_activation": "None", # 输出层不使用激活函数
                "n_neurons": hidden_size,   # 隐藏层神经元数量
                "n_hidden_layers": n_blocks - 1,  # 隐藏层数量
            }
        )

        self.rgb_mlp = tcnn.Network(
            n_input_dims=2 * c_dim,  # 输入维度
            n_output_dims=3,         # 输出维度（RGB三个通道）
            network_config={
                "otype": "FullyFusedMLP",  # 使用完全融合的MLP
                "activation": "ReLU",       # 隐藏层使用ReLU
                "output_activation": "Sigmoid",  # 输出层使用Sigmoid将值限制在[0,1]范围内
                "n_neurons": hidden_size,   # 隐藏层神经元数量
                "n_hidden_layers": n_blocks - 1,  # 隐藏层数量
            }
        )
        
    def extract_camera_params(self,cfg):
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        
    def init_planes(self, cfg):
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        self.coarse_planes_res = cfg['planes_res']['coarse']
        self.fine_planes_res = cfg['planes_res']['fine']

        self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
        self.fine_c_planes_res = cfg['c_planes_res']['fine']

        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        ####### Initializing Planes ############
        planes_xy, planes_xz, planes_yz = [], [], []
        c_planes_xy, c_planes_xz, c_planes_yz = [], [], []
        planes_res = [self.coarse_planes_res, self.fine_planes_res]
        c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]

        planes_dim = c_dim
        for grid_res in planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for grid_res in c_planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        self.sdf_planes_xy = planes_xy
        self.sdf_planes_xz = planes_xz
        self.sdf_planes_yz = planes_yz

        self.rgb_planes_xy = c_planes_xy
        self.rgb_planes_xz = c_planes_xz
        self.rgb_planes_yz = c_planes_yz