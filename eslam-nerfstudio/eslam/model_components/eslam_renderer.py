import torch
import torch.nn as nn
from renderer_iterator import RaySamplesDepthIterator
from nerfstudio.fields.field_utils import FieldHeadNames
class SDFRenderer(nn.Module):
        def __init__(self):
            super().__init__()
            
        def render_batch_ray(self, ray_samples, outputs, beta):
            sample_positions = ray_samples.frustums.get_positions()
            p_shape = sample_positions.shape
            
            raw = torch.cat([outputs[FieldHeadNames.RGB], outputs[FieldHeadNames.SDF].unsqueeze(-1)], dim=-1)
            raw = raw.reshape(*p_shape[:-1], -1)
            
            alpha = self.sdf2alpha(raw[..., -1], beta)
            weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), ray_samples)
                                                    , (1. - alpha + 1e-10)], -1), -1)[:, :-1]

            rendered_rgb = torch.sum(weights[..., None] * raw[..., :3], -2)
            rendered_depth = torch.sum(weights * sample_positions, -1)

            return rendered_depth, rendered_rgb, raw[..., -1], sample_positions
    
        def sdf2alpha(self, sdf, beta=10):
            return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))
        
        def render_img(self, H, W, ray_samples, ray_batch_size, sdf, beta, gt_depth=None):
            """
            Renders out depth and color images.
            Args:
                all_planes (Tuple): feature planes
                decoders (torch.nn.Module): decoders for TSDF and color.
                c2w (tensor, 4*4): camera pose.
                truncation (float): truncation distance.
                device (torch.device): device to run on.
                gt_depth (tensor, H*W): ground truth depth image.
            Returns:
                rendered_depth (tensor, H*W): rendered depth image.
                rendered_rgb (tensor, H*W*3): rendered color image.

            """
            with torch.no_grad():
                depth_list = []
                color_list = []
                gt_depth = gt_depth.reshape(-1)
                
                for ray_samples in RaySamplesDepthIterator(ray_samples, ray_batch_size, gt_depth):
                    if gt_depth is None:
                        ret = self.render_batch_ray(ray_samples, sdf, beta)
                    else:
                        gt_depth_batch = gt_depth[i:i+ray_batch_size]
                        ret = self.render_batch_ray(ray_samples, sdf, beta, gt_depth=gt_depth_batch)

                    depth, color, _, _ = ret
                    depth_list.append(depth.double())
                    color_list.append(color)

                depth = torch.cat(depth_list, dim=0)
                color = torch.cat(color_list, dim=0)

                depth = depth.reshape(H, W)
                color = color.reshape(H, W, 3)

                return depth, color