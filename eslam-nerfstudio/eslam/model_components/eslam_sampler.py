import torch
from nerfstudio.model_components.ray_samplers import Sampler
from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from typing import Any, Callable, List, Optional, Protocol, Tuple, Union
from eslam.common import normalize_3d_coordinate, sample_pdf
class ESLAMSampler(Sampler):
    def __init__(
        self,
        num_samples: Optional[int] = None,
        num_stratified: Optional[int] = None,
        num_importance: Optional[int] = None,
        perturb: Optional[bool] = None,
    ) -> None:
        super().__init__(num_samples=num_samples)
        assert num_stratified is not None
        assert num_importance is not None
        assert perturb is not None
        self.num_stratified = num_stratified
        self.num_importance = num_importance
        self.perturb = perturb
        
    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand
    
    # 将sdf转换为alpha值, beta是sdf的系数, 用于控制sdf的平滑程度
    def sdf2alpha(self, sdf, beta=10):
        return 1. - torch.exp(-beta * torch.sigmoid(-sdf * beta))
    
    def generate_ray_samples(
            self,
            truncation: float,
            gt_depth: Optional[torch.Tensor] = None,
            ray_bundle: Optional[RayBundle] = None,
            field: Optional[Callable] = None,
        ) -> RaySamples:
            """Generates position samples according to spacing function.

            Args:
                ray_bundle: Rays to generate samples for
                num_samples: Number of samples per ray

            Returns:
                Positions and deltas for samples along a ray
            """
            assert ray_bundle is not None
            assert ray_bundle.nears is not None
            assert ray_bundle.fars is not None

            num_rays = ray_bundle.origins.shape[0]
            
            depth_vals = torch.empty([num_rays, self.num_stratified + self.num_importance], device=ray_bundle.origins.device)
            t_vals_along_ray = torch.linspace(0., 1., steps=self.num_stratified, device=ray_bundle.origins.device)
            t_vals_surface = torch.linspace(0., 1., steps=self.num_importance, device=ray_bundle.origins.device)

            ### pixels with gt depth:
            # 获取gt_depth中大于0的值, 并reshape为(num_rays, 1)
            gt_depth = gt_depth.reshape(-1, 1)
            gt_mask = (gt_depth > 0).squeeze()
            gt_nonezero = gt_depth[gt_mask]
            
            # 获取对应的nears值
            valid_nears = ray_bundle.nears[gt_mask].unsqueeze(-1).expand(-1, self.num_stratified)
            
            ## Sampling points around the gt depth (surface)
            gt_depth_surface = gt_nonezero.expand(-1, self.num_importance)
            t_vals_surface = t_vals_surface.unsqueeze(0).expand(gt_depth_surface.shape[0], -1)
            depth_vals_surface = gt_depth_surface - (1.5 * truncation) + (3 * truncation * t_vals_surface)
                        
            gt_depth_along_ray = gt_nonezero.expand(-1, self.num_stratified)
            t_vals_along_ray = t_vals_along_ray.unsqueeze(0).expand(gt_depth_along_ray.shape[0], -1)
            depth_vals_ray = valid_nears + 1.2 * gt_depth_along_ray * t_vals_along_ray

            depth_vals_nonzero, _ = torch.sort(torch.cat([depth_vals_ray, depth_vals_surface], dim=-1), dim=-1)
            if self.perturb:
                depth_vals_nonzero = self.perturbation(depth_vals_nonzero)
            depth_vals[gt_mask] = depth_vals_nonzero

            ### pixels without gt depth (importance sampling):
            if not gt_mask.all():
                with torch.no_grad():
                    rays_origin = ray_bundle.origins[~gt_mask].detach().unsqueeze(-1)  # (
                    rays_direction = ray_bundle.directions[~gt_mask].detach().unsqueeze(-1)
                    t = (self.bound.unsqueeze(0) - rays_origin)/rays_direction  # (N, 3, 2)
                    far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    far_bb = far_bb.unsqueeze(-1)
                    far_bb += 0.01

                    depth_vals_ray = ray_bundle.nears[~gt_mask] * (1. - t_vals_along_ray) + far_bb * t_vals_along_ray
                    if self.perturb:
                        depth = self.perturbation(depth_vals_ray)
                    samples_uni = rays_origin.unsqueeze(1) + rays_direction.unsqueeze(1) * depth_vals_ray.unsqueeze(-1)  # [n_rays, n_stratified, 3]
                    samples_uni_norm = normalize_3d_coordinate(samples_uni.clone(), self.bound) 
                    
                    #TODO: 在field中记得实现
                    sdf_uni = field.get_raw_sdf(samples_uni_norm)
                    sdf_uni = sdf_uni.reshape(*samples_uni.shape[0:2])
                    alpha_uni = self.sdf2alpha(sdf_uni, field.beta)
                    weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=ray_bundle.origins.device)
                                                            , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]

                    depth_vals_ray_mid = .5 * (depth_vals_ray[..., 1:] + depth_vals_ray[..., :-1])
                    depth_samples_importance = sample_pdf(depth_vals_ray_mid, weights_uni[..., 1:-1], self.num_importance, det=False, device=ray_bundle.origins.device)
                    depth_vals_ray, ind = torch.sort(torch.cat([depth_vals_ray, depth_samples_importance], -1), -1)
                    depth_vals[~gt_mask] = depth_vals_ray
        
            bin_starts = depth_vals[...,:-1] 
            bin_ends = depth_vals[..., 1:]
            
            ray_samples = ray_bundle.get_ray_samples(bin_starts, bin_ends)
            return ray_samples
        

    def forward(self, *args, **kwargs) -> Any:
        """Generate ray samples"""
        return self.generate_ray_samples(*args, **kwargs)