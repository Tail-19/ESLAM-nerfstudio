import torch
def sdf_losses(self, sdf, ray_samples, gt_depth):
    """
    Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

    Args:
    - self: instance of the class containing this method
    - sdf: a tensor of shape (R, N) representing the SDF values
    - sample_positions: a tensor of shape (R, N) representing the depth values
    - gt_depth: a tensor of shape (R,) containing the ground truth depth values

    Returns:
    - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
    """
    sample_positions = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
    front_mask = torch.where(sample_positions < (gt_depth[:, None] - self.truncation),
                                torch.ones_like(sample_positions), torch.zeros_like(sample_positions)).bool()

    back_mask = torch.where(sample_positions > (gt_depth[:, None] + self.truncation),
                            torch.ones_like(sample_positions), torch.zeros_like(sample_positions)).bool()

    center_mask = torch.where((sample_positions > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                (sample_positions < (gt_depth[:, None] + 0.4 * self.truncation)),
                                torch.ones_like(sample_positions), torch.zeros_like(sample_positions)).bool()

    tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

    fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
    center_loss = torch.mean(torch.square(
        (sample_positions + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(sample_positions.shape)[center_mask]))
    tail_loss = torch.mean(torch.square(
        (sample_positions + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(sample_positions.shape)[tail_mask]))

    sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

    return sdf_losses

def color_loss(self, color, gt_color, depth_mask):
    ## Color Loss
    return torch.square(gt_color - color)[depth_mask].mean()

def depth_loss(self, depth, gt_depth, depth_mask):
    ### Depth loss
    return torch.square(gt_depth[depth_mask] - depth[depth_mask]).mean()
