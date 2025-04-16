from nerfstudio.cameras.rays import RaySamples

class RaySamplesIterator:
    def __init__(self, ray_samples: RaySamples, batch_size: int):
        self.ray_samples = ray_samples
        self.batch_size = batch_size
        self.total_samples = len(ray_samples.frustums.origins)
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self) -> RaySamples:
        if self.current_idx >= self.total_samples:
            raise StopIteration
        
        # 计算当前批次的大小
        current_batch_size = min(self.batch_size, self.total_samples - self.current_idx)
        
        # 直接使用切片获取当前批次的RaySamples
        batch_ray_samples = self.ray_samples[self.current_idx:self.current_idx + current_batch_size]
        
        # 更新索引
        self.current_idx += current_batch_size
        
        return batch_ray_samples
    
class RaySamplesDepthIterator(RaySamplesIterator):
    def __init__(self, ray_samples: RaySamples, batch_size: int, gt_depth: torch.Tensor):
        super().__init__(ray_samples, batch_size)
        self.gt_depth = gt_depth

    def __next__(self) -> RaySamples:
        ray_samples = super().__next__()
        gt_depth = self.gt_depth[self.current_idx:self.current_idx + current_batch_size]
        return ray_samples, gt_depth

