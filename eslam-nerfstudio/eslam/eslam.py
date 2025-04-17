"""
ESLAM Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.models.base_model import Model, ModelConfig  # for custom Model

from eslam_field import ESLAMField
from model_components.eslam_renderer import ESLAMRenderer
from model_components.eslam_losses import ESLAMLosses
from model_components.eslam_sampler import ESLAMSampler

@dataclass
class ESLAMModelConfig(ModelConfig):
    """ESLAM Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: ESLAMModel)


class ESLAMModel(Model):
    """ESLAM Model."""

    config: ESLAMModelConfig

    def populate_modules(self):
        """Set the fields and modules."""

        # Ray Samplers
        self.ray_sampler = ESLAMSampler()
        
        #Camera Optimizer
        self.camera_optimizer = CameraOptimizer(
            num_cameras=self.num_train_cameras,
            config=self.config,
        )

        # Fields
        self.field = ESLAMField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_images,
            cfg=self.config,
        )

        # Renderers
        self.renderer = ESLAMRenderer()

        # Losses
        self.losses = ESLAMLosses()

        # Metrics

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimizer your model components."""
        param_groups = {}
        param_groups["field"] = list(self.field.parameters())
        param_groups["camera_optimizer"] = list(self.camera_optimizer.parameters())
        return param_groups
    
    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
        """Returns the training callbacks, such as updating a density grid for Instant NGP."""

    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples = self.ray_sampler(ray_bundle)
        field_outputs = self.field(ray_samples)
        outputs = self.renderer(ray_samples, field_outputs)
        return outputs
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        metrics_dict["psnr"] = self.losses.psnr(outputs, batch)
        return metrics_dict
        
    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        
        """Returns a dictionary of losses to be summed which will be your loss."""

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""

    # TODO: Override any potential functions/methods to implement your own method
    # or subclass from "Model" and define all mandatory fields.
