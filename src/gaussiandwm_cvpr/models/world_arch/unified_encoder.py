import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .convnext import convnext_supertiny


class LayoutCondEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, img_shape=(960, 384), out_dim=1024, in_chans=3, condition_list=None):
        super().__init__()
        self.resize_input_width = img_shape[0]
        self.resize_input_height = img_shape[1]
        self.out_dim = out_dim
        self.conditions = condition_list
        self.down_factor = 32
        self.convnext_tiny_backbone = convnext_supertiny(
            in_chans=in_chans, depths=[3, 3, 9, 3], dims=[32, 64, 128, 256]
        )

    def forward(self, box_info=None):
        if box_info is None:
            raise ValueError("LayoutCondEncoder requires box_info input.")
        return self.convnext_tiny_backbone(box_info)
