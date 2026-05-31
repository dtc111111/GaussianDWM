from __future__ import annotations

import torch
import torch.nn as nn


class AEDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)


class GaussAligner(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.ae_decoder = AEDecoder()
        self.language_projector = nn.Linear(512, hidden_dim)
        self.positional_projector = nn.Linear(2048, hidden_dim)
        self.scale_projector = nn.Linear(3, hidden_dim)
        self.opacity_projector = nn.Linear(1, hidden_dim)
        self.rotation_projector = nn.Linear(4, hidden_dim)

        self.w_xyz = nn.Linear(1, 1, bias=False)
        self.w_scaling = nn.Linear(1, 1, bias=False)
        self.w_rotation = nn.Linear(1, 1, bias=False)
        self.w_opacity = nn.Linear(1, 1, bias=False)
        self.w_language = nn.Linear(1, 1, bias=False)

        self.aligner = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.B = nn.Linear(3, 1024, bias=False)

    def positional_embedding(self, xyz: torch.Tensor) -> torch.Tensor:
        loc_b = self.B(xyz)
        loc_b_2pi = 2 * torch.pi * loc_b
        return torch.cat([torch.sin(loc_b_2pi), torch.cos(loc_b_2pi)], dim=-1)

    def decode_language_clip_features(self, gauss_values: torch.Tensor) -> torch.Tensor:
        """Decode Gaussian language coefficients into CLIP feature space."""
        if gauss_values.ndim != 3 or gauss_values.shape[-1] != 14:
            raise ValueError(f"gauss_values must have shape [B,N,14], got {gauss_values.shape}")
        module_dtype = self.B.weight.dtype
        if gauss_values.dtype != module_dtype:
            gauss_values = gauss_values.to(module_dtype)
        feat_language = gauss_values[:, :, 11:]
        return self.ae_decoder(feat_language)

    def forward(self, gauss_values: torch.Tensor) -> torch.Tensor:
        if gauss_values.ndim != 3 or gauss_values.shape[-1] != 14:
            raise ValueError(f"gauss_values must have shape [B,N,14], got {gauss_values.shape}")
        module_dtype = self.B.weight.dtype
        if gauss_values.dtype != module_dtype:
            gauss_values = gauss_values.to(module_dtype)

        feat_xyz = gauss_values[:, :, :3]
        feat_scaling = gauss_values[:, :, 3:6]
        feat_rotation = gauss_values[:, :, 6:10]
        feat_opacity = gauss_values[:, :, 10:11]
        feat_language = gauss_values[:, :, 11:]

        feat_xyz = self.positional_projector(self.positional_embedding(feat_xyz))
        feat_scaling = self.scale_projector(feat_scaling)
        feat_rotation = self.rotation_projector(feat_rotation)
        feat_opacity = self.opacity_projector(feat_opacity)
        feat_language = self.language_projector(self.ae_decoder(feat_language))

        weighted = (
            self.w_xyz(torch.ones_like(feat_xyz[:, :, :1])) * feat_xyz
            + self.w_scaling(torch.ones_like(feat_scaling[:, :, :1])) * feat_scaling
            + self.w_rotation(torch.ones_like(feat_rotation[:, :, :1])) * feat_rotation
            + self.w_opacity(torch.ones_like(feat_opacity[:, :, :1])) * feat_opacity
            + self.w_language(torch.ones_like(feat_language[:, :, :1])) * feat_language
        )
        return weighted + self.aligner(weighted)
