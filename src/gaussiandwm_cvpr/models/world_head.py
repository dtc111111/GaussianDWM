from __future__ import annotations

from inspect import signature
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import EulerDiscreteScheduler
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_scheduler(
    config_path: str | None = None,
) -> EulerDiscreteScheduler:
    if config_path is not None:
        config = EulerDiscreteScheduler.load_config(str(config_path))
        return EulerDiscreteScheduler.from_config(config)
    return EulerDiscreteScheduler(num_train_timesteps=1000)


def vae_decode_chunked(
    vae: nn.Module,
    latents: torch.Tensor,
    scaling_factor: float,
    decode_chunk_size: int = 14,
) -> torch.Tensor:
    latents = latents / scaling_factor
    accepts_num_frames = "num_frames" in signature(getattr(vae, "_orig_mod", vae).forward).parameters
    frames = []
    for i in range(0, latents.shape[0], decode_chunk_size):
        chunk = latents[i : i + decode_chunk_size]
        kwargs = {"num_frames": chunk.shape[0]} if accepts_num_frames else {}
        out = vae.decode(chunk, **kwargs)
        frames.append(out.sample)
    return torch.cat(frames, dim=0)


class CondFusion(nn.Module):
    def __init__(
        self,
        *,
        cond_dim: int,
        text_cond_dim: int,
        ref_cond_dim: int | None = None,
        image_encoder: nn.Module,
    ) -> None:
        super().__init__()
        ref_cond_dim = cond_dim if ref_cond_dim is None else ref_cond_dim
        self.cond_dim = int(cond_dim)
        self.text_cond_dim = int(text_cond_dim)
        self.ref_cond_dim = int(ref_cond_dim)
        if image_encoder is None:
            raise ValueError("CondFusion.image_encoder is required and cannot be None.")
        if not isinstance(image_encoder, nn.Module):
            raise TypeError(
                "CondFusion.image_encoder must be an instance of torch.nn.Module, "
                f"got {type(image_encoder)!r}."
            )
        self.image_encoder = image_encoder
        self.ref_proj = (
            nn.Identity()
            if ref_cond_dim == cond_dim
            else nn.Linear(ref_cond_dim, cond_dim, bias=False)
        )
        self.cond_proj = nn.Linear(text_cond_dim, cond_dim, bias=False)
        self.text_norm = nn.LayerNorm(cond_dim)

    def forward(
        self,
        *,
        ref_pixel_values: torch.Tensor,
        text_cond_seed: torch.Tensor,
    ) -> torch.Tensor:
        image_cond = self.image_condition(ref_pixel_values=ref_pixel_values).squeeze(1)
        text_cond = self.text_norm(self.cond_proj(text_cond_seed))
        return torch.stack([image_cond, text_cond], dim=1)

    def image_condition(self, *, ref_pixel_values: torch.Tensor) -> torch.Tensor:
        ref_features = self._encode_ref_image(ref_pixel_values)
        image_cond = self.ref_proj(ref_features)
        return image_cond.unsqueeze(1)

    def _encode_ref_image(self, ref_pixel_values: torch.Tensor) -> torch.Tensor:
        image = self._prepare_clip_inputs(ref_pixel_values)
        outputs = self.image_encoder(image)
        if hasattr(outputs, "image_embeds"):
            return outputs.image_embeds
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise TypeError(f"Unsupported image encoder output type: {type(outputs)!r}")

    @staticmethod
    def _prepare_clip_inputs(ref_pixel_values: torch.Tensor) -> torch.Tensor:
        image = ref_pixel_values.to(dtype=torch.float32)
        if image.ndim != 4:
            raise ValueError(f"ref_pixel_values must have shape [B,3,H,W], got {tuple(image.shape)}")
        image = image * 2.0 - 1.0
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        mean = image.new_tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std = image.new_tensor(_CLIP_STD).view(1, 3, 1, 1)
        return (image - mean) / std


class WorldUNetBundle(nn.Module):
    def __init__(
        self,
        *,
        unet: nn.Module,
        layout_encoder: nn.Module,
        layout_encoder_depth: nn.Module,
        vae: nn.Module,
    ) -> None:
        super().__init__()
        components = {
            "WorldUNetBundle.unet": unet,
            "WorldUNetBundle.layout_encoder": layout_encoder,
            "WorldUNetBundle.layout_encoder_depth": layout_encoder_depth,
            "WorldUNetBundle.vae": vae,
        }
        for component_name, module in components.items():
            if module is None:
                raise ValueError(f"{component_name} is required and cannot be None.")
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"{component_name} must be an instance of torch.nn.Module, got {type(module)!r}."
                )
        self.unet = unet
        self.layout_encoder = layout_encoder
        self.layout_encoder_depth = layout_encoder_depth
        self.vae = vae

    @property
    def latent_channels(self) -> int:
        return int(self.unet.config.out_channels)


@dataclass
class WorldHeadOutput:
    loss: torch.Tensor
    pred_noise: torch.Tensor


class WorldDiffusionHead(nn.Module):
    def __init__(
        self,
        *,
        bundle: WorldUNetBundle,
        scheduler_config_path: str | None = None,
        conditioning_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if bundle is None:
            raise ValueError("WorldDiffusionHead.bundle is required and cannot be None.")
        if not isinstance(bundle, WorldUNetBundle):
            raise TypeError(
                "WorldDiffusionHead.bundle must be an instance of WorldUNetBundle, "
                f"got {type(bundle)!r}."
            )
        self.bundle = bundle
        self.conditioning_dropout_prob = float(conditioning_dropout_prob)
        self.scheduler_config_path = (
            str(scheduler_config_path) if scheduler_config_path is not None else None
        )
        # Scheduler is only used during inference (EulerDiscreteScheduler).
        # Training uses EDM-style manual noise injection.
        self.scheduler = build_scheduler(config_path=self.scheduler_config_path)

    def _require_bundle(self) -> None:
        if self.bundle is None:
            raise RuntimeError(
                "WorldDiffusionHead.bundle must be a concrete WorldUNetBundle before calling forward()."
            )

    @property
    def latent_channels(self) -> int:
        self._require_bundle()
        return self.bundle.latent_channels

    @property
    def vae_scaling_factor(self) -> float:
        self._require_bundle()
        return float(self.bundle.vae.config.scaling_factor)

    def freeze_unet(self) -> None:
        """Freeze UNet and VAE for Stage C (cond alignment warmup)."""
        self._require_bundle()
        self.bundle.unet.requires_grad_(False)
        self.bundle.vae.requires_grad_(False)

    def export_config(self) -> dict[str, object]:
        return {
            "conditioning_dropout_prob": self.conditioning_dropout_prob,
            "scheduler_config": dict(self.scheduler.config),
        }

    @staticmethod
    def _module_floating_dtype(module: nn.Module, *, fallback: torch.dtype) -> torch.dtype:
        for tensor in list(module.parameters()) + list(module.buffers()):
            if torch.is_floating_point(tensor):
                return tensor.dtype
        return fallback

    def forward(
        self,
        *,
        target_latents: dict[str, torch.Tensor],
        pseudo_pixel_values: torch.Tensor,
        pseudo_depth_values: torch.Tensor,
        cond_embeddings: torch.Tensor,
        world_meta: dict[str, object],
        added_time_ids: torch.Tensor | None = None,
        scaling_factor: float | None = None,
    ) -> WorldHeadOutput:
        self._require_bundle()
        rgb_latents = self._ensure_video_latents(target_latents["rgb_latents"])
        depth_latents = self._ensure_video_latents(
            target_latents.get("depth_latents", target_latents["rgb_latents"])
        )
        batch_size, num_frames = rgb_latents.shape[:2]
        sf = scaling_factor if scaling_factor is not None else self.vae_scaling_factor

        # Sample noise_aug_strength once per batch (scalar), matching DiST line 1861:
        # noise_aug_strength = math.exp(random.normalvariate(mu=-3, sigma=0.5))
        if self.training:
            noise_aug_strength_scalar = math.exp(random.normalvariate(-3, 0.5))
        else:
            noise_aug_strength_scalar = 0.02  # inference default

        if added_time_ids is None:
            added_time_ids = self._build_added_time_ids(
                world_meta,
                noise_aug_strength=noise_aug_strength_scalar,
                batch_size=batch_size,
                device=rgb_latents.device,
                dtype=rgb_latents.dtype,
            )

        encoder_hidden_states = self._expand_encoder_hidden_states(
            cond_embeddings, num_frames=num_frames
        )

        # Language dropout (DiST main loop lines 1845-1852): zero out encoder_hidden_states
        if self.training and self.conditioning_dropout_prob > 0:
            lang_mask = (
                torch.rand(batch_size, device=rgb_latents.device) < self.conditioning_dropout_prob
            )
            lang_mask = lang_mask.reshape(batch_size, 1, 1)
            # encoder_hidden_states is [B*T, 2, C]; expand mask to [B*T, 1, 1]
            lang_mask_expanded = lang_mask.repeat_interleave(num_frames, dim=0)
            encoder_hidden_states = torch.where(
                lang_mask_expanded, torch.zeros_like(encoder_hidden_states), encoder_hidden_states
            )

        layout_cond, depth_cond = self._encode_layout_conditions(
            pseudo_pixel_values=pseudo_pixel_values,
            pseudo_depth_values=pseudo_depth_values,
            batch_size=batch_size,
            num_frames=num_frames,
            device=rgb_latents.device,
            dtype=rgb_latents.dtype,
            noise_aug_strength=noise_aug_strength_scalar if self.training else None,
        )

        # Joint RGB+depth: concat on channel dim -> [B, T, 8, H, W] (DiST line 1773)
        latents = torch.cat([rgb_latents, depth_latents], dim=2)

        pred_noise, loss = self._edm_diffusion_loss(
            latents,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            layout_cond=layout_cond,
            depth_cond=depth_cond,
            scaling_factor=sf,
        )
        return WorldHeadOutput(loss=loss, pred_noise=pred_noise)

    @torch.no_grad()
    def generate(
        self,
        *,
        pseudo_pixel_values: torch.Tensor,
        pseudo_depth_values: torch.Tensor,
        cond_embeddings: torch.Tensor,
        world_meta: dict[str, object],
        num_inference_steps: int,
        added_time_ids: torch.Tensor | None = None,
        guidance_scale: float = 2.0,
        noise_aug_strength: float = 0.02,
        scaling_factor: float | None = None,
    ) -> dict[str, torch.Tensor]:
        self._require_bundle()
        batch_size = pseudo_pixel_values.shape[0]
        num_frames = pseudo_pixel_values.shape[1] if pseudo_pixel_values.ndim == 5 else 1
        sf = scaling_factor if scaling_factor is not None else self.vae_scaling_factor

        layout_cond, depth_cond = self._encode_layout_conditions(
            pseudo_pixel_values=pseudo_pixel_values,
            pseudo_depth_values=pseudo_depth_values,
            batch_size=batch_size,
            num_frames=num_frames,
            device=cond_embeddings.device,
            dtype=cond_embeddings.dtype,
            noise_aug_strength=float(noise_aug_strength),
        )
        if added_time_ids is None:
            added_time_ids = self._build_added_time_ids(
                world_meta,
                noise_aug_strength=float(noise_aug_strength),
                batch_size=batch_size,
                device=cond_embeddings.device,
                dtype=cond_embeddings.dtype,
            )
        encoder_hidden_states = self._expand_encoder_hidden_states(
            cond_embeddings, num_frames=num_frames
        )

        latent_height, latent_width = layout_cond.shape[-2:]
        latents = self._denoise_latents(
            batch_size=batch_size,
            num_frames=num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
            layout_cond=layout_cond,
            depth_cond=depth_cond,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            scaling_factor=sf,
        )

        # Split 8-channel output: [:4] = RGB, [4:] = depth (DiST pipeline)
        rgb_latents = latents[:, :, :4]
        depth_latents = latents[:, :, 4:]

        decoded_rgb = self._decode_latents(rgb_latents, scaling_factor=sf)
        decoded_depth = self._decode_latents(depth_latents, scaling_factor=sf)
        depth_map = decoded_depth.mean(dim=2, keepdim=True)
        return {
            "rgb": decoded_rgb,
            "depth": depth_map,
        }

    def _edm_diffusion_loss(
        self,
        latents: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        layout_cond: torch.Tensor,
        depth_cond: torch.Tensor,
        scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EDM-style training loss matching DiST (train_svd_nus_mm_qwen3.py lines 1760-1893)."""
        batch_size = latents.shape[0]
        num_frames = latents.shape[1]

        # EDM noise sampling (DiST lines 1760-1763)
        P_std = 1.6
        P_mean = 0.7
        rnd_normal = torch.randn([batch_size, 1, 1, 1, 1], device=latents.device)
        sigma = (rnd_normal * P_std + P_mean).exp()

        # EDM preconditioning coefficients (DiST lines 1765-1770)
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out = -sigma / (sigma**2 + 1.0) ** 0.5
        c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
        c_noise = (sigma.log() / 4.0).reshape([batch_size])
        loss_weight = (sigma**2 + 1.0) / sigma**2
        unet_dtype = self._module_floating_dtype(self.bundle.unet, fallback=latents.dtype)
        # `scaling_factor` applies to target VAE latents. DiST feeds layout/depth
        # encoder outputs to the UNet at their raw scale.

        # Add noise to joint RGB+depth latents (DiST line 1774)
        noisy_latents = latents + torch.randn_like(latents) * sigma

        # Spatial dropout on layout conditions (DiST PackedModule lines 951-961):
        # per-sample mask, same mask for RGB and depth condition.
        if self.training and self.conditioning_dropout_prob > 0:
            rand_p = torch.rand(batch_size, device=layout_cond.device)
            spatial_mask = (rand_p < self.conditioning_dropout_prob).reshape(batch_size, 1, 1, 1, 1)
            layout_cond = torch.where(spatial_mask, torch.zeros_like(layout_cond), layout_cond)
            depth_cond = torch.where(spatial_mask, torch.zeros_like(depth_cond), depth_cond)

        # Assemble UNet input: [B, T, 8+4+4, H, W] (DiST PackedModule line 967)
        model_input = torch.cat([
            c_in * noisy_latents,
            layout_cond,
            depth_cond,
        ], dim=2).to(dtype=unet_dtype)
        if encoder_hidden_states.dtype != unet_dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=unet_dtype)

        model_pred = self.bundle.unet(
            model_input,
            c_noise,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
        ).sample
        if model_pred.dtype != latents.dtype:
            model_pred = model_pred.to(dtype=latents.dtype)

        # EDM denoised prediction loss (DiST lines 1891-1893)
        pred_final = c_out * model_pred + c_skip * noisy_latents
        loss = ((pred_final - latents) ** 2 * loss_weight).mean()

        return model_pred, loss

    def _denoise_latents(
        self,
        *,
        batch_size: int,
        num_frames: int,
        latent_height: int,
        latent_width: int,
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        layout_cond: torch.Tensor,
        depth_cond: torch.Tensor,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """Iterative denoising with EulerDiscreteScheduler (matches DiST pipeline)."""
        out_channels = self.bundle.unet.config.out_channels
        self.scheduler.set_timesteps(num_inference_steps, device=encoder_hidden_states.device)
        timesteps = self.scheduler.timesteps

        sample = torch.randn(
            batch_size,
            num_frames,
            out_channels,
            latent_height,
            latent_width,
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )
        sample = sample * self.scheduler.init_noise_sigma

        do_cfg = guidance_scale > 1.0

        for timestep in timesteps:
            scaled_sample = self.scheduler.scale_model_input(sample, timestep)

            if do_cfg:
                # Keep unconditional CFG spatial conditions zeroed, and feed
                # conditional layout/depth latents at the same raw scale as DiST.
                uncond_input = torch.cat([
                    scaled_sample,
                    torch.zeros_like(layout_cond),
                    torch.zeros_like(depth_cond),
                ], dim=2)
                cond_input = torch.cat([
                    scaled_sample,
                    layout_cond,
                    depth_cond,
                ], dim=2)
                model_input = torch.cat([uncond_input, cond_input])
                enc_hidden = torch.cat([encoder_hidden_states] * 2)
                time_ids = torch.cat([added_time_ids] * 2)
            else:
                model_input = torch.cat([
                    scaled_sample,
                    layout_cond,
                    depth_cond,
                ], dim=2)
                enc_hidden = encoder_hidden_states
                time_ids = added_time_ids

            model_output = self.bundle.unet(
                model_input,
                timestep,
                encoder_hidden_states=enc_hidden,
                added_time_ids=time_ids,
            ).sample

            if do_cfg:
                noise_uncond, noise_cond = model_output.chunk(2)
                model_output = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            sample = self.scheduler.step(model_output, timestep, sample).prev_sample

        return sample

    @staticmethod
    def _build_added_time_ids(
        world_meta: dict[str, object],
        *,
        noise_aug_strength: float,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build added_time_ids [B, 3] = [fps, motion_bucket_id, noise_aug_strength].

        `world_meta` comes from the world collator and carries one fps and one
        motion bucket id per sample.  Keeping these values data-driven makes
        annotation changes visible to the world UNet instead of silently using
        stale DiST defaults.
        """
        if "fps" not in world_meta:
            raise KeyError("world_meta must contain fps for world added_time_ids.")
        if "motion_bucket_id" not in world_meta:
            raise KeyError("world_meta must contain motion_bucket_id for world added_time_ids.")

        fps = torch.as_tensor(world_meta["fps"], device=device, dtype=dtype).reshape(-1)
        motion_bucket_id = torch.as_tensor(
            world_meta["motion_bucket_id"],
            device=device,
            dtype=dtype,
        ).reshape(-1)
        if fps.numel() != batch_size:
            raise ValueError(
                f"world_meta['fps'] must contain {batch_size} values, got {fps.numel()}."
            )
        if motion_bucket_id.numel() != batch_size:
            raise ValueError(
                "world_meta['motion_bucket_id'] must contain "
                f"{batch_size} values, got {motion_bucket_id.numel()}."
            )

        noise_aug_strengths = torch.full(
            (batch_size,),
            float(noise_aug_strength),
            device=device,
            dtype=dtype,
        )
        return torch.stack([fps, motion_bucket_id, noise_aug_strengths], dim=1)

    def _encode_layout_conditions(
        self,
        *,
        pseudo_pixel_values: torch.Tensor,
        pseudo_depth_values: torch.Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        dtype: torch.dtype,
        noise_aug_strength: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode layout conditions for all T frames independently.

        Matches DiST (train_svd_nus_mm_qwen3.py lines 1862-1867):
          first_frame_image = batch["pseudo_pixel_values"].squeeze(0)  # [T, C, H, W] — ALL frames
          first_frame_image = first_frame_image + noise_aug_strength * randn_like(...)
          layout_encoder processes all T frames independently.

        The world adapter is responsible for producing DiST-aligned tensors:
        pseudo RGB and pseudo depth must both be 3-channel normalized inputs.
        """
        # pseudo_pixel_values: [B, T, C, H, W]
        if pseudo_pixel_values.ndim == 5:
            all_rgb = pseudo_pixel_values.flatten(0, 1).to(device=device, dtype=dtype)  # [B*T, C, H, W]
        elif pseudo_pixel_values.ndim == 4:
            all_rgb = pseudo_pixel_values.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported pseudo_pixel_values shape: {tuple(pseudo_pixel_values.shape)}")
        if all_rgb.ndim != 4 or all_rgb.shape[1] != 3:
            raise ValueError(
                "world adapter must produce DiST-aligned pseudo RGB tensors with shape [B*T, 3, H, W]; "
                f"got {tuple(all_rgb.shape)}"
            )

        # pseudo_depth_values: [B, T, C, H, W]
        if pseudo_depth_values.ndim == 5:
            all_depth = pseudo_depth_values.flatten(0, 1).to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported pseudo_depth_values shape: {tuple(pseudo_depth_values.shape)}")
        if all_depth.ndim != 4 or all_depth.shape[1] != 3:
            raise ValueError(
                "world adapter must produce DiST-aligned pseudo depth tensors with shape [B*T, 3, H, W]; "
                f"got {tuple(all_depth.shape)}"
            )

        # Add noise augmentation during training (DiST lines 1863, 1867)
        if noise_aug_strength is not None:
            all_rgb = all_rgb + noise_aug_strength * torch.randn_like(all_rgb)
            all_depth = all_depth + noise_aug_strength * torch.randn_like(all_depth)

        # Encode all B*T frames through layout encoders independently
        layout_cond = self.bundle.layout_encoder(box_info=all_rgb)        # [B*T, C', H', W']
        depth_cond = self.bundle.layout_encoder_depth(box_info=all_depth)  # [B*T, C', H', W']

        # Reshape back to [B, T, C', H', W']; DiST feeds these raw layout latents to the UNet.
        layout_cond = layout_cond.reshape(batch_size, num_frames, *layout_cond.shape[1:])
        depth_cond = depth_cond.reshape(batch_size, num_frames, *depth_cond.shape[1:])

        return layout_cond, depth_cond

    def _expand_encoder_hidden_states(
        self, cond_embeddings: torch.Tensor, *, num_frames: int
    ) -> torch.Tensor:
        """Expand [B, S, C] → [B*T, S, C] for UNet cross-attention.

        repeat_interleave(num_frames, dim=0) produces [b0,b0,...,b1,b1,...] which
        correctly aligns with the UNet's internal flatten(0,1) of [B, T, ...].
        """
        if cond_embeddings.ndim != 3:
            raise ValueError(
                f"cond_embeddings must have shape [B,S,C], got {tuple(cond_embeddings.shape)}"
            )
        if num_frames == 1:
            return cond_embeddings
        return cond_embeddings.repeat_interleave(num_frames, dim=0)

    @staticmethod
    def _ensure_video_latents(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 5:
            return latents
        if latents.ndim == 4:
            return latents.unsqueeze(1)
        raise ValueError(
            f"target latents must have shape [B,C,H,W] or [B,T,C,H,W], got {tuple(latents.shape)}"
        )

    def _decode_latents(
        self, latents: torch.Tensor, *, scaling_factor: float | None = None
    ) -> torch.Tensor:
        """Decode video latents with chunked temporal decoding.

        Matches DiST pipeline decode_latents (pipeline_stable_video_diffusion_custom_mm.py:1093-1125):
        - Flatten [B, T, C, H, W] → [B*T, C, H, W]
        - Chunked decode with decode_chunk_size=14 to avoid OOM
        - Conditionally pass num_frames (AutoencoderKLTemporalDecoder requires it)
        - Reshape back to [B, T, C_out, H_out, W_out]
        """
        batch_size, num_frames = latents.shape[:2]
        flat_latents = latents.flatten(0, 1)
        sf = scaling_factor if scaling_factor is not None else self.vae_scaling_factor
        with torch.no_grad():
            decoded = vae_decode_chunked(self.bundle.vae, flat_latents, scaling_factor=sf)
        decoded = decoded.reshape(batch_size, num_frames, *decoded.shape[1:])
        return decoded
