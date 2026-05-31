from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gaussiandwm_cvpr.data.dataset import GaussianDWMSample, IGNORE_INDEX


@dataclass
class GaussianDWMCollator:
    tokenizer: Any
    mode: str = "train"

    def __call__(self, samples: list[GaussianDWMSample]) -> dict[str, Any]:
        if not samples:
            raise ValueError("samples must not be empty")

        task = samples[0].task
        if any(sample.task != task for sample in samples):
            raise ValueError("Collator received a mixed task batch")

        batch: dict[str, Any] = {
            "task": task,
            "sample_uids": [sample.sample_uid for sample in samples],
            "dataset_names": [sample.dataset_name for sample in samples],
            "qwen_inputs": self._collate_qwen_inputs(samples),
            "meta": self._collate_meta(samples),
        }

        if task == "qa":
            has_labels = ["labels" in sample.targets for sample in samples]
            if any(has_labels):
                if not all(has_labels):
                    raise ValueError("QA batch mixes labeled and unlabeled samples")
                batch["labels"] = self._pad_1d_tensors(
                    [sample.targets["labels"] for sample in samples],
                    pad_value=IGNORE_INDEX,
                    padding_side="right",
                )
            return batch

        if task == "world":
            batch["ref_pixel_values"] = torch.stack(
                [sample.task_inputs["ref_pixel_values"] for sample in samples],
                dim=0,
            )
            batch["pseudo_pixel_values"] = torch.stack(
                [sample.task_inputs["pseudo_pixel_values"] for sample in samples],
                dim=0,
            )
            batch["pseudo_depth_values"] = torch.stack(
                [sample.task_inputs["pseudo_depth_values"] for sample in samples],
                dim=0,
            )
            has_target_latents = ["target_latents" in sample.targets for sample in samples]
            if any(has_target_latents):
                if not all(has_target_latents):
                    raise ValueError("World batch mixes samples with and without target_latents")
                batch["target_latents"] = {
                    "rgb_latents": torch.stack(
                        [sample.targets["target_latents"]["rgb_latents"] for sample in samples],
                        dim=0,
                    ),
                    "depth_latents": torch.stack(
                        [sample.targets["target_latents"]["depth_latents"] for sample in samples],
                        dim=0,
                    ),
                }
            batch["world_meta"] = {
                "fps": [int(sample.meta["fps"]) for sample in samples],
                "motion_bucket_id": [int(sample.meta["motion_bucket_id"]) for sample in samples],
                "pseudo_source_name": [sample.meta["pseudo_source_name"] for sample in samples],
                "pseudo_source_kind": [sample.meta["pseudo_source_kind"] for sample in samples],
            }
            return batch

        raise ValueError(f"Unsupported task={task!r}")

    def _collate_qwen_inputs(self, samples: list[GaussianDWMSample]) -> dict[str, Any]:
        padding_side = "left" if self.mode == "infer" else "right"
        qwen_inputs: dict[str, Any] = {
            "input_ids": self._pad_1d_tensors(
                [sample.qwen_inputs["input_ids"] for sample in samples],
                pad_value=self.tokenizer.pad_token_id,
                padding_side=padding_side,
            ),
            "attention_mask": self._pad_1d_tensors(
                [sample.qwen_inputs["attention_mask"] for sample in samples],
                pad_value=0,
                padding_side=padding_side,
            ),
        }

        if any("pixel_values" in sample.qwen_inputs for sample in samples):
            all_image_outputs = all(
                "pixel_values" in sample.qwen_inputs and "image_grid_thw" in sample.qwen_inputs
                for sample in samples
            )
            if not all_image_outputs:
                raise ValueError("Batch mixes samples with and without image processor outputs")
            qwen_inputs["pixel_values"] = torch.cat(
                [sample.qwen_inputs["pixel_values"] for sample in samples],
                dim=0,
            )
            qwen_inputs["image_grid_thw"] = torch.cat(
                [sample.qwen_inputs["image_grid_thw"] for sample in samples],
                dim=0,
            )

        gauss_values = [sample.qwen_inputs.get("coarse_gauss_values") for sample in samples]
        if any(value is not None for value in gauss_values):
            if not all(isinstance(value, torch.Tensor) for value in gauss_values):
                raise ValueError("Batch mixes samples with and without coarse_gauss_values")
            coarse_gauss_values, coarse_gauss_mask = self._pad_gaussian_values(gauss_values)  # type: ignore[arg-type]
            qwen_inputs["coarse_gauss_values"] = coarse_gauss_values
            qwen_inputs["coarse_gauss_mask"] = coarse_gauss_mask

        clip_text_embeds = [sample.qwen_inputs.get("clip_text_embed") for sample in samples]
        if any(value is not None for value in clip_text_embeds):
            if not all(isinstance(value, torch.Tensor) for value in clip_text_embeds):
                raise ValueError("Batch mixes samples with and without clip_text_embed")
            for value in clip_text_embeds:
                if value.shape != (512,):
                    raise ValueError(f"clip_text_embed must have shape (512,), got {tuple(value.shape)}")
            qwen_inputs["clip_text_embed"] = torch.stack(clip_text_embeds, dim=0)  # type: ignore[arg-type]

        return qwen_inputs

    @staticmethod
    def _pad_gaussian_values(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_k = max(value.shape[0] for value in values)
        feature_dim = values[0].shape[-1]
        padded = []
        masks = []
        for value in values:
            if value.ndim != 2 or value.shape[-1] != feature_dim:
                raise ValueError(f"Gaussian candidates must share shape [K,{feature_dim}], got {tuple(value.shape)}")
            pad_k = max_k - value.shape[0]
            if pad_k > 0:
                pad = torch.zeros((pad_k, feature_dim), dtype=value.dtype)
                padded.append(torch.cat([value, pad], dim=0))
                masks.append(
                    torch.cat(
                        [
                            torch.ones(value.shape[0], dtype=torch.bool),
                            torch.zeros(pad_k, dtype=torch.bool),
                        ],
                        dim=0,
                    )
                )
            else:
                padded.append(value)
                masks.append(torch.ones(value.shape[0], dtype=torch.bool))
        return torch.stack(padded, dim=0), torch.stack(masks, dim=0)

    @staticmethod
    def _pad_1d_tensors(
        values: list[torch.Tensor],
        *,
        pad_value: int,
        padding_side: str,
    ) -> torch.Tensor:
        max_len = max(value.shape[0] for value in values)
        padded = []
        for value in values:
            pad_len = max_len - value.shape[0]
            if pad_len <= 0:
                padded.append(value)
                continue
            pad = torch.full((pad_len,), pad_value, dtype=value.dtype)
            if padding_side == "left":
                padded.append(torch.cat([pad, value], dim=0))
            elif padding_side == "right":
                padded.append(torch.cat([value, pad], dim=0))
            else:
                raise ValueError(f"padding_side must be left or right, got {padding_side!r}")
        return torch.stack(padded, dim=0)

    @staticmethod
    def _collate_meta(samples: list[GaussianDWMSample]) -> dict[str, list[Any]]:
        keys = set()
        for sample in samples:
            keys.update(sample.meta.keys())
        return {key: [sample.meta.get(key) for sample in samples] for key in sorted(keys)}
