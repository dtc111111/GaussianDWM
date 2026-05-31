from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLCausalLMOutputWithPast, Qwen3VLModelOutputWithPast
from transformers.utils import is_torchdynamo_compiling

from .gauss_aligner import GaussAligner
from .selectors import IdentitySelector, SimilaritySelector


@dataclass
class BackboneRequest:
    qwen_inputs: dict[str, torch.Tensor | None]
    need_token_hidden: bool
    need_global_condition: bool
    task_type: str
    global_step: int = 0
    use_gumbel: bool = True


@dataclass
class BackboneOutput:
    token_hidden_states: torch.Tensor | None
    token_attention_mask: torch.Tensor | None
    global_condition: torch.Tensor | None
    text_token_mask: torch.Tensor
    gauss_token_mask: torch.Tensor
    past_key_values: Any | None = None
    rope_deltas: Any | None = None


class QwenGaussBackbone(nn.Module):
    def __init__(
        self,
        *,
        qwen_model: nn.Module,
        hidden_size: int,
        fine_k_by_task: dict[str, int],
        gaussian_pad_token_id: int,
        gaussian_special_token_ids: list[int],
        pad_token_id: int,
        pooling_mode: str = "mixed",
        fine_method: str = "identity",
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        self.hidden_size = int(hidden_size)
        self.fine_k_by_task = dict(fine_k_by_task)
        self.gaussian_pad_token_id = int(gaussian_pad_token_id)
        self.gaussian_special_token_ids = set(int(x) for x in gaussian_special_token_ids)
        self.pad_token_id = int(pad_token_id)
        self.pooling_mode = pooling_mode
        self._fine_method = fine_method

        self.gauss_aligner = GaussAligner(hidden_dim=hidden_size)
        self.fine_selector = self._build_selector(fine_method, hidden_size)

    @staticmethod
    def _build_selector(method: str, hidden_size: int) -> nn.Module:
        del hidden_size
        if method == "identity":
            return IdentitySelector()
        if method == "similarity":
            return SimilaritySelector()
        raise ValueError(f"CVPR package supports fine_method='identity' or 'similarity', got {method!r}")

    def build_text_token_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mask language tokens while excluding padding, image/video, and Gaussian placeholder ids."""
        mask = attention_mask.to(torch.bool)
        excluded = input_ids.eq(self.pad_token_id)
        image_token_id = getattr(self.qwen.config, "image_token_id", None)
        video_token_id = getattr(self.qwen.config, "video_token_id", None)
        if image_token_id is not None:
            excluded = excluded | input_ids.eq(int(image_token_id))
        if video_token_id is not None:
            excluded = excluded | input_ids.eq(int(video_token_id))
        for token_id in self.gaussian_special_token_ids:
            excluded = excluded | input_ids.eq(token_id)
        return mask & ~excluded

    def build_gauss_token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.eq(self.gaussian_pad_token_id)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (values * weights).sum(dim=1) / denom

    def forward_backbone(self, request: BackboneRequest) -> BackboneOutput:
        q = request.qwen_inputs
        input_ids = self._ensure_2d(q["input_ids"], "input_ids")
        attention_mask = self._ensure_2d(q["attention_mask"], "attention_mask")
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask must match input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}"
            )
        qwen_embed_model = self._resolve_qwen_module("get_input_embeddings")
        qwen_mm_model = self._resolve_qwen_module(
            "get_image_features",
            "get_placeholder_mask",
            "language_model",
        )
        inputs_embeds = qwen_embed_model.get_input_embeddings()(input_ids)

        image_mask = None
        deepstack_visual_embeds = None
        if q.get("pixel_values") is not None:
            image_embeds, deepstack_image_embeds = qwen_mm_model.get_image_features(
                q["pixel_values"],  # type: ignore[arg-type]
                q.get("image_grid_thw"),
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = qwen_mm_model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            image_mask = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds

        text_token_mask = self.build_text_token_mask(input_ids, attention_mask)
        gauss_token_mask = self.build_gauss_token_mask(input_ids)

        fine_k = int(self.fine_k_by_task[request.task_type])
        coarse_gauss_values = q.get("coarse_gauss_values")
        coarse_gauss_mask = q.get("coarse_gauss_mask")
        if coarse_gauss_values is None:
            if coarse_gauss_mask is not None:
                raise ValueError("coarse_gauss_mask must be None when coarse_gauss_values is None.")
            counts = gauss_token_mask.sum(dim=1).tolist()
            if any(count != 0 for count in counts):
                raise ValueError(f"Prompt gaussian placeholder count must be 0 when gaussian input is disabled; got {counts}")
        else:
            coarse_gauss_values = self._ensure_3d(coarse_gauss_values, "coarse_gauss_values")
            coarse_gauss_mask = self._ensure_2d(coarse_gauss_mask, "coarse_gauss_mask").to(torch.bool)
            if coarse_gauss_values.shape[-1] != 14:
                raise ValueError(
                    f"coarse_gauss_values must have shape [B,K,14], got {tuple(coarse_gauss_values.shape)}"
                )
            if coarse_gauss_mask.shape != coarse_gauss_values.shape[:2]:
                raise ValueError(
                    "coarse_gauss_mask must match coarse_gauss_values batch and candidate dimensions"
                )
            if not torch.all(gauss_token_mask.sum(dim=1) == fine_k):
                counts = gauss_token_mask.sum(dim=1).tolist()
                raise ValueError(f"Prompt gaussian placeholder count must equal fine_k={fine_k}; got {counts}")

            gauss_hidden = self.gauss_aligner(coarse_gauss_values.to(inputs_embeds.device, inputs_embeds.dtype))

            if self._fine_method == "similarity":
                clip_text_embed = q.get("clip_text_embed")
                if clip_text_embed is None:
                    raise ValueError("clip_text_embed is required when fine_method='similarity'.")
                clip_text_embed = self._ensure_2d(clip_text_embed, "clip_text_embed").to(inputs_embeds.device)
                if clip_text_embed.shape[-1] != 512:
                    raise ValueError(f"clip_text_embed must have shape [B,512], got {tuple(clip_text_embed.shape)}")
                gauss_clip_features = self.gauss_aligner.decode_language_clip_features(
                    coarse_gauss_values.to(inputs_embeds.device, inputs_embeds.dtype)
                )
                selected_gauss = self.fine_selector(
                    clip_text_embed=clip_text_embed,
                    gauss_clip_features=gauss_clip_features,
                    coarse_gauss_hidden=gauss_hidden,
                    coarse_gauss_mask=coarse_gauss_mask.to(inputs_embeds.device),
                    fine_k=fine_k,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
            else:
                selected_gauss = self.fine_selector(
                    coarse_gauss_hidden=gauss_hidden,
                    coarse_gauss_mask=coarse_gauss_mask.to(inputs_embeds.device),
                    fine_k=fine_k,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = self.inject_selected_gauss(inputs_embeds, gauss_token_mask, selected_gauss)

        position_ids = self._build_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=q.get("image_grid_thw"),
        )
        outputs = qwen_mm_model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        outputs = Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=getattr(self.qwen, "rope_deltas", None),
        )

        global_condition = None
        if request.need_global_condition:
            if self.pooling_mode == "mixed":
                global_condition = self.masked_mean(outputs.last_hidden_state, attention_mask.to(torch.bool))
            elif self.pooling_mode == "text_only":
                global_condition = self.masked_mean(outputs.last_hidden_state, text_token_mask)
            else:
                raise ValueError(f"Unsupported pooling_mode={self.pooling_mode!r}")

        token_hidden_states = outputs.last_hidden_state if request.need_token_hidden else None
        return BackboneOutput(
            token_hidden_states=token_hidden_states,
            token_attention_mask=attention_mask,
            global_condition=global_condition,
            text_token_mask=text_token_mask,
            gauss_token_mask=gauss_token_mask,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def forward_for_generation(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        coarse_gauss_values: torch.Tensor | None = None,
        coarse_gauss_mask: torch.Tensor | None = None,
        clip_text_embed: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        use_cache: bool = True,
        return_dict: bool = True,
        task_type: str = "qa",
        **_: Any,
    ) -> Qwen3VLCausalLMOutputWithPast:
        """Run QA prefill with Gaussian token injection, then delegate cached decoding to Qwen."""
        del return_dict
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        is_prefill = cache_position is None or int(cache_position[0].item()) == 0
        if not is_prefill:
            return self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
            )

        output = self.forward_backbone(
            BackboneRequest(
                qwen_inputs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_grid_thw": image_grid_thw,
                    "coarse_gauss_values": coarse_gauss_values,
                    "coarse_gauss_mask": coarse_gauss_mask,
                    "clip_text_embed": clip_text_embed,
                },
                need_token_hidden=True,
                need_global_condition=False,
                task_type=task_type,
                use_gumbel=False,
            )
        )
        hidden_states = output.token_hidden_states
        if hidden_states is None:
            raise RuntimeError("Generation prefill requires token_hidden_states.")
        qwen_causal_lm = self._resolve_qwen_module("lm_head")
        lm_head = qwen_causal_lm.lm_head
        logits = lm_head(hidden_states.to(lm_head.weight.dtype))
        return Qwen3VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=output.past_key_values,
            rope_deltas=output.rope_deltas,
        )

    def inject_selected_gauss(
        self,
        inputs_embeds: torch.Tensor,
        gauss_token_mask: torch.Tensor,
        selected_gauss: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = inputs_embeds.shape[0]
        output = inputs_embeds.clone()
        for batch_idx in range(batch_size):
            positions = torch.where(gauss_token_mask[batch_idx])[0]
            if positions.numel() != selected_gauss.shape[1]:
                raise ValueError(
                    f"Batch {batch_idx} has {positions.numel()} gauss placeholders but selected {selected_gauss.shape[1]}"
                )
            output[batch_idx, positions] = selected_gauss[batch_idx]
        return output

    def _build_position_ids(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Use Qwen3-VL RoPE indexing after Gaussian placeholders have been replaced by embeddings."""
        attention_mask_tensor = attention_mask
        if attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and inputs_embeds.shape[1] != 1
        prefill_noncompiled_stage = not is_torchdynamo_compiling()
        if prefill_compiled_stage or prefill_noncompiled_stage or getattr(self.qwen, "rope_deltas", None) is None:
            qwen_rope_model = self._resolve_qwen_module("get_rope_index")
            position_ids, rope_deltas = qwen_rope_model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,
                attention_mask=attention_mask_tensor,
            )
            self.qwen.rope_deltas = rope_deltas
            return position_ids
        raise RuntimeError("Unexpected position id state for backbone forward.")

    def _resolve_qwen_module(self, *required_attrs: str) -> Any:
        module: Any = self.qwen
        visited: set[int] = set()
        while True:
            if all(hasattr(module, attr) for attr in required_attrs):
                return module
            visited.add(id(module))
            next_module = getattr(module, "model", None)
            if next_module is None or id(next_module) in visited:
                break
            module = next_module
        raise AttributeError(
            "Could not resolve Qwen module with attributes "
            f"{required_attrs!r} from type {type(self.qwen).__name__}."
        )

    @staticmethod
    def _ensure_2d(value: torch.Tensor | None, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} is required.")
        if value.ndim == 1:
            return value.unsqueeze(0)
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [B,L], got {value.shape}")
        return value

    @staticmethod
    def _ensure_3d(value: torch.Tensor | None, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} is required.")
        if value.ndim == 2:
            return value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape [B,K,D], got {value.shape}")
        return value
