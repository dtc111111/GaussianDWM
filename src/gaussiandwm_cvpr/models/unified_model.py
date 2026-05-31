from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from transformers.modeling_utils import no_init_weights
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from .qwen_gauss_backbone import BackboneRequest, QwenGaussBackbone
from .qa_head import QAHead
from .world_head import CondFusion, WorldDiffusionHead, WorldUNetBundle
from .world_arch.unet_spatio_temporal import UNetSpatioTemporalConditionModelV2_inchannel16
from .world_arch.unified_encoder import LayoutCondEncoder


_CLASS_REGISTRY: dict[str, type[Any]] = {
    "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
    "CLIPVisionModelWithProjection": CLIPVisionModelWithProjection,
    "UNetSpatioTemporalConditionModelV2_inchannel16": UNetSpatioTemporalConditionModelV2_inchannel16,
    "LayoutCondEncoder": LayoutCondEncoder,
    "AutoencoderKLTemporalDecoder": AutoencoderKLTemporalDecoder,
    "QwenGaussBackbone": QwenGaussBackbone,
    "QAHead": QAHead,
    "WorldDiffusionHead": WorldDiffusionHead,
    "WorldUNetBundle": WorldUNetBundle,
    "CondFusion": CondFusion,
}

_CONFIG_REGISTRY: dict[str, type[Any]] = {
    "Qwen3VLForConditionalGeneration": Qwen3VLConfig,
    "CLIPVisionModelWithProjection": CLIPVisionConfig,
}

_DEFAULT_HF_MODEL_ID = "dtc111/GaussianDWM"


def _save_torch_state(state_dict: dict[str, torch.Tensor], path: Path) -> str:
    from safetensors.torch import save_file

    weight_path = path.with_suffix(".safetensors")
    save_file(state_dict, str(weight_path))
    return weight_path.name


def _load_torch_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix != ".safetensors":
        raise ValueError(
            "CVPR package expects root-level model.safetensors weights; "
            f"got {path.name!r}."
        )
    from safetensors.torch import load_file

    return load_file(str(path))


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _require_config_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Unified model config field {key!r} must be a dict, got {type(value)!r}.")
    return value


def _resolve_pretrained_root(model_id_or_path: str | Path, *, revision: str = "main") -> Path:
    local_root = Path(model_id_or_path).expanduser()
    if local_root.exists():
        return local_root

    from huggingface_hub import hf_hub_download

    repo_id = str(model_id_or_path)
    config_path = Path(hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = _require_config_dict(config, "meta")
    weight_filename = str(meta["weight_filename"])
    if weight_filename != "model.safetensors":
        raise ValueError(
            "CVPR package expects root-level model.safetensors weights; "
            f"config requested {weight_filename!r}."
        )
    hf_hub_download(repo_id=repo_id, filename=weight_filename, revision=revision)
    return config_path.parent


def _infer_floating_dtype(state_dict: dict[str, torch.Tensor]) -> torch.dtype | None:
    for value in state_dict.values():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            return value.dtype
    return None


def _resolve_registered_class(class_name: str) -> type[Any]:
    cls = _CLASS_REGISTRY.get(class_name)
    if cls is None:
        raise KeyError(f"Unknown unified model component class: {class_name}")
    return cls


def _instantiate_leaf_module(payload: dict[str, Any]) -> nn.Module:
    class_name = str(payload["class_name"])
    config = payload.get("config") or {}
    cls = _resolve_registered_class(class_name)
    from_config = getattr(cls, "from_config", None)
    if callable(from_config):
        return from_config(config)
    config_cls = _CONFIG_REGISTRY.get(class_name)
    if config_cls is not None:
        return cls(config_cls.from_dict(config))
    return cls(**config)


@dataclass
class UnifiedModelOutput:
    loss: torch.Tensor
    logits: torch.Tensor | None = None
    world_pred_noise: torch.Tensor | None = None


class UnifiedGaussianDWM(nn.Module):
    def __init__(
        self,
        *,
        backbone: QwenGaussBackbone,
        qa_head: QAHead,
        world_head: WorldDiffusionHead,
        cond_fusion: CondFusion,
        model_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.qa_head = qa_head
        self.world_head = world_head
        self.cond_fusion = cond_fusion
        self.model_config = model_config

    @property
    def qwen(self) -> nn.Module:
        return self.backbone.qwen

    def export_config(self, *, weight_filename: str = "model.safetensors") -> dict[str, object]:
        tokens = self.model_config.get("tokens", {})
        return {
            "meta": {
                "format_version": 1,
                "model_type": "unified_gaussian_dwm",
                "weight_filename": weight_filename,
            },
            "qwen": {
                "class_name": type(self.backbone.qwen).__name__,
                "config": self.backbone.qwen.config.to_dict(),
            },
            "backbone": {
                "class_name": type(self.backbone).__name__,
                "config": {
                    "hidden_size": self.backbone.hidden_size,
                    "fine_k_by_task": dict(self.backbone.fine_k_by_task),
                    "gaussian_pad_token_id": self.backbone.gaussian_pad_token_id,
                    "gaussian_special_token_ids": sorted(self.backbone.gaussian_special_token_ids),
                    "pad_token_id": self.backbone.pad_token_id,
                    "pooling_mode": self.backbone.pooling_mode,
                    "fine_method": self.backbone._fine_method,
                },
            },
            "world": {
                "head_class_name": type(self.world_head).__name__,
                "bundle_class_name": type(self.world_head.bundle).__name__,
                "head_config": {
                    "conditioning_dropout_prob": self.world_head.conditioning_dropout_prob,
                    "scheduler_config": dict(self.world_head.scheduler.config),
                },
                "unet": {
                    "class_name": type(self.world_head.bundle.unet).__name__,
                    "config": dict(self.world_head.bundle.unet.config),
                },
                "layout_encoder": {
                    "class_name": type(self.world_head.bundle.layout_encoder).__name__,
                    "config": self.world_head.bundle.layout_encoder.config.to_dict()
                    if hasattr(self.world_head.bundle.layout_encoder.config, "to_dict")
                    else dict(self.world_head.bundle.layout_encoder.config),
                },
                "layout_encoder_depth": {
                    "class_name": type(self.world_head.bundle.layout_encoder_depth).__name__,
                    "config": self.world_head.bundle.layout_encoder_depth.config.to_dict()
                    if hasattr(self.world_head.bundle.layout_encoder_depth.config, "to_dict")
                    else dict(self.world_head.bundle.layout_encoder_depth.config),
                },
                "vae": {
                    "class_name": type(self.world_head.bundle.vae).__name__,
                    "config": dict(self.world_head.bundle.vae.config),
                },
            },
            "heads": {
                "qa_head": {
                    "class_name": type(self.qa_head).__name__,
                    "config": {},
                },
            },
            "cond_fusion": {
                "class_name": type(self.cond_fusion).__name__,
                "config": {
                    "cond_dim": self.cond_fusion.cond_dim,
                    "text_cond_dim": self.cond_fusion.text_cond_dim,
                    "ref_cond_dim": self.cond_fusion.ref_cond_dim,
                },
                "image_encoder": {
                    "class_name": type(self.cond_fusion.image_encoder).__name__,
                    "config": self.cond_fusion.image_encoder.config.to_dict()
                    if hasattr(self.cond_fusion.image_encoder.config, "to_dict")
                    else dict(self.cond_fusion.image_encoder.config),
                },
            },
            "tokens": {
                "gaussian_pad_token_id": self.backbone.gaussian_pad_token_id,
                "gaussian_special_token_ids": sorted(self.backbone.gaussian_special_token_ids),
                "condition_token_ids": list(tokens.get("condition_token_ids", [])),
                "condition_token_names": list(tokens.get("condition_token_names", [])),
            },
        }

    def forward(self, task: str, **kwargs: Any) -> UnifiedModelOutput:
        global_step = int(kwargs.pop("global_step", 0))
        use_gumbel = bool(kwargs.pop("use_gumbel", True))
        if task == "qa":
            return self.forward_qa(global_step=global_step, use_gumbel=use_gumbel, **kwargs)
        if task == "world":
            return self.forward_world(global_step=global_step, use_gumbel=use_gumbel, **kwargs)
        raise ValueError(f"CVPR package supports task='qa' or task='world', got {task!r}")

    def forward_qa(self, *, qwen_inputs: dict[str, Any], labels: torch.Tensor | None = None, global_step: int = 0, use_gumbel: bool = True, **_: Any) -> UnifiedModelOutput:
        bb = self.backbone.forward_backbone(
            BackboneRequest(
                qwen_inputs=qwen_inputs,
                need_token_hidden=True,
                need_global_condition=False,
                task_type="qa",
                global_step=global_step,
                use_gumbel=use_gumbel,
            )
        )
        qa = self.qa_head(
            token_hidden_states=bb.token_hidden_states,  # type: ignore[arg-type]
            lm_head=self.backbone.qwen.lm_head,
            labels=labels,
        )
        if qa.loss is None:
            raise RuntimeError("QA training requires labels to produce a loss.")
        return UnifiedModelOutput(loss=qa.loss, logits=qa.logits)

    def forward_world(
        self,
        *,
        qwen_inputs: dict[str, Any],
        ref_pixel_values: torch.Tensor,
        pseudo_pixel_values: torch.Tensor,
        pseudo_depth_values: torch.Tensor,
        target_latents: dict[str, torch.Tensor],
        world_meta: dict[str, object],
        global_step: int = 0,
        use_gumbel: bool = True,
        **_: Any,
    ) -> UnifiedModelOutput:
        bb = self.backbone.forward_backbone(
            BackboneRequest(
                qwen_inputs=qwen_inputs,
                need_token_hidden=False,
                need_global_condition=True,
                task_type="world",
                global_step=global_step,
                use_gumbel=use_gumbel,
            )
        )
        if bb.global_condition is None:
            raise RuntimeError("World forward requires global_condition.")
        cond_embeddings = self.cond_fusion(
            ref_pixel_values=ref_pixel_values,
            text_cond_seed=bb.global_condition,
        )
        world = self.world_head(
            target_latents=target_latents,
            pseudo_pixel_values=pseudo_pixel_values,
            pseudo_depth_values=pseudo_depth_values,
            cond_embeddings=cond_embeddings,
            world_meta=world_meta,
        )
        return UnifiedModelOutput(loss=world.loss, world_pred_noise=world.pred_noise)

    def save_pretrained(self, model_init_dir: str | Path) -> dict[str, str]:
        target = Path(model_init_dir)
        target.mkdir(parents=True, exist_ok=True)
        weight_name = _save_torch_state(self.state_dict(), target / "model")
        _save_json(self.export_config(weight_filename=weight_name), target / "config.json")
        return {
            "config": "config.json",
            "weights": weight_name,
        }

    @classmethod
    def build_from_config(cls, payload: dict[str, Any]) -> "UnifiedGaussianDWM":
        qwen_payload = _require_config_dict(payload, "qwen")
        backbone_payload = _require_config_dict(payload, "backbone")
        world_payload = _require_config_dict(payload, "world")
        heads_payload = _require_config_dict(payload, "heads")
        cond_fusion_payload = _require_config_dict(payload, "cond_fusion")
        tokens_payload = _require_config_dict(payload, "tokens")

        qwen_model = _instantiate_leaf_module(qwen_payload)
        if not isinstance(qwen_model, nn.Module):
            raise TypeError(f"qwen component must resolve to nn.Module, got {type(qwen_model)!r}.")

        backbone_config = _require_config_dict(backbone_payload, "config")
        backbone = QwenGaussBackbone(
            qwen_model=qwen_model,
            hidden_size=int(backbone_config["hidden_size"]),
            fine_k_by_task=dict(backbone_config["fine_k_by_task"]),
            gaussian_pad_token_id=int(backbone_config["gaussian_pad_token_id"]),
            gaussian_special_token_ids=list(backbone_config["gaussian_special_token_ids"]),
            pad_token_id=int(backbone_config["pad_token_id"]),
            pooling_mode=str(backbone_config["pooling_mode"]),
            fine_method=str(backbone_config["fine_method"]),
        )

        qa_head_config = _require_config_dict(_require_config_dict(heads_payload, "qa_head"), "config")
        del qa_head_config
        qa_head = QAHead()

        vae = _instantiate_leaf_module(_require_config_dict(world_payload, "vae"))
        vae.requires_grad_(False)
        world_bundle = WorldUNetBundle(
            unet=_instantiate_leaf_module(_require_config_dict(world_payload, "unet")),
            layout_encoder=_instantiate_leaf_module(_require_config_dict(world_payload, "layout_encoder")),
            layout_encoder_depth=_instantiate_leaf_module(
                _require_config_dict(world_payload, "layout_encoder_depth")
            ),
            vae=vae,
        )
        world_head_config = _require_config_dict(world_payload, "head_config")
        world_head = WorldDiffusionHead(
            bundle=world_bundle,
            conditioning_dropout_prob=float(world_head_config.get("conditioning_dropout_prob", 0.0)),
        )
        scheduler_config = world_head_config.get("scheduler_config")
        if scheduler_config is not None:
            world_head.scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
        world_head.scheduler_config_path = None

        cond_fusion_config = _require_config_dict(cond_fusion_payload, "config")
        image_encoder = _instantiate_leaf_module(_require_config_dict(cond_fusion_payload, "image_encoder"))
        image_encoder.requires_grad_(False)
        cond_fusion = CondFusion(
            cond_dim=int(cond_fusion_config["cond_dim"]),
            text_cond_dim=int(cond_fusion_config["text_cond_dim"]),
            ref_cond_dim=int(cond_fusion_config["ref_cond_dim"]),
            image_encoder=image_encoder,
        )

        token_config = {
            "fine_k": dict(backbone_config["fine_k_by_task"]),
            "fine_method": str(backbone_config["fine_method"]),
            "gaussian_pad_token_id": int(backbone_config["gaussian_pad_token_id"]),
            "gaussian_special_token_ids": list(backbone_config["gaussian_special_token_ids"]),
            "pad_token_id": int(backbone_config["pad_token_id"]),
            "condition_token_ids": list(tokens_payload.get("condition_token_ids", [])),
            "condition_token_names": list(tokens_payload.get("condition_token_names", [])),
        }
        model_config = {
            "tokens": dict(tokens_payload),
            "token_config": token_config,
        }
        return cls(
            backbone=backbone,
            qa_head=qa_head,
            world_head=world_head,
            cond_fusion=cond_fusion,
            model_config=model_config,
        )

    @classmethod
    def from_config_only(
        cls,
        model_init_dir: str | Path = _DEFAULT_HF_MODEL_ID,
        *,
        revision: str = "main",
        device: str | torch.device | None = None,
    ) -> "UnifiedGaussianDWM":
        root = _resolve_pretrained_root(model_init_dir, revision=revision)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        with no_init_weights():
            model = cls.build_from_config(config)
        model.model_config = dict(model.model_config)
        model.model_config.setdefault("model_init_dir", str(root.resolve()))
        if device is not None:
            model.to(device)
        return model

    @staticmethod
    def load_pretrained_state_dict(
        model_init_dir: str | Path = _DEFAULT_HF_MODEL_ID,
        *,
        revision: str = "main",
    ) -> dict[str, torch.Tensor]:
        root = _resolve_pretrained_root(model_init_dir, revision=revision)
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        meta = _require_config_dict(config, "meta")
        weight_filename = str(meta["weight_filename"])
        if weight_filename != "model.safetensors":
            raise ValueError(
                "CVPR package expects root-level model.safetensors weights; "
                f"config requested {weight_filename!r}."
            )
        return _load_torch_state(root / weight_filename)

    def load_pretrained_weights(
        self,
        model_init_dir: str | Path,
        *,
        revision: str = "main",
        strict: bool = True,
    ) -> None:
        state = self.load_pretrained_state_dict(model_init_dir, revision=revision)
        state_dtype = _infer_floating_dtype(state)
        if state_dtype is not None:
            self.to(dtype=state_dtype)
        self.load_state_dict(state, strict=strict)

    @classmethod
    def from_pretrained(
        cls,
        model_init_dir: str | Path = _DEFAULT_HF_MODEL_ID,
        *,
        revision: str = "main",
        device: str | torch.device | None = None,
    ) -> "UnifiedGaussianDWM":
        root = _resolve_pretrained_root(model_init_dir, revision=revision)
        model = cls.from_config_only(root, revision=revision, device=device)
        model.load_pretrained_weights(root, revision=revision, strict=True)
        return model
