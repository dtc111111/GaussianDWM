from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


CVPR_TRAINABLE_GROUPS: tuple[str, ...] = (
    "gauss_token_rows",
    "gauss_aligner_core",
    "qwen_backbone_lora",
    "qa_lm_head_lora",
    "cond_fusion",
    "world_unet_lora",
)

LORA_GROUPS = {"qwen_backbone_lora", "qa_lm_head_lora", "world_unet_lora"}
TOKEN_ROW_GROUPS = ("gauss_token_rows",)


@dataclass(frozen=True)
class TrainableGroupReport:
    groups: tuple[str, ...]
    enabled_parameter_count: int
    enabled_parameter_names: tuple[str, ...]


def inject_lora_from_config(model: nn.Module, lora_config: dict[str, Any]) -> nn.Module:
    """Inject only the LoRA adapters exposed by the public QA/world configs."""

    if not bool(lora_config.get("enabled", False)):
        return model
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("PEFT is required when public training config sets lora.enabled=true.") from exc

    qwen_config = lora_config.get("qwen")
    if isinstance(qwen_config, dict) and bool(qwen_config.get("enabled", False)):
        qwen = getattr(getattr(model, "backbone", None), "qwen", None)
        if qwen is None:
            raise AttributeError("model.backbone.qwen is required for qwen LoRA training.")
        if getattr(qwen, "peft_config", None) is None:
            target_modules = list(
                dict.fromkeys(
                    [
                        *[str(item) for item in qwen_config.get("backbone_target_modules", [])],
                        *[str(item) for item in qwen_config.get("lm_head_target_modules", [])],
                    ]
                )
            )
            if not target_modules:
                raise ValueError("qwen LoRA config must define target modules.")
            model.backbone.qwen = get_peft_model(
                qwen,
                LoraConfig(
                    r=int(qwen_config.get("rank", 8)),
                    lora_alpha=int(qwen_config.get("alpha", 16)),
                    lora_dropout=float(qwen_config.get("dropout", 0.05)),
                    target_modules=target_modules,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )

    world_config = lora_config.get("world_unet")
    if isinstance(world_config, dict) and bool(world_config.get("enabled", False)):
        world_unet = _world_unet(model)
        if getattr(world_unet, "peft_config", None) is None:
            target_modules = [str(item) for item in world_config.get("target_modules", [])]
            if not target_modules:
                raise ValueError("world_unet LoRA config must define target modules.")
            model.world_head.bundle.unet = get_peft_model(
                world_unet,
                LoraConfig(
                    r=int(world_config.get("rank", 8)),
                    lora_alpha=int(world_config.get("alpha", 16)),
                    lora_dropout=float(world_config.get("dropout", 0.05)),
                    target_modules=target_modules,
                    bias="none",
                ),
            )

    return model


def apply_trainable_groups(model: nn.Module, trainable_groups: Iterable[str]) -> TrainableGroupReport:
    requested = tuple(str(group) for group in trainable_groups)
    unknown = sorted(set(requested) - set(CVPR_TRAINABLE_GROUPS))
    if unknown:
        raise ValueError(f"Unsupported CVPR trainable modules: {', '.join(unknown)}")

    _clear_token_row_mask(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    enabled_names: list[str] = []
    if "gauss_token_rows" in requested:
        enabled_names.extend(_enable_gaussian_token_rows(model))
    if "gauss_aligner_core" in requested:
        enabled_names.extend(_enable_module_parameters(model.backbone.gauss_aligner, "backbone.gauss_aligner"))
    if "qwen_backbone_lora" in requested:
        enabled_names.extend(_enable_lora_group(model, "qwen_backbone_lora"))
    if "qa_lm_head_lora" in requested:
        enabled_names.extend(_enable_lora_group(model, "qa_lm_head_lora"))
    if "cond_fusion" in requested:
        enabled_names.extend(_enable_cond_fusion(model))
    if "world_unet_lora" in requested:
        enabled_names.extend(_enable_lora_group(model, "world_unet_lora"))

    enabled_count = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    return TrainableGroupReport(
        groups=requested,
        enabled_parameter_count=enabled_count,
        enabled_parameter_names=tuple(enabled_names),
    )


def planned_token_row_groups(trainable_groups: Iterable[str]) -> tuple[str, ...]:
    groups = {str(group) for group in trainable_groups}
    return tuple(group for group in TOKEN_ROW_GROUPS if group in groups)


def _enable_lora_group(model: nn.Module, group_name: str) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if _logical_lora_group(name) != group_name:
            continue
        parameter.requires_grad_(True)
        names.append(name)
    if not names:
        raise RuntimeError(
            f"Requested {group_name}, but the published model has no matching LoRA parameters. "
            "Inject adapters from config before applying the stage, or remove this trainable module."
        )
    return names


def _logical_lora_group(parameter_name: str) -> str | None:
    name = parameter_name.removeprefix("module.")
    if "lora_" not in name:
        return None
    if name.startswith("world_head.bundle.unet."):
        return "world_unet_lora"
    if ".lm_head." in name or name.endswith(".lm_head.lora_A.default.weight") or ".lm_head.lora_" in name:
        return "qa_lm_head_lora"
    if "backbone.qwen." in name:
        return "qwen_backbone_lora"
    return None


def _enable_gaussian_token_rows(model: nn.Module) -> list[str]:
    token_ids = _gaussian_token_ids(model)
    embedding = model.backbone.qwen.get_input_embeddings()
    embedding.weight.requires_grad_(True)

    indices = torch.tensor(token_ids, dtype=torch.long)

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(grad)
        mask.index_fill_(0, indices.to(grad.device), 1.0)
        return grad * mask

    hook = embedding.weight.register_hook(_mask_grad)
    setattr(model, "_cvpr_gauss_token_row_hook", hook)
    setattr(model, "_cvpr_gauss_token_row_ids", tuple(token_ids))
    return ["backbone.qwen.input_embeddings.gaussian_token_rows"]


def _gaussian_token_ids(model: nn.Module) -> tuple[int, ...]:
    tokens = {}
    model_config = getattr(model, "model_config", None)
    if isinstance(model_config, dict) and isinstance(model_config.get("tokens"), dict):
        tokens = dict(model_config["tokens"])
    ids: set[int] = set()
    if tokens.get("gaussian_pad_token_id") is not None:
        ids.add(int(tokens["gaussian_pad_token_id"]))
    for token_id in tokens.get("gaussian_special_token_ids", []):
        ids.add(int(token_id))
    if not ids:
        backbone = getattr(model, "backbone", None)
        if getattr(backbone, "gaussian_pad_token_id", None) is not None:
            ids.add(int(backbone.gaussian_pad_token_id))
        for token_id in getattr(backbone, "gaussian_special_token_ids", []):
            ids.add(int(token_id))
    if not ids:
        raise RuntimeError("gauss_token_rows requested, but no Gaussian token ids are recorded on the model.")
    return tuple(sorted(ids))


def _enable_module_parameters(module: nn.Module, prefix: str) -> list[str]:
    names: list[str] = []
    for name, parameter in module.named_parameters():
        parameter.requires_grad_(True)
        names.append(f"{prefix}.{name}" if name else prefix)
    if not names:
        raise RuntimeError(f"Requested {prefix}, but it exposes no parameters.")
    return names


def _enable_cond_fusion(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, parameter in model.cond_fusion.named_parameters():
        if name.startswith("image_encoder."):
            continue
        parameter.requires_grad_(True)
        names.append(f"cond_fusion.{name}")
    if not names:
        raise RuntimeError("cond_fusion requested, but it exposes no trainable non-image-encoder parameters.")
    return names


def _clear_token_row_mask(model: nn.Module) -> None:
    hook = getattr(model, "_cvpr_gauss_token_row_hook", None)
    if hook is not None:
        hook.remove()
    setattr(model, "_cvpr_gauss_token_row_hook", None)
    setattr(model, "_cvpr_gauss_token_row_ids", ())


def _world_unet(model: nn.Module) -> nn.Module:
    world_head = getattr(model, "world_head", None)
    bundle = getattr(world_head, "bundle", None)
    world_unet = getattr(bundle, "unet", None)
    if world_unet is None:
        raise AttributeError("model.world_head.bundle.unet is required for world_unet_lora training.")
    return world_unet
