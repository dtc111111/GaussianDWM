from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from .trainable_groups import CVPR_TRAINABLE_GROUPS, apply_trainable_groups, planned_token_row_groups


CVPR_TRAINABLE_MODULES: tuple[str, ...] = (
    "gauss_token_rows",
    "gauss_aligner_core",
    "qwen_backbone_lora",
    "qa_lm_head_lora",
    "cond_fusion",
    "world_unet_lora",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    start_step: int
    end_step: int | None
    trainable_modules: tuple[str, ...]
    loss_balancer_mode: str = "static"


class StageScheduler:
    def __init__(self, plan: dict[str, Any], per_device_batch_size: dict[str, int]) -> None:
        stages_payload = plan.get("stages")
        if not isinstance(stages_payload, list) or not stages_payload:
            raise ValueError("stage plan must contain a non-empty stages list.")

        self.per_device_batch_size = {str(task): int(size) for task, size in per_device_batch_size.items()}
        self.stages: list[StageSpec] = []
        for index, item in enumerate(stages_payload):
            if not isinstance(item, dict):
                raise TypeError(f"stages[{index}] must be a mapping.")
            modules = tuple(str(module) for module in item.get("trainable_modules", []))
            if not modules:
                raise ValueError(f"Stage {item.get('name', index)!r} must define non-empty trainable_modules.")
            traj_modules = [module for module in modules if "traj" in module.lower()]
            if traj_modules:
                raise ValueError(
                    "CVPR public training supports QA/world only and rejects traj trainable modules: "
                    f"{', '.join(sorted(traj_modules))}"
                )
            unknown = sorted(set(modules) - set(CVPR_TRAINABLE_MODULES))
            if unknown:
                raise ValueError(f"Unsupported CVPR trainable modules: {', '.join(unknown)}")
            if set(CVPR_TRAINABLE_MODULES) != set(CVPR_TRAINABLE_GROUPS):
                raise RuntimeError("CVPR_TRAINABLE_MODULES and trainable group implementation drifted.")
            loss_balancer_mode = str(item.get("loss_balancer_mode", "static"))
            if loss_balancer_mode != "static":
                raise ValueError(
                    "CVPR public training supports only static loss balancing; "
                    f"unsupported loss balancer mode {loss_balancer_mode!r}. GradNorm is not included."
                )

            self.stages.append(
                StageSpec(
                    name=str(item["name"]),
                    start_step=int(item["start_step"]),
                    end_step=None if item.get("end_step") is None else int(item["end_step"]),
                    trainable_modules=modules,
                    loss_balancer_mode=loss_balancer_mode,
                )
            )

    def get_stage(self, step: int) -> StageSpec:
        for stage in self.stages:
            if int(step) < stage.start_step:
                continue
            if stage.end_step is None or int(step) < stage.end_step:
                return stage
        return self.stages[-1]

    def get_batch_sizes(self, step: int) -> dict[str, int]:
        del step
        return dict(self.per_device_batch_size)

    def planned_token_row_groups(self) -> tuple[str, ...]:
        groups: list[str] = []
        for stage in self.stages:
            groups.extend(stage.trainable_modules)
        return planned_token_row_groups(groups)

    def planned_trainable_modules(self) -> tuple[str, ...]:
        groups: list[str] = []
        for stage in self.stages:
            groups.extend(stage.trainable_modules)
        return tuple(dict.fromkeys(groups))

    def apply(self, model: nn.Module, step: int) -> StageSpec:
        stage = self.get_stage(step)
        apply_trainable_groups(model, stage.trainable_modules)
        return stage
