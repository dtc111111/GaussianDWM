from __future__ import annotations

from typing import Any

import torch


CVPR_TASKS = {"qa", "world"}


class StaticLossManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        payload = dict(config or {})
        mode = str(payload.get("mode", "static"))
        if mode != "static":
            raise ValueError(f"CVPR public training supports only static loss mode, got {mode!r}.")
        weights = payload.get("static_weights", {"qa": 1.0, "world": 1.0})
        if not isinstance(weights, dict) or not weights:
            raise ValueError("loss.static_weights must be a non-empty mapping.")
        unknown = sorted(str(task) for task in weights if str(task) not in CVPR_TASKS)
        if unknown:
            raise ValueError(f"CVPR public training supports only qa/world losses, got: {', '.join(unknown)}")
        self.static_weights = {str(task): float(weight) for task, weight in weights.items()}

    def scale(self, task: str, loss: torch.Tensor) -> torch.Tensor:
        if task not in CVPR_TASKS:
            raise ValueError(f"CVPR public training supports only qa/world losses, got task={task!r}.")
        if task not in self.static_weights:
            raise KeyError(f"No static loss weight configured for task={task!r}.")
        return loss * self.static_weights[task]

    def weight_for(self, task: str) -> float:
        if task not in CVPR_TASKS:
            raise ValueError(f"CVPR public training supports only qa/world losses, got task={task!r}.")
        return float(self.static_weights[task])
