from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def save_checkpoint(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None = None,
    run_dir: str | Path,
    step: int,
    keep_last_k: int | None = None,
) -> Path:
    checkpoint_dir = Path(run_dir) / "checkpoints" / f"checkpoint-{int(step)}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(checkpoint_dir / "model_state.safetensors"))
    training_state: dict[str, Any] = {
        "step": int(step),
        "optimizer": optimizer.state_dict(),
    }
    if lr_scheduler is not None:
        training_state["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(
        training_state,
        checkpoint_dir / "training_state.pt",
    )
    (checkpoint_dir / "checkpoint_meta.json").write_text(
        json.dumps({"step": int(step)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if keep_last_k is not None and int(keep_last_k) > 0:
        _prune_old_checkpoints(Path(run_dir) / "checkpoints", keep_last_k=int(keep_last_k))
    return checkpoint_dir


def load_resume_state(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None = None,
    checkpoint_dir: str | Path,
) -> int:
    checkpoint = Path(checkpoint_dir)
    model_state_path = checkpoint / "model_state.safetensors"
    state_path = checkpoint / "training_state.pt"
    if not model_state_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing model_state.safetensors: {checkpoint}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint is missing training_state.pt: {checkpoint}")
    model.load_state_dict(load_file(str(model_state_path)), strict=True)
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict) or "optimizer" not in state or "step" not in state:
        raise ValueError(f"Invalid resume training state: {state_path}")
    optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None:
        if "lr_scheduler" not in state:
            raise ValueError(f"Resume training state is missing lr_scheduler: {state_path}")
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    return int(state["step"])


def save_final_model(model: Any, output_dir: str | Path) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _merge_peft_module(getattr(getattr(model, "backbone", None), "qwen", None), model, ("backbone", "qwen"))
    world_bundle = getattr(getattr(model, "world_head", None), "bundle", None)
    _merge_peft_module(getattr(world_bundle, "unet", None), world_bundle, ("unet",))
    model.save_pretrained(target)
    return target


def _prune_old_checkpoints(checkpoint_root: Path, *, keep_last_k: int) -> None:
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_root.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        checkpoints.append((step, path))
    for _, path in sorted(checkpoints)[:-keep_last_k]:
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        path.rmdir()


def _merge_peft_module(module: Any, owner: Any, attribute_path: tuple[str, ...]) -> None:
    merge_and_unload = getattr(module, "merge_and_unload", None)
    if not callable(merge_and_unload):
        return
    merged = merge_and_unload()
    target = owner
    for attribute in attribute_path[:-1]:
        target = getattr(target, attribute)
    setattr(target, attribute_path[-1], merged)
