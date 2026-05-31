from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_cosine_schedule_with_warmup

from gaussiandwm_cvpr.data.collator import GaussianDWMCollator
from gaussiandwm_cvpr.data.dataset import GaussianDWMDataset
from gaussiandwm_cvpr.data.gauss_cache import GaussFeatureCache
from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer
from gaussiandwm_cvpr.data.records import load_records
from gaussiandwm_cvpr.data.taxonomy import load_taxonomy
from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM
from gaussiandwm_cvpr.train.checkpoint import load_resume_state, save_checkpoint, save_final_model
from gaussiandwm_cvpr.train.loss import StaticLossManager
from gaussiandwm_cvpr.train.stage_scheduler import StageScheduler
from gaussiandwm_cvpr.train.trainable_groups import apply_trainable_groups, inject_lora_from_config
from gaussiandwm_cvpr.utils import dump_json, dump_yaml, load_yaml, resolve_path


CVPR_TASKS = {"qa", "world"}
DEFAULT_MODEL_ID = "dtc111/GaussianDWM"
DEFAULT_REVISION = "main"


def run_training_from_config_dir(
    task: str,
    config_dir: str | Path,
    run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
    data_root: str | Path | None = None,
    annotation_path: str | Path | None = None,
    gauss_cache_root: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
) -> Path:
    if task not in CVPR_TASKS:
        raise ValueError(f"CVPR public training supports only task='qa' or task='world', got {task!r}.")
    if run_dir is not None and output_dir is not None:
        raise ValueError("Specify only one of run_dir and output_dir; output_dir is used only when run_dir is omitted.")

    config_root = Path(config_dir)
    budgets_path = config_root / "budgets.yaml"
    model_config = load_yaml(config_root / "model.yaml")
    data_config = load_yaml(config_root / "data.yaml")
    gaussian_config = load_yaml(config_root / "gaussian_token.yaml")
    train_config = load_yaml(config_root / "train.yaml")
    stage_plan = load_yaml(config_root / "stage_plan.yaml")
    budgets_config = load_yaml(budgets_path)

    if model_id == DEFAULT_MODEL_ID and model_config.get("model_id"):
        model_id = str(model_config["model_id"])
    if revision == DEFAULT_REVISION and model_config.get("revision"):
        revision = str(model_config["revision"])

    if data_root is not None:
        _override_dataset_roots(data_config, Path(data_root))
    if annotation_path is not None:
        _override_annotation_path(data_config, task, Path(annotation_path))
    if gauss_cache_root is not None:
        gaussian_config["cache_root"] = str(gauss_cache_root)

    if run_dir is None:
        run_dir_path = Path(output_dir) if output_dir is not None else Path("outputs/gaussiandwm_cvpr")
    else:
        run_dir_path = Path(run_dir)
    final_output_dir = run_dir_path / "final_model"
    run_dir_path.mkdir(parents=True, exist_ok=True)
    (run_dir_path / "config_snapshot").mkdir(exist_ok=True)
    (run_dir_path / "meta").mkdir(exist_ok=True)
    (run_dir_path / "checkpoints").mkdir(exist_ok=True)

    for name, payload in (
        ("model.yaml", model_config),
        ("data.yaml", data_config),
        ("gaussian_token.yaml", gaussian_config),
        ("train.yaml", train_config),
        ("stage_plan.yaml", stage_plan),
        ("budgets.yaml", budgets_config),
    ):
        dump_yaml(payload, run_dir_path / "config_snapshot" / name)
    dump_json(
        {
            "task": task,
            "model_id": model_id,
            "revision": revision,
            "run_dir": str(run_dir_path),
            "output_dir": str(final_output_dir),
        },
        run_dir_path / "meta" / "run_manifest.json",
    )

    seed = int(train_config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    package_root = config_root.parent.parent
    taxonomy = load_taxonomy(budgets_path)
    records = load_records(
        data_config,
        taxonomy=taxonomy,
        split="train",
        task_filter=[task],
        package_root=package_root,
        require_world_targets=task == "world",
    )

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = UnifiedGaussianDWM.from_pretrained(model_id, revision=revision)
    lora_config = model_config.get("lora")
    if isinstance(lora_config, dict):
        model = inject_lora_from_config(model, lora_config)
    fine_k = gaussian_config.get("fine_k")
    if not isinstance(fine_k, dict):
        raise TypeError("gaussian_token.yaml must define fine_k as a mapping.")
    cache_root = Path(str(gaussian_config.get("cache_root", "outputs/cache/gauss")))
    if not cache_root.is_absolute():
        cache_root = run_dir_path / cache_root
    gauss_data_config = data_config.get("gauss", {})
    if not isinstance(gauss_data_config, dict):
        raise TypeError("data.yaml must define gauss as a mapping when present.")
    pose_dir = gauss_data_config.get("pose_dir")
    normalizer = GaussNormalizer(
        pose_dir=None if not pose_dir else str(resolve_path(str(pose_dir), base=package_root)),
        pose_template=str(gauss_data_config.get("pose_template", "{scene_id}.txt")),
    )
    gauss_cache = GaussFeatureCache(
        cache_root=cache_root,
        coarse_method=str(gaussian_config["coarse_method"]),
        coarse_k=int(gaussian_config["coarse_k"]),
        fine_k_by_task={str(name): int(value) for name, value in fine_k.items()},
        normalizer_version=str(gaussian_config.get("normalizer_version", "v2")),
        normalizer=normalizer,
    )
    dataset = GaussianDWMDataset(
        records=records,
        processor=processor,
        gauss_cache=gauss_cache,
        gaussian_config=gaussian_config,
        world_config=data_config.get("world") if task == "world" else None,
        mode="train",
    )

    scheduler = StageScheduler(stage_plan, per_device_batch_size=train_config["per_device_batch_size"])
    loss_manager = StaticLossManager(train_config.get("loss", {"mode": "static", "static_weights": {task: 1.0}}))
    apply_trainable_groups(model, scheduler.planned_trainable_modules())

    batch_sizes = scheduler.get_batch_sizes(0)
    if task not in batch_sizes:
        raise KeyError(f"per_device_batch_size must define {task!r}.")
    collator = GaussianDWMCollator(tokenizer=processor.tokenizer, mode="train")
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_sizes[task]),
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 0)),
        collate_fn=collator,
    )

    max_steps = int(train_config.get("max_steps", 1))
    optimizer = torch.optim.AdamW(_optimizer_groups(model, train_config.get("optimizer", {})))
    warmup_steps = int(float(train_config.get("scheduler", {}).get("warmup_ratio", 0.0)) * max_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    start_step = 0
    if resume_from_checkpoint is not None:
        start_step = load_resume_state(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_dir=resume_from_checkpoint,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    logging_steps = int(train_config.get("logging_steps", 50))
    checkpoint_config = train_config.get("checkpoint", {})
    save_every_steps = int(checkpoint_config.get("save_every_steps", 0)) if isinstance(checkpoint_config, dict) else 0
    keep_last_k = int(checkpoint_config.get("keep_last_k", 0)) if isinstance(checkpoint_config, dict) else 0
    log_path = run_dir_path / "logs.jsonl"

    step = start_step
    while step < max_steps:
        for batch in dataloader:
            if step >= max_steps:
                break
            scheduler.apply(model, step)
            task_from_batch = str(batch.pop("task"))
            if task_from_batch != task:
                raise ValueError(f"Expected {task!r} batch, got {task_from_batch!r}.")
            model_inputs = {
                key: _move_to_device(value, device)
                for key, value in batch.items()
                if key not in {"sample_uids", "dataset_names", "meta"}
            }
            model_inputs["global_step"] = step
            outputs = model(task, **model_inputs)
            if outputs.loss is None:
                raise RuntimeError(f"Model did not return a loss for task={task!r}.")
            loss = loss_manager.scale(task, outputs.loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if logging_steps > 0 and step % logging_steps == 0:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "task": task, "loss": float(loss.detach().cpu())}))
                    handle.write("\n")
            if save_every_steps > 0 and step % save_every_steps == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    run_dir=run_dir_path,
                    step=step,
                    keep_last_k=keep_last_k if keep_last_k > 0 else None,
                )

    save_final_model(model, final_output_dir)
    return run_dir_path


def _optimizer_groups(model: torch.nn.Module, optimizer_config: Any) -> list[dict[str, Any]]:
    config = dict(optimizer_config or {})
    default_lr = float(config.get("lr", 1.0e-5))
    groups_by_lr: dict[float, list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = default_lr
        if name.startswith("backbone.gauss_aligner"):
            lr = float(config.get("lr_gauss_aligner", default_lr))
        elif name.startswith("cond_fusion"):
            lr = float(config.get("lr_cond_fusion", default_lr))
        elif name.startswith("world_head.bundle.unet") and "lora_" in name:
            lr = float(config.get("lr_world_unet_lora", default_lr))
        elif "lm_head" in name and "lora_" in name:
            lr = float(config.get("lr_qa_lm_head_lora", default_lr))
        elif "lora_" in name:
            lr = float(config.get("lr_qwen_backbone_lora", default_lr))
        groups_by_lr.setdefault(lr, []).append(parameter)
    if not groups_by_lr:
        raise RuntimeError("No trainable parameters are enabled for optimizer construction.")
    return [{"params": params, "lr": lr} for lr, params in sorted(groups_by_lr.items())]


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _override_dataset_roots(data_config: dict[str, Any], data_root: Path) -> None:
    datasets = data_config.get("datasets")
    if not isinstance(datasets, dict):
        raise TypeError("data.yaml must define datasets as a mapping.")
    for dataset in datasets.values():
        if not isinstance(dataset, dict):
            continue
        dataset["annotation_root"] = str(data_root)
        dataset["image_root"] = str(data_root)
        dataset["gauss_root"] = str(data_root)
    clip_config = data_config.get("clip_text_feature")
    if isinstance(clip_config, dict):
        clip_config["root"] = str(data_root)


def _override_annotation_path(data_config: dict[str, Any], task: str, annotation_path: Path) -> None:
    datasets = data_config.get("datasets")
    if not isinstance(datasets, dict):
        raise TypeError("data.yaml must define datasets as a mapping.")
    changed = False
    for dataset in datasets.values():
        if not isinstance(dataset, dict):
            continue
        if dataset.get("major_task") == task and dataset.get("split") == "train":
            dataset["annotation_root"] = str(annotation_path.parent)
            dataset["annotation_path"] = annotation_path.name
            changed = True
    if not changed:
        raise ValueError(f"No train dataset found for task={task!r} to override annotation_path.")
