from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from gaussiandwm_cvpr.data.collator import GaussianDWMCollator
from gaussiandwm_cvpr.data.dataset import GaussianDWMDataset
from gaussiandwm_cvpr.data.gauss_cache import GaussFeatureCache
from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer
from gaussiandwm_cvpr.data.records import load_records
from gaussiandwm_cvpr.data.taxonomy import load_taxonomy
from gaussiandwm_cvpr.models.qwen_gauss_backbone import BackboneRequest
from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM
from gaussiandwm_cvpr.utils import dump_json, load_yaml, resolve_path, timestamp_now


def run_world_inference_from_config_dir(
    config_dir: str | Path,
    run_dir: str | Path | None = None,
    model_id: str | Path | None = None,
    revision: str | None = None,
    data_root: str | Path | None = None,
    annotation_path: str | Path | None = None,
    gauss_cache_root: str | Path | None = None,
) -> Path:
    """Run world generation and write prediction metadata plus RGB/depth files."""
    config_root = Path(config_dir).resolve()
    if run_dir is None:
        run_root = Path("outputs/gaussiandwm_cvpr").resolve()
    else:
        run_root = Path(run_dir).resolve()

    data_cfg = load_yaml(config_root / "data.yaml")
    gaussian_cfg = load_yaml(config_root / "gaussian_token.yaml")
    infer_cfg = load_yaml(config_root / "infer.yaml")
    model_cfg = load_yaml(config_root / "model.yaml")
    taxonomy = load_taxonomy(config_root / "budgets.yaml")

    model_name = str(model_id if model_id is not None else model_cfg["model_id"])
    model_revision = str(revision if revision is not None else model_cfg.get("revision", "main"))
    split = str(infer_cfg.get("split", "val"))
    dataset_names = [str(name) for name in infer_cfg.get("dataset_names", [])]
    if not dataset_names:
        raise ValueError("infer.yaml must provide at least one dataset name.")

    if data_root is not None:
        root_text = str(Path(data_root).expanduser())
        for dataset_name in dataset_names:
            dataset_cfg = data_cfg["datasets"][dataset_name]
            dataset_cfg["annotation_root"] = root_text
            dataset_cfg["image_root"] = root_text
            dataset_cfg["gauss_root"] = root_text
    if annotation_path is not None:
        for dataset_name in dataset_names:
            data_cfg["datasets"][dataset_name]["annotation_path"] = str(Path(annotation_path).expanduser())
    if gauss_cache_root is not None:
        gaussian_cfg["cache_root"] = str(Path(gauss_cache_root).expanduser())

    world_cfg = dict(data_cfg.get("world", {}))
    world_cfg["require_target_latents"] = False
    package_root = _package_root_for_config(config_root)
    processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
    records = load_records(
        data_cfg,
        taxonomy=taxonomy,
        split=split,
        dataset_names=dataset_names,
        task_filter=["world"],
        package_root=package_root,
        require_world_targets=False,
    )
    gauss_data_cfg = data_cfg.get("gauss", {})
    if not isinstance(gauss_data_cfg, dict):
        raise TypeError("data.yaml must define gauss as a mapping when present.")
    pose_dir = gauss_data_cfg.get("pose_dir")
    gauss_cache = GaussFeatureCache(
        cache_root=gaussian_cfg["cache_root"],
        coarse_method=str(gaussian_cfg["coarse_method"]),
        coarse_k=int(gaussian_cfg["coarse_k"]),
        fine_k_by_task={str(k): int(v) for k, v in gaussian_cfg["fine_k"].items()},
        normalizer_version=str(gaussian_cfg.get("normalizer_version", "v2")),
        normalizer=GaussNormalizer(
            pose_dir=None if not pose_dir else str(resolve_path(str(pose_dir), base=package_root)),
            pose_template=str(gauss_data_cfg.get("pose_template", "{scene_id}.txt")),
        ),
    )
    dataset = GaussianDWMDataset(
        records=records,
        processor=processor,
        gauss_cache=gauss_cache,
        gaussian_config=gaussian_cfg,
        world_config=world_cfg,
        mode="infer",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(infer_cfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=GaussianDWMCollator(processor.tokenizer, mode="infer"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedGaussianDWM.from_pretrained(model_name, revision=model_revision)
    model.to(device)
    model.eval()

    infer_id = str(infer_cfg.get("infer_id", "auto"))
    if infer_id == "auto":
        infer_id = timestamp_now()
    output_dir = run_root / "infer" / f"world_predictions_{infer_id}"
    num_inference_steps = int(infer_cfg.get("world", {}).get("num_inference_steps", 25))
    world_condition_mode = str(infer_cfg.get("world", {}).get("condition_mode", "learned"))
    text_condition_scale = float(infer_cfg.get("world", {}).get("text_condition_scale", 0.05))
    rgb_save_mode = str(infer_cfg.get("world", {}).get("rgb_save_mode", "svd_postprocess"))
    guidance_scale = float(infer_cfg.get("world", {}).get("guidance_scale", 2.0))
    noise_aug_strength = float(infer_cfg.get("world", {}).get("noise_aug_strength", 0.02))
    if world_condition_mode not in {"learned", "image_only"}:
        raise ValueError(f"Unsupported world_condition_mode={world_condition_mode!r}")
    if rgb_save_mode not in {"svd_postprocess", "legacy_minmax"}:
        raise ValueError(f"Unsupported rgb_save_mode={rgb_save_mode!r}")

    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            if world_condition_mode == "learned":
                qwen_inputs = _move_tensors_to_device(batch["qwen_inputs"], device)
                bb = model.backbone.forward_backbone(
                    BackboneRequest(
                        qwen_inputs=qwen_inputs,
                        need_token_hidden=False,
                        need_global_condition=True,
                        task_type="world",
                        use_gumbel=False,
                    )
                )
                if bb.global_condition is None:
                    raise RuntimeError("learned world condition requires global_condition from the backbone")
                text_cond_seed = bb.global_condition * float(text_condition_scale)
                cond_embeddings = model.cond_fusion(
                    ref_pixel_values=batch["ref_pixel_values"].to(device),
                    text_cond_seed=text_cond_seed,
                )
            else:
                cond_embeddings = model.cond_fusion.image_condition(
                    ref_pixel_values=batch["ref_pixel_values"].to(device),
                )

            generated = model.world_head.generate(
                pseudo_pixel_values=batch["pseudo_pixel_values"].to(device),
                pseudo_depth_values=batch["pseudo_depth_values"].to(device),
                cond_embeddings=cond_embeddings,
                world_meta=batch["world_meta"],
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                noise_aug_strength=float(noise_aug_strength),
            )
            rgb = generated["rgb"].detach().cpu()
            depth = generated["depth"].detach().cpu()
            gt_rgb_paths = batch["meta"].get("gt_rgb_frame_paths", [[] for _ in range(rgb.shape[0])])
            gt_depth_paths = batch["meta"].get("gt_depth_npz_paths", [[] for _ in range(rgb.shape[0])])

            for index in range(rgb.shape[0]):
                sample_uid = str(batch["sample_uids"][index])
                sample_dir = media_dir / sample_uid
                sample_dir.mkdir(parents=True, exist_ok=True)
                sample_gt_rgb_paths = list(gt_rgb_paths[index]) if index < len(gt_rgb_paths) else []
                sample_gt_depth_paths = list(gt_depth_paths[index]) if index < len(gt_depth_paths) else []
                frame_count = max(
                    len(sample_gt_rgb_paths),
                    len(sample_gt_depth_paths),
                    _generated_frame_count(rgb[index]),
                    _generated_frame_count(depth[index]),
                )
                rgb_frames = _expand_frame_tensor(rgb[index], frame_count)
                depth_frames = _expand_frame_tensor(depth[index], frame_count)
                pred_rgb_paths: list[str] = []
                pred_depth_paths: list[str] = []
                for frame_idx in range(frame_count):
                    rgb_path = sample_dir / f"frame_{frame_idx:03d}_rgb.png"
                    depth_path = sample_dir / f"frame_{frame_idx:03d}_depth.npz"
                    _save_rgb_tensor(
                        rgb_frames[min(frame_idx, len(rgb_frames) - 1)],
                        rgb_path,
                        mode=rgb_save_mode,
                    )
                    np.savez_compressed(
                        depth_path,
                        depth=_to_depth_array(depth_frames[min(frame_idx, len(depth_frames) - 1)]),
                    )
                    pred_rgb_paths.append(str(rgb_path.relative_to(output_dir)))
                    pred_depth_paths.append(str(depth_path.relative_to(output_dir)))
                predictions.append(
                    {
                        "task_type": "world",
                        "sample_uid": sample_uid,
                        "dataset_name": batch["dataset_names"][index],
                        "pred": {
                            "rgb_frame_paths": pred_rgb_paths,
                            "depth_npz_paths": pred_depth_paths,
                        },
                        "gt": {
                            "rgb_frame_paths": sample_gt_rgb_paths,
                            "depth_npz_paths": sample_gt_depth_paths,
                        },
                        "meta": {
                            "fps": _batch_value(batch["world_meta"].get("fps"), index),
                            "motion_bucket_id": _batch_value(batch["world_meta"].get("motion_bucket_id"), index),
                            "pseudo_source_name": _batch_value(batch["world_meta"].get("pseudo_source_name"), index),
                            "pseudo_source_kind": _batch_value(batch["world_meta"].get("pseudo_source_kind"), index),
                            "world_condition_mode": world_condition_mode,
                            "text_condition_scale": float(text_condition_scale),
                            "rgb_save_mode": rgb_save_mode,
                            "guidance_scale": float(guidance_scale),
                            "noise_aug_strength": float(noise_aug_strength),
                        },
                    },
                )

    output_path = output_dir / "predictions.json"
    dump_json(
        {
            "task_type": "world",
            "model_id": model_name,
            "revision": model_revision,
            "config_dir": str(config_root),
            "split": split,
            "dataset_names": dataset_names,
            "count": len(predictions),
            "items": predictions,
        },
        output_path,
    )
    return output_path


def _move_tensors_to_device(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in payload.items()}


def _save_rgb_tensor(tensor: torch.Tensor, path: Path, *, mode: str) -> None:
    image = tensor.detach().to(dtype=torch.float32)
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image[:3]
    if mode == "svd_postprocess":
        image = (image / 2.0 + 0.5).clamp(0.0, 1.0)
    elif mode == "legacy_minmax":
        image = image - image.min()
        image = image / max(float(image.max()), 1e-6)
    array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def _expand_frame_tensor(tensor: torch.Tensor, frame_count: int) -> list[torch.Tensor]:
    if tensor.ndim == 4:
        return [tensor[i] for i in range(tensor.shape[0])]
    if tensor.ndim == 3:
        return [tensor for _ in range(frame_count)]
    if tensor.ndim == 2:
        return [tensor.unsqueeze(0) for _ in range(frame_count)]
    raise ValueError(f"Unsupported world frame tensor shape: {tuple(tensor.shape)}")


def _generated_frame_count(tensor: torch.Tensor) -> int:
    if tensor.ndim == 4:
        return int(tensor.shape[0])
    if tensor.ndim in {2, 3}:
        return 1
    raise ValueError(f"Unsupported world frame tensor shape: {tuple(tensor.shape)}")


def _to_depth_array(tensor: torch.Tensor) -> np.ndarray:
    depth = tensor.detach().to(dtype=torch.float32)
    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    return depth.numpy()


def _batch_value(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        item = value[index]
        return item.item() if item.ndim == 0 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _package_root_for_config(config_root: Path) -> Path:
    if config_root.parent.name == "configs":
        return config_root.parent.parent
    return config_root
