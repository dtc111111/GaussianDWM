from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussiandwm_cvpr.utils import dump_json, load_json_or_jsonl, load_yaml, timestamp_now


def evaluate_world_from_config_dir(
    config_dir: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> Path:
    config_root = Path(config_dir).resolve()
    output_root = Path(output_dir).resolve()
    eval_cfg = load_yaml(config_root / "eval.yaml")
    items = load_json_or_jsonl(predictions_path)
    task_filter = set(str(task) for task in eval_cfg.get("task_filter", ["world"]))
    dataset_filter = set(str(name) for name in eval_cfg.get("dataset_filter", []))
    filtered = [
        item
        for item in items
        if str(item.get("task_type")) in task_filter
        and (not dataset_filter or str(item.get("dataset_name")) in dataset_filter)
    ]

    summary, per_items = evaluate_world_items(filtered, infer_root=Path(predictions_path).resolve().parent)
    output_path = output_root / f"world_metrics_{timestamp_now()}.json"
    dump_json(
        {
            "task_type": "world",
            "config_dir": str(config_root),
            "predictions_path": str(predictions_path),
            "count": len(per_items),
            "summary": summary,
            "items": per_items,
        },
        output_path,
    )
    return output_path


def evaluate_world_items(
    items: list[dict[str, Any]],
    *,
    infer_root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(infer_root)
    per_items: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    abs_rel_values: list[float] = []
    rmse_values: list[float] = []

    for item in items:
        pred_rgb_paths = list(item.get("pred", {}).get("rgb_frame_paths", []))
        gt_rgb_paths = list(item.get("gt", {}).get("rgb_frame_paths", []))
        pred_depth_paths = list(item.get("pred", {}).get("depth_npz_paths", []))
        gt_depth_paths = list(item.get("gt", {}).get("depth_npz_paths", []))
        missing = (
            not pred_rgb_paths
            or not gt_rgb_paths
            or len(pred_rgb_paths) != len(gt_rgb_paths)
            or len(pred_depth_paths) != len(gt_depth_paths)
        )
        clip_psnr: list[float] = []
        clip_ssim: list[float] = []
        clip_abs_rel: list[float] = []
        clip_rmse: list[float] = []

        if not missing:
            for pred_rgb_rel, gt_rgb_rel in zip(pred_rgb_paths, gt_rgb_paths):
                pred_rgb_path = _resolve_path(root, str(pred_rgb_rel))
                gt_rgb_path = _resolve_path(root, str(gt_rgb_rel))
                if not (pred_rgb_path.exists() and gt_rgb_path.exists()):
                    missing = True
                    break
                pred_rgb = _load_image_tensor(pred_rgb_path)
                gt_rgb = _load_image_tensor(gt_rgb_path, size_wh=(int(pred_rgb.shape[-1]), int(pred_rgb.shape[-2])))
                clip_psnr.append(_psnr(pred_rgb, gt_rgb))
                clip_ssim.append(_ssim(pred_rgb, gt_rgb))

        if not missing and pred_depth_paths:
            for pred_depth_rel, gt_depth_rel in zip(pred_depth_paths, gt_depth_paths):
                pred_depth_path = _resolve_path(root, str(pred_depth_rel))
                gt_depth_path = _resolve_path(root, str(gt_depth_rel))
                if not (pred_depth_path.exists() and gt_depth_path.exists()):
                    missing = True
                    break
                pred_depth = _load_npz(pred_depth_path)
                gt_depth = _load_npz(gt_depth_path)
                if pred_depth.shape != gt_depth.shape:
                    gt_depth = _resize_depth_array(gt_depth, pred_depth.shape)
                valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (np.abs(gt_depth) > 1e-6)
                if np.any(valid):
                    pred_valid = pred_depth[valid]
                    gt_valid = gt_depth[valid]
                    clip_abs_rel.append(float(np.mean(np.abs(pred_valid - gt_valid) / np.clip(np.abs(gt_valid), 1e-6, None))))
                    clip_rmse.append(float(np.sqrt(np.mean((pred_valid - gt_valid) ** 2))))

        item_psnr = _mean(clip_psnr)
        item_ssim = _mean(clip_ssim)
        item_abs_rel = _mean(clip_abs_rel)
        item_rmse = _mean(clip_rmse)
        if not missing:
            psnr_values.append(item_psnr)
            ssim_values.append(item_ssim)
            if clip_abs_rel:
                abs_rel_values.append(item_abs_rel)
                rmse_values.append(item_rmse)
        failure_score = 1000.0 if missing else (1.0 - item_ssim) + item_abs_rel
        per_items.append(
            {
                "task_type": "world",
                "sample_uid": item.get("sample_uid"),
                "dataset_name": item.get("dataset_name"),
                "is_failure": missing,
                "failure_score": float(failure_score),
                "metrics": {
                    "missing_required_files": missing,
                    "psnr_rgb": item_psnr,
                    "ssim_rgb": item_ssim,
                    "abs_rel_depth": item_abs_rel,
                    "rmse_depth": item_rmse,
                },
                "pred": item.get("pred"),
                "gt": item.get("gt"),
                "meta": item.get("meta") if isinstance(item.get("meta"), dict) else {},
            }
        )

    return {
        "count": len(items),
        "psnr_rgb": _mean(psnr_values),
        "ssim_rgb": _mean(ssim_values),
        "abs_rel_depth": _mean(abs_rel_values),
        "rmse_depth": _mean(rmse_values),
    }, per_items


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _load_image_tensor(path: Path, *, size_wh: tuple[int, int] | None = None) -> torch.Tensor:
    image_pil = Image.open(path).convert("RGB")
    if size_wh is not None:
        image_pil = image_pil.resize(size_wh)
    image = np.asarray(image_pil, dtype=np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)


def _load_npz(path: Path) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        for key in ("depth", "depth_gt", "depth_proj"):
            if key in payload:
                return np.asarray(payload[key], dtype=np.float32)
        return np.asarray(payload[payload.files[0]], dtype=np.float32)
    return np.asarray(payload, dtype=np.float32)


def _resize_depth_array(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if array.ndim != 2 or len(target_shape) != 2:
        return array
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    resized = Image.fromarray(array.astype(np.float32), mode="F").resize(
        (target_w, target_h),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=np.float32)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = float(torch.mean((pred - gt) ** 2).item())
    if mse <= 1e-12:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    window = _gaussian_window(device=pred.device, dtype=pred.dtype)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.conv2d(pred, window, padding=5, groups=3)
    mu_y = F.conv2d(gt, window, padding=5, groups=3)
    sigma_x = F.conv2d(pred * pred, window, padding=5, groups=3) - mu_x.pow(2)
    sigma_y = F.conv2d(gt * gt, window, padding=5, groups=3) - mu_y.pow(2)
    sigma_xy = F.conv2d(pred * gt, window, padding=5, groups=3) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator.clamp(min=1e-12)).mean().item())


def _gaussian_window(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    kernel_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(3, 1, 11, 11).contiguous()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
