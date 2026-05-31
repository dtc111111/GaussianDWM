from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gaussiandwm_cvpr.utils import timestamp_now


SOURCE_RE = re.compile(r"^frame_movecam2egoy(?P<sign>[+-])(?P<shift>\d+)$")


@dataclass(frozen=True)
class PredictionRef:
    root: Path
    item: dict[str, Any]


@dataclass(frozen=True)
class ExportFrame:
    shift: str
    sign: str
    scene_id: str
    camera: str
    token: str
    pred_rgb_path: Path
    gt_rgb_path: Path


def export_layout(
    *,
    annotation_path: str | Path,
    predictions_json_paths: list[str | Path],
    output_root: str | Path,
    camera: str,
    cameras: list[str] | None = None,
    layout_name: str | None = None,
    shifts: set[str],
    missing_prediction: str = "error",
    overwrite: str = "error",
) -> dict[str, Any]:
    if missing_prediction not in {"error", "skip"}:
        raise ValueError("missing_prediction must be 'error' or 'skip'.")
    if overwrite not in {"error", "replace"}:
        raise ValueError("overwrite must be 'error' or 'replace'.")

    annotation_file = Path(annotation_path)
    prediction_files = [Path(path) for path in predictions_json_paths]
    output_dir = Path(output_root)
    if output_dir.exists():
        if overwrite == "error":
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(prediction_files)
    annotation_items = _load_json_or_jsonl(annotation_file)
    requested_cameras = list(cameras or [camera])
    requested_camera_set = set(requested_cameras)
    if layout_name is None:
        layout_name = "dists_camfront" if requested_camera_set == {"CAM_FRONT"} else "dists"

    frames: list[ExportFrame] = []
    skipped_camera = 0
    skipped_missing_prediction = 0
    missing_examples: list[str] = []
    for item in annotation_items:
        item_views = [str(view) for view in item.get("views", [])]
        item_camera = item_views[0] if len(item_views) == 1 else None
        if item_camera not in requested_camera_set:
            skipped_camera += 1
            continue
        source_match = SOURCE_RE.match(str(item.get("pseudo_source_name") or ""))
        if source_match is None or source_match.group("shift") not in shifts:
            continue

        sample_uid = str(item["sample_uid"])
        prediction = predictions.get(sample_uid)
        if prediction is None:
            if missing_prediction == "skip":
                skipped_missing_prediction += 1
                if len(missing_examples) < 20:
                    missing_examples.append(sample_uid)
                continue
            raise KeyError(f"Missing prediction for sample_uid={sample_uid!r}")

        pred_rgb_paths = list(prediction.item.get("pred", {}).get("rgb_frame_paths", []))
        gt_rgb_paths = [str(path) for path in item.get("gt_rgb_frame_paths", [])]
        tokens = _pseudo_tokens_for_item(item)
        if not (len(pred_rgb_paths) == len(gt_rgb_paths) == len(tokens)):
            raise ValueError(
                f"Length mismatch for sample_uid={sample_uid!r}: "
                f"pred={len(pred_rgb_paths)}, gt={len(gt_rgb_paths)}, tokens={len(tokens)}"
            )

        scene_id = str(item.get("source_scene_id", item.get("scene_idx", "unknown_scene")))
        for index, token in enumerate(tokens):
            frames.append(
                ExportFrame(
                    shift=source_match.group("shift"),
                    sign=source_match.group("sign"),
                    scene_id=scene_id,
                    camera=str(item_camera),
                    token=token,
                    pred_rgb_path=_resolve_prediction_path(prediction.root, str(pred_rgb_paths[index])),
                    gt_rgb_path=_resolve_annotation_path(annotation_file.parent, gt_rgb_paths[index]),
                )
            )

    per_shift: dict[str, dict[str, Any]] = {}
    total_real = 0
    total_fake = 0
    for shift in sorted(shifts, key=_natural_sort_key):
        shift_frames = [frame for frame in frames if frame.shift == shift]
        scenes_by_sign = {
            "-": {frame.scene_id for frame in shift_frames if frame.sign == "-"},
            "+": {frame.scene_id for frame in shift_frames if frame.sign == "+"},
        }
        paired_scenes = scenes_by_sign["-"] & scenes_by_sign["+"]
        shift_summary = _write_shift_links(
            output_dir,
            shift=shift,
            frames=shift_frames,
            paired_scenes=paired_scenes,
            layout_name=layout_name,
        )
        total_real += shift_summary["fid_real_count"]
        total_fake += shift_summary["fid_fake_count"]
        per_shift[shift] = {
            "paired_scene_count": len(paired_scenes),
            "paired_scenes": sorted(paired_scenes, key=_natural_sort_key),
            **shift_summary,
        }

    manifest = {
        "created_at": timestamp_now(),
        "annotation_path": str(annotation_file),
        "predictions_json_paths": [str(path) for path in prediction_files],
        "output_root": str(output_dir),
        "camera": camera,
        "requested_cameras": requested_cameras,
        "requested_shifts": sorted(shifts, key=_natural_sort_key),
        "annotation_item_count": len(annotation_items),
        "prediction_item_count": len(predictions),
        "export_frame_count": len(frames),
        "layout_name": layout_name,
        "fid_real_count": total_real,
        "fid_fake_count": total_fake,
        "skipped_camera": skipped_camera,
        "missing_prediction_policy": missing_prediction,
        "skipped_missing_prediction": skipped_missing_prediction,
        "missing_prediction_examples": missing_examples,
        "per_shift": per_shift,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _load_predictions(paths: list[Path]) -> dict[str, PredictionRef]:
    predictions: dict[str, PredictionRef] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
        if not isinstance(items, list):
            raise TypeError(f"Predictions payload must be a list or contain an items list: {path}")
        for item in items:
            sample_uid = str(item["sample_uid"])
            if sample_uid in predictions:
                raise ValueError(f"Duplicate prediction sample_uid={sample_uid!r}")
            predictions[sample_uid] = PredictionRef(root=path.parent, item=item)
    return predictions


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"] if isinstance(payload, dict) and "items" in payload else payload
    if not isinstance(items, list):
        raise TypeError(f"Annotation payload must be a list or contain an items list: {path}")
    return items


def _pseudo_tokens_for_item(item: dict[str, Any]) -> list[str]:
    raw_tokens = item.get("pseudo_tokens")
    if isinstance(raw_tokens, list) and raw_tokens:
        return [str(token) for token in raw_tokens]
    paths = item.get("pseudo_video_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"Annotation item {item.get('sample_uid')!r} lacks pseudo token metadata.")
    return [Path(str(path)).stem for path in paths]


def _resolve_prediction_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _resolve_annotation_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _write_shift_links(
    output_root: Path,
    *,
    shift: str,
    frames: list[ExportFrame],
    paired_scenes: set[str],
    layout_name: str,
) -> dict[str, int]:
    fid_root = output_root / "fid_imgs" / f"{layout_name}_pm{shift}"
    real_all = fid_root / "real_all"
    fake_all = fid_root / "fake_all"
    fake_source_links = 0
    fid_real_count = 0
    fid_fake_count = 0
    seen_real: set[tuple[str, str, str]] = set()
    seen_fake_fid: set[tuple[str, str, str, str]] = set()
    seen_fake_source: set[tuple[str, str, str, str, str]] = set()

    for frame in frames:
        if frame.scene_id not in paired_scenes:
            continue
        direction_label = "M" if frame.sign == "-" else "P"
        source_name = f"frame_movecam2egoy{frame.sign}{shift}"
        source_root = output_root / "fake_sources" / f"pm{shift}" / source_name / frame.scene_id

        fake_filename = f"{frame.token}{frame.pred_rgb_path.suffix.lower()}"
        for camera_root in (source_root / frame.camera, source_root / "color" / frame.camera):
            fake_source_key = (str(camera_root), frame.scene_id, frame.camera, frame.token, direction_label)
            if fake_source_key not in seen_fake_source:
                _symlink(frame.pred_rgb_path, camera_root / fake_filename)
                fake_source_links += 1
                seen_fake_source.add(fake_source_key)

        real_key = (frame.scene_id, frame.camera, frame.token)
        if real_key not in seen_real:
            real_name = f"{frame.scene_id}__{frame.camera}__{frame.token}{frame.gt_rgb_path.suffix.lower()}"
            _symlink(frame.gt_rgb_path, real_all / real_name)
            fid_real_count += 1
            seen_real.add(real_key)

        fake_fid_key = (direction_label, frame.scene_id, frame.camera, frame.token)
        if fake_fid_key not in seen_fake_fid:
            fake_name = (
                f"{direction_label}{shift}__{frame.scene_id}__{frame.camera}__"
                f"{frame.token}{frame.pred_rgb_path.suffix.lower()}"
            )
            _symlink(frame.pred_rgb_path, fake_all / fake_name)
            fid_fake_count += 1
            seen_fake_fid.add(fake_fid_key)

    return {
        "fake_source_link_count": fake_source_links,
        "fid_real_count": fid_real_count,
        "fid_fake_count": fid_fake_count,
    }


def _symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def _natural_sort_key(value: str) -> tuple[int, int, str] | tuple[int, str]:
    try:
        return (0, int(value), value)
    except ValueError:
        return (1, value)
