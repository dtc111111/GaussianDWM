from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gaussiandwm_cvpr.eval.export_dists_layout import export_layout


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path)


def test_export_layout_writes_camfront_pm1_fid_layout(tmp_path: Path) -> None:
    gt_paths: list[str] = []
    pred_paths: list[str] = []
    for index in range(6):
        gt_path = tmp_path / "gt" / f"gt_{index:03d}.png"
        pred_path = tmp_path / "pred" / f"pred_{index:03d}.png"
        _write_image(gt_path, (index, 20, 40))
        _write_image(pred_path, (40, 20, index))
        gt_paths.append(str(gt_path.relative_to(tmp_path)))
        pred_paths.append(str(pred_path.relative_to(tmp_path)))

    annotation_path = tmp_path / "annotations.jsonl"
    annotation_records = [
        {
            "task": "world",
            "sample_uid": "sample_plus_000001",
            "source_scene_id": "scene_000001",
            "views": ["CAM_FRONT"],
            "pseudo_source_name": "frame_movecam2egoy+1",
            "pseudo_tokens": [f"{index:06d}" for index in range(3)],
            "gt_rgb_frame_paths": gt_paths[:3],
        },
        {
            "task": "world",
            "sample_uid": "sample_minus_000001",
            "source_scene_id": "scene_000001",
            "views": ["CAM_FRONT"],
            "pseudo_source_name": "frame_movecam2egoy-1",
            "pseudo_tokens": [f"{index:06d}" for index in range(3, 6)],
            "gt_rgb_frame_paths": gt_paths[3:],
        },
    ]
    annotation_path.write_text(
        "\n".join(json.dumps(record) for record in annotation_records) + "\n",
        encoding="utf-8",
    )

    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_type": "world",
                        "sample_uid": "sample_plus_000001",
                        "pred": {"rgb_frame_paths": pred_paths[:3]},
                    },
                    {
                        "task_type": "world",
                        "sample_uid": "sample_minus_000001",
                        "pred": {"rgb_frame_paths": pred_paths[3:]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "output"
    manifest = export_layout(
        annotation_path=annotation_path,
        predictions_json_paths=[predictions_path],
        output_root=output_root,
        camera="CAM_FRONT",
        cameras=None,
        layout_name=None,
        shifts={"1"},
        missing_prediction="error",
        overwrite="replace",
    )

    assert manifest["per_shift"]["1"]["fid_real_count"] == 6
    assert manifest["per_shift"]["1"]["fid_fake_count"] == 6
    fake_all = output_root / "fid_imgs" / "dists_camfront_pm1" / "fake_all"
    real_all = output_root / "fid_imgs" / "dists_camfront_pm1" / "real_all"
    assert fake_all.is_dir()
    assert real_all.is_dir()
    assert sorted(path.name for path in fake_all.iterdir()) == [
        *(f"M1__scene_000001__CAM_FRONT__{index:06d}.png" for index in range(3, 6)),
        *(f"P1__scene_000001__CAM_FRONT__{index:06d}.png" for index in range(3)),
    ]
    assert sorted(path.name for path in real_all.iterdir()) == [
        f"scene_000001__CAM_FRONT__{index:06d}.png" for index in range(6)
    ]


def test_export_layout_skips_unpaired_shift_scene(tmp_path: Path) -> None:
    gt_paths: list[str] = []
    pred_paths: list[str] = []
    for index in range(6):
        gt_path = tmp_path / "gt" / f"gt_{index:03d}.png"
        pred_path = tmp_path / "pred" / f"pred_{index:03d}.png"
        _write_image(gt_path, (index, 20, 40))
        _write_image(pred_path, (40, 20, index))
        gt_paths.append(str(gt_path.relative_to(tmp_path)))
        pred_paths.append(str(pred_path.relative_to(tmp_path)))

    annotation_path = tmp_path / "annotations.jsonl"
    annotation_path.write_text(
        json.dumps(
            {
                "task": "world",
                "sample_uid": "sample_plus_000001",
                "source_scene_id": "scene_000001",
                "views": ["CAM_FRONT"],
                "pseudo_source_name": "frame_movecam2egoy+1",
                "pseudo_tokens": [f"{index:06d}" for index in range(6)],
                "gt_rgb_frame_paths": gt_paths,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_type": "world",
                        "sample_uid": "sample_plus_000001",
                        "pred": {"rgb_frame_paths": pred_paths},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "output"
    manifest = export_layout(
        annotation_path=annotation_path,
        predictions_json_paths=[predictions_path],
        output_root=output_root,
        camera="CAM_FRONT",
        cameras=None,
        layout_name=None,
        shifts={"1"},
        missing_prediction="error",
        overwrite="replace",
    )

    assert manifest["per_shift"]["1"]["fid_real_count"] == 0
    assert manifest["per_shift"]["1"]["fid_fake_count"] == 0
