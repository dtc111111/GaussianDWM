from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import torch

from gaussiandwm_cvpr.eval.qa_metrics import evaluate_qa_from_config_dir
from gaussiandwm_cvpr.infer.qa import run_qa_inference_from_config_dir
from gaussiandwm_cvpr.infer.world import run_world_inference_from_config_dir


def test_public_inference_entries_accept_run_dir_none() -> None:
    qa_sig = signature(run_qa_inference_from_config_dir)
    world_sig = signature(run_world_inference_from_config_dir)

    assert qa_sig.parameters["run_dir"].default is None
    assert world_sig.parameters["run_dir"].default is None


def test_qa_metrics_output_labels_public_approximate_protocol(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "eval.yaml").write_text(
        "\n".join(
            [
                "task_filter: [qa]",
                "dataset_filter: [qa_val]",
                "qa:",
                "  enabled_groups: [NuInteract_Caption]",
            ]
        ),
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "task_type": "qa",
                        "sample_uid": "sample_000001",
                        "dataset_name": "qa_val",
                        "qa_group": "NuInteract_Caption",
                        "pred": "a car is stopped",
                        "gt": "a car is stopped",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    metrics_path = evaluate_qa_from_config_dir(
        config_dir=config_dir,
        predictions_path=predictions_path,
        output_dir=tmp_path / "metrics",
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert payload["metric_protocol"] == "public_approximate"
    assert "not official paper" in payload["metric_note"]
    assert payload["summary"]["groups"]["NuInteract_Caption"]["metric_protocol"] == "public_approximate"


def test_world_inference_without_gt_paths_writes_all_generated_frames(tmp_path: Path, monkeypatch) -> None:
    from gaussiandwm_cvpr.infer import world as world_infer

    config_dir = tmp_path / "config"
    config_dir.mkdir()

    def fake_load_yaml(path: str | Path) -> dict:
        filename = Path(path).name
        if filename == "data.yaml":
            return {"world": {}, "datasets": {"world_val": {}}}
        if filename == "gaussian_token.yaml":
            return {
                "cache_root": str(tmp_path / "cache"),
                "coarse_method": "voxel_topk",
                "coarse_k": 512,
                "fine_k": {"world": 512},
                "normalizer_version": "v2",
            }
        if filename == "infer.yaml":
            return {
                "split": "val",
                "dataset_names": ["world_val"],
                "infer_id": "unit",
                "batch_size": 1,
                "world": {"condition_mode": "image_only", "rgb_save_mode": "svd_postprocess"},
            }
        if filename == "model.yaml":
            return {"model_id": "fake-model", "revision": "main"}
        raise AssertionError(path)

    class FakeProcessor:
        tokenizer = object()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> FakeProcessor:
            return FakeProcessor()

    class FakeCondFusion:
        def image_condition(self, ref_pixel_values: torch.Tensor) -> torch.Tensor:
            return torch.zeros(ref_pixel_values.shape[0], 1)

    class FakeWorldHead:
        def generate(self, **_: object) -> dict[str, torch.Tensor]:
            return {
                "rgb": torch.zeros(1, 3, 3, 4, 4),
                "depth": torch.ones(1, 3, 1, 4, 4),
            }

    class FakeModel:
        cond_fusion = FakeCondFusion()
        world_head = FakeWorldHead()

        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> "FakeModel":
            return FakeModel()

        def to(self, device: torch.device) -> "FakeModel":
            return self

        def eval(self) -> None:
            return None

    batch = {
        "ref_pixel_values": torch.zeros(1, 3, 4, 4),
        "pseudo_pixel_values": torch.zeros(1, 3, 4, 4),
        "pseudo_depth_values": torch.zeros(1, 1, 4, 4),
        "world_meta": {
            "fps": [12],
            "motion_bucket_id": [127],
            "pseudo_source_name": ["frame_movecam2egoy+1"],
            "pseudo_source_kind": ["move"],
        },
        "sample_uids": ["sample_no_gt"],
        "dataset_names": ["world_val"],
        "meta": {"gt_rgb_frame_paths": [[]], "gt_depth_npz_paths": [[]]},
    }

    monkeypatch.setattr(world_infer, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(world_infer, "load_taxonomy", lambda path: object())
    monkeypatch.setattr(world_infer, "load_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(world_infer, "AutoProcessor", FakeAutoProcessor)
    monkeypatch.setattr(world_infer, "UnifiedGaussianDWM", FakeModel)
    monkeypatch.setattr(world_infer, "GaussFeatureCache", lambda *args, **kwargs: object())
    monkeypatch.setattr(world_infer, "GaussianDWMDataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(world_infer, "DataLoader", lambda *args, **kwargs: [batch])

    predictions_path = run_world_inference_from_config_dir(
        config_dir=config_dir,
        run_dir=tmp_path / "run",
        model_id="fake-model",
        revision="main",
    )
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    item = payload["items"][0]

    assert item["gt"] == {"rgb_frame_paths": [], "depth_npz_paths": []}
    assert len(item["pred"]["rgb_frame_paths"]) == 3
    assert len(item["pred"]["depth_npz_paths"]) == 3
    for rel_path in item["pred"]["rgb_frame_paths"] + item["pred"]["depth_npz_paths"]:
        assert (predictions_path.parent / rel_path).is_file()
