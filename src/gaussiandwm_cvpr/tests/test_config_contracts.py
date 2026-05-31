from __future__ import annotations

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "configs"
PRIVATE_PATH_TOKEN = "/" + "mnt" + "/"
PRIVATE_NAME_TOKENS = ["chen" + suffix for suffix in ("xf", "qu", "yi")]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict), f"{path} must contain a YAML mapping"
    return payload


def test_public_configs_exist_for_qa_and_world() -> None:
    required = {
        "model.yaml",
        "data.yaml",
        "gaussian_token.yaml",
        "train.yaml",
        "infer.yaml",
        "eval.yaml",
        "budgets.yaml",
        "stage_plan.yaml",
    }
    for task_name in ("qa", "world"):
        task_dir = CONFIG_ROOT / task_name
        assert task_dir.is_dir()
        assert {path.name for path in task_dir.glob("*.yaml")} == required


def test_public_configs_have_no_private_mnt_defaults() -> None:
    for path in CONFIG_ROOT.rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert PRIVATE_PATH_TOKEN not in text
        for token in PRIVATE_NAME_TOKENS:
            assert token not in text


def test_model_configs_default_to_hf_repo_root() -> None:
    for task_name in ("qa", "world"):
        payload = _load_yaml(CONFIG_ROOT / task_name / "model.yaml")
        assert payload["model_id"] == "dtc111/GaussianDWM"
        assert payload["revision"] == "main"
        assert payload.get("subfolder") is None


def test_selector_scope_is_cvpr_only() -> None:
    qa_cfg = _load_yaml(CONFIG_ROOT / "qa" / "gaussian_token.yaml")
    world_cfg = _load_yaml(CONFIG_ROOT / "world" / "gaussian_token.yaml")
    assert qa_cfg["coarse_method"] == "voxel_topk"
    assert qa_cfg["coarse_k"] == 4096
    assert qa_cfg["fine_method"] == "similarity"
    assert qa_cfg["fine_k"]["qa"] == 512
    assert world_cfg["coarse_method"] == "voxel_topk"
    assert world_cfg["coarse_k"] == 512
    assert world_cfg["fine_method"] == "identity"
    assert world_cfg["fine_k"]["world"] == 512
    assert "attention" not in {qa_cfg["fine_method"], world_cfg["fine_method"]}


def test_budgets_exclude_traj() -> None:
    for task_name in ("qa", "world"):
        payload = _load_yaml(CONFIG_ROOT / task_name / "budgets.yaml")
        assert payload["major_tasks"] == ["qa", "world"]
        assert "traj" not in payload["task_budgets"]


def test_world_infer_defaults_use_selected_cvpr_conditioning() -> None:
    payload = _load_yaml(CONFIG_ROOT / "world" / "infer.yaml")
    assert payload["world"]["condition_mode"] == "learned"
    assert payload["world"]["text_condition_scale"] == 0.05
    assert payload["world"]["rgb_save_mode"] == "svd_postprocess"
