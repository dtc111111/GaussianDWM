from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


SCRIPT_MODULES = [
    "gaussiandwm_cvpr.scripts.train_qa",
    "gaussiandwm_cvpr.scripts.train_world",
    "gaussiandwm_cvpr.scripts.infer_qa",
    "gaussiandwm_cvpr.scripts.infer_world",
    "gaussiandwm_cvpr.scripts.eval_qa",
    "gaussiandwm_cvpr.scripts.eval_world",
    "gaussiandwm_cvpr.scripts.export_world_dists_layout",
]


def test_public_script_modules_expose_main() -> None:
    for module_name in SCRIPT_MODULES:
        module = importlib.import_module(module_name)
        assert callable(module.main)


def test_train_scripts_do_not_forward_default_output_dir_with_explicit_run_dir(monkeypatch, tmp_path, capsys) -> None:
    from gaussiandwm_cvpr.scripts import train_qa, train_world

    calls: list[dict[str, object]] = []

    def fake_train(**kwargs: object) -> Path:
        calls.append(kwargs)
        return tmp_path / str(kwargs["task"])

    monkeypatch.setattr(train_qa, "run_training_from_config_dir", fake_train)
    monkeypatch.setattr(train_world, "run_training_from_config_dir", fake_train)

    monkeypatch.setattr(sys, "argv", ["train_qa", "--run_dir", str(tmp_path / "qa_run")])
    train_qa.main()
    monkeypatch.setattr(sys, "argv", ["train_world", "--run_dir", str(tmp_path / "world_run")])
    train_world.main()
    capsys.readouterr()

    assert calls[0]["task"] == "qa"
    assert calls[0]["run_dir"] == str(tmp_path / "qa_run")
    assert calls[0]["output_dir"] is None
    assert calls[1]["task"] == "world"
    assert calls[1]["run_dir"] == str(tmp_path / "world_run")
    assert calls[1]["output_dir"] is None


def test_train_scripts_forward_default_output_dir_without_run_dir(monkeypatch, tmp_path, capsys) -> None:
    from gaussiandwm_cvpr.scripts import train_qa, train_world

    calls: list[dict[str, object]] = []

    def fake_train(**kwargs: object) -> Path:
        calls.append(kwargs)
        return tmp_path / str(kwargs["task"])

    monkeypatch.setattr(train_qa, "run_training_from_config_dir", fake_train)
    monkeypatch.setattr(train_world, "run_training_from_config_dir", fake_train)

    monkeypatch.setattr(sys, "argv", ["train_qa"])
    train_qa.main()
    monkeypatch.setattr(sys, "argv", ["train_world"])
    train_world.main()
    capsys.readouterr()

    assert calls[0]["task"] == "qa"
    assert calls[0]["run_dir"] is None
    assert calls[0]["output_dir"] == train_qa.DEFAULT_OUTPUT_DIR
    assert calls[1]["task"] == "world"
    assert calls[1]["run_dir"] is None
    assert calls[1]["output_dir"] == train_world.DEFAULT_OUTPUT_DIR


def test_train_scripts_forward_explicit_run_dir_output_dir_conflict(monkeypatch, tmp_path) -> None:
    from gaussiandwm_cvpr.scripts import train_qa, train_world

    def reject_conflict(**kwargs: object) -> Path:
        assert kwargs["run_dir"] is not None
        assert kwargs["output_dir"] is not None
        raise ValueError("Specify only one of run_dir and output_dir")

    monkeypatch.setattr(train_qa, "run_training_from_config_dir", reject_conflict)
    monkeypatch.setattr(train_world, "run_training_from_config_dir", reject_conflict)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_qa",
            "--run_dir",
            str(tmp_path / "qa_run"),
            "--output_dir",
            str(tmp_path / "qa_out"),
        ],
    )
    with pytest.raises(ValueError, match="Specify only one"):
        train_qa.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_world",
            "--run_dir",
            str(tmp_path / "world_run"),
            "--output_dir",
            str(tmp_path / "world_out"),
        ],
    )
    with pytest.raises(ValueError, match="Specify only one"):
        train_world.main()
