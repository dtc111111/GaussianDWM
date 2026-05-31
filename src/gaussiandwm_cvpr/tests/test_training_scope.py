from __future__ import annotations

from inspect import signature
from pathlib import Path

import pytest
import torch


def test_stage_scheduler_rejects_traj_trainable_module() -> None:
    from gaussiandwm_cvpr.train.stage_scheduler import StageScheduler

    plan = {
        "stages": [
            {
                "name": "bad_traj_stage",
                "start_step": 0,
                "end_step": None,
                "trainable_modules": ["traj_head"],
            }
        ]
    }

    with pytest.raises(ValueError, match="traj"):
        StageScheduler(plan, per_device_batch_size={"qa": 4})


def test_stage_scheduler_accepts_cvpr_qa_modules_and_batch_size() -> None:
    from gaussiandwm_cvpr.train.stage_scheduler import StageScheduler

    scheduler = StageScheduler(
        {
            "stages": [
                {
                    "name": "qa_cvpr_finetune",
                    "start_step": 0,
                    "end_step": None,
                    "trainable_modules": [
                        "gauss_token_rows",
                        "gauss_aligner_core",
                        "qwen_backbone_lora",
                        "qa_lm_head_lora",
                    ],
                }
            ]
        },
        per_device_batch_size={"qa": 4},
    )

    assert scheduler.get_batch_sizes(0) == {"qa": 4}
    assert scheduler.get_stage(0).name == "qa_cvpr_finetune"
    assert scheduler.planned_token_row_groups() == ("gauss_token_rows",)


def test_stage_scheduler_rejects_unknown_trainable_module() -> None:
    from gaussiandwm_cvpr.train.stage_scheduler import StageScheduler

    with pytest.raises(ValueError, match="Unsupported CVPR trainable modules: unknown_group"):
        StageScheduler(
            {
                "stages": [
                    {
                        "name": "bad_unknown_stage",
                        "start_step": 0,
                        "end_step": None,
                        "trainable_modules": ["unknown_group"],
                    }
                ]
            },
            per_device_batch_size={"qa": 4},
        )


def test_stage_scheduler_rejects_gradnorm_loss_balancer_mode() -> None:
    from gaussiandwm_cvpr.train.stage_scheduler import StageScheduler

    with pytest.raises(ValueError, match="static|GradNorm|unsupported loss balancer"):
        StageScheduler(
            {
                "stages": [
                    {
                        "name": "bad_loss_balancer_stage",
                        "start_step": 0,
                        "end_step": None,
                        "trainable_modules": ["gauss_aligner_core"],
                        "loss_balancer_mode": "gradnorm",
                    }
                ]
            },
            per_device_batch_size={"qa": 4},
        )


def test_run_training_from_config_dir_signature_supports_hf_checkpoint_root() -> None:
    from gaussiandwm_cvpr.train.trainer import run_training_from_config_dir

    params = signature(run_training_from_config_dir).parameters
    assert list(params) == [
        "task",
        "config_dir",
        "run_dir",
        "output_dir",
        "model_id",
        "revision",
        "data_root",
        "annotation_path",
        "gauss_cache_root",
        "resume_from_checkpoint",
    ]
    assert params["model_id"].default == "dtc111/GaussianDWM"
    assert params["revision"].default == "main"


def test_run_training_from_config_dir_rejects_run_dir_and_output_dir_before_config_loading(tmp_path, monkeypatch) -> None:
    from gaussiandwm_cvpr.train import trainer

    def fail_load_yaml(path: object) -> object:
        raise AssertionError(f"config should not be loaded after invalid output arguments: {path}")

    monkeypatch.setattr(trainer, "load_yaml", fail_load_yaml)

    with pytest.raises(ValueError, match="run_dir and output_dir"):
        trainer.run_training_from_config_dir(
            task="qa",
            config_dir=tmp_path / "missing_config",
            run_dir=tmp_path / "run",
            output_dir=tmp_path / "out",
        )


def test_run_training_from_config_dir_loads_taxonomy_from_budgets_path_before_hf_work(tmp_path, monkeypatch) -> None:
    from gaussiandwm_cvpr.train import trainer

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "model.yaml",
        "data.yaml",
        "gaussian_token.yaml",
        "train.yaml",
        "stage_plan.yaml",
        "budgets.yaml",
    ):
        (config_dir / name).write_text("{}\n", encoding="utf-8")

    def fake_load_yaml(path: object) -> dict[str, object]:
        filename = Path(path).name
        if filename == "model.yaml":
            return {}
        if filename == "data.yaml":
            return {"datasets": {}}
        if filename == "gaussian_token.yaml":
            return {"fine_k": {"qa": 1}, "coarse_method": "topk", "coarse_k": 1}
        if filename == "train.yaml":
            return {"per_device_batch_size": {"qa": 1}, "max_steps": 0}
        if filename == "stage_plan.yaml":
            return {"stages": []}
        if filename == "budgets.yaml":
            return {
                "major_tasks": ["qa"],
                "qa_group_budgets": {"planning": 1},
                "qa_subtask_budgets": {"planning": {"left": 1}},
            }
        raise AssertionError(path)

    def fake_load_taxonomy(path: object) -> object:
        assert path == config_dir / "budgets.yaml"
        raise RuntimeError("taxonomy path checked before HF work")

    class FailAutoProcessor:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise AssertionError("HF processor should not load before taxonomy validation")

    class FailModel:
        @staticmethod
        def from_pretrained(*args: object, **kwargs: object) -> object:
            raise AssertionError("model should not load before taxonomy validation")

    monkeypatch.setattr(trainer, "load_yaml", fake_load_yaml)
    monkeypatch.setattr(trainer, "load_taxonomy", fake_load_taxonomy)
    monkeypatch.setattr(trainer, "AutoProcessor", FailAutoProcessor)
    monkeypatch.setattr(trainer, "UnifiedGaussianDWM", FailModel)

    with pytest.raises(RuntimeError, match="taxonomy path checked"):
        trainer.run_training_from_config_dir(
            task="qa",
            config_dir=config_dir,
            run_dir=tmp_path / "run",
        )


def test_checkpoint_resume_restores_scheduler_last_epoch(tmp_path) -> None:
    from gaussiandwm_cvpr.train.checkpoint import load_resume_state, save_checkpoint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 / (step + 1))
    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()

    checkpoint_dir = save_checkpoint(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        run_dir=tmp_path,
        step=2,
    )

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.1)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lr_lambda=lambda step: 1.0 / (step + 1))

    loaded_step = load_resume_state(
        model=resumed_model,
        optimizer=resumed_optimizer,
        lr_scheduler=resumed_scheduler,
        checkpoint_dir=checkpoint_dir,
    )

    assert loaded_step == 2
    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_scheduler.state_dict()["_last_lr"] == scheduler.state_dict()["_last_lr"]
