from __future__ import annotations

import json
from inspect import signature
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_public_package_excludes_pruned_model_runtime_strings() -> None:
    private_mount = "/" + "mnt" + "/"
    private_names = ["chen" + suffix for suffix in ("xf", "qu", "yi")]
    private_implementation = "_".join(("unified", "dwm"))
    private_import = ".".join(("src", "unified"))
    attention_class = "Fine" + "Attention"
    attention_label = "attention" + " selector"
    trajectory_head = "Traj" + "Head"
    trajectory_forward = "_".join(("forward", "traj"))
    trajectory_output = "_".join(("traj", "pred"))
    forbidden = (
        attention_class,
        attention_label,
        trajectory_head,
        trajectory_forward,
        trajectory_output,
        private_implementation,
        private_import,
        private_mount,
        *private_names,
    )
    root = _package_root()

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.relative_to(root).parts:
            continue
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                offenders.append(f"{path.relative_to(root)} contains {needle!r}")

    assert offenders == []


def test_unified_model_exposes_root_from_pretrained() -> None:
    from gaussiandwm_cvpr.models import unified_model

    UnifiedGaussianDWM = unified_model.UnifiedGaussianDWM

    assert hasattr(UnifiedGaussianDWM, "from_pretrained")
    pretrained_signature = signature(UnifiedGaussianDWM.from_pretrained)
    assert pretrained_signature.parameters["model_init_dir"].default == "dtc111/GaussianDWM"
    assert pretrained_signature.parameters["revision"].default == "main"
    assert signature(unified_model._resolve_pretrained_root).parameters["revision"].default == "main"
    assert signature(UnifiedGaussianDWM.from_config_only).parameters["revision"].default == "main"
    assert signature(UnifiedGaussianDWM.load_pretrained_state_dict).parameters["revision"].default == "main"
    assert signature(UnifiedGaussianDWM.load_pretrained_weights).parameters["revision"].default == "main"


def test_hf_root_resolver_passes_revision_to_config_and_weight_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gaussiandwm_cvpr.models import unified_model

    cache_root = tmp_path / "hf-cache"
    cache_root.mkdir()
    _write_root_config(cache_root)
    (cache_root / "model.safetensors").write_bytes(b"unused")
    calls: list[tuple[str, str]] = []

    def fake_hf_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        assert repo_id == "dtc111/GaussianDWM"
        calls.append((filename, revision))
        return str(cache_root / filename)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    root = unified_model._resolve_pretrained_root("dtc111/GaussianDWM", revision="review-branch")

    assert root == cache_root
    assert calls == [
        ("config.json", "review-branch"),
        ("model.safetensors", "review-branch"),
    ]


def _write_root_config(root: Path, weight_filename: str = "model.safetensors") -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "meta": {
                    "format_version": 1,
                    "model_type": "unified_gaussian_dwm",
                    "weight_filename": weight_filename,
                }
            }
        ),
        encoding="utf-8",
    )


def _tiny_unified_class(base_class: type[nn.Module]) -> type[nn.Module]:
    class TinyUnifiedDWM(base_class):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.weight = nn.Parameter(torch.zeros(1))
            self.model_config: dict[str, object] = {}

    return TinyUnifiedDWM


def test_from_pretrained_loads_root_safetensors_strictly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM

    _write_root_config(tmp_path)
    save_file({"weight": torch.tensor([7.0])}, str(tmp_path / "model.safetensors"))
    tiny_cls = _tiny_unified_class(UnifiedGaussianDWM)

    monkeypatch.setattr(
        UnifiedGaussianDWM,
        "build_from_config",
        classmethod(lambda cls, payload: tiny_cls()),
    )

    model = UnifiedGaussianDWM.from_pretrained(tmp_path, revision="local-path-ignored")

    assert isinstance(model, tiny_cls)
    assert torch.equal(model.weight.detach(), torch.tensor([7.0]))


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({}, "Missing key"),
        ({"weight": torch.tensor([1.0]), "unexpected": torch.tensor([2.0])}, "Unexpected key"),
    ],
)
def test_from_pretrained_rejects_missing_or_unexpected_root_safetensors_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: dict[str, torch.Tensor],
    message: str,
) -> None:
    from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM

    _write_root_config(tmp_path)
    save_file(state, str(tmp_path / "model.safetensors"))
    tiny_cls = _tiny_unified_class(UnifiedGaussianDWM)
    monkeypatch.setattr(
        UnifiedGaussianDWM,
        "build_from_config",
        classmethod(lambda cls, payload: tiny_cls()),
    )

    with pytest.raises(RuntimeError, match=message):
        UnifiedGaussianDWM.from_pretrained(tmp_path)


def test_load_pretrained_state_dict_rejects_non_safetensors_root_weights(tmp_path: Path) -> None:
    from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM

    _write_root_config(tmp_path, weight_filename="model.pt")
    torch.save({"weight": torch.tensor([1.0])}, tmp_path / "model.pt")

    with pytest.raises(ValueError, match="model.safetensors"):
        UnifiedGaussianDWM.load_pretrained_state_dict(tmp_path)
