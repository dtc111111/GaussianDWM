from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from gaussiandwm_cvpr.data.coarse_selectors import select_coarse_candidates, voxel_topk_select
from gaussiandwm_cvpr.data.gauss_cache import GaussFeatureCache
from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer
from gaussiandwm_cvpr.models.selectors import IdentitySelector, SimilaritySelector


@dataclass
class _FakeRecord:
    sample_uid: str
    task: str
    scene_idx: int
    frame_idx: int
    views: list[str]
    gauss_paths: list[str]


class _FakeNormalizer:
    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def load_and_normalize(self, gauss_paths: list[str], *, scene_idx: int, frame_idx: int) -> torch.Tensor:
        del gauss_paths, scene_idx, frame_idx
        return self.values.clone()


def test_identity_selector_requires_enough_valid_candidates() -> None:
    hidden = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
    mask = torch.tensor([[True, True, True, False]])
    selector = IdentitySelector()

    selected = selector(coarse_gauss_hidden=hidden, coarse_gauss_mask=mask, fine_k=3)

    assert selected.shape == (1, 3, 3)
    assert torch.allclose(selected, hidden[:, :3])


def test_identity_selector_accepts_non_bool_mask() -> None:
    hidden = torch.arange(1 * 4 * 2, dtype=torch.float32).reshape(1, 4, 2)
    mask = torch.tensor([[1, 0, 1, 1]], dtype=torch.int64)
    selector = IdentitySelector()

    selected = selector(coarse_gauss_hidden=hidden, coarse_gauss_mask=mask, fine_k=2)

    assert torch.allclose(selected, hidden[:, [0, 2]])


def test_identity_selector_rejects_too_few_candidates() -> None:
    hidden = torch.zeros(1, 2, 3)
    mask = torch.tensor([[True, False]])
    selector = IdentitySelector()

    with pytest.raises(ValueError, match="valid Gaussian candidates"):
        selector(coarse_gauss_hidden=hidden, coarse_gauss_mask=mask, fine_k=2)


def test_identity_selector_rejects_unsupported_extra_kwargs() -> None:
    hidden = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    selector = IdentitySelector()

    with pytest.raises(TypeError):
        selector(coarse_gauss_hidden=hidden, coarse_gauss_mask=mask, fine_k=1, attention=True)


def test_similarity_selector_uses_clip_text_feature_shape() -> None:
    selector = SimilaritySelector()
    coarse_hidden = torch.randn(2, 5, 4)
    gauss_clip = torch.nn.functional.normalize(torch.randn(2, 5, 512), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(2, 512), dim=-1)
    mask = torch.ones(2, 5, dtype=torch.bool)

    selected = selector(
        clip_text_embed=text,
        gauss_clip_features=gauss_clip,
        coarse_gauss_hidden=coarse_hidden,
        coarse_gauss_mask=mask,
        fine_k=3,
    )

    assert selected.shape == (2, 3, 4)


def test_similarity_selector_scores_float32_and_keeps_hidden_dtype() -> None:
    selector = SimilaritySelector()
    coarse_hidden = torch.randn(1, 4, 3, dtype=torch.float16)
    gauss_clip = torch.nn.functional.normalize(torch.randn(1, 4, 512, dtype=torch.float32), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(1, 512, dtype=torch.float64), dim=-1)
    mask = torch.ones(1, 4, dtype=torch.bool)

    selected = selector(
        clip_text_embed=text,
        gauss_clip_features=gauss_clip,
        coarse_gauss_hidden=coarse_hidden,
        coarse_gauss_mask=mask,
        fine_k=2,
    )

    assert selected.shape == (1, 2, 3)
    assert selected.dtype == coarse_hidden.dtype


def test_similarity_selector_rejects_wrong_clip_text_shape() -> None:
    selector = SimilaritySelector()
    coarse_hidden = torch.randn(1, 5, 4)
    gauss_clip = torch.randn(1, 5, 512)
    mask = torch.ones(1, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="clip_text_embed must be \\[B,512\\]"):
        selector(
            clip_text_embed=torch.randn(1, 256),
            gauss_clip_features=gauss_clip,
            coarse_gauss_hidden=coarse_hidden,
            coarse_gauss_mask=mask,
            fine_k=3,
        )


def test_similarity_selector_rejects_too_few_candidates() -> None:
    selector = SimilaritySelector()
    coarse_hidden = torch.randn(1, 3, 4)
    gauss_clip = torch.randn(1, 3, 512)
    text = torch.randn(1, 512)
    mask = torch.tensor([[True, False, True]])

    with pytest.raises(ValueError, match="valid Gaussian candidates"):
        selector(
            clip_text_embed=text,
            gauss_clip_features=gauss_clip,
            coarse_gauss_hidden=coarse_hidden,
            coarse_gauss_mask=mask,
            fine_k=3,
        )


def test_similarity_selector_rejects_unsupported_extra_kwargs() -> None:
    selector = SimilaritySelector()
    coarse_hidden = torch.randn(1, 3, 4)
    gauss_clip = torch.randn(1, 3, 512)
    text = torch.randn(1, 512)
    mask = torch.ones(1, 3, dtype=torch.bool)

    with pytest.raises(TypeError):
        selector(
            clip_text_embed=text,
            gauss_clip_features=gauss_clip,
            coarse_gauss_hidden=coarse_hidden,
            coarse_gauss_mask=mask,
            fine_k=2,
            traj=True,
        )


def test_select_coarse_candidates_requires_width_14_before_small_tensor_return() -> None:
    with pytest.raises(ValueError, match=r"\[N,14\]|width 14"):
        select_coarse_candidates(torch.zeros(2, 2), method="voxel_topk", coarse_k=4)


def test_gauss_normalizer_applies_pose_and_field_normalization(tmp_path: Path) -> None:
    gauss_path = tmp_path / "gauss.pth"
    torch.save(
        {
            "_xyz": torch.tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]]),
            "_scaling": torch.tensor([[-10.0, 1.0, 10.0], [8.0, -8.0, 0.0]]),
            "_rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "_opacity": torch.tensor([[0.0], [2.0]]),
            "_language_feature": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        },
        gauss_path,
    )
    pose_dir = tmp_path / "poses"
    pose_dir.mkdir()
    pose_dir.joinpath("7.txt").write_text(
        "\n".join(
            [
                "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
                "1 0 0 10 0 1 0 20 0 0 1 30 0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )

    values = GaussNormalizer(pose_dir=str(pose_dir)).load_and_normalize(
        [str(gauss_path)],
        scene_idx=7,
        frame_idx=1,
    )

    assert values.shape == (2, 14)
    assert torch.allclose(values[:, :3], torch.tensor([[11.0, 22.0, 33.0], [10.0, 19.0, 32.0]]))
    assert torch.allclose(values[:, 3:6], torch.tensor([[-7.5, 1.0, 7.5], [7.5, -7.5, 0.0]]))
    assert torch.allclose(values[:, 6:10], torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
    assert torch.allclose(values[:, 10:11], torch.sigmoid(torch.tensor([[0.0], [2.0]])))
    assert torch.allclose(values[:, 11:14], torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))


def test_gauss_feature_cache_rejects_integer_cached_tensor(tmp_path: Path) -> None:
    record = _FakeRecord(
        sample_uid="sample",
        task="qa",
        scene_idx=3,
        frame_idx=5,
        views=["CAM_FRONT"],
        gauss_paths=["unused.pth"],
    )
    cache = GaussFeatureCache(
        cache_root=tmp_path,
        coarse_method="voxel_topk",
        coarse_k=4,
        fine_k_by_task={"qa": 2},
        normalizer=_FakeNormalizer(torch.randn(4, 14, dtype=torch.float32)),  # type: ignore[arg-type]
    )

    values = cache.load_for_record(record)
    assert values.dtype == torch.float32
    cache_path = tmp_path / "cache_voxel_topk_k4_norm_v2" / "3_5_CAM_FRONT.pt"
    assert cache_path.exists()

    torch.save(torch.ones(4, 14, dtype=torch.int64), cache_path)
    with pytest.raises(ValueError, match="floating dtype"):
        cache.load_for_record(record)


def test_gauss_feature_cache_rejects_oversized_cached_tensor(tmp_path: Path) -> None:
    record = _FakeRecord(
        sample_uid="sample",
        task="qa",
        scene_idx=4,
        frame_idx=6,
        views=["CAM_FRONT"],
        gauss_paths=["unused.pth"],
    )
    cache = GaussFeatureCache(
        cache_root=tmp_path,
        coarse_method="voxel_topk",
        coarse_k=4,
        fine_k_by_task={"qa": 2},
        normalizer=_FakeNormalizer(torch.randn(4, 14, dtype=torch.float32)),  # type: ignore[arg-type]
    )
    cache_path = tmp_path / "cache_voxel_topk_k4_norm_v2" / "4_6_CAM_FRONT.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(torch.randn(7, 14, dtype=torch.float32), cache_path)

    with pytest.raises(ValueError, match="coarse_k|coarse"):
        cache.load_for_record(record)


def test_voxel_topk_select_is_deterministic_and_deduplicated() -> None:
    values = torch.zeros(8, 14)
    values[:, :3] = torch.arange(8, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    values[:, 3:6] = 1.0
    values[:, 6] = 1.0
    values[:, 10] = torch.linspace(0.1, 0.8, steps=8)

    first = voxel_topk_select(values, coarse_k=4)
    second = voxel_topk_select(values, coarse_k=4)

    assert torch.equal(first, second)
    assert first.numel() == torch.unique(first).numel()
    assert first.numel() <= 4
