from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Protocol

import torch

from gaussiandwm_cvpr.data.coarse_selectors import select_coarse_candidates
from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer


class GaussRecord(Protocol):
    sample_uid: str
    task: str
    scene_idx: int
    frame_idx: int
    views: list[str]
    gauss_paths: list[str]


class GaussFeatureCache:
    """Cache normalized coarse Gaussian tensors for CVPR QA and world records.

    `load_for_record(record)` returns `[Kc,14]`, where `Kc <= coarse_k` and
    `Kc >= fine_k_by_task[record.task]`.
    """

    def __init__(
        self,
        *,
        cache_root: str | Path,
        coarse_method: str,
        coarse_k: int,
        fine_k_by_task: dict[str, int],
        normalizer_version: str = "v2",
        normalizer: GaussNormalizer | None = None,
    ) -> None:
        if coarse_method != "voxel_topk":
            raise ValueError(f"CVPR package only supports coarse_method='voxel_topk', got {coarse_method!r}")
        if coarse_k <= 0:
            raise ValueError(f"coarse_k must be positive, got {coarse_k}")
        if not fine_k_by_task:
            raise ValueError("fine_k_by_task must not be empty")

        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.coarse_method = coarse_method
        self.coarse_k = int(coarse_k)
        self.fine_k_by_task: dict[str, int] = {}
        for task, fine_k in fine_k_by_task.items():
            if fine_k <= 0:
                raise ValueError(f"fine_k for task {task!r} must be positive, got {fine_k}")
            if fine_k > self.coarse_k:
                raise ValueError(
                    f"fine_k for task {task!r} must be <= coarse_k={self.coarse_k}, got {fine_k}"
                )
            self.fine_k_by_task[str(task)] = int(fine_k)
        self.normalizer_version = str(normalizer_version)
        self.normalizer = normalizer or GaussNormalizer()

    def load_for_record(self, record: GaussRecord) -> torch.Tensor:
        if record.task not in self.fine_k_by_task:
            raise KeyError(f"No fine_k configured for task {record.task!r}")

        fine_k = self.fine_k_by_task[record.task]
        cache_path = self._cache_path(record)
        if cache_path.exists():
            values = torch.load(cache_path, map_location="cpu")
            self._validate_cached_values(values, record, fine_k, self.coarse_k, cache_path)
            return values

        values = self.normalizer.load_and_normalize(
            record.gauss_paths,
            scene_idx=record.scene_idx,
            frame_idx=record.frame_idx,
        )
        if values.shape[0] < fine_k:
            raise ValueError(
                f"{record.sample_uid} has {values.shape[0]} Gaussian candidates, need fine_k={fine_k}"
            )

        values = select_coarse_candidates(values, method=self.coarse_method, coarse_k=self.coarse_k)
        if values.shape[0] < fine_k:
            raise ValueError(
                f"{record.sample_uid} has {values.shape[0]} selected Gaussian candidates, need fine_k={fine_k}"
            )
        if not values.is_floating_point():
            raise ValueError(f"{record.sample_uid} selected Gaussian candidates must have a floating dtype")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pt", dir=str(cache_path.parent), delete=False) as handle:
                tmp_path = Path(handle.name)
            torch.save(values, tmp_path)
            tmp_path.replace(cache_path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()
        return values

    def _cache_path(self, record: GaussRecord) -> Path:
        views = "-".join(sorted(record.views))
        return (
            self.cache_root
            / f"cache_{self.coarse_method}_k{self.coarse_k}_norm_{self.normalizer_version}"
            / f"{record.scene_idx}_{record.frame_idx}_{views}.pt"
        )

    @staticmethod
    def _validate_cached_values(
        values: object,
        record: GaussRecord,
        fine_k: int,
        coarse_k: int,
        cache_path: Path,
    ) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"Gaussian cache must contain a tensor: {cache_path}")
        if values.ndim != 2 or values.shape[-1] != 14:
            raise ValueError(f"Gaussian cache must have shape [Kc,14], got {tuple(values.shape)}: {cache_path}")
        if not values.is_floating_point():
            raise ValueError(f"Gaussian cache must have a floating dtype: {cache_path}")
        if values.shape[0] < fine_k:
            raise ValueError(
                f"{record.sample_uid} cache has {values.shape[0]} Gaussian candidates, need fine_k={fine_k}: "
                f"{cache_path}"
            )
        if values.shape[0] > coarse_k:
            raise ValueError(
                f"{record.sample_uid} cache has {values.shape[0]} Gaussian candidates, exceeds coarse_k={coarse_k}: "
                f"{cache_path}"
            )
