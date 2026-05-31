from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class GaussNormalizer:
    """Load LangSplat payloads and return normalized Gaussian rows `[N,14]`.

    The returned column order is:
    `xyz(3), scaling(3), rotation(4), opacity(1), language_feature(3)`.

    Required raw payload fields:
    - `_xyz`: `[N,3]`
    - `_scaling`: `[N,3]`
    - `_rotation`: `[N,4]`
    - `_opacity`: `[N,1]`
    - `_language_feature`: `[N,3]`

    Normalization matches the conference training path:
    - transform `_xyz` with `pose_dir/{scene_idx}.txt[frame_idx]` when a pose
      directory is configured;
    - clamp `_scaling` to `[-scaling_clip, scaling_clip]`;
    - map `_opacity` logits through `sigmoid`.
    """

    def __init__(
        self,
        *,
        pose_dir: str | None = None,
        pose_template: str = "{scene_id}.txt",
        scaling_clip: float = 7.5,
    ) -> None:
        self.pose_dir = Path(pose_dir).resolve() if pose_dir else None
        self.pose_template = pose_template
        self.scaling_clip = float(scaling_clip)
        self._pose_cache: dict[tuple[int, int], torch.Tensor] = {}

    def load_and_normalize(
        self,
        gauss_paths: list[str],
        *,
        scene_idx: int | None,
        frame_idx: int | None,
    ) -> torch.Tensor:
        """Pack record Gaussian files into `[N,14]`.

        `scene_idx` and `frame_idx` identify the row in the optional pose file.
        Without a configured `pose_dir`, `_xyz` is kept in the payload frame.
        """
        if not gauss_paths:
            raise ValueError("gauss_paths must not be empty")
        tensors = [self._load_one(Path(path), scene_idx=scene_idx, frame_idx=frame_idx) for path in gauss_paths]
        return torch.cat(tensors, dim=0)

    def _load_one(
        self,
        path: Path,
        *,
        scene_idx: int | None,
        frame_idx: int | None,
    ) -> torch.Tensor:
        if path.suffix in {".pt", ".pth"}:
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):  # type: ignore[attr-defined]
                payload = torch.load(path, map_location="cpu", weights_only=False)
        elif path.suffix == ".npz":
            payload = dict(np.load(path, allow_pickle=True))
        elif path.suffix == ".npy":
            payload = np.load(path, allow_pickle=True)
        else:
            raise ValueError(f"Unsupported Gaussian payload suffix {path.suffix!r}: {path}")

        if isinstance(payload, np.ndarray):
            state: dict[str, Any] = {"packed": payload}
        elif isinstance(payload, dict):
            state = payload.get("gaussians_state", payload)
        else:
            raise TypeError(f"Gaussian payload must be a mapping or [N,14] array: {path}")

        if not isinstance(state, dict):
            raise TypeError(f"Gaussian payload state must be a mapping: {path}")
        return self._pack_state(state, path=path, scene_idx=scene_idx, frame_idx=frame_idx)

    def _pack_state(
        self,
        state: dict[str, Any],
        *,
        path: Path,
        scene_idx: int | None,
        frame_idx: int | None,
    ) -> torch.Tensor:
        if "packed" in state:
            packed = torch.as_tensor(state["packed"], dtype=torch.float32)
            if packed.ndim != 2 or packed.shape[-1] != 14:
                raise ValueError(f"Expected packed Gaussian tensor [N,14], got {tuple(packed.shape)} from {path}")
            return packed

        parts = [
            self._transform_xyz(
                self._field(state, "_xyz", 3, path),
                scene_idx=scene_idx,
                frame_idx=frame_idx,
            ),
            torch.clamp(self._field(state, "_scaling", 3, path), min=-self.scaling_clip, max=self.scaling_clip),
            self._field(state, "_rotation", 4, path),
            torch.sigmoid(self._field(state, "_opacity", 1, path)),
            self._field(state, "_language_feature", 3, path),
        ]
        row_count = parts[0].shape[0]
        for value in parts[1:]:
            if value.shape[0] != row_count:
                raise ValueError(f"Gaussian fields must share N rows in {path}")

        values = torch.cat(parts, dim=-1).to(dtype=torch.float32)
        if values.ndim != 2 or values.shape[-1] != 14:
            raise ValueError(f"Gaussian tensor must have shape [N,14], got {tuple(values.shape)} from {path}")
        return values

    def _transform_xyz(
        self,
        xyz: torch.Tensor,
        *,
        scene_idx: int | None,
        frame_idx: int | None,
    ) -> torch.Tensor:
        if self.pose_dir is None or scene_idx is None or frame_idx is None:
            return xyz
        transform = self._load_pose(scene_idx, frame_idx).to(device=xyz.device, dtype=xyz.dtype)
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
        transformed = (transform @ xyz_h.T).T
        return transformed[:, :3]

    def _load_pose(self, scene_idx: int, frame_idx: int) -> torch.Tensor:
        cache_key = (scene_idx, frame_idx)
        if cache_key in self._pose_cache:
            return self._pose_cache[cache_key]

        if self.pose_dir is None:
            raise RuntimeError("pose_dir is not configured")
        pose_path = self.pose_dir / self.pose_template.format(scene_id=scene_idx)
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file does not exist: {pose_path}")

        lines = pose_path.read_text(encoding="utf-8").strip().splitlines()
        if frame_idx >= len(lines):
            raise IndexError(f"frame_idx={frame_idx} exceeds pose lines in {pose_path}")
        values = [float(x) for x in lines[frame_idx].strip().split()]
        if len(values) != 16:
            raise ValueError(f"Expected 16 floats per pose line in {pose_path}, got {len(values)}")

        transform = torch.tensor(values, dtype=torch.float32).reshape(4, 4)
        self._pose_cache[cache_key] = transform
        return transform

    @staticmethod
    def _field(payload: dict[str, Any], key: str, width: int, path: Path) -> torch.Tensor:
        if key not in payload:
            raise KeyError(f"Gaussian payload {path} is missing {key}")
        value = torch.as_tensor(payload[key], dtype=torch.float32)
        if value.ndim != 2 or value.shape[-1] != width:
            raise ValueError(
                f"Gaussian field {key} must have shape [N,{width}], got {tuple(value.shape)} from {path}"
            )
        return value
