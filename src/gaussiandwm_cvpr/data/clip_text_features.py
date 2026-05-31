from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=32)
def _load_shard(path: str) -> np.ndarray:
    shard = np.load(path)
    if shard.ndim != 2 or shard.shape[1] != 512:
        raise ValueError(f"CLIP text shard must have shape [N, 512]: {path}")
    return shard


def load_clip_text_feature(path: str | Path, row: int) -> torch.Tensor:
    if not isinstance(row, int) or isinstance(row, bool) or row < 0:
        raise ValueError(f"CLIP text row must be a non-negative int: {row!r}")
    shard = _load_shard(str(Path(path).resolve()))
    if row >= shard.shape[0]:
        raise IndexError(f"CLIP text row {row} is out of bounds for {path}")
    feature = np.array(shard[row], dtype=np.float32, copy=True)
    return torch.from_numpy(feature)
