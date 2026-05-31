from __future__ import annotations

import torch


def quality_topk_select(gauss_values: torch.Tensor, topk_count: int) -> torch.Tensor:
    """Return indices for high-quality Gaussian candidates from `[N,14]` values."""
    if gauss_values.ndim != 2:
        raise ValueError(f"gauss_values must have shape [N,D], got {tuple(gauss_values.shape)}")
    if gauss_values.shape[-1] < 11:
        raise ValueError(f"gauss_values must contain at least 11 dims, got {tuple(gauss_values.shape)}")
    if topk_count <= 0:
        raise ValueError(f"topk_count must be positive, got {topk_count}")

    pos = gauss_values[:, :3]
    scale = gauss_values[:, 3:6]
    rotation = gauss_values[:, 6:10]
    opacity = gauss_values[:, 10]
    quality = opacity * (scale.abs().sum(dim=-1) + 1.0)
    quality = quality / (1.0 + pos.norm(dim=-1) + 1.0e-4)
    quality = quality / (1.0 + (rotation.norm(dim=-1) - 1.0).abs())
    return torch.topk(quality, k=min(int(topk_count), gauss_values.shape[0]), dim=0).indices


def voxel_cover_select(gauss_values: torch.Tensor, cover_count: int) -> torch.Tensor:
    """Return one strong candidate per occupied voxel, capped to `cover_count`."""
    if gauss_values.ndim != 2:
        raise ValueError(f"gauss_values must have shape [N,D], got {tuple(gauss_values.shape)}")
    if gauss_values.shape[-1] < 11:
        raise ValueError(f"gauss_values must contain at least 11 dims, got {tuple(gauss_values.shape)}")
    if cover_count <= 0:
        raise ValueError(f"cover_count must be positive, got {cover_count}")
    if gauss_values.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=gauss_values.device)

    positions = gauss_values[:, :3]
    mins = positions.min(dim=0).values
    maxs = positions.max(dim=0).values
    extent = (maxs - mins).clamp(min=1.0e-6)
    cell_size = (extent.prod().clamp(min=1.0e-6) / int(cover_count)) ** (1.0 / 3.0)
    voxels = torch.floor((positions - mins) / cell_size).to(torch.long)
    max_voxel = voxels.max(dim=0).values + 1
    codes = voxels[:, 0] + max_voxel[0] * (voxels[:, 1] + max_voxel[1] * voxels[:, 2])

    distance = positions.norm(dim=-1) + 1.0e-4
    scores = gauss_values[:, 10] * (gauss_values[:, 3:6].abs().sum(dim=-1) + 1.0) / (1.0 + distance)
    chosen: dict[int, tuple[float, int]] = {}
    for idx, code in enumerate(codes.tolist()):
        score = float(scores[idx].item())
        if code not in chosen or score > chosen[code][0]:
            chosen[code] = (score, idx)

    indices = torch.tensor([value[1] for value in chosen.values()], dtype=torch.long, device=gauss_values.device)
    if indices.numel() <= cover_count:
        return indices
    keep = quality_topk_select(gauss_values.index_select(0, indices), cover_count)
    return indices.index_select(0, keep)


def voxel_topk_select(gauss_values: torch.Tensor, coarse_k: int) -> torch.Tensor:
    """Return `[<=coarse_k]` indices using the CVPR voxel coverage plus quality selector."""
    if gauss_values.ndim != 2:
        raise ValueError(f"gauss_values must have shape [N,D], got {tuple(gauss_values.shape)}")
    if coarse_k <= 0:
        raise ValueError(f"coarse_k must be positive, got {coarse_k}")

    voxel_idx = voxel_cover_select(gauss_values, coarse_k)
    topk_idx = quality_topk_select(gauss_values, coarse_k).to(device=gauss_values.device, dtype=torch.long)
    combined = torch.cat([voxel_idx.to(dtype=torch.long), topk_idx])
    if combined.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=gauss_values.device)

    generator = torch.Generator(device=combined.device)
    generator.manual_seed(int(coarse_k))
    order = torch.randperm(combined.numel(), generator=generator, device=combined.device)
    shuffled = combined.index_select(0, order)
    selected: list[int] = []
    seen: set[int] = set()
    for idx in shuffled.tolist():
        if idx in seen:
            continue
        seen.add(idx)
        selected.append(idx)
        if len(selected) >= coarse_k:
            break
    return torch.tensor(selected, dtype=torch.long, device=gauss_values.device)


def select_coarse_candidates(gauss_values: torch.Tensor, *, method: str, coarse_k: int) -> torch.Tensor:
    """Return selected Gaussian candidate rows for the CVPR release scope."""
    if method != "voxel_topk":
        raise ValueError(f"CVPR package only supports coarse_method='voxel_topk', got {method!r}")
    if coarse_k <= 0:
        raise ValueError(f"coarse_k must be positive, got {coarse_k}")
    if gauss_values.ndim != 2:
        raise ValueError(f"gauss_values must have shape [N,D], got {tuple(gauss_values.shape)}")
    if gauss_values.shape[-1] != 14:
        raise ValueError(f"gauss_values must have shape [N,14], got {tuple(gauss_values.shape)}")
    if gauss_values.shape[0] == 0:
        raise ValueError("gauss_values is empty")
    if gauss_values.shape[0] <= coarse_k:
        return gauss_values
    return gauss_values.index_select(0, voxel_topk_select(gauss_values, coarse_k))
