from __future__ import annotations

import torch
import torch.nn as nn


class IdentitySelector(nn.Module):
    """Return the first `fine_k` valid Gaussian hidden states.

    Inputs:
    - `coarse_gauss_hidden`: `[B,Kc,H]`
    - `coarse_gauss_mask`: `[B,Kc]`

    Output:
    - selected hidden states: `[B,fine_k,H]`
    """

    def forward(
        self,
        *,
        coarse_gauss_hidden: torch.Tensor,
        coarse_gauss_mask: torch.Tensor,
        fine_k: int,
    ) -> torch.Tensor:
        if coarse_gauss_hidden.ndim != 3:
            raise ValueError(
                f"coarse_gauss_hidden must be [B,Kc,H], got {tuple(coarse_gauss_hidden.shape)}"
            )
        if coarse_gauss_mask.shape != coarse_gauss_hidden.shape[:2]:
            raise ValueError(
                "coarse_gauss_mask must match coarse_gauss_hidden batch and candidate dimensions"
            )
        if fine_k <= 0:
            raise ValueError(f"fine_k must be positive, got {fine_k}")

        mask = coarse_gauss_mask.to(device=coarse_gauss_hidden.device, dtype=torch.bool)
        outputs = []
        for batch_idx in range(coarse_gauss_hidden.shape[0]):
            valid = coarse_gauss_hidden[batch_idx][mask[batch_idx]]
            if valid.shape[0] < fine_k:
                raise ValueError(
                    f"Batch {batch_idx} has {valid.shape[0]} valid Gaussian candidates, need fine_k={fine_k}"
                )
            outputs.append(valid[:fine_k])
        return torch.stack(outputs, dim=0)


class SimilaritySelector(nn.Module):
    """Select Gaussian hidden states by CLIP-space text/Gaussian similarity.

    Inputs:
    - `clip_text_embed`: `[B,512]`
    - `gauss_clip_features`: `[B,Kc,512]`
    - `coarse_gauss_hidden`: `[B,Kc,H]`
    - `coarse_gauss_mask`: `[B,Kc]`

    Output:
    - selected hidden states: `[B,fine_k,H]`
    """

    def forward(
        self,
        *,
        clip_text_embed: torch.Tensor,
        gauss_clip_features: torch.Tensor,
        coarse_gauss_hidden: torch.Tensor,
        coarse_gauss_mask: torch.Tensor,
        fine_k: int,
    ) -> torch.Tensor:
        if clip_text_embed.ndim != 2 or clip_text_embed.shape[-1] != 512:
            raise ValueError(f"clip_text_embed must be [B,512], got {tuple(clip_text_embed.shape)}")
        if gauss_clip_features.ndim != 3 or gauss_clip_features.shape[-1] != 512:
            raise ValueError(
                f"gauss_clip_features must be [B,Kc,512], got {tuple(gauss_clip_features.shape)}"
            )
        if coarse_gauss_hidden.ndim != 3:
            raise ValueError(
                f"coarse_gauss_hidden must be [B,Kc,H], got {tuple(coarse_gauss_hidden.shape)}"
            )
        if clip_text_embed.shape[0] != gauss_clip_features.shape[0]:
            raise ValueError("clip_text_embed and gauss_clip_features must share batch dimension")
        if coarse_gauss_hidden.shape[:2] != gauss_clip_features.shape[:2]:
            raise ValueError("coarse_gauss_hidden and gauss_clip_features must share [B,Kc]")
        if coarse_gauss_mask.shape != gauss_clip_features.shape[:2]:
            raise ValueError("coarse_gauss_mask must match [B,Kc]")
        if fine_k <= 0:
            raise ValueError(f"fine_k must be positive, got {fine_k}")

        device = coarse_gauss_hidden.device
        mask = coarse_gauss_mask.to(device=device, dtype=torch.bool)
        valid_counts = mask.sum(dim=1)
        if torch.any(valid_counts < fine_k):
            raise ValueError(
                "Every batch item must have at least "
                f"fine_k={fine_k} valid Gaussian candidates; got {valid_counts.tolist()}"
            )

        text = torch.nn.functional.normalize(clip_text_embed.to(device=device, dtype=torch.float32), dim=-1)
        gauss = torch.nn.functional.normalize(gauss_clip_features.to(device=device, dtype=torch.float32), dim=-1)
        scores = torch.einsum("bd,bkd->bk", text, gauss)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        indices = torch.topk(scores, k=fine_k, dim=1).indices
        gather_index = indices.unsqueeze(-1).expand(-1, -1, coarse_gauss_hidden.shape[-1])
        return torch.gather(coarse_gauss_hidden, dim=1, index=gather_index)
