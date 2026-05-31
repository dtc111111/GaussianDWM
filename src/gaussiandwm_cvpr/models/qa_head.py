from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QAHeadOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor


class QAHead(nn.Module):
    def forward(
        self,
        *,
        token_hidden_states: torch.Tensor,
        lm_head: nn.Module,
        labels: torch.Tensor | None,
    ) -> QAHeadOutput:
        weight = getattr(lm_head, "weight", None)
        if isinstance(weight, torch.Tensor) and token_hidden_states.dtype != weight.dtype:
            token_hidden_states = token_hidden_states.to(weight.dtype)
        logits = lm_head(token_hidden_states)
        loss = None
        if labels is not None:
            # Standard causal-LM next-token loss; labels use -100 for ignored positions.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return QAHeadOutput(loss=loss, logits=logits)
