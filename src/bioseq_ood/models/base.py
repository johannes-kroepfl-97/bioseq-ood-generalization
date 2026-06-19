from __future__ import annotations

import torch
import torch.nn.functional as F


class _SequenceInputMixin:
    def _to_one_hot(self, x: torch.Tensor, vocab_size: int) -> torch.Tensor:
        if x.ndim == 2:
            if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.long):
                x = x.long()
            return F.one_hot(x, num_classes=vocab_size).to(torch.float32)
        if x.ndim == 3:
            if x.shape[-1] != vocab_size:
                raise ValueError(
                    f"Expected one-hot input with last dim {vocab_size}, got {tuple(x.shape)}"
                )
            return x.to(torch.float32)
        raise ValueError(
            f"Expected input with shape (batch, seq_len) or (batch, seq_len, vocab_size), got {tuple(x.shape)}"
        )
