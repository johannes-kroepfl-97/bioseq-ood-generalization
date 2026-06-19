from __future__ import annotations

import torch
import torch.nn as nn

VALID_NORMALIZATION_STRATEGIES = {
    # Compared in the Phase B normalization runoff (None vs architecture_native_norm);
    # the better-val_id variant of the tuned architecture becomes the shared baseline:
    None,                       # no normalisation anywhere
    "architecture_native_norm", # BN after every conv/linear (CNN, MLP, Transformer);
                                # LayerNorm on the LSTM output (LSTM, hybrids)
    # Method-specific override (not a baseline choice):
    "native_norm_plus_final_bn", # LSTM + AdaBN only: keeps LN AND adds a final BN
                                 # so AdaBN has a BatchNorm layer to adapt
}


def validate_normalization_strategy(strategy: str | None) -> str | None:
    if strategy == "none":
        strategy = None

    if strategy not in VALID_NORMALIZATION_STRATEGIES:
        raise ValueError(
            f"Unknown normalization_strategy={strategy}. "
            f"Expected one of {VALID_NORMALIZATION_STRATEGIES}."
        )

    return strategy


class BatchNormLastDim(nn.Module):
    """
    BatchNorm over feature dimension for tensors shaped:
    - (batch, features)
    - (batch, seq_len, features)

    This avoids applying BN across LSTM time steps incorrectly.
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.bn(x)

        if x.ndim == 3:
            x = x.transpose(1, 2)
            x = self.bn(x)
            return x.transpose(1, 2)

        raise ValueError(f"Expected 2D or 3D tensor, got shape={tuple(x.shape)}")


def maybe_batch_norm_1d(num_features: int, active: bool) -> nn.Module:
    return nn.BatchNorm1d(num_features) if active else nn.Identity()


def maybe_batch_norm_last_dim(num_features: int, active: bool) -> nn.Module:
    return BatchNormLastDim(num_features) if active else nn.Identity()


def maybe_layer_norm(num_features: int, active: bool) -> nn.Module:
    return nn.LayerNorm(num_features) if active else nn.Identity()
