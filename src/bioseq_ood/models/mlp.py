from __future__ import annotations

import torch
import torch.nn as nn

from .base import _SequenceInputMixin
from .normalization import validate_normalization_strategy


class MLPRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dims: int | list[int],
        num_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers

        if len(hidden_dims) != num_layers:
            raise ValueError("hidden_dims length must match num_layers.")

        self.vocab_size = int(vocab_size)
        self.seq_len = int(seq_len)
        self.input_dim = self.vocab_size * self.seq_len
        self.normalization_strategy = normalization_strategy

        self.input_bn = (
            nn.BatchNorm1d(self.input_dim)
            if normalization_strategy == "input_bn"
            else nn.Identity()
        )

        layers: list[nn.Module] = [nn.Dropout(float(dropout_input))]
        dims = [self.input_dim] + list(hidden_dims)

        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            if normalization_strategy == "architecture_native_norm":
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            layers.extend(
                [
                    nn.ReLU(),
                    nn.Dropout(float(dropout_hidden)),
                ]
            )

        self.feature_extractor = nn.Sequential(*layers)

        self.final_bn = (
            nn.BatchNorm1d(dims[-1])
            if normalization_strategy == "final_bn"
            else nn.Identity()
        )

        self.output_layer = nn.Linear(dims[-1], 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self._to_one_hot(x, self.vocab_size)

        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.shape[1]}.")

        x = x.reshape(x.shape[0], -1)
        x = self.input_bn(x)
        h_raw = self.feature_extractor(x)
        h_raw = self.final_bn(h_raw)
        y = self.output_layer(h_raw)
        if return_features:
            h_cmd = torch.sigmoid(h_raw)
            return y, h_cmd
        return y
