from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _SequenceInputMixin
from .normalization import (
    maybe_batch_norm_last_dim,
    maybe_layer_norm,
    validate_normalization_strategy,
)


class LSTMRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        bidirectional: bool,
        fc_dim: int,
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.normalization_strategy = normalization_strategy

        self.input_bn = maybe_batch_norm_last_dim(
            vocab_size,
            normalization_strategy == "input_bn",
        )

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_hidden if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)

        # "native_norm_plus_final_bn" keeps the architecture-native LayerNorm AND
        # adds a BatchNorm1d so AdaBN has a BN layer to adapt without discarding the LN.
        self.native_ln = maybe_layer_norm(
            output_dim,
            normalization_strategy in ("architecture_native_norm", "native_norm_plus_final_bn"),
        )

        self.final_bn = (
            nn.BatchNorm1d(output_dim)
            if normalization_strategy in ("final_bn", "native_norm_plus_final_bn")
            else nn.Identity()
        )


        self.fc_hidden = nn.Linear(output_dim, fc_dim)
        self.hidden_dropout = nn.Dropout(dropout_hidden)
        self.output_layer = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_bn(x)
        x = self.input_dropout(x)

        _, (hn, _) = self.lstm(x)

        if self.lstm.bidirectional:
            z = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            z = hn[-1]

        z = self.native_ln(z)
        z = self.final_bn(z)

        h_raw = self.fc_hidden(z)

        h_pred = F.relu(h_raw)
        h_pred = self.hidden_dropout(h_pred)
        y = self.output_layer(h_pred)

        if return_features:
            h_cmd = torch.sigmoid(h_raw)
            return y, h_cmd

        return y
