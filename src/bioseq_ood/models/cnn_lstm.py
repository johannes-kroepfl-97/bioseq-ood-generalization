from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import _SequenceInputMixin
from .normalization import maybe_layer_norm, validate_normalization_strategy


class CNNLSTMRegressor(nn.Module, _SequenceInputMixin):
    def __init__(
        self,
        vocab_size: int,
        cnn_channels: list[int],
        kernel_size: int,
        lstm_hidden_dim: int,
        lstm_layers: int,
        dropout_input: float,
        dropout_hidden: float,
        bidirectional: bool,
        fc_dim: int,
        normalization_strategy: str | None = None,
    ) -> None:
        super().__init__()

        normalization_strategy = validate_normalization_strategy(normalization_strategy)

        if not cnn_channels:
            raise ValueError("cnn_channels must contain at least one value")

        self.vocab_size = vocab_size
        self.input_dropout = nn.Dropout(dropout_input)
        self.normalization_strategy = normalization_strategy

        conv_layers: list[nn.Module] = []
        in_channels = vocab_size

        for i, out_channels in enumerate(cnn_channels):
            conv_layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )

            if normalization_strategy == "architecture_native_norm":
                conv_layers.append(nn.BatchNorm1d(out_channels))

            elif normalization_strategy == "input_bn" and i == 0:
                conv_layers.append(nn.BatchNorm1d(out_channels))

            conv_layers.extend(
                [
                    nn.ReLU(),
                    nn.Dropout(dropout_hidden),
                ]
            )

            in_channels = out_channels

        self.conv = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout_hidden if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)

        self.native_ln = maybe_layer_norm(
            lstm_output_dim,
            normalization_strategy == "architecture_native_norm",
        )

        self.final_bn = (
            nn.BatchNorm1d(lstm_output_dim)
            if normalization_strategy == "final_bn"
            else nn.Identity()
        )

        self.fc_hidden = nn.Linear(lstm_output_dim, fc_dim)
        self.hidden_dropout = nn.Dropout(dropout_hidden)
        self.output_layer = nn.Linear(fc_dim, 1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self._to_one_hot(x, self.vocab_size)
        x = self.input_dropout(x)

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        z = lstm_out[:, -1, :]

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
