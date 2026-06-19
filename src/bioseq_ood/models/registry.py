from __future__ import annotations

from typing import Any

from .cnn import CNNRegressor
from .cnn_lstm import CNNLSTMRegressor
from .lstm import LSTMRegressor
from .lstm_cnn import LSTMCNNRegressor
from .mlp import MLPRegressor
from .transformer import TransformerRegressor

MODEL_REGISTRY = {
    "mlp": MLPRegressor,
    "cnn": CNNRegressor,
    "lstm": LSTMRegressor,
    "cnn_lstm": CNNLSTMRegressor,
    "lstm_cnn": LSTMCNNRegressor,
    "transformer": TransformerRegressor,
}


def build_model(model_name: str, model_cfg: dict[str, Any], vocab_size: int, seq_len: int):
    model_name = str(model_name).lower()
    normalization_strategy = model_cfg.get("normalization_strategy", None)

    if model_name == "mlp":
        return MLPRegressor(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dims=model_cfg["hidden_dims"],
            num_layers=int(model_cfg["num_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            normalization_strategy=normalization_strategy,
        )

    if model_name == "cnn":
        return CNNRegressor(
            vocab_size=vocab_size,
            channels=list(model_cfg["channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            fc_dim=int(model_cfg["fc_dim"]),
            normalization_strategy=normalization_strategy,
        )

    if model_name == "lstm":
        return LSTMRegressor(
            vocab_size=vocab_size,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            fc_dim=int(model_cfg["fc_dim"]),
            normalization_strategy=normalization_strategy,
        )

    if model_name == "cnn_lstm":
        return CNNLSTMRegressor(
            vocab_size=vocab_size,
            cnn_channels=list(model_cfg["cnn_channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
            lstm_hidden_dim=int(model_cfg["lstm_hidden_dim"]),
            lstm_layers=int(model_cfg["lstm_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            fc_dim=int(model_cfg["fc_dim"]),
            normalization_strategy=normalization_strategy,
        )

    if model_name == "lstm_cnn":
        return LSTMCNNRegressor(
            vocab_size=vocab_size,
            lstm_hidden_dim=int(model_cfg["lstm_hidden_dim"]),
            lstm_layers=int(model_cfg["lstm_layers"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
            cnn_channels=list(model_cfg["cnn_channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
            fc_dim=int(model_cfg["fc_dim"]),
            normalization_strategy=normalization_strategy,
        )

    if model_name == "transformer":
        return TransformerRegressor(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=int(model_cfg["d_model"]),
            nhead=int(model_cfg["nhead"]),
            num_layers=int(model_cfg["num_layers"]),
            dim_feedforward=int(model_cfg["dim_feedforward"]),
            dropout_input=float(model_cfg.get("dropout_input", 0.0)),
            dropout_hidden=float(model_cfg.get("dropout_hidden", 0.0)),
            pooling=str(model_cfg.get("pooling", "last")),
            normalization_strategy=normalization_strategy,
        )

    valid = ", ".join(sorted(MODEL_REGISTRY))
    raise ValueError(f"Unsupported model_name={model_name!r}. Expected one of: {valid}")
