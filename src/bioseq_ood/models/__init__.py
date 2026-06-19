from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "MODEL_REGISTRY": "registry",
    "build_model": "registry",
    "MLPRegressor": "mlp",
    "CNNRegressor": "cnn",
    "LSTMRegressor": "lstm",
    "CNNLSTMRegressor": "cnn_lstm",
    "LSTMCNNRegressor": "lstm_cnn",
    "TransformerRegressor": "transformer",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
