from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "std_abs_error": float("nan"), "n_samples": 0}
    err = y_pred - y_true
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    return {"mae": float(np.mean(abs_err)), "mse": mse, "rmse": float(np.sqrt(mse)), "std_abs_error": float(np.std(abs_err)), "n_samples": int(y_true.size)}
