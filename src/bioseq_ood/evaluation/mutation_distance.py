from __future__ import annotations

import pandas as pd

from .metrics import regression_metrics


def metrics_by_mutation_distance(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"mut_dist", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns for mutation-distance evaluation: {sorted(missing)}")
    rows = []
    for mut_dist, group in predictions.groupby("mut_dist", dropna=True):
        row = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        row["mut_dist"] = int(mut_dist)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mut_dist").reset_index(drop=True)
