from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bioseq_ood.data.datasets import SequenceRegressionDataset


@dataclass(frozen=True)
class EvaluationArtifacts:
    predictions_path: Path
    metrics_path: Path
    per_mut_dist_metrics_path: Path


def _inverse_scale_tensor(
    values: torch.Tensor,
    y_scaler: dict[str, float] | None,
) -> torch.Tensor:
    if y_scaler is None:
        return values
    mean = torch.tensor(y_scaler["mean"], dtype=values.dtype, device=values.device)
    std = torch.tensor(y_scaler["std"], dtype=values.dtype, device=values.device)
    return values * std + mean


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.shape[0] == 0:
        return {
            "mae": float("nan"),
            "mse": float("nan"),
            "rmse": float("nan"),
            "std_abs_error": float("nan"),
            "n_samples": 0,
        }

    err = y_pred - y_true
    abs_err = np.abs(err)
    mse = float(np.mean(err ** 2))
    return {
        "mae": float(np.mean(abs_err)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "std_abs_error": float(np.std(abs_err)),
        "n_samples": int(y_true.shape[0]),
    }


def evaluate_regression_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    y_scaler: dict[str, float] | None,
    device: torch.device | str,
    split_name: str,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Evaluate a model and return overall, per-row, and per-mutation-distance metrics.

    The dataloader may yield either `(x, y)` or `(x, y, mut_dist)`. If mutation
    distances are present, the returned per-row dataframe contains them and the
    grouped metrics dataframe reports one row per mutation count.
    """
    model = model.eval().to(device)

    preds_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    mut_dist_all: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
                mut_dist = None
            elif len(batch) == 3:
                x, y, mut_dist = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements.")

            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds_all.append(preds.detach().cpu())
            targets_all.append(y.detach().cpu())
            if mut_dist is not None:
                mut_dist_all.append(mut_dist.detach().cpu())

    preds_scaled = torch.cat(preds_all, dim=0)
    targets_scaled = torch.cat(targets_all, dim=0)

    preds = _inverse_scale_tensor(preds_scaled, y_scaler).numpy().reshape(-1)
    targets = _inverse_scale_tensor(targets_scaled, y_scaler).numpy().reshape(-1)

    pred_df = pd.DataFrame(
        {
            "split": split_name,
            "y_true": targets,
            "y_pred": preds,
            "error": preds - targets,
            "abs_error": np.abs(preds - targets),
        }
    )

    if mut_dist_all:
        pred_df["mut_dist"] = torch.cat(mut_dist_all, dim=0).numpy().reshape(-1).astype(int)
    else:
        pred_df["mut_dist"] = np.nan

    overall = _regression_metrics(targets, preds)
    overall["split"] = split_name

    if pred_df["mut_dist"].notna().any():
        rows = []
        for mut_dist, group in pred_df.groupby("mut_dist", dropna=True):
            row = _regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
            row["split"] = split_name
            row["mut_dist"] = int(mut_dist)
            rows.append(row)
        per_mut_dist_df = pd.DataFrame(rows).sort_values(["split", "mut_dist"]).reset_index(drop=True)
    else:
        per_mut_dist_df = pd.DataFrame(
            columns=["split", "mut_dist", "mae", "mse", "rmse", "std_abs_error", "n_samples"]
        )

    return overall, pred_df, per_mut_dist_df


def evaluate_splits(
    model: torch.nn.Module,
    data_module,
    *,
    split_names: Iterable[str],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device | str,
    output_dir: str | Path,
) -> tuple[dict[str, dict[str, float | int]], pd.DataFrame, pd.DataFrame, EvaluationArtifacts]:
    """Evaluate multiple labeled splits and persist predictions + metrics artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics: dict[str, dict[str, float | int]] = {}
    all_predictions: list[pd.DataFrame] = []
    all_per_mut_dist: list[pd.DataFrame] = []

    for split_name in split_names:
        split_artifact = data_module.get_split_artifact(split_name)
        if split_artifact is None or split_artifact.get("y") is None:
            continue

        dataset = SequenceRegressionDataset(
            split_artifact["x"],
            split_artifact["y"],
            dataset_name=data_module.dataset_name,
            mut_dist=split_artifact.get("mut_dist"),
            input_encoding=data_module.input_encoding,
            vocab_size=data_module.bundle.vocab_size,
            return_mut_dist=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        metrics, pred_df, per_mut_dist_df = evaluate_regression_model(
            model,
            dataloader,
            y_scaler=data_module.bundle.y_scaler,
            device=device,
            split_name=split_name,
        )
        overall_metrics[split_name] = metrics
        all_predictions.append(pred_df)
        all_per_mut_dist.append(per_mut_dist_df)

    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    per_mut_dist_df = pd.concat(all_per_mut_dist, ignore_index=True) if all_per_mut_dist else pd.DataFrame()
    metrics_df = pd.DataFrame(list(overall_metrics.values()))

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "evaluation_metrics.csv"
    per_mut_dist_path = output_dir / "evaluation_metrics_by_mut_dist.csv"

    predictions_df.to_csv(predictions_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    per_mut_dist_df.to_csv(per_mut_dist_path, index=False)

    artifacts = EvaluationArtifacts(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        per_mut_dist_metrics_path=per_mut_dist_path,
    )
    return overall_metrics, predictions_df, per_mut_dist_df, artifacts
