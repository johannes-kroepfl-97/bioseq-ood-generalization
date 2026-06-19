from __future__ import annotations

from pathlib import Path
from typing import Any

from bioseq_ood.config.loader import load_config
from bioseq_ood.training.trainer import train_single_run


def run_single_experiment(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    override_path: str | Path | None = None,
    method_name: str | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
    run_name: str | None = None,
    use_mlflow: bool | None = None,
):
    if config is None:
        if config_path is None:
            raise ValueError("Provide either config or config_path.")
        config = load_config(config_path, override_path)
    else:
        config = dict(config)

    if dataset_name is not None:
        config.setdefault("dataset", {})["name"] = dataset_name
    if model_name is not None:
        config["model_name"] = model_name
    if method_name is not None:
        config.setdefault("training", {})["method"] = method_name
    if run_name is not None:
        config.setdefault("output", {})["run_name"] = run_name
    if use_mlflow is not None:
        config.setdefault("mlflow", {})["enabled"] = bool(use_mlflow)

    return train_single_run(config)
