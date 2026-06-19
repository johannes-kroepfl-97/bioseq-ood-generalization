from __future__ import annotations

import argparse
import json
import random
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from bioseq_ood.config.loader import deep_update, load_config
from bioseq_ood.training.trainer import train_single_run


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must contain a dictionary at the top level.")
    return data


def sample_from_space(space: dict[str, Any]) -> dict[str, Any]:
    sampled: dict[str, Any] = {}
    for key, value in space.items():
        if isinstance(value, dict):
            sampled[key] = sample_from_space(value)
        elif isinstance(value, list):
            if not value:
                raise ValueError(f"Search space list for key '{key}' is empty.")
            sampled[key] = random.choice(value)
        else:
            sampled[key] = value
    return sampled


def copy_best_artifacts(best_run_dir: Path, best_dir: Path) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ["config.yaml", "model_state_dict.pt", "metrics.json", "y_scaler.json", "mlflow_run_id.txt"]:
        src = best_run_dir / file_name
        if src.exists():
            shutil.copy2(src, best_dir / file_name)


def run_random_search(base_config: dict[str, Any], search_space: dict[str, Any], n_trials: int) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    model_name = str(base_config.get("model_name", "cnn"))
    dataset_name = str(base_config["dataset"]["name"])

    for trial_idx in range(n_trials):
        sampled_override = sample_from_space(search_space)
        trial_config = deep_update(deepcopy(base_config), sampled_override)
        trial_config.setdefault("output", {})["run_name"] = f"trial_{trial_idx:03d}"
        trial_config.setdefault("mlflow", {})["run_name"] = f"{model_name}__{dataset_name}__trial_{trial_idx:03d}"
        metrics, artifacts = train_single_run(trial_config)
        result = {
            "trial": trial_idx,
            "selected_metric": metrics["selected_metric"],
            "val_id_mae": metrics["val_id"]["mae"],
            "val_ood_mae": metrics["val_ood"]["mae"],
            "run_dir": str(artifacts.run_dir),
            "config": trial_config,
        }
        trials.append(result)
        if best_result is None or result["selected_metric"] < best_result["selected_metric"]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Random search did not run any trials.")

    best_run_dir = Path(best_result["run_dir"])
    output_dir = Path(base_config.get("output", {}).get("base_dir", "results/training"))
    best_dir = output_dir / model_name / dataset_name / "best"
    copy_best_artifacts(best_run_dir, best_dir)
    summary = {"n_trials": n_trials, "best_trial": best_result, "trials": trials, "best_dir": str(best_dir)}
    with open(output_dir / model_name / dataset_name / "random_search_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random search for one deep learning model.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--override", default=None, help="Optional dataset override YAML.")
    parser.add_argument("--search-space", required=True, help="Path to YAML search space.")
    parser.add_argument("--trials", type=int, default=20, help="Number of random search trials.")
    args = parser.parse_args()
    base_config = load_config(args.config, args.override)
    search_space = load_yaml(args.search_space)
    summary = run_random_search(base_config, search_space, args.trials)
    print(json.dumps(summary["best_trial"], indent=2))


if __name__ == "__main__":
    main()
