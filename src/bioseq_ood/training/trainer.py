from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger, MLFlowLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

import yaml

from bioseq_ood.config.loader import load_config
from bioseq_ood.data.datasets import SequenceDataModule
from bioseq_ood.evaluation.evaluate import evaluate_splits
from bioseq_ood.methods.registry import build_method_from_config
from bioseq_ood.models.registry import build_model
from bioseq_ood.training.lightning_module import LightningSequenceRegressor, set_dropout_train_bn_eval
from bioseq_ood.training.selection import SplitPlan, plan_from_config
from bioseq_ood.utils.logging import flatten_dict
from bioseq_ood.utils.paths import ensure_dir
from bioseq_ood.utils.seed import set_seed


@dataclass
class RunArtifacts:
    run_dir: Path
    best_checkpoint_path: Path | None
    model_state_dict_path: Path
    metrics_path: Path
    config_path: Path
    y_scaler_path: Path | None
    mlflow_run_id_path: Path | None
    predictions_path: Path | None = None
    evaluation_metrics_path: Path | None = None
    per_mut_dist_metrics_path: Path | None = None


class NullMlflowLogger:
    def __init__(self) -> None:
        self.run_id: str | None = None

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None


def build_run_dir(base_output_dir: str | Path, model_name: str, dataset_name: str, run_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or timestamp
    return ensure_dir(Path(base_output_dir) / model_name / dataset_name / run_id)


def create_mlflow_logger(config: dict[str, Any], model_name: str, dataset_name: str):
    mlflow_cfg = config.get("mlflow", {})
    if not mlflow_cfg.get("enabled", True):
        return NullMlflowLogger()

    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name", f"ssl-ood-{model_name}")
    run_name = mlflow_cfg.get("run_name", f"{model_name}__{dataset_name}")
    try:
        return MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            run_name=run_name,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not initialize MLflow logger: {exc}")
        return NullMlflowLogger()


def _log_mlflow_artifacts(mlflow_logger, paths: list[Path]) -> None:
    if isinstance(mlflow_logger, NullMlflowLogger):
        return
    run_id = getattr(mlflow_logger, "run_id", None)
    experiment = getattr(mlflow_logger, "experiment", None)
    if run_id is None or experiment is None:
        return
    for path in paths:
        if path is not None and Path(path).exists():
            try:
                experiment.log_artifact(run_id, str(path))
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Could not log MLflow artifact {path}: {exc}")


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def save_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _git_sha() -> str | None:
    """Short commit hash of the working tree (for run provenance), or None.

    Lets every run record which code produced it -- the thing that caused the
    old-vs-new-code confusion. Falls back to the GIT_SHA env var when git is not
    available (e.g. a shallow copy on the pod)."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return os.environ.get("GIT_SHA")


def _safe_len(ds: Any) -> int | None:
    try:
        return int(len(ds)) if ds is not None else None
    except Exception:
        return None


def _write_run_record(
    *,
    run_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    data_module: SequenceDataModule,
    stage: str,
    epochs_run: int,
    max_epochs: int,
    stopped_early: bool,
) -> Path:
    """Write the canonical, self-contained per-run record (run_record.json).

    This is the single source of truth that collect.py walks to build all_runs.csv.
    Everything here is derived from the already-built ``metrics`` dict plus a little
    provenance, so it stays in lock-step with metrics.json. ``config["provenance"]``
    (set by the pipeline: protocol / adapt_pool / test_pool / trial) is copied through
    verbatim; the trainer does not know the protocol naming itself.
    """
    training_cfg = config.get("training", {}) or {}
    prov = config.get("provenance", {}) if isinstance(config.get("provenance"), dict) else {}
    method_name = (metrics.get("method", {}) or {}).get("name") or training_cfg.get("method", "erm")
    seed = metrics.get("seed")
    eval_metrics = metrics.get("evaluation", {}) or {}

    # adapt pool: explicit provenance, else inferred from the unlabeled target file(s).
    adapt_pool = prov.get("adapt_pool")
    if adapt_pool is None:
        target_files = training_cfg.get("target_split_files") or []
        if target_files:
            adapt_pool = Path(str(target_files[0])).stem

    method_hparams = metrics.get(method_name) if isinstance(metrics.get(method_name), dict) else {}
    protocol = prov.get("protocol")
    run_name = Path(run_dir).name
    run_id = "__".join(
        str(x) for x in [metrics.get("dataset"), metrics.get("model_name"),
                         method_name, protocol or "na", f"seed{seed}", run_name]
    )

    record = {
        # --- identity / provenance ---
        "run_id": run_id,
        "run_dir": str(run_dir),
        "stage": stage,                          # "search" | "final" | "unknown"
        "git_sha": _git_sha(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        # --- experiment keys ---
        "dataset": metrics.get("dataset"),
        "model": metrics.get("model_name"),
        "method": method_name,
        "protocol": protocol,
        "adapt_pool": adapt_pool,
        "test_pool": prov.get("test_pool"),
        "seed": seed,
        "trial": prov.get("trial"),
        # --- config / data shape ---
        "normalization_strategy": metrics.get("normalization_strategy"),
        "input_encoding": metrics.get("input_encoding"),
        "seq_len": metrics.get("seq_len"),
        "vocab_size": metrics.get("vocab_size"),
        "n_train": _safe_len(getattr(data_module, "train_dataset", None)),
        "n_val_id": _safe_len(getattr(data_module, "val_id_dataset", None)),
        "n_adapt": _safe_len(getattr(data_module, "target_unlabeled_dataset", None)),
        "training_cfg": training_cfg,
        "model_cfg": config.get("model", {}),
        "method_meta": metrics.get("method", {}),
        "method_hparams": method_hparams,
        "selection": metrics.get("selection", {}),
        # --- training dynamics (scalars) ---
        "best_epoch": metrics.get("best_epoch"),
        "epochs_run": epochs_run,
        "max_epochs": max_epochs,
        "stopped_early": stopped_early,
        "training_dynamics": metrics.get("train_losses", {}),
        # --- outcomes (per evaluated split: mae/rmse/spearman/r2/naive_mae/n_samples/...) ---
        "metrics": eval_metrics,
        "selected_split": metrics.get("selected_split"),
        "selected_metric": metrics.get("selected_metric"),
        "report_split": (metrics.get("selection", {}) or {}).get("report_split"),
        "report_metric": metrics.get("report_metric"),
        # --- status / pointers ---
        "status": "ok",
        "error": None,
        "paths": {
            "metrics_json": str(Path(run_dir) / "metrics.json"),
            "config_yaml": str(Path(run_dir) / "config.yaml"),
            "predictions_csv": str(Path(run_dir) / "evaluation" / "predictions.csv"),
            "per_mut_dist_csv": str(Path(run_dir) / "evaluation" / "evaluation_metrics_by_mut_dist.csv"),
        },
    }
    return save_json(Path(run_dir) / "run_record.json", record)


def _safe_split_metric(overall_eval_metrics: dict[str, Any], split: str, metric_name: str) -> float | None:
    split_metrics = overall_eval_metrics.get(split)
    if not isinstance(split_metrics, dict) or metric_name not in split_metrics:
        return None
    value = split_metrics[metric_name]
    return float(value) if isinstance(value, (int, float)) else None


def _count_bn_layers(model: torch.nn.Module) -> int:
    """Return the number of BatchNorm layers in the model (for logging transparency)."""
    return sum(1 for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))


def _apply_adabn(model: torch.nn.Module, target_loader, device: torch.device | str) -> int:
    """Update BatchNorm running statistics on unlabeled target inputs (AdaBN). """
    has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    if not has_bn:
        return 0    # No BatchNorm in the architecture --> AdaBN does nothing.

    was_training = model.training
    model.to(device)
    model.train()

    # PyTorch with momentum=None, Without this, PyTorch uses EMA (momentum=0.1) which is biased toward the last
    # batches and carries the source running_mean as its initialisation.

    saved_momenta: list[tuple[torch.nn.modules.batchnorm._BatchNorm, float | None]] = []
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            saved_momenta.append((m, m.momentum))
            m.momentum = None
            m.reset_running_stats()

    n_batches = 0
    with torch.no_grad():
        for batch in target_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            _ = model(x)
            n_batches += 1

    for m, orig_momentum in saved_momenta:
        m.momentum = orig_momentum

    model.train(was_training)
    return n_batches


def _build_run_contract(
    *,
    config: dict,
    method_meta: dict,
    dataset_name: str,
    model_name: str,
    seed: int,
    plan: SplitPlan,
) -> dict:
    """Create compact metadata describing exactly what this run used."""
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    debug_cfg = config.get("debug", {})
    mlflow_cfg = config.get("mlflow", {})
    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "method_name": training_cfg.get("method", "erm"),
        "seed": seed,
        "selection_split": plan.selection_split,
        "report_split": plan.report_split,
        "checkpoint_monitor": plan.monitor_metric,
        "early_stop_monitor": plan.early_stop_metric,
        "source_split": method_meta.get("source_split", "train"),
        "target_split": method_meta.get("target_setting"),
        "validation_split": method_meta.get("validation_split", "val_id"),
        "test_split": method_meta.get("test_split", "test"),
        "target_labels_used": method_meta.get("target_labels_used", False),
        "target_labels_ignored": method_meta.get("target_labels_ignored", False),
        "input_encoding": dataset_cfg.get("input_encoding", config.get("data", {}).get("input_encoding", "one_hot")),
        "debug_enabled": bool(debug_cfg.get("enabled", False)),
        "debug_max_samples_per_split": debug_cfg.get("max_samples_per_split"),
        "debug_seed": debug_cfg.get("seed"),
        "mlflow_enabled": bool(mlflow_cfg.get("enabled", False)),
    }


def _get_pseudo_cfg(training_cfg: dict[str, Any], run_seed: int = 42) -> dict[str, Any]:
    raw = training_cfg.get("pseudo_labeling", {})
    pseudo_cfg = raw if isinstance(raw, dict) else {}
    # The run seed lives at config["seed"], not training_cfg, so it must be passed in.
    # Deriving mc_seed from it makes the MC-dropout pseudo-label draw differ per seed
    # instead of defaulting to a constant (which made every seed bit-identical).
    seed_default = int(training_cfg.get("seed", run_seed))
    return {
        "keep_ratio": float(training_cfg.get("keep_ratio", pseudo_cfg.get("keep_ratio", 0.5))),
        "mc_passes": int(training_cfg.get("mc_passes", pseudo_cfg.get("mc_passes", 20))),
        "lambda_pseudo_max": float(training_cfg.get("lambda_pseudo_max", pseudo_cfg.get("lambda_pseudo_max", 0.3))),
        "rampup_epochs": int(training_cfg.get("rampup_epochs", pseudo_cfg.get("rampup_epochs", 10))),
        "retrain_from_scratch": bool(training_cfg.get("retrain_from_scratch", pseudo_cfg.get("retrain_from_scratch", True))),
        "mc_seed": int(training_cfg.get("mc_seed", pseudo_cfg.get("mc_seed", seed_default + 10_000))),
        "pretrained_checkpoint": training_cfg.get("pretrained_checkpoint", pseudo_cfg.get("pretrained_checkpoint")),
    }


def _predict_mc_dropout(
    model: torch.nn.Module,
    dataloader,
    *,
    mc_passes: int,
    device: torch.device | str,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predictive mean and std over T stochastic MC-dropout passes. """
    if mc_passes <= 0:
        raise ValueError("mc_passes must be a positive integer.")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    model.to(device)
    all_passes: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(mc_passes):                       # T stochastic forward passes
            set_dropout_train_bn_eval(model)             # dropout on, BN frozen
            preds_one_pass: list[torch.Tensor] = []
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                preds_one_pass.append(model(x).detach().cpu())   # ŷ_t for this pass
            all_passes.append(torch.cat(preds_one_pass, dim=0))
    stacked = torch.stack(all_passes, dim=0)             # (T, n_samples, 1)
    mean = stacked.mean(dim=0)                           # E[y*] ≈ (1/T) Σ ŷ_t
    std = stacked.std(dim=0, unbiased=False)             # sqrt(sample variance); τ^{-1} dropped
    model.eval()
    return mean, std


def _build_validation(data_module: SequenceDataModule) -> tuple[list[Any], list[str]]:
    """Validation dataloaders + stage names. Selection always monitors val_id."""
    return [data_module.val_id_dataloader(), data_module.val_ood_dataloader()], ["val_id", "val_ood"]


def _make_trainer(
    *,
    training_cfg: dict[str, Any],
    run_dir: Path,
    checkpoints_dir: Path,
    mlflow_logger,
    filename: str = "best",
    monitor: str | None = None,
    early_stop_monitor: str | None = None,
) -> tuple[Any, Any, Any]:
    monitor = monitor or str(training_cfg.get("checkpoint_monitor", "val_id_mae"))
    early_stop_monitor = early_stop_monitor or str(training_cfg.get("early_stopping_monitor", "val_id_mae"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename=filename,
        monitor=monitor,
        mode="min",
        save_top_k=1,
        # We only reload the best weights to evaluate; the optimizer/scheduler state is
        # never resumed, so dropping it makes each checkpoint write ~3x smaller.
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor=early_stop_monitor,
        mode="min",
        patience=int(training_cfg.get("early_stopping_patience", 20)),
    )
    # Always persist the per-epoch metrics the LightningModule logs (val_id_mae,
    # train_loss, train_lambda_consistency/_fixmatch, *_consistency_loss, ...) to
    # <run_dir>/csv_logs/.../metrics.csv. Every run -- search trials included -- needs an
    # inspectable training curve; it is most important precisely when a method
    # underperforms (did it diverge, never ramp up, early-stop too soon?). The CSV is a
    # few KB per run, so this is cheap. When MLflow is enabled it logs in parallel.
    csv_logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")
    if isinstance(mlflow_logger, NullMlflowLogger):
        run_logger = csv_logger
    else:
        run_logger = [mlflow_logger, csv_logger]

    trainer = pl.Trainer(
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", "auto"),
        max_epochs=int(training_cfg.get("epochs", 100)),
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 10)),
        logger=run_logger,
        callbacks=[checkpoint_callback, early_stopping],
        deterministic=bool(training_cfg.get("deterministic", True)),
        enable_progress_bar=bool(training_cfg.get("enable_progress_bar", True)),
        default_root_dir=str(run_dir),
    )
    return trainer, checkpoint_callback, early_stopping


def _save_standard_outputs(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
    best_model: torch.nn.Module,
    data_module: SequenceDataModule,
    mlflow_logger,
    eval_artifacts,
    best_ckpt_path: str | Path | None,
    run_contract: dict[str, Any],
) -> RunArtifacts:
    model_state_dict_path = run_dir / "model_state_dict.pt"
    if config.get("training", {}).get("save_model_state_dict", True):
        torch.save(best_model.state_dict(), model_state_dict_path)
    metrics_path = save_json(run_dir / "metrics.json", metrics)
    config_path = save_yaml(run_dir / "config.yaml", config)
    run_contract_path = save_json(run_dir / "run_contract.json", run_contract)

    y_scaler_path = None
    if data_module.bundle is not None and data_module.bundle.y_scaler is not None:
        y_scaler_path = save_json(run_dir / "y_scaler.json", data_module.bundle.y_scaler)

    mlflow_run_id_path = None
    run_id = getattr(mlflow_logger, "run_id", None)
    if run_id is not None:
        mlflow_run_id_path = run_dir / "mlflow_run_id.txt"
        mlflow_run_id_path.write_text(str(run_id), encoding="utf-8")

    _log_mlflow_artifacts(
        mlflow_logger,
        [
            metrics_path,
            config_path,
            run_contract_path,
            model_state_dict_path,
            eval_artifacts.predictions_path,
            eval_artifacts.metrics_path,
            eval_artifacts.per_mut_dist_metrics_path,
        ],
    )

    return RunArtifacts(
        run_dir=run_dir,
        best_checkpoint_path=None if best_ckpt_path is None else Path(best_ckpt_path),
        model_state_dict_path=model_state_dict_path,
        metrics_path=metrics_path,
        config_path=config_path,
        y_scaler_path=y_scaler_path,
        mlflow_run_id_path=mlflow_run_id_path,
        predictions_path=eval_artifacts.predictions_path,
        evaluation_metrics_path=eval_artifacts.metrics_path,
        per_mut_dist_metrics_path=eval_artifacts.per_mut_dist_metrics_path,
    )


def _train_pseudo_labeling_run(config: dict[str, Any], method) -> tuple[dict[str, Any], RunArtifacts]:
    """Run the two-stage pseudo-labeling protocol."""
    config = deepcopy(config)
    model_name = str(config.get("model_name", "cnn"))
    dataset_name = str(config["dataset"]["name"])
    training_cfg = config["training"]
    training_cfg["use_cmd"] = False
    training_cfg["lambda_cmd"] = 0.0

    plan = plan_from_config(config)

    seed = int(config.get("seed", 42))
    stage = str(config.get("stage", "unknown"))
    pseudo_cfg = _get_pseudo_cfg(training_cfg, run_seed=seed)
    keep_ratio = pseudo_cfg["keep_ratio"]
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("Pseudo-label keep_ratio must be in (0, 1].")

    output_dir = config.get("output", {}).get("base_dir", "results/training")
    run_name = config.get("output", {}).get("run_name")
    run_dir = build_run_dir(output_dir, model_name, dataset_name, run_name=run_name)
    pretrain_ckpt_dir = ensure_dir(run_dir / "checkpoints" / "pretrain")
    final_ckpt_dir = ensure_dir(run_dir / "checkpoints" / "final")
    eval_dir = ensure_dir(run_dir / "evaluation")
    pseudo_dir = ensure_dir(run_dir / "pseudo_labels")

    set_seed(seed)

    data_module = SequenceDataModule(
        dataset_name=dataset_name,
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        include_test=True,
        include_eval_targets=True,
        include_target_unlabeled=True,
        target_split_files=method.target_split_files,
        allow_test_as_target=method.allow_test_as_target,
        target_drop_last=False,
        input_encoding=str(config.get("data", {}).get("input_encoding", "one_hot")),
        debug_enabled=bool(config.get("debug", {}).get("enabled", False)),
        debug_max_samples_per_split=config.get("debug", {}).get("max_samples_per_split"),
        debug_seed=int(config.get("debug", {}).get("seed", seed)),
    )
    data_module.setup()
    assert data_module.bundle is not None
    if data_module.target_unlabeled_dataset is None:
        raise RuntimeError("Pseudo-labeling requires target data, but none was loaded.")

    mlflow_logger = create_mlflow_logger(config, model_name=model_name, dataset_name=dataset_name)
    method_meta = method.to_metadata(data_module)
    method_meta.update({
        "pseudo_keep_ratio": keep_ratio,
        "pseudo_mc_passes": pseudo_cfg["mc_passes"],
        "pseudo_lambda_pseudo_max": pseudo_cfg["lambda_pseudo_max"],
        "pseudo_rampup_epochs": pseudo_cfg["rampup_epochs"],
        "pseudo_retrain_from_scratch": pseudo_cfg["retrain_from_scratch"],
    })
    run_contract = _build_run_contract(
        config=config,
        method_meta=method_meta,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        plan=plan,
    )

    if hasattr(mlflow_logger, "log_hyperparams"):
        mlflow_logger.log_hyperparams(flatten_dict(config))
        mlflow_logger.log_hyperparams(flatten_dict({"run_contract": run_contract, "method": method_meta}))

    val_dataloaders, val_stage_names = _build_validation(data_module)

    pretrain_model = build_model(model_name, config["model"], data_module.bundle.vocab_size, data_module.bundle.seq_len)
    pretrain_cfg = deepcopy(training_cfg)
    pretrain_cfg["use_cmd"] = False
    pretrain_cfg["lambda_cmd"] = 0.0
    pretrain_cfg["use_pseudo_labeling"] = False
    pretrain_module = LightningSequenceRegressor(
        model=pretrain_model,
        training_config=pretrain_cfg,
        y_scaler=data_module.bundle.y_scaler,
        val_stage_names=val_stage_names,
    )
    pretrainer, pretrain_checkpoint, _ = _make_trainer(
        training_cfg=pretrain_cfg,
        run_dir=run_dir,
        checkpoints_dir=pretrain_ckpt_dir,
        mlflow_logger=mlflow_logger,
        filename="pretrain_best",
        monitor=plan.monitor_metric,
        early_stop_monitor=plan.early_stop_metric,
    )

    pretrained_ckpt_path = pseudo_cfg.get("pretrained_checkpoint")
    stage1_reused = False
    if pretrained_ckpt_path and Path(str(pretrained_ckpt_path)).exists():

        try:
            state = torch.load(str(pretrained_ckpt_path), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            try:
                pretrain_model.load_state_dict(state)
            except Exception:
                # Lightning checkpoints prefix weights with "model."; strip it and retry.
                stripped = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
                pretrain_model.load_state_dict(stripped)
            stage1_reused = True
        except Exception as exc:
            print(
                f"[WARN] Could not reuse pretrained checkpoint {pretrained_ckpt_path} "
                f"({exc}); training Stage 1 from scratch instead."
            )

    if stage1_reused:
        print(f"[pseudo] Stage-1 reused: loaded baseline {pretrained_ckpt_path} (no Stage-1 training)")
        pretrain_best_ckpt = str(pretrained_ckpt_path)
        pretrain_best_module = pretrain_module
        gen_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        pretrainer.fit(
            model=pretrain_module,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=val_dataloaders,
        )
        pretrain_best_ckpt = pretrain_checkpoint.best_model_path or None
        if pretrain_best_ckpt:
            pretrain_best_module = LightningSequenceRegressor.load_from_checkpoint(
                pretrain_best_ckpt,
                model=pretrain_model,
                training_config=pretrain_cfg,
                y_scaler=data_module.bundle.y_scaler,
                val_stage_names=val_stage_names,
            )
        else:
            pretrain_best_module = pretrain_module
        gen_device = pretrainer.strategy.root_device
    pretrain_best_module = pretrain_best_module.to(gen_device)

    target_eval_loader = DataLoader(
        data_module.target_unlabeled_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        drop_last=False,
    )

    pseudo_mean, pseudo_std = _predict_mc_dropout(
        pretrain_best_module.model,
        target_eval_loader,
        mc_passes=int(pseudo_cfg["mc_passes"]),
        device=gen_device,
        seed=int(pseudo_cfg["mc_seed"]),
    )

    uncertainty = pseudo_std.view(-1)
    n_target = int(uncertainty.numel())
    n_keep = max(1, int(np.ceil(n_target * keep_ratio)))
    selected_idx = torch.argsort(uncertainty)[:n_keep]

    target_x = data_module.target_unlabeled_dataset.x.detach().cpu()
    selected_x = target_x[selected_idx]
    selected_y = pseudo_mean.detach().cpu()[selected_idx].view(-1, 1)
    selected_uncertainty = uncertainty.detach().cpu()[selected_idx].view(-1)

    pseudo_summary = pd.DataFrame(
        {
            "target_index": selected_idx.detach().cpu().numpy().astype(int),
            "pseudo_label_scaled": selected_y.numpy().reshape(-1),
            "uncertainty_scaled": selected_uncertainty.numpy().reshape(-1),
        }
    )
    pseudo_labels_path = pseudo_dir / "pseudo_labels_filtered.csv"
    pseudo_summary.to_csv(pseudo_labels_path, index=False)

    pseudo_dataset = TensorDataset(selected_x.to(torch.float32) if selected_x.ndim == 3 else selected_x.long(), selected_y.to(torch.float32))
    pseudo_batch_size = int(training_cfg["batch_size"])
    pseudo_loader = DataLoader(
        pseudo_dataset,
        batch_size=pseudo_batch_size,
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        # A singleton trailing batch breaks BatchNorm in train mode; drop a lone last
        # sample (matches SequenceDataModule._loader). Only triggers when needed.
        drop_last=(len(pseudo_dataset) % pseudo_batch_size == 1),
    )


    # _predict_mc_dropout reset the global RNG to the (fixed) mc_seed, and Stage 1 is a
    # reused checkpoint, so without re-seeding here the Stage-2 weight init, data shuffling
    # and dropout would be identical across run seeds. Re-seed so two seeds are genuinely
    # different runs.
    set_seed(seed)
    final_model = build_model(model_name, config["model"], data_module.bundle.vocab_size, data_module.bundle.seq_len)
    if not pseudo_cfg["retrain_from_scratch"]:
        final_model.load_state_dict(pretrain_best_module.model.state_dict())

    final_cfg = deepcopy(training_cfg)
    final_cfg["use_cmd"] = False
    final_cfg["lambda_cmd"] = 0.0
    final_cfg["use_pseudo_labeling"] = True
    final_cfg["lambda_pseudo_max"] = float(pseudo_cfg["lambda_pseudo_max"])
    final_cfg["rampup_epochs"] = int(pseudo_cfg["rampup_epochs"])

    final_module = LightningSequenceRegressor(
        model=final_model,
        training_config=final_cfg,
        y_scaler=data_module.bundle.y_scaler,
        val_stage_names=val_stage_names,
    )
    final_trainer, final_checkpoint, final_es = _make_trainer(
        training_cfg=final_cfg,
        run_dir=run_dir,
        checkpoints_dir=final_ckpt_dir,
        mlflow_logger=mlflow_logger,
        filename="final_best",
        monitor=plan.monitor_metric,
        early_stop_monitor=plan.early_stop_metric,
    )
    final_trainer.fit(
        model=final_module,
        train_dataloaders={"source": data_module.train_dataloader(), "pseudo": pseudo_loader},
        val_dataloaders=val_dataloaders,
    )

    final_best_ckpt = final_checkpoint.best_model_path or None
    if final_best_ckpt:
        best_module = LightningSequenceRegressor.load_from_checkpoint(
            final_best_ckpt,
            model=final_model,
            training_config=final_cfg,
            y_scaler=data_module.bundle.y_scaler,
            val_stage_names=val_stage_names,
        )
    else:
        best_module = final_module
    best_module = best_module.to(final_trainer.strategy.root_device)

    evaluation_cfg = config.get("evaluation", {})
    eval_splits = list(evaluation_cfg.get("splits", ["val_id", "val_ood", "target_close", "target_test", "test"]))
    for required_split in (plan.selection_split, plan.report_split):
        if required_split not in eval_splits:
            eval_splits.append(required_split)
    overall_eval_metrics, predictions_df, per_mut_dist_df, eval_artifacts = evaluate_splits(
        best_module.model,
        data_module,
        split_names=eval_splits,
        batch_size=int(evaluation_cfg.get("batch_size", training_cfg["batch_size"])),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        device=final_trainer.strategy.root_device,
        output_dir=eval_dir,
    )

    selected_split = plan.selection_split
    selected_metric_name = "mae"
    selected_metric = float(overall_eval_metrics[selected_split][selected_metric_name])

    metrics: dict[str, Any] = {
        "dataset": dataset_name,
        "model_name": model_name,
        "normalization_strategy": config["model"].get("normalization_strategy", None),
        "seed": seed,
        "seq_len": data_module.bundle.seq_len,
        "vocab_size": data_module.bundle.vocab_size,
        "input_encoding": data_module.input_encoding,
        "best_checkpoint_monitor": plan.monitor_metric,
        "best_epoch": int(getattr(final_trainer, "current_epoch", 0)),
        "method": method_meta,
        "run_contract": run_contract,
        "evaluation": overall_eval_metrics,
        "selection": {
            "selection_split": plan.selection_split,
            "report_split": plan.report_split,
            "monitor_metric": plan.monitor_metric,
            "early_stop_metric": plan.early_stop_metric,
        },
        "selected_split": selected_split,
        "selected_metric_name": selected_metric_name,
        "selected_metric": selected_metric,
        "report_metric": _safe_split_metric(overall_eval_metrics, plan.report_split, "mae"),
        "val_id": overall_eval_metrics.get("val_id"),
        "val_ood": overall_eval_metrics.get("val_ood"),
        "test": overall_eval_metrics.get("test"),
        "pseudo_labeling": {
            "enabled": True,
            "pretrain_best_checkpoint": str(pretrain_best_ckpt) if pretrain_best_ckpt else None,
            "final_best_checkpoint": str(final_best_ckpt) if final_best_ckpt else None,
            "stage1_reused_checkpoint": bool(stage1_reused),
            "mc_passes": int(pseudo_cfg["mc_passes"]),
            "mc_seed": int(pseudo_cfg["mc_seed"]),
            "keep_ratio": keep_ratio,
            "n_target_total": n_target,
            "n_target_kept": n_keep,
            "lambda_pseudo_max": float(pseudo_cfg["lambda_pseudo_max"]),
            "rampup_epochs": int(pseudo_cfg["rampup_epochs"]),
            "retrain_from_scratch": bool(pseudo_cfg["retrain_from_scratch"]),
            "mean_uncertainty_kept": float(selected_uncertainty.mean().item()),
            "max_uncertainty_kept": float(selected_uncertainty.max().item()),
            "pseudo_labels_path": str(pseudo_labels_path),
        },
        "cmd": {"enabled": False, "use_cmd": False, "lambda_cmd": 0.0},
        "adabn_batches": 0,
    }

    final_logged_losses = {}
    for key in ["train_loss", "train_loss_total", "train_source_mae_loss", "train_pseudo_mae_loss", "train_lambda_pseudo"]:
        value = final_trainer.callback_metrics.get(key)
        if value is not None:
            final_logged_losses[key] = float(value.detach().cpu().item())
    metrics["train_losses"] = final_logged_losses

    log_metrics = {
        "selected_metric": selected_metric,
        "pseudo_labeling_enabled": 1.0,
        "pseudo_keep_ratio": float(keep_ratio),
        "pseudo_n_target_kept": float(n_keep),
        "pseudo_mean_uncertainty_kept": float(selected_uncertainty.mean().item()),
    }
    for split_name, split_metrics in overall_eval_metrics.items():
        for metric_name, value in split_metrics.items():
            if isinstance(value, (int, float)):
                log_metrics[f"{split_name}_{metric_name}"] = float(value)
    if hasattr(mlflow_logger, "log_metrics"):
        mlflow_logger.log_metrics(log_metrics)

    # Include pseudo-label CSV in MLflow artifacts.
    _log_mlflow_artifacts(mlflow_logger, [pseudo_labels_path])

    artifacts = _save_standard_outputs(
        config=config,
        metrics=metrics,
        run_dir=run_dir,
        best_model=best_module.model,
        data_module=data_module,
        mlflow_logger=mlflow_logger,
        eval_artifacts=eval_artifacts,
        best_ckpt_path=final_best_ckpt,
        run_contract=run_contract,
    )

    max_epochs = int(final_cfg.get("epochs", training_cfg.get("epochs", 100)))
    epochs_run = int(getattr(final_trainer, "current_epoch", 0))
    stopped_early = bool(getattr(final_es, "stopped_epoch", 0)) or epochs_run < max_epochs
    _write_run_record(
        run_dir=run_dir, config=config, metrics=metrics, data_module=data_module,
        stage=stage, epochs_run=epochs_run, max_epochs=max_epochs, stopped_early=stopped_early,
    )
    return metrics, artifacts


def train_single_run(config: dict[str, Any]) -> tuple[dict[str, Any], RunArtifacts]:
    config = deepcopy(config)
    model_name = str(config.get("model_name", "cnn"))
    dataset_name = str(config["dataset"]["name"])
    training_cfg = config["training"]
    stage = str(config.get("stage", "unknown"))
    plan = plan_from_config(config)
    if "target_split_files" not in training_cfg:
        training_cfg["target_split_files"] = ["target_close.csv"]
    method = build_method_from_config(training_cfg)
    if method.name == "pseudo_labeling":
        return _train_pseudo_labeling_run(config, method)

    if method.name == "mean_teacher":
        mt_cfg = training_cfg.get("mean_teacher", {}) if isinstance(training_cfg.get("mean_teacher", {}), dict) else {}
        training_cfg["use_mean_teacher"] = True
        training_cfg["use_fixmatch"] = False
        training_cfg["use_cmd"] = False
        training_cfg["lambda_cmd"] = 0.0
        training_cfg.setdefault("lambda_consistency_max", mt_cfg.get("lambda_consistency_max", 1.0))
        training_cfg.setdefault("consistency_rampup_epochs", mt_cfg.get("rampup_epochs", 10))
        training_cfg.setdefault("ema_decay", mt_cfg.get("ema_decay", 0.999))
        training_cfg.setdefault("ema_decay_warmup", mt_cfg.get("ema_decay_warmup", 0.99))
        training_cfg.setdefault("mt_input_dropout_p", mt_cfg.get("input_dropout_p", 0.0))
        training_cfg.setdefault("mt_consistency_on_source", mt_cfg.get("consistency_on_source", True))
    else:
        training_cfg["use_mean_teacher"] = False

    if method.name == "fixmatch":
        fm_cfg = training_cfg.get("fixmatch", {}) if isinstance(training_cfg.get("fixmatch", {}), dict) else {}
        training_cfg["use_fixmatch"] = True
        training_cfg["use_cmd"] = False
        training_cfg["lambda_cmd"] = 0.0
        training_cfg.setdefault("lambda_fixmatch_max", fm_cfg.get("lambda_fixmatch_max", 1.0))
        training_cfg.setdefault("fixmatch_rampup_epochs", fm_cfg.get("rampup_epochs", 10))
        training_cfg.setdefault("fm_weak_noise_sigma", fm_cfg.get("weak_noise_sigma", 0.0))
        training_cfg.setdefault("fm_strong_noise_sigma", fm_cfg.get("strong_noise_sigma", 0.1))
        training_cfg.setdefault("fm_mc_passes", fm_cfg.get("mc_passes", 10))
        training_cfg.setdefault("fm_keep_ratio", fm_cfg.get("keep_ratio", 0.5))
        if "uncertainty_threshold" in fm_cfg and "fm_uncertainty_threshold" not in training_cfg:
            training_cfg["fm_uncertainty_threshold"] = fm_cfg["uncertainty_threshold"]
    else:
        training_cfg["use_fixmatch"] = False

    if method.name == "cmd":
        training_cfg["use_cmd"] = True
        # lambda_cmd / cmd_n_moments come from config/methods.yaml (or the Phase E
        # tuned values); the lightning module reads them and fails loudly if absent.
        # No silent coercion of lambda_cmd here -- a wrong/zero value must surface.
    else:
        training_cfg["use_cmd"] = False
        training_cfg["lambda_cmd"] = 0.0

    output_dir = config.get("output", {}).get("base_dir", "results/training")
    run_name = config.get("output", {}).get("run_name")
    run_dir = build_run_dir(output_dir, model_name, dataset_name, run_name=run_name)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    eval_dir = ensure_dir(run_dir / "evaluation")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_module = SequenceDataModule(
        dataset_name=dataset_name,
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        include_test=True,
        include_eval_targets=True,
        include_target_unlabeled=method.include_target_unlabeled,
        target_split_files=method.target_split_files,
        allow_test_as_target=method.allow_test_as_target,
        target_drop_last=method.target_drop_last,
        input_encoding=str(config.get("data", {}).get("input_encoding", "one_hot")),
        debug_enabled=bool(config.get("debug", {}).get("enabled", False)),
        debug_max_samples_per_split=config.get("debug", {}).get("max_samples_per_split"),
        debug_seed=int(config.get("debug", {}).get("seed", seed)),
    )
    data_module.setup()
    assert data_module.bundle is not None

    if method.include_target_unlabeled:
        if data_module.target_unlabeled_dataset is None:
            raise RuntimeError(f"Method '{method.name}' requires target data but none was loaded.")
        if len(data_module.target_unlabeled_dataset) < int(training_cfg["batch_size"]):
            raise ValueError(
                "Target data has fewer samples than batch_size. "
                "Reduce training.batch_size or provide more target data."
            )
        assert data_module.target_unlabeled_dataset.y is None

    model = build_model(
        model_name,
        config["model"],
        data_module.bundle.vocab_size,
        data_module.bundle.seq_len,
    )
    val_dataloaders, val_stage_names = _build_validation(data_module)
    lightning_module = LightningSequenceRegressor(
        model=model,
        training_config=config["training"],
        y_scaler=data_module.bundle.y_scaler,
        val_stage_names=val_stage_names,
    )

    mlflow_logger = create_mlflow_logger(config, model_name=model_name, dataset_name=dataset_name)
    method_meta = method.to_metadata(data_module)
    run_contract = _build_run_contract(
        config=config,
        method_meta=method_meta,
        dataset_name=dataset_name,
        model_name=model_name,
        seed=seed,
        plan=plan,
    )

    if hasattr(mlflow_logger, "log_hyperparams"):
        mlflow_logger.log_hyperparams(flatten_dict(config))
        mlflow_logger.log_hyperparams(flatten_dict({"run_contract": run_contract, "method": method_meta}))

    trainer, checkpoint_callback, early_stopping = _make_trainer(
        training_cfg=training_cfg,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        mlflow_logger=mlflow_logger,
        filename="best",
        monitor=plan.monitor_metric,
        early_stop_monitor=plan.early_stop_metric,
    )

    trainer.fit(
        model=lightning_module,
        train_dataloaders=method.build_train_dataloaders(data_module),
        val_dataloaders=val_dataloaders,
    )

    best_ckpt_path = checkpoint_callback.best_model_path or None
    if best_ckpt_path:
        best_module = LightningSequenceRegressor.load_from_checkpoint(
            best_ckpt_path,
            model=model,
            training_config=config["training"],
            y_scaler=data_module.bundle.y_scaler,
            val_stage_names=val_stage_names,
        )
    else:
        best_module = lightning_module
    best_module = best_module.to(trainer.strategy.root_device)

    adabn_batches = 0
    if method.name == "adabn":
        adabn_batches = _apply_adabn(
            best_module.model,
            data_module.target_dataloader(),
            trainer.strategy.root_device,
        )
        if adabn_batches == 0:
            raise RuntimeError(
                "AdaBN ran but found no BatchNorm layers in the model. The adaptation "
                "step was a complete no-op, so this result is identical to ERM. "
                "Set model.normalization_strategy to 'architecture_native_norm' (CNN/MLP/hybrids) "
                "or 'native_norm_plus_final_bn' (LSTM) when using the AdaBN method."
            )

    inference_model = best_module.get_inference_model()

    evaluation_cfg = config.get("evaluation", {})
    eval_splits = list(evaluation_cfg.get("splits", ["val_id", "val_ood", "target_close", "target_test", "test"]))

    for required_split in (plan.selection_split, plan.report_split):
        if required_split not in eval_splits:
            eval_splits.append(required_split)
    overall_eval_metrics, predictions_df, per_mut_dist_df, eval_artifacts = evaluate_splits(
        inference_model,
        data_module,
        split_names=eval_splits,
        batch_size=int(evaluation_cfg.get("batch_size", training_cfg["batch_size"])),
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=bool(training_cfg.get("pin_memory", True)),
        device=trainer.strategy.root_device,
        output_dir=eval_dir,
    )


    selected_split = plan.selection_split
    selected_metric_name = "mae"
    selected_metric = float(overall_eval_metrics[selected_split][selected_metric_name])

    metrics: dict[str, Any] = {
        "dataset": dataset_name,
        "model_name": model_name,
        "normalization_strategy": config["model"].get("normalization_strategy", None),
        "seed": seed,
        "seq_len": data_module.bundle.seq_len,
        "vocab_size": data_module.bundle.vocab_size,
        "input_encoding": data_module.input_encoding,
        "best_checkpoint_monitor": plan.monitor_metric,
        "best_epoch": int(getattr(trainer, "current_epoch", 0)),
        "method": method_meta,
        "adabn_batches": adabn_batches if method.name == "adabn" else 0,
        "adabn_bn_layer_count": _count_bn_layers(best_module.model) if method.name == "adabn" else 0,
        "run_contract": run_contract,
        "evaluation": overall_eval_metrics,
        "selection": {
            "selection_split": plan.selection_split,
            "report_split": plan.report_split,
            "monitor_metric": plan.monitor_metric,
            "early_stop_metric": plan.early_stop_metric,
        },
        "selected_split": selected_split,
        "selected_metric_name": selected_metric_name,
        "selected_metric": selected_metric,
        "report_metric": _safe_split_metric(overall_eval_metrics, plan.report_split, "mae"),

        "val_id": overall_eval_metrics.get("val_id"),
        "val_ood": overall_eval_metrics.get("val_ood"),
        "test": overall_eval_metrics.get("test"),
    }

    metrics["cmd"] = {
        "enabled": method.name == "cmd",
        "use_cmd": bool(training_cfg.get("use_cmd", method.name == "cmd")),
        "lambda_cmd": float(training_cfg.get("lambda_cmd", 0.0)),
        "n_moments": int(training_cfg.get("cmd_n_moments", 5)),
        "a": float(training_cfg.get("cmd_a", 0.0)),
        "b": float(training_cfg.get("cmd_b", 1.0)),
        "features_bounded_by": "torch.sigmoid(h_raw)",
        "target_split_files": method_meta.get("target_split_files", []),
        "target_labels_ignored": method_meta.get("target_labels_ignored", False),
        "target_n_samples": method_meta.get("target_n_samples"),
    }

    metrics["mean_teacher"] = {
        "enabled": method.name == "mean_teacher",
        "use_mean_teacher": bool(training_cfg.get("use_mean_teacher", False)),
        "lambda_consistency_max": float(training_cfg.get("lambda_consistency_max", 1.0)),
        "consistency_rampup_epochs": int(training_cfg.get("consistency_rampup_epochs", 10)),
        "ema_decay": float(training_cfg.get("ema_decay", 0.999)),
        "ema_decay_warmup": float(training_cfg.get("ema_decay_warmup", 0.99)),
        "input_dropout_p": float(training_cfg.get("mt_input_dropout_p", 0.0)),
        "consistency_on_source": bool(training_cfg.get("mt_consistency_on_source", True)),
        "use_teacher_for_eval": bool(training_cfg.get("use_teacher_for_eval", True)),
        "target_split_files": method_meta.get("target_split_files", []),
        "target_labels_ignored": method_meta.get("target_labels_ignored", False),
        "target_n_samples": method_meta.get("target_n_samples"),
    }

    metrics["fixmatch"] = {
        "enabled": method.name == "fixmatch",
        "use_fixmatch": bool(training_cfg.get("use_fixmatch", False)),
        "lambda_fixmatch_max": float(training_cfg.get("lambda_fixmatch_max", 1.0)),
        "rampup_epochs": int(training_cfg.get("fixmatch_rampup_epochs", 10)),
        "weak_noise_sigma": float(training_cfg.get("fm_weak_noise_sigma", 0.0)),
        "strong_noise_sigma": float(training_cfg.get("fm_strong_noise_sigma", 0.1)),
        "mc_passes": int(training_cfg.get("fm_mc_passes", 10)),
        "keep_ratio": float(training_cfg.get("fm_keep_ratio", 0.5)),
        "uncertainty_threshold": training_cfg.get("fm_uncertainty_threshold", None),
        "target_split_files": method_meta.get("target_split_files", []),
        "target_labels_ignored": method_meta.get("target_labels_ignored", False),
        "target_n_samples": method_meta.get("target_n_samples"),
        "regression_adaptation": (
            "pseudo-target = MC-dropout predictive mean over T weak passes (UPS-style); "
            "strong view = Gaussian noise on the one-hot input; gate = quantile on MC std; "
            "λ_u ramped (sigmoid) to restore the curriculum the fixed-fraction gate loses; "
            "ℓ_u normalized by full unlabeled batch size μB"
        ),
    }

    final_logged_losses = {}
    for key in ["train_loss", "train_loss_pred", "train_loss_cmd", "train_loss_total", "train_source_loss", "train_consistency_loss", "train_lambda_consistency", "train_ema_decay", "train_fixmatch_loss", "train_lambda_fixmatch", "train_fixmatch_mask_fraction", "train_fixmatch_uncertainty"]:
        value = trainer.callback_metrics.get(key)
        if value is not None:
            final_logged_losses[key] = float(value.detach().cpu().item())
    metrics["train_losses"] = final_logged_losses

    model_state_dict_path = run_dir / "model_state_dict.pt"
    if training_cfg.get("save_model_state_dict", True):
        torch.save(inference_model.state_dict(), model_state_dict_path)
        if method.name == "mean_teacher" and best_module.teacher_model is not None:
            torch.save(best_module.model.state_dict(), run_dir / "student_model_state_dict.pt")
            torch.save(best_module.teacher_model.state_dict(), run_dir / "teacher_model_state_dict.pt")
    metrics_path = save_json(run_dir / "metrics.json", metrics)
    config_path = save_yaml(run_dir / "config.yaml", config)
    run_contract_path = save_json(run_dir / "run_contract.json", run_contract)

    y_scaler_path = None
    if data_module.bundle.y_scaler is not None:
        y_scaler_path = save_json(run_dir / "y_scaler.json", data_module.bundle.y_scaler)

    mlflow_run_id_path = None
    run_id = getattr(mlflow_logger, "run_id", None)
    if run_id is not None:
        mlflow_run_id_path = run_dir / "mlflow_run_id.txt"
        mlflow_run_id_path.write_text(str(run_id), encoding="utf-8")

    log_metrics = {
        "selected_metric": selected_metric,
        "cmd_enabled": float(method.name == "cmd"),
        "adabn_enabled": float(method.name == "adabn"),
        "mean_teacher_enabled": float(method.name == "mean_teacher"),
        "fixmatch_enabled": float(method.name == "fixmatch"),
        "adabn_batches": float(adabn_batches if method.name == "adabn" else 0),
        "cmd_lambda": float(training_cfg.get("lambda_cmd", 0.0)),
        "cmd_n_moments": float(training_cfg.get("cmd_n_moments", 5)),
        "cmd_a": float(training_cfg.get("cmd_a", 0.0)),
        "cmd_b": float(training_cfg.get("cmd_b", 1.0)),
    }
    for split_name, split_metrics in overall_eval_metrics.items():
        for metric_name, value in split_metrics.items():
            if isinstance(value, (int, float)):
                log_metrics[f"{split_name}_{metric_name}"] = float(value)
    for key, value in metrics.get("train_losses", {}).items():
        log_metrics[key] = float(value)

    if hasattr(mlflow_logger, "log_metrics"):
        mlflow_logger.log_metrics(log_metrics)

    _log_mlflow_artifacts(
        mlflow_logger,
        [
            metrics_path,
            config_path,
            run_contract_path,
            model_state_dict_path,
            eval_artifacts.predictions_path,
            eval_artifacts.metrics_path,
            eval_artifacts.per_mut_dist_metrics_path,
        ],
    )

    max_epochs = int(training_cfg.get("epochs", 100))
    epochs_run = int(getattr(trainer, "current_epoch", 0))
    stopped_early = bool(getattr(early_stopping, "stopped_epoch", 0)) or epochs_run < max_epochs
    _write_run_record(
        run_dir=run_dir, config=config, metrics=metrics, data_module=data_module,
        stage=stage, epochs_run=epochs_run, max_epochs=max_epochs, stopped_early=stopped_early,
    )

    artifacts = RunArtifacts(
        run_dir=run_dir,
        best_checkpoint_path=None if best_ckpt_path is None else Path(best_ckpt_path),
        model_state_dict_path=model_state_dict_path,
        metrics_path=metrics_path,
        config_path=config_path,
        y_scaler_path=y_scaler_path,
        mlflow_run_id_path=mlflow_run_id_path,
        predictions_path=eval_artifacts.predictions_path,
        evaluation_metrics_path=eval_artifacts.metrics_path,
        per_mut_dist_metrics_path=eval_artifacts.per_mut_dist_metrics_path,
    )
    return metrics, artifacts


def train_one_run(
    base_config_path,
    override_config_paths=None,
    run_name=None,
    use_mlflow=True,
):
    config = load_config(base_config_path, override_config_paths[0] if override_config_paths else None)
    config.setdefault("output", {})
    if run_name is not None:
        config["output"]["run_name"] = run_name
    config.setdefault("mlflow", {})
    config["mlflow"]["enabled"] = use_mlflow
    metrics, artifacts = train_single_run(config)
    return {
        "metrics": metrics,
        "run_dir": artifacts.run_dir,
        "best_checkpoint_path": artifacts.best_checkpoint_path,
        "state_dict_path": artifacts.model_state_dict_path,
        "config_path": artifacts.config_path,
        "metrics_path": artifacts.metrics_path,
        "y_scaler_path": artifacts.y_scaler_path,
        "mlflow_run_id_path": artifacts.mlflow_run_id_path,
        "predictions_path": artifacts.predictions_path,
        "evaluation_metrics_path": artifacts.evaluation_metrics_path,
        "per_mut_dist_metrics_path": artifacts.per_mut_dist_metrics_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one deep learning model run with PyTorch Lightning.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--override", default=None, help="Optional YAML override path.")
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    metrics, artifacts = train_single_run(config)
    print(json.dumps({"metrics": metrics, "run_dir": str(artifacts.run_dir)}, indent=2))


if __name__ == "__main__":
    main()
