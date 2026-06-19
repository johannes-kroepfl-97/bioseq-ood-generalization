"""Single source of truth for method hyperparameters.

Two concerns live here, deliberately kept separate from model/training config:

* ``apply_method_hparams`` fills a run config with the *static defaults* declared
  in ``config/methods.yaml`` (missing keys only; committed or searched values
  always win).
* ``sample_method_hparams`` draws from each method's *search distribution*. These
  are sampling logic (log-uniform draws, conditional ranges) and cannot be
  expressed as plain YAML without inventing a DSL, so they stay in code.

``config/methods.yaml`` therefore holds only flat data; this module holds the
logic. ``pipeline_phases.py`` imports both from here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# config/ sits at the repository root: methods.py -> methods -> bioseq_ood -> src -> root.
_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


@lru_cache(maxsize=1)
def _method_defaults() -> dict[str, dict[str, Any]]:
    with open(_CONFIG_DIR / "methods.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def adabn_norm_strategy(model_name: str) -> str:
    """The normalization_strategy AdaBN requires for a given model."""
    return "native_norm_plus_final_bn" if model_name == "lstm" else "architecture_native_norm"


def _recursive_setdefault(target: dict, defaults: dict) -> None:
    """Fill keys from ``defaults`` into ``target`` only where missing (recursive)."""
    for k, v in defaults.items():
        if isinstance(v, dict):
            _recursive_setdefault(target.setdefault(k, {}), v)
        else:
            target.setdefault(k, v)


def apply_method_hparams(cfg: dict, method_name: str) -> None:
    """Fill in static defaults for ``method_name`` from ``config/methods.yaml``.

    Idempotent and non-destructive: existing keys are kept, so values supplied by
    the search or by a committed tuned/ file are never overwritten.
    """
    training = cfg.setdefault("training", {})
    _recursive_setdefault(training, _method_defaults().get(method_name, {}))
    if method_name == "adabn":
        # AdaBN adapts BatchNorm running statistics, so the encoder must contain BN.
        # If the tuned baseline already carries BN (it picked architecture_native_norm
        # in the Phase B runoff -> BN for cnn/mlp/transformer/hybrids), AdaBN runs on
        # that SAME baseline so the lift is not confounded by adding BN. Only override
        # when the baseline has no BN to adapt: a no-normalisation baseline, or a
        # pure LSTM (whose native norm is LayerNorm, which has no running stats).
        model = cfg.setdefault("model", {})
        model_name = cfg.get("model_name", "cnn")
        if model_name == "lstm" or model.get("normalization_strategy") is None:
            model["normalization_strategy"] = adabn_norm_strategy(model_name)


def sample_method_hparams(method_name: str, rng: np.random.Generator) -> dict:
    """Random draw from the per-method hyperparameter space (Phase E search)."""
    if method_name in ("erm", "adabn"):
        return {}
    if method_name == "cmd":
        return {"training": {
            "lambda_cmd":    float(10 ** rng.uniform(-2, 0)),
            "cmd_n_moments": int(rng.choice([3, 5, 7])),
        }}
    if method_name == "pseudo_labeling":
        return {"training": {"pseudo_labeling": {
            "keep_ratio":           float(rng.uniform(0.3, 0.9)),  # <1.0: MC-dropout selection always keeps a strict fraction
            "lambda_pseudo_max":    float(10 ** rng.uniform(-1, 0)),
            "rampup_epochs":        int(rng.choice([5, 10, 20])),
            "mc_passes":            int(rng.choice([10, 20])),
            "retrain_from_scratch": bool(rng.choice([True, False])),
        }}}
    if method_name == "mean_teacher":
        return {"training": {"mean_teacher": {
            "lambda_consistency_max": float(10 ** rng.uniform(-1, 0)),
            "ema_decay":              float(rng.choice([0.99, 0.999])),
            "rampup_epochs":          int(rng.choice([5, 10, 20])),
            "consistency_on_source":  bool(rng.choice([True, False])),
        }}}
    if method_name == "fixmatch":
        return {"training": {"fixmatch": {
            "lambda_fixmatch_max":  float(10 ** rng.uniform(-1, 0)),
            "strong_noise_sigma":   float(rng.uniform(0.05, 0.2)),
            "keep_ratio":           float(rng.uniform(0.3, 0.9)),  # <1.0: confidence gate is never a no-op
            "mc_passes":            int(rng.choice([5, 10])),
            "rampup_epochs":        int(rng.choice([5, 10, 20])),
        }}}
    raise ValueError(f"Unknown method: {method_name}")
