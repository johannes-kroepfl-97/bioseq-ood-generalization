"""Protocol-based orchestrator for the OOD-generalization study.

Open this file in VS Code with the Python extension and step through the `# %%`
cells one at a time. Each cell corresponds to one phase (A-G) of the
methodology described in `methodology_paper/sections/methodology.tex`.

The single execution unit is :func:`run_protocol`. Every higher-level loop calls
it; you can also call it directly for an ad-hoc run -- see the example at the
bottom of this file.

This is the sole orchestrator for the study. Configuration lives in `config/`
(see `config/README.md`): `training.yaml` for shared training defaults,
`models.yaml` for architectures, `methods.yaml` for method defaults, and
`config/tuned/` for the search winners written by Phase B / Phase E.
"""
# %% Imports + bootstrap
from __future__ import annotations

import gc
import json
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
for p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bioseq_ood.config.loader import load_config
from bioseq_ood.data.load_data import _get_data_root, load_all_data
from bioseq_ood.data.preprocess_data import load_y_scaler, preprocess_all_data, preprocess_split
from bioseq_ood.methods.hparams import apply_method_hparams, sample_method_hparams
from bioseq_ood.training.run_single import run_single_experiment
from bioseq_ood.training.search import run_random_search

CONFIG_DIR = PROJECT_ROOT / "config"
TUNED_DIR = CONFIG_DIR / "tuned"      # committed search winners (written by Phase B/E)
RESULTS_DIR = PROJECT_ROOT / "results_phases"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("Project root:", PROJECT_ROOT)
print("Results dir: ", RESULTS_DIR)


# %% Knobs - the MODE switch flips every default atomically
MODE   = "gfp_sanity" # "tutorial" | "smoke" | "gfp_sanity" | "real"
RESUME = True         # if True, phases load their CSV instead of re-running when it already exists
# STRICT_MODE defaults to True: a crashed run or a non-finite metric raises immediately
# instead of being swallowed into a silent None row. Set it False per mode only if you
# deliberately want a best-effort sweep (failures are then printed loudly and recorded
# with an `error`, never dropped silently).

_DEFAULTS = {
    "tutorial": dict(
        DATASETS=("gfp",),
        MODELS=("cnn", "mlp", "lstm"),
        N_SEARCH_TRIALS=3, N_METHOD_TRIALS=3,
        SEEDS=(42,), EPOCHS=30,
        DEBUG=True, DEBUG_MAX_SAMPLES=2000,
        TOP_K=3, STRICT_MODE=True,
    ),
    "smoke": dict(
        DATASETS=("gfp",),
        MODELS=("cnn", "mlp", "lstm"),
        N_SEARCH_TRIALS=5, N_METHOD_TRIALS=5,
        SEEDS=(42,), EPOCHS=30,
        DEBUG=False, DEBUG_MAX_SAMPLES=2000,
        TOP_K=3, STRICT_MODE=True,
    ),
    "gfp_sanity": dict(
        # Full-data pre-flight on gfp before the big poster run. Asymmetric search
        # budget: 5 encoder trials (the encoder is a shared backbone, so it needn't be
        # exhaustively tuned), 10 method trials (where the real signal is). 2 seeds so
        # the error bars populate; top-2 models go through the method phases.
        DATASETS=("gfp",),
        MODELS=("cnn", "mlp", "lstm"),
        N_SEARCH_TRIALS=5, N_METHOD_TRIALS=10,
        SEEDS=(42, 0), EPOCHS=150,
        DEBUG=False, DEBUG_MAX_SAMPLES=2000,
        TOP_K=2, STRICT_MODE=False,   # paid run: complete + record errors, don't abort
    ),
    "real": dict(
        DATASETS=("gfp", "aav", "tfbind8"),
        MODELS=("cnn", "mlp", "lstm", "cnn_lstm", "lstm_cnn", "transformer"),
        N_SEARCH_TRIALS=20, N_METHOD_TRIALS=20,
        SEEDS=(42, 0, 1), EPOCHS=200,
        DEBUG=False, DEBUG_MAX_SAMPLES=2000,
        TOP_K=3, STRICT_MODE=True,
    ),
}
globals().update(_DEFAULTS[MODE])

# Module-level constants that don't vary by mode.
METHODS            = ("erm", "cmd", "adabn", "pseudo_labeling", "mean_teacher", "fixmatch")
ADAPTATION_METHODS = tuple(m for m in METHODS if m != "erm")
SEARCH_SEED         = 42    # seed used inside method/encoder random search (fixed for reproducibility)
CLOSE_SUBSPLIT_SEED = 42    # seed used to split target_close into T_close + U_close

# The five protocols: (adapt_pool, test_pool). All share L_train as the supervised pool
# and L_val_id as the selection signal. See methodology_paper/sections/methodology.tex.
PROTOCOLS: dict[str, tuple[str, str]] = {
    "E1": ("T_close", "T_close"),    # transductive UDA, close
    "E2": ("U_close", "T_close"),    # inductive   UDA, close
    "E3": ("U_close", "T_far"),      # extrapolation
    "E4": ("U_far",   "T_far"),      # inductive   UDA, far
    "E5": ("T_far",   "T_far"),      # transductive UDA, far
}

# Pool name -> CSV file(s) the trainer reads as the unlabelled adaptation pool.
# All four CSVs are materialized by Phase A in paper notation:
#   T_close, U_close = disjoint halves of B_close = (val_ood ∪ target_close)
#   T_far,   U_far   = renamed copies of test, target_test
ADAPT_POOL_TO_SPLIT_FILES = {
    "T_close": ["T_close.csv"],
    "U_close": ["U_close.csv"],
    "U_far":   ["U_far.csv"],
    "T_far":   ["T_far.csv"],
}

# Pool name -> the split name the trainer reports MAE on (must exist on DatasetBundle).
TEST_POOL_TO_EVAL_SPLIT = {
    "T_close": "T_close",
    "T_far":   "T_far",
}

print(
    f"MODE={MODE}  RESUME={RESUME}\n"
    f"  DATASETS={DATASETS}  MODELS={MODELS}\n"
    f"  N_SEARCH_TRIALS={N_SEARCH_TRIALS}  N_METHOD_TRIALS={N_METHOD_TRIALS}  SEEDS={SEEDS}  EPOCHS={EPOCHS}\n"
    f"  DEBUG={DEBUG}  TOP_K={TOP_K}  STRICT_MODE={STRICT_MODE}"
)


# %% Shared helpers
@dataclass
class ProtocolResult:
    method: str
    model: str
    dataset: str
    seed: int
    adapt_pool: str
    test_pool: str
    val_id_mae: float | None
    report_mae: float | None
    run_dir: str            # informational only; not consumed by the analysis layer
    error: str | None = None
    per_mut: list[dict] | None = None   # per-mutation-distance metrics for the report split


def _read_per_mut_for_split(artifacts, split_name: str) -> list[dict]:
    """Per-mutation-distance metrics for one split, read from the run's
    evaluation_metrics_by_mut_dist.csv (mae + std_abs_error + n_samples per mut_dist).

    Returns [] when the artifact is missing; the run is then dropped from the per-mut
    output rather than failing the whole phase.
    """
    path = getattr(artifacts, "per_mut_dist_metrics_path", None)
    if not path or not Path(path).exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "split" not in df.columns:
        return []
    sub = df[df["split"] == split_name]
    cols = [c for c in ("mut_dist", "mae", "std_abs_error", "n_samples") if c in sub.columns]
    return sub[cols].to_dict("records")


@lru_cache(maxsize=1)
def _training_yaml() -> dict:
    """config/training.yaml: shared run config + the shared encoder search grid."""
    with open(CONFIG_DIR / "training.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _models_yaml() -> dict:
    """config/models.yaml: per-model architecture defaults + search grid."""
    with open(CONFIG_DIR / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def base_cfg(model_name: str, dataset_name: str) -> dict:
    """Assemble a run config from the shared training.yaml plus the model's block.

    Study-wide constants (selection mode, the legacy `setting` field, the mlflow
    toggle) are applied separately by `_apply_study_constants`, so there is one
    place that owns them. Method defaults are filled by `apply_method_hparams`.
    """
    models = _models_yaml()
    if model_name not in models:
        raise KeyError(f"Unknown model {model_name!r}; not in config/models.yaml.")
    cfg = deepcopy(_training_yaml())
    cfg.pop("search_space", None)          # the search grid is not part of a run config
    cfg["model_name"] = model_name
    cfg["model"] = deepcopy(models[model_name]["defaults"])
    cfg.setdefault("dataset", {})["name"] = dataset_name
    return cfg


def search_space_for(model_name: str) -> dict:
    """Shared (training) + architecture (model) search grid for the encoder search."""
    return {
        "training": deepcopy(_training_yaml().get("search_space", {})),
        "model":    deepcopy(_models_yaml()[model_name].get("search_space", {})),
    }


def _apply_study_constants(cfg: dict) -> None:
    """Fixed methodological choices shared by every run -- not user knobs.

    `selection.mode = val_id` is the honest-selection rule (DomainBed): only
    in-distribution validation labels drive checkpoint/HP selection. `setting` is
    a legacy field the trainer's SplitPlan still validates; the protocol is
    actually determined by adapt_pool/test_pool (target_split_files), so we pin it
    to a valid constant. The mlflow toggle comes from training.yaml.
    """
    cfg.setdefault("selection", {})
    cfg["selection"]["mode"]   = "val_id"
    cfg["selection"]["metric"] = "mae"
    cfg["setting"] = "extrapolative"
    cfg.setdefault("mlflow", {})["enabled"] = bool(
        _training_yaml().get("mlflow", {}).get("enabled", False)
    )


def deep_update(target: dict, src: dict) -> dict:
    """Recursive merge of src into target (mutates target)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            deep_update(target[k], v)
        else:
            target[k] = v
    return target


def banner(label: str, *lines) -> None:
    print()
    print("=" * 72)
    print(f"  [{label}] " + " | ".join(str(l) for l in lines))
    print("=" * 72)


def save_results(rows, filename: str) -> Path:
    out = RESULTS_DIR / filename
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  -> {out.relative_to(PROJECT_ROOT)}")
    return out


def append_result(filename: str, row: dict) -> None:
    """Append ONE row to a results CSV, writing the header only when the file is new.

    Each call flushes to disk, so an interrupted run keeps every row already produced
    (and you can ``tail``/``watch`` the file grow live). Phase F uses this instead of a
    single end-of-phase write so a crash loses at most one run, not the whole phase.
    """
    out = RESULTS_DIR / filename
    pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


# --- The tuned/ store: committed, human-readable search winners ----------------
# results_phases/*.csv stay the raw search log; config/tuned/*.yaml are the small,
# version-controlled decisions that Phase C/E/F consume. This replaces the old
# resume mechanism (reloading a run's config.yaml via best_dir, and parsing the
# hparams JSON column out of the Phase E CSV).
TUNED_MODELS_PATH  = TUNED_DIR / "models.yaml"     # dataset -> model -> encoder cfg
TUNED_METHODS_PATH = TUNED_DIR / "methods.yaml"    # dataset -> model -> method -> hparams

_TUNED_MODELS_HEADER = (
    "# AUTO-GENERATED by pipeline_phases.py (phase_B). Committed for reproducibility; do not hand-edit.\n"
    "# dataset -> model -> tuned encoder architecture (the model.* block of the search winner).\n"
)
_TUNED_METHODS_HEADER = (
    "# AUTO-GENERATED by pipeline_phases.py (phase_E). Committed for reproducibility; do not hand-edit.\n"
    "# dataset -> model -> method -> tuned hyperparameters (deep-merged under the run config).\n"
)


def _write_tuned(path: Path, header: str, data: dict) -> None:
    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(data, f, sort_keys=True, default_flow_style=False)
    print(f"  -> {path.relative_to(PROJECT_ROOT)}")


def _load_tuned(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _encoder_cfgs_to_nested(encoder_cfgs: dict) -> dict:
    """{(model, dataset): cfg} -> {dataset: {model: cfg}} for on-disk storage."""
    out: dict[str, dict] = {}
    for (model, dataset), mcfg in encoder_cfgs.items():
        out.setdefault(dataset, {})[model] = mcfg
    return out


def _nested_to_encoder_cfgs(nested: dict) -> dict:
    return {(model, dataset): mcfg
            for dataset, models in nested.items()
            for model, mcfg in models.items()}


def _best_hparams_to_nested(best: dict) -> dict:
    """{(method, model, dataset): hp} -> {dataset: {model: {method: hp}}}."""
    out: dict[str, dict] = {}
    for (method, model, dataset), hp in best.items():
        out.setdefault(dataset, {}).setdefault(model, {})[method] = hp
    return out


def _nested_to_best_hparams(nested: dict) -> dict:
    return {(method, model, dataset): hp
            for dataset, models in nested.items()
            for model, methods in models.items()
            for method, hp in methods.items()}


def _assert_close_subsplits(dataset: str) -> None:
    """Phase E/F sanity check: T_close.npz and U_close.npz exist for the dataset."""
    npz_dir = _get_data_root() / dataset / "preprocessed"
    for split in ("T_close", "U_close"):
        if not (npz_dir / f"{split}.npz").exists():
            raise RuntimeError(
                f"{split}.npz missing for {dataset!r}. "
                f"Re-run Phase A (phase_A()) to materialize the close-band subsplits."
            )


print("Helpers defined.")


# %% THE execution unit: run_protocol
def run_protocol(
    *,
    method: str,
    model: str,
    dataset: str,
    encoder_cfg: dict | None,
    hparams: dict,
    seed: int,
    adapt_pool: str,
    test_pool: str,
    epochs: int | None = None,
    run_name: str | None = None,
    pretrained_checkpoint: str | None = None,
) -> ProtocolResult:
    """One training + evaluation run with the given protocol parameters.

    Used by Phase B (ERM via the existing random search), Phase C (ERM reference),
    Phase E (method hparam search, one call per trial), and Phase F (one call per
    seed x protocol with the committed best hparams).

    Selection rule: always L_val_id MAE for early stopping and checkpoint selection.
    """
    epochs = epochs if epochs is not None else EPOCHS

    if adapt_pool not in ADAPT_POOL_TO_SPLIT_FILES:
        raise ValueError(f"Unknown adapt_pool: {adapt_pool}. "
                         f"Expected one of {list(ADAPT_POOL_TO_SPLIT_FILES)}.")
    if test_pool not in TEST_POOL_TO_EVAL_SPLIT:
        raise ValueError(f"Unknown test_pool: {test_pool}. "
                         f"Expected one of {list(TEST_POOL_TO_EVAL_SPLIT)}.")

    cfg = base_cfg(model, dataset)
    cfg.setdefault("training", {})
    cfg.setdefault("evaluation", {})
    _apply_study_constants(cfg)

    cfg["seed"] = seed
    cfg["training"]["method"]   = method
    cfg["training"]["epochs"]   = epochs
    cfg["training"]["early_stopping_patience"] = max(3, epochs // 5)
    cfg["training"]["num_workers"] = 0
    cfg["training"]["enable_progress_bar"] = False

    if DEBUG:
        cfg.setdefault("debug", {})
        cfg["debug"]["enabled"] = True
        cfg["debug"]["max_samples_per_split"] = DEBUG_MAX_SAMPLES
        cfg["debug"]["seed"] = seed

    if encoder_cfg is not None:
        cfg["model"] = deepcopy(encoder_cfg)

    cfg["training"]["target_split_files"] = list(ADAPT_POOL_TO_SPLIT_FILES[adapt_pool])

    if hparams:
        deep_update(cfg, deepcopy(hparams))
    apply_method_hparams(cfg, method)

    # Pseudo-labeling reuses the tuned ERM baseline for this (model, dataset) to
    # generate the MC-dropout pseudo-labels, instead of retraining Stage 1 from
    # scratch. The trainer's `pretrained_checkpoint` hook loads it and skips Stage-1
    # training; Stage-2 retraining on the kept pseudo-labels is unchanged.
    if pretrained_checkpoint and method == "pseudo_labeling":
        cfg["training"].setdefault("pseudo_labeling", {})["pretrained_checkpoint"] = str(pretrained_checkpoint)

    # The three splits we report on, named exactly as TEST_POOL_TO_EVAL_SPLIT
    # looks them up: val_id (selection diagnostic), T_close (E1/E2), T_far (E3/E4/E5).
    # NB: must list "T_far" -- not the legacy "test" alias -- or far-band protocols
    # silently report NaN (the lookup key would not exist in eval_metrics).
    cfg["evaluation"]["splits"] = ["val_id", "T_close", "T_far"]

    if run_name is not None:
        cfg.setdefault("output", {})["run_name"] = run_name

    try:
        metrics, artifacts = run_single_experiment(config=cfg)
    except Exception as exc:
        if STRICT_MODE:
            raise
        traceback.print_exc()
        return ProtocolResult(
            method=method, model=model, dataset=dataset, seed=seed,
            adapt_pool=adapt_pool, test_pool=test_pool,
            val_id_mae=None, report_mae=None, run_dir="", error=repr(exc),
        )

    eval_metrics    = metrics.get("evaluation", {}) or {}
    test_split_name = TEST_POOL_TO_EVAL_SPLIT[test_pool]
    report_mae      = (eval_metrics.get(test_split_name) or {}).get("mae")
    val_id_mae      = (eval_metrics.get("val_id")        or {}).get("mae")

    # Fail loudly on a missing / non-finite metric. The run did not raise, but it
    # produced no usable number -- without this guard the result silently becomes a
    # None row that vanishes into the seed-mean (exactly how the T_far eval-split
    # naming bug stayed hidden). Every protocol must yield finite val_id and report MAE.
    def _finite(x: object) -> bool:
        return x is not None and bool(np.isfinite(float(x)))

    if not (_finite(val_id_mae) and _finite(report_mae)):
        msg = (
            f"Non-finite metric with no exception for {method}/{model}/{dataset} "
            f"adapt={adapt_pool} test={test_pool}: val_id_mae={val_id_mae}, "
            f"report_mae={report_mae}. Looked up eval split {test_split_name!r}; "
            f"available eval splits: {sorted(eval_metrics)}."
        )
        if STRICT_MODE:
            raise RuntimeError(msg)
        print("!!! NON-FINITE METRIC: " + msg)   # loud even when not strict
        return ProtocolResult(
            method=method, model=model, dataset=dataset, seed=seed,
            adapt_pool=adapt_pool, test_pool=test_pool,
            val_id_mae=float(val_id_mae) if _finite(val_id_mae) else None,
            report_mae=float(report_mae) if _finite(report_mae) else None,
            run_dir=str(getattr(artifacts, "run_dir", "")), error=msg,
        )

    return ProtocolResult(
        method=method, model=model, dataset=dataset, seed=seed,
        adapt_pool=adapt_pool, test_pool=test_pool,
        val_id_mae=float(val_id_mae),
        report_mae=float(report_mae),
        run_dir=str(getattr(artifacts, "run_dir", "")),
        error=None,
        per_mut=_read_per_mut_for_split(artifacts, test_split_name),
    )


print("run_protocol defined.")


# %% Phase A . Data prep -- bring splits into paper notation and clean up legacy files
import shutil

# Legacy CSV / NPZ files that the protocol-based design does not use.
# They are deleted (not renamed) by phase_A; train, val_id, val_ood, target_close,
# target_test and test remain in place as source data for the materialization
# step below.
_LEGACY_TO_DELETE = [
    "target_unlabeled.csv",  "target_unlabeled.npz",
    "target_close_full.csv", "target_close_full.npz",
    "target_test_full.csv",  "target_test_full.npz",
]

# (src_split_name, dst_split_name): paper-notation copies of the far-band pools.
_FAR_RENAMES = [("test", "T_far"), ("target_test", "U_far")]


def phase_A():
    """Load + preprocess raw data, materialize the paper-notation pools, clean up legacy files.

    After this runs, each dataset's splits/ and preprocessed/ folders contain:
      train, val_id            (L_train, L_val_id; unchanged)
      val_ood, target_close    (kept as source data; merged below into the close-band pool)
      target_test, test        (kept as source data; copied below into U_far / T_far)
      T_close, U_close         (disjoint halves of B_close = val_ood + target_close)
      U_far,   T_far           (renamed copies of target_test, test)
      y_scaler.npz             (universal label scaler; unchanged)
    """
    print("Phase A - data prep")
    load_all_data()
    preprocess_all_data(datasets=DATASETS, overwrite=False)

    data_root = _get_data_root()
    for ds in DATASETS:
        splits_dir       = data_root / ds / "splits"
        preprocessed_dir = data_root / ds / "preprocessed"
        try:
            y_scaler = load_y_scaler(ds)
        except FileNotFoundError:
            y_scaler = None

        # --- 1. Materialize T_close + U_close from B_close = val_ood ∪ target_close.
        T_csv, U_csv = splits_dir / "T_close.csv", splits_dir / "U_close.csv"
        T_npz, U_npz = preprocessed_dir / "T_close.npz", preprocessed_dir / "U_close.npz"
        if not (T_csv.exists() and U_csv.exists() and T_npz.exists() and U_npz.exists()):
            parts = []
            for src_name in ("val_ood.csv", "target_close.csv"):
                p = splits_dir / src_name
                if p.exists():
                    parts.append(pd.read_csv(p))
            if not parts:
                raise FileNotFoundError(
                    f"{splits_dir}: neither val_ood.csv nor target_close.csv present. "
                    f"Run load_all_data() first."
                )
            df = pd.concat(parts, ignore_index=True)
            rng = np.random.default_rng(CLOSE_SUBSPLIT_SEED)
            perm = rng.permutation(len(df))
            half = len(df) // 2
            df.iloc[sorted(perm[:half].tolist())].to_csv(T_csv, index=False)
            df.iloc[sorted(perm[half:].tolist())].to_csv(U_csv, index=False)
            preprocess_split(ds, "T_close", y_scaler=y_scaler, overwrite=True)
            preprocess_split(ds, "U_close", y_scaler=y_scaler, overwrite=True)

        # --- 2. Materialize T_far + U_far by copying the existing far-band CSVs.
        for src_name, dst_name in _FAR_RENAMES:
            dst_csv = splits_dir / f"{dst_name}.csv"
            dst_npz = preprocessed_dir / f"{dst_name}.npz"
            if dst_csv.exists() and dst_npz.exists():
                continue
            src_csv = splits_dir / f"{src_name}.csv"
            if not src_csv.exists():
                raise FileNotFoundError(
                    f"{src_csv} missing -- cannot materialize {dst_name} for {ds}."
                )
            shutil.copy2(src_csv, dst_csv)
            preprocess_split(ds, dst_name, y_scaler=y_scaler, overwrite=True)

        # --- 3. Delete legacy artifacts that the new pipeline does not use.
        for legacy in _LEGACY_TO_DELETE:
            for p in (splits_dir / legacy, preprocessed_dir / legacy):
                if p.exists():
                    p.unlink()

    banner("Phase A", f"Data ready for {DATASETS}")


phase_A()


# %% Phase B . Encoder search per (model, dataset)
def phase_B() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    out_csv = RESULTS_DIR / "phaseB_encoder_search.csv"
    if RESUME and TUNED_MODELS_PATH.exists():
        print(f"Phase B - resume: loading {TUNED_MODELS_PATH.relative_to(PROJECT_ROOT)}")
        encoder_cfgs = _nested_to_encoder_cfgs(_load_tuned(TUNED_MODELS_PATH))
        rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
        banner("Phase B", f"resumed {len(encoder_cfgs)} tuned encoders")
        return rows, encoder_cfgs

    rows: list[dict] = []
    all_trial_rows: list[dict] = []
    encoder_cfgs: dict[tuple[str, str], dict] = {}
    for dataset in DATASETS:
        for model in MODELS:
            torch.cuda.empty_cache(); gc.collect()
            print(f"\n=== Phase B | {dataset} | {model} | {N_SEARCH_TRIALS} ERM trials ===")
            try:
                cfg = base_cfg(model, dataset)
                cfg.setdefault("training", {})
                _apply_study_constants(cfg)
                cfg["training"]["method"]                  = "erm"
                cfg["training"]["epochs"]                  = EPOCHS
                cfg["training"]["early_stopping_patience"] = max(3, EPOCHS // 5)
                cfg["training"]["num_workers"]             = 0
                cfg["training"]["enable_progress_bar"]     = False
                if DEBUG:
                    cfg.setdefault("debug", {})
                    cfg["debug"]["enabled"]               = True
                    cfg["debug"]["max_samples_per_split"] = DEBUG_MAX_SAMPLES
                    cfg["debug"]["seed"]                  = SEARCH_SEED

                space   = search_space_for(model)
                summary = run_random_search(cfg, space, n_trials=N_SEARCH_TRIALS)
                # Keep every trial, not just the winner, so the sampled hyperparameter
                # values and their val_id/val_ood can be inspected for range mistakes.
                for t in summary.get("trials", []):
                    tcfg = t.get("config", {})
                    all_trial_rows.append({
                        "dataset": dataset, "model": model,
                        "trial": t.get("trial"),
                        "val_id_mae": t.get("val_id_mae"),
                        "val_ood_mae": t.get("val_ood_mae"),
                        "run_dir": t.get("run_dir"),
                        "training_cfg": json.dumps(tcfg.get("training", {}), default=str),
                        "model_cfg": json.dumps(tcfg.get("model", {}), default=str),
                    })
                best    = summary["best_trial"]
                best_dir = Path(summary["best_dir"])
                best_cfg = load_config(best_dir / "config.yaml")
                base_val_id = float(best["val_id_mae"])

                # --- Normalization runoff. The search above runs without normalization;
                # here we retrain the same best architecture with architecture_native_norm
                # and adopt it as the shared baseline only if it improves val_id. This
                # stops normalization (a known free win, and a prerequisite for AdaBN)
                # from being excluded from the baseline. +1 ERM trial per (model, dataset).
                norm_cfg = deepcopy(best_cfg)
                norm_cfg.setdefault("model", {})["normalization_strategy"] = "architecture_native_norm"
                norm_cfg.setdefault("output", {})["run_name"] = "best_norm_runoff"
                norm_metrics, norm_artifacts = run_single_experiment(config=norm_cfg)
                norm_val_id = (norm_metrics.get("evaluation", {}).get("val_id") or {}).get("mae")
                use_norm = norm_val_id is not None and float(norm_val_id) < base_val_id

                chosen_model_cfg = (norm_cfg if use_norm else best_cfg)["model"]
                encoder_cfgs[(model, dataset)] = chosen_model_cfg
                rows.append({
                    "dataset": dataset, "model": model,
                    "val_id_mae_no_norm":   base_val_id,
                    "val_id_mae_with_norm": float(norm_val_id) if norm_val_id is not None else None,
                    "chosen_normalization": chosen_model_cfg.get("normalization_strategy"),
                    "best_val_id_mae":      float(norm_val_id) if use_norm else base_val_id,
                    "best_dir":             str(norm_artifacts.run_dir) if use_norm else str(best_dir),
                    "best_dir_no_norm":     str(best_dir),
                    "best_dir_with_norm":   str(norm_artifacts.run_dir) if use_norm else None,
                    "status": "ok", "error": None,
                })
                print(f"  ok  val_id no_norm={base_val_id:.4f}  with_norm="
                      f"{f'{float(norm_val_id):.4f}' if norm_val_id is not None else 'NA'}"
                      f"  -> {'architecture_native_norm' if use_norm else 'None'}")
            except Exception as exc:
                if STRICT_MODE: raise
                rows.append({"dataset": dataset, "model": model,
                             "status": "failed", "error": repr(exc),
                             "traceback": traceback.format_exc()})
                traceback.print_exc()

    save_results(rows, "phaseB_encoder_search.csv")
    if all_trial_rows:
        save_results(all_trial_rows, "phaseB_all_trials.csv")
    _write_tuned(TUNED_MODELS_PATH, _TUNED_MODELS_HEADER, _encoder_cfgs_to_nested(encoder_cfgs))
    banner("Phase B", f"{sum(1 for r in rows if r.get('status')=='ok')}/{len(rows)} architectures searched ok")
    return rows, encoder_cfgs


phaseB_rows, ENCODER_CFGS = phase_B()


def _baseline_ckpts_from_rows(rows: list[dict]) -> dict[tuple[str, str], str]:
    """(model, dataset) -> tuned ERM baseline checkpoint (best_dir/model_state_dict.pt).

    Pseudo-labeling loads this baseline to generate its MC-dropout pseudo-labels
    instead of retraining Stage 1 from scratch. best_dir is recorded by phase_B and
    survives resume (it is read back from phaseB_encoder_search.csv). If a checkpoint
    is missing the entry is omitted and pseudo-labeling falls back to training Stage 1
    (handled by the trainer hook).
    """
    out: dict[tuple[str, str], str] = {}
    for r in rows:
        best_dir = r.get("best_dir")
        if isinstance(best_dir, str) and best_dir:
            ckpt = Path(best_dir) / "model_state_dict.pt"
            if ckpt.exists():
                out[(r["model"], r["dataset"])] = str(ckpt)
    return out


BASELINE_CKPTS = _baseline_ckpts_from_rows(phaseB_rows)
print(f"Baseline checkpoints for pseudo-labeling: {len(BASELINE_CKPTS)}/{len(ENCODER_CFGS)} (model, dataset) pairs")


# %% Phase C . Baseline ERM reference
def phase_C() -> list[dict]:
    out_csv = RESULTS_DIR / "phaseC_baseline_reference.csv"
    if RESUME and out_csv.exists():
        print(f"Phase C - resume: loading {out_csv.name}")
        rows = pd.read_csv(out_csv).to_dict("records")
        banner("Phase C", f"resumed {len(rows)} rows")
        return rows

    rows: list[dict] = []
    for (model, dataset), enc_cfg in ENCODER_CFGS.items():
        for seed in SEEDS:
            torch.cuda.empty_cache(); gc.collect()
            print(f"\n=== Phase C | {dataset} | {model} | seed={seed} ===")
            r_close = run_protocol(
                method="erm", model=model, dataset=dataset,
                encoder_cfg=enc_cfg, hparams={}, seed=seed,
                adapt_pool="U_close", test_pool="T_close",
            )
            r_far = run_protocol(
                method="erm", model=model, dataset=dataset,
                encoder_cfg=enc_cfg, hparams={}, seed=seed,
                adapt_pool="U_far",   test_pool="T_far",
            )
            rows.append({
                "dataset": dataset, "model": model, "seed": seed,
                "ERM_close_mae": r_close.report_mae,
                "ERM_far_mae":   r_far.report_mae,
                "val_id_mae":    r_close.val_id_mae,
                "error_close":   r_close.error, "error_far": r_far.error,
            })
            print(f"  ERM_close = {r_close.report_mae}  ERM_far = {r_far.report_mae}")

    save_results(rows, "phaseC_baseline_reference.csv")
    banner("Phase C", f"{len(rows)} ERM baseline rows")
    return rows


phaseC_rows = phase_C()


# %% Phase D . Top-K cut per dataset
def phase_D() -> dict[str, list[str]]:
    df = pd.DataFrame(phaseC_rows)
    df["avg_mae"] = df[["ERM_close_mae", "ERM_far_mae"]].mean(axis=1)
    agg = df.groupby(["dataset", "model"])["avg_mae"].mean().reset_index()

    top_models: dict[str, list[str]] = {}
    for ds in DATASETS:
        ranked = agg[agg["dataset"] == ds].sort_values("avg_mae")["model"].tolist()
        top_models[ds] = ranked[:TOP_K]

    save_results(
        [{"dataset": ds, "rank": i + 1, "model": m}
         for ds, ms in top_models.items() for i, m in enumerate(ms)],
        "phaseD_top_models.csv",
    )
    banner("Phase D", f"Top-{TOP_K} per dataset:", top_models)
    return top_models


TOP_MODELS = phase_D()


# %% Phase E . Method hyperparameter search (on E2)
def phase_E() -> tuple[list[dict], dict[tuple[str, str, str], dict]]:
    for ds in DATASETS:
        _assert_close_subsplits(ds)

    out_csv = RESULTS_DIR / "phaseE_method_search.csv"
    if RESUME and TUNED_METHODS_PATH.exists():
        print(f"Phase E - resume: loading {TUNED_METHODS_PATH.relative_to(PROJECT_ROOT)}")
        best = _nested_to_best_hparams(_load_tuned(TUNED_METHODS_PATH))
        rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
        banner("Phase E", f"resumed {len(best)} (method, model, dataset) triples")
        return rows, best

    rows: list[dict] = []
    best: dict[tuple[str, str, str], dict] = {}
    for dataset in DATASETS:
        for model in TOP_MODELS[dataset]:
            enc_cfg = ENCODER_CFGS.get((model, dataset))
            if enc_cfg is None:
                continue
            for method in ADAPTATION_METHODS:
                trials = 1 if method == "adabn" else N_METHOD_TRIALS    # AdaBN is parameter-free
                print(f"\n=== Phase E | {dataset} | {model} | {method} | trials={trials} ===")
                rng = np.random.default_rng(abs(hash((method, model, dataset))) % (2**32))
                best_trial: dict | None = None
                for trial in range(trials):
                    torch.cuda.empty_cache(); gc.collect()
                    hparams = sample_method_hparams(method, rng)
                    res = run_protocol(
                        method=method, model=model, dataset=dataset,
                        encoder_cfg=enc_cfg, hparams=hparams, seed=SEARCH_SEED,
                        adapt_pool="U_close", test_pool="T_close",   # <- E2
                        pretrained_checkpoint=BASELINE_CKPTS.get((model, dataset)),
                    )
                    rows.append({
                        "dataset": dataset, "model": model, "method": method,
                        "trial": trial,
                        "val_id_mae":    res.val_id_mae,
                        "report_mae_E2": res.report_mae,
                        "hparams":       json.dumps(hparams),
                        "error":         res.error,
                    })
                    print(f"  trial {trial}: val_id_mae = {res.val_id_mae}")
                    if res.val_id_mae is not None and (best_trial is None or res.val_id_mae < best_trial["val_id_mae"]):
                        best_trial = {"val_id_mae": res.val_id_mae, "hparams": hparams}
                if best_trial is not None:
                    best[(method, model, dataset)] = best_trial["hparams"]
                    print(f"  best val_id_mae = {best_trial['val_id_mae']}")

    save_results(rows, "phaseE_method_search.csv")
    _write_tuned(TUNED_METHODS_PATH, _TUNED_METHODS_HEADER, _best_hparams_to_nested(best))
    banner("Phase E", f"{len(best)} (method, model, dataset) triples tuned")
    return rows, best


phaseE_rows, BEST_HPARAMS = phase_E()


# %% Phase F . Run all five protocols with the committed hparams
def phase_F() -> tuple[list[dict], list[dict]]:
    for ds in DATASETS:
        _assert_close_subsplits(ds)

    out_csv = RESULTS_DIR / "phaseF_all_protocols.csv"
    mut_csv = RESULTS_DIR / "phaseF_by_mut_dist.csv"

    # Incremental + row-level resume. Each finished run is appended (and flushed) to the
    # CSVs immediately, so a crash loses at most ONE run and you can watch progress live.
    # On resume we reload finished rows and skip those cells; on a fresh run we clear any
    # stale CSVs so the appends start from a clean file.
    if RESUME and out_csv.exists():
        rows = pd.read_csv(out_csv).to_dict("records")
        per_mut_rows = pd.read_csv(mut_csv).to_dict("records") if mut_csv.exists() else []
        print(f"Phase F - resume: {len(rows)} runs already on disk, continuing")
    else:
        rows, per_mut_rows = [], []
        out_csv.unlink(missing_ok=True)
        mut_csv.unlink(missing_ok=True)
    done = {(r["dataset"], r["model"], r["method"], r["protocol"], r["seed"]) for r in rows}

    for dataset in DATASETS:
        for model in TOP_MODELS[dataset]:
            enc_cfg = ENCODER_CFGS.get((model, dataset))
            if enc_cfg is None:
                continue
            for method in ADAPTATION_METHODS:
                hparams = BEST_HPARAMS.get((method, model, dataset), {})
                for protocol_name, (adapt, test) in PROTOCOLS.items():
                    for seed in SEEDS:
                        if (dataset, model, method, protocol_name, seed) in done:
                            continue   # already finished in an earlier (interrupted) attempt
                        torch.cuda.empty_cache(); gc.collect()
                        print(f"\n=== Phase F | {dataset} | {model} | {method} | {protocol_name} | seed={seed} ===")
                        res = run_protocol(
                            method=method, model=model, dataset=dataset,
                            encoder_cfg=enc_cfg, hparams=hparams, seed=seed,
                            adapt_pool=adapt, test_pool=test,
                            pretrained_checkpoint=BASELINE_CKPTS.get((model, dataset)),
                        )
                        row = {
                            "dataset": dataset, "model": model, "method": method,
                            "protocol": protocol_name, "adapt_pool": adapt, "test_pool": test,
                            "seed": seed,
                            "val_id_mae": res.val_id_mae,
                            "report_mae": res.report_mae,
                            "error": res.error,
                        }
                        rows.append(row)
                        append_result("phaseF_all_protocols.csv", row)   # incremental + flushed
                        for pm in (res.per_mut or []):
                            mrow = {
                                "dataset": dataset, "model": model, "method": method,
                                "protocol": protocol_name, "seed": seed,
                                "mut_dist": pm.get("mut_dist"),
                                "report_mae": pm.get("mae"),
                                "std_abs_error": pm.get("std_abs_error"),
                                "n_samples": pm.get("n_samples"),
                            }
                            per_mut_rows.append(mrow)
                            append_result("phaseF_by_mut_dist.csv", mrow)   # incremental + flushed
                        print(f"  {protocol_name}: report_mae = {res.report_mae}")

    banner("Phase F", f"{sum(1 for r in rows if not r.get('error'))}/{len(rows)} protocol runs ok")
    return rows, per_mut_rows


phaseF_rows, phaseF_by_mut = phase_F()


# %% Phase G . Analysis quantities + headline summary
def phase_G() -> pd.DataFrame:
    fdf = pd.DataFrame(phaseF_rows)
    fdf = fdf[fdf["error"].isna() if "error" in fdf.columns else slice(None)]
    method_mae = (
        fdf.groupby(["dataset", "model", "method", "protocol"])["report_mae"]
           .mean().unstack("protocol").reset_index()
    )
    erm = (
        pd.DataFrame(phaseC_rows)
          .groupby(["dataset", "model"])[["ERM_close_mae", "ERM_far_mae"]]
          .mean().reset_index()
    )
    merged = method_mae.merge(erm, on=["dataset", "model"], how="left")

    for col in ("E1", "E2", "E3", "E4", "E5"):
        if col not in merged.columns:
            merged[col] = float("nan")

    merged["lift_close"]  = merged["ERM_close_mae"] - merged["E2"]
    merged["lift_far"]    = merged["ERM_far_mae"]   - merged["E4"]
    merged["delta_close"] = merged["E2"] - merged["E1"]
    merged["delta_far"]   = merged["E4"] - merged["E5"]
    merged["G_extrap"]    = merged["E3"] - merged["E4"]
    merged["D_shift"]     = merged["E3"] - merged["E2"]

    save_results(merged.to_dict("records"), "phaseG_analysis.csv")

    # Per-mutation-distance curves aggregated across seeds (plot-ready, with error
    # bars). report_mae_std is the spread ACROSS seeds (NaN for a single seed);
    # mean_std_abs_error is the mean WITHIN-bin absolute-error spread per run.
    if phaseF_by_mut:
        mdf = pd.DataFrame(phaseF_by_mut)
        if "report_mae" in mdf.columns:
            mdf = mdf[mdf["report_mae"].notna()]
        if not mdf.empty:
            mut_agg = (
                mdf.groupby(["dataset", "model", "method", "protocol", "mut_dist"])
                   .agg(report_mae_mean=("report_mae", "mean"),
                        report_mae_std=("report_mae", "std"),
                        mean_std_abs_error=("std_abs_error", "mean"),
                        n_samples=("n_samples", "mean"),
                        n_seeds=("report_mae", "size"))
                   .reset_index()
            )
            save_results(mut_agg.to_dict("records"), "phaseG_by_mut_dist.csv")

    print("\n" + "=" * 72)
    print("  HEADLINE SUMMARY  (mean over models within each dataset)")
    print("=" * 72)
    headline_cols = ["lift_close", "lift_far", "G_extrap", "D_shift", "delta_close", "delta_far"]
    for ds in DATASETS:
        sub = merged[merged["dataset"] == ds]
        if sub.empty:
            continue
        print(f"\n  {ds}")
        print(sub.groupby("method")[headline_cols].mean().round(4).to_string())

    banner("Phase G", f"Analysis written for {len(merged)} (dataset, model, method) cells")
    return merged


phaseG_table = phase_G()


# %% Ad-hoc example (commented). Single line to reproduce an arbitrary protocol.
# result = run_protocol(
#     method="cmd", model="cnn", dataset="gfp",
#     encoder_cfg=ENCODER_CFGS.get(("cnn", "gfp")),
#     hparams=BEST_HPARAMS.get(("cmd", "cnn", "gfp"), {}),
#     seed=42,
#     adapt_pool="U_far", test_pool="T_far",
# )
# print(result)
