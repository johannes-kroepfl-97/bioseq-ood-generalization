"""Collect per-run records into flat, inspectable tables.

Walks an artifacts tree for ``run_record.json`` files (written by the trainer) and
produces three tidy CSVs plus a raw JSONL dump:

    all_runs.csv          one row per run; per-split metrics flattened to columns
                          (mae_T_far, spearman_T_far, naive_mae_val_id, ...). Config
                          blocks are kept as JSON-string columns.
    all_per_epoch.csv     one row per (run, epoch): the per-epoch training curves,
                          read from <run>/csv_logs|lightning_logs/*/metrics.csv.
    all_per_mut_dist.csv  one row per (run, split, mut_dist): the per-distance metrics.
    all_runs.jsonl        the untouched run records, one JSON object per line.

run_record.json is the single source of truth; everything here is derived, so you can
re-run this any time without re-training. Aggregate over seeds with aggregate.py.

Usage:
    python collect.py                              # root=results/training, out=collected
    python collect.py --root results/training --out collected
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Config/provenance blocks kept as JSON strings rather than exploded into columns.
_JSON_COLS = (
    "training_cfg", "model_cfg", "method_meta", "method_hparams",
    "selection", "training_dynamics", "paths",
)
# Scalar top-level fields copied straight through to all_runs.csv.
_SCALAR_COLS = (
    "run_id", "run_dir", "stage", "git_sha", "timestamp",
    "dataset", "model", "method", "protocol", "adapt_pool", "test_pool",
    "seed", "trial",
    "normalization_strategy", "input_encoding", "seq_len", "vocab_size",
    "n_train", "n_val_id", "n_adapt",
    "best_epoch", "epochs_run", "max_epochs", "stopped_early",
    "selected_split", "selected_metric", "report_split", "report_metric",
    "status", "error",
)


def _flatten_record(rec: dict) -> dict:
    """One run record -> one flat row (per-split metrics become <metric>_<split>)."""
    row: dict = {col: rec.get(col) for col in _SCALAR_COLS}
    for split, split_metrics in (rec.get("metrics") or {}).items():
        if isinstance(split_metrics, dict):
            for metric_name, value in split_metrics.items():
                if metric_name == "split":
                    continue
                row[f"{metric_name}_{split}"] = value
    for col in _JSON_COLS:
        if rec.get(col) is not None:
            row[col] = json.dumps(rec[col], default=str)
    return row


def _read_per_epoch(run_dir: Path) -> pd.DataFrame:
    """Per-epoch curves from csv_logs (preferred) or lightning_logs metrics.csv."""
    for sub in ("csv_logs", "lightning_logs"):
        hits = sorted((run_dir / sub).rglob("metrics.csv")) if (run_dir / sub).exists() else []
        if hits:
            df = pd.read_csv(hits[-1])   # last version_* is the final-stage run
            if "epoch" in df.columns:    # collapse step-rows into one row per epoch
                df = df.groupby("epoch", as_index=False).mean(numeric_only=True)
            return df
    return pd.DataFrame()


def collect(root: Path, out: Path) -> None:
    records = sorted(root.rglob("run_record.json"))
    print(f"found {len(records)} run_record.json under {root}")
    if not records:
        return

    rows, per_epoch, per_mut, raw = [], [], [], []
    for rpath in records:
        try:
            rec = json.loads(rpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  skip {rpath}: {exc}")
            continue
        raw.append(rec)
        rows.append(_flatten_record(rec))

        run_dir = rpath.parent           # resolve artifacts relative to the record itself,
        run_id = rec.get("run_id")       # so it works after the tree is downloaded/moved
        keys = {k: rec.get(k) for k in
                ("run_id", "dataset", "model", "method", "protocol", "seed", "stage")}

        ep = _read_per_epoch(run_dir)
        if not ep.empty:
            for k, v in keys.items():
                ep[k] = v
            per_epoch.append(ep)

        mut_path = run_dir / "evaluation" / "evaluation_metrics_by_mut_dist.csv"
        if mut_path.exists() and mut_path.stat().st_size > 0:
            md = pd.read_csv(mut_path)
            if not md.empty:
                md["run_id"] = run_id
                for k, v in keys.items():
                    if k != "run_id":
                        md[k] = v
                per_mut.append(md)

    out.mkdir(parents=True, exist_ok=True)
    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(out / "all_runs.csv", index=False)
    (out / "all_runs.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in raw), encoding="utf-8"
    )
    if per_epoch:
        pd.concat(per_epoch, ignore_index=True).to_csv(out / "all_per_epoch.csv", index=False)
    if per_mut:
        pd.concat(per_mut, ignore_index=True).to_csv(out / "all_per_mut_dist.csv", index=False)

    print(f"  -> {out/'all_runs.csv'}  ({all_runs.shape[0]} runs, {all_runs.shape[1]} cols)")
    print(f"     stages: {all_runs['stage'].value_counts().to_dict()}")
    print(f"     per-epoch runs: {len(per_epoch)} | per-mut-dist runs: {len(per_mut)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/training", help="Artifacts dir to walk for run_record.json.")
    ap.add_argument("--out", default="collected", help="Output dir for the combined tables.")
    args = ap.parse_args()
    collect(Path(args.root), Path(args.out))


if __name__ == "__main__":
    main()
