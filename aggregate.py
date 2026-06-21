"""Aggregate the collected per-run table over seeds.

Reads all_runs.csv (from collect.py), keeps the FINAL runs (stage == "final"), and
produces two summaries:

    aggregated.csv   mean / std / sem / n_seeds of every metric, per configuration
                     (dataset, model, method, protocol, adapt_pool, test_pool).
    method_lift.csv  each method's report-MAE lift over the ERM baseline, paired on
                     seed and matched on test_pool (lift > 0 means it beats ERM).

Aggregating *after* collection (rather than inline during the run) is deliberate: you
can re-select, drop a bad seed, or fix an aggregation bug without re-training. Search
trials (stage == "search") are excluded -- those are selected by val_id, not averaged.

Usage:
    python aggregate.py                                  # collected/all_runs.csv -> collected/
    python aggregate.py --runs collected/all_runs.csv --out collected
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_KEYS = ["dataset", "model", "method", "protocol", "adapt_pool", "test_pool"]
# A column is a metric if its leading token is one of these (e.g. mae_T_far, r2_val_id).
_METRIC_PREFIXES = {"mae", "rmse", "mse", "spearman", "pearson", "r2", "naive"}


def _metric_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.split("_")[0] in _METRIC_PREFIXES]
    for extra in ("report_metric", "selected_metric"):
        if extra in df.columns:
            cols.append(extra)
    # keep only numeric, de-duplicated, stable order
    seen, out = set(), []
    for c in cols:
        if c not in seen and pd.api.types.is_numeric_dtype(df[c]):
            seen.add(c)
            out.append(c)
    return out


def summarize_over_seeds(final: pd.DataFrame) -> pd.DataFrame:
    metric_cols = _metric_cols(final)
    grouped = final.groupby(_KEYS, dropna=False)
    parts = [grouped["seed"].nunique().rename("n_seeds")]
    for col in metric_cols:
        stats = grouped[col].agg(["mean", "std", "count"])
        sem = stats["std"] / np.sqrt(stats["count"].clip(lower=1))
        parts.append(stats["mean"].rename(f"{col}_mean"))
        parts.append(stats["std"].rename(f"{col}_std"))
        parts.append(sem.rename(f"{col}_sem"))
    return pd.concat(parts, axis=1).reset_index()


def method_lift(final: pd.DataFrame, baseline: str = "erm") -> pd.DataFrame:
    """Report-MAE lift of each method over ERM, paired on seed, matched on test_pool."""
    if "report_metric" not in final.columns:
        return pd.DataFrame()
    erm = (final[final["method"] == baseline]
           [["dataset", "model", "test_pool", "seed", "report_metric"]]
           .rename(columns={"report_metric": "erm_report_metric"}))
    meth = final[final["method"] != baseline]
    if erm.empty or meth.empty:
        return pd.DataFrame()
    merged = meth.merge(erm, on=["dataset", "model", "test_pool", "seed"], how="left")
    merged["lift"] = merged["erm_report_metric"] - merged["report_metric"]  # >0 beats ERM
    out = (merged.groupby(["dataset", "model", "method", "protocol", "test_pool"], dropna=False)
                 .agg(report_mae_mean=("report_metric", "mean"),
                      erm_mae_mean=("erm_report_metric", "mean"),
                      lift_mean=("lift", "mean"),
                      lift_std=("lift", "std"),
                      n_seeds=("seed", "nunique"))
                 .reset_index())
    out["pct_improvement"] = 100.0 * out["lift_mean"] / out["erm_mae_mean"]
    return out.sort_values(["dataset", "model", "test_pool", "lift_mean"],
                           ascending=[True, True, True, False]).reset_index(drop=True)


def run(runs_path: str | Path, out_dir: str | Path, *, verbose: bool = True) -> pd.DataFrame:
    """Aggregate all_runs.csv -> aggregated.csv + method_lift.csv. Returns the agg table.

    Importable so the pipeline (phase G) can auto-generate the summaries; also driven
    from the CLI via main().
    """
    runs = pd.read_csv(runs_path)
    if "stage" in runs.columns:
        final = runs[runs["stage"] == "final"].copy()
    else:
        final = runs.copy()
    if "status" in final.columns:
        final = final[final["status"].isna() | (final["status"] == "ok")]
    if verbose:
        print(f"{len(final)} final runs / {len(runs)} total")
    if final.empty:
        print("No final runs to aggregate.")
        return pd.DataFrame()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg = summarize_over_seeds(final)
    lift = method_lift(final)
    agg.to_csv(out / "aggregated.csv", index=False)
    if not lift.empty:
        lift.to_csv(out / "method_lift.csv", index=False)

    if verbose:
        with pd.option_context("display.max_columns", None, "display.width", 220):
            keep = [c for c in ["dataset", "model", "method", "protocol", "test_pool",
                                "n_seeds", "report_metric_mean", "report_metric_std"] if c in agg.columns]
            print("\n=== aggregated over seeds (report_metric) ===")
            print(agg[keep].round(4).to_string(index=False))
            if not lift.empty:
                print("\n=== method lift over ERM (lift>0 beats baseline) ===")
                print(lift.round(4).to_string(index=False))
        print(f"\n  -> {out/'aggregated.csv'}" + ("" if lift.empty else f"\n  -> {out/'method_lift.csv'}"))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="collected/all_runs.csv", help="all_runs.csv from collect.py.")
    ap.add_argument("--out", default="collected", help="Output dir for the summary CSVs.")
    args = ap.parse_args()
    run(args.runs, args.out)


if __name__ == "__main__":
    main()
