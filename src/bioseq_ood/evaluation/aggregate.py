from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

# Identity columns of an experiment result CSV (one row per evaluated cell).
_CELL_KEYS = ["dataset", "model", "method", "setting"]


def load_results(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate one or more experiment result CSVs into a single dataframe."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy() if df["error"].notna().any() else df.drop(columns=["error"])
    return df


def summarize_over_seeds(df: pd.DataFrame, *, metric: str = "report_metric") -> pd.DataFrame:
    """Mean +/- std of a metric over seeds for every (cell, selection_mode)."""
    keys = _CELL_KEYS + ["selection_mode"]
    grouped = df.groupby(keys)[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "n_seeds"})
    return out.sort_values(keys).reset_index(drop=True)


def selection_gap_table(df: pd.DataFrame, *, metric: str = "report_metric") -> pd.DataFrame:
    """Report-split error per selection mode, plus the model-selection penalty.

    For each (dataset, model, method, setting) the report-split error is averaged
    over seeds and pivoted by selection mode. The penalty columns quantify how much
    is lost by selecting a model without far-OOD labels:

        selection_penalty_val_ood = reported val_ood - reported oracle
        selection_penalty_val_id  = reported val_id  - reported oracle

    Positive penalty = realistic selection is worse than the oracle upper bound.
    """
    summary = summarize_over_seeds(df, metric=metric)
    mean_col = f"{metric}_mean"
    pivot = summary.pivot_table(index=_CELL_KEYS, columns="selection_mode", values=mean_col)
    pivot = pivot.rename(columns={m: f"report@{m}" for m in pivot.columns})

    if "report@oracle" in pivot.columns:
        for mode in ("val_ood", "val_id"):
            col = f"report@{mode}"
            if col in pivot.columns:
                pivot[f"selection_penalty_{mode}"] = pivot[col] - pivot["report@oracle"]
    return pivot.reset_index()


def method_lift_table(df: pd.DataFrame, *, metric: str = "report_metric", baseline: str = "erm") -> pd.DataFrame:
    """Report-split lift of each method over the ERM baseline, per selection mode.

    Lift is defined as a reduction in error: baseline_error - method_error (positive
    means the method beats ERM). Computed within matching (dataset, model, setting,
    selection_mode) groups, averaged over seeds.
    """
    summary = summarize_over_seeds(df, metric=metric)
    mean_col = f"{metric}_mean"
    base = summary[summary["method"] == baseline]
    base = base[["dataset", "model", "setting", "selection_mode", mean_col]].rename(
        columns={mean_col: "baseline_error"}
    )
    merged = summary.merge(base, on=["dataset", "model", "setting", "selection_mode"], how="left")
    merged["lift_over_erm"] = merged["baseline_error"] - merged[mean_col]
    merged = merged[merged["method"] != baseline].copy()
    return merged.sort_values(_CELL_KEYS + ["selection_mode"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment results into summary tables.")
    parser.add_argument("results", nargs="+", help="One or more experiment_results_*.csv files.")
    parser.add_argument("--metric", default="report_metric", help="Metric column to aggregate.")
    parser.add_argument("--output-dir", default=None, help="Directory to write summary CSVs.")
    args = parser.parse_args()

    df = load_results(args.results)
    if df.empty:
        print("No (non-error) results to aggregate.")
        return

    gap = selection_gap_table(df, metric=args.metric)
    lift = method_lift_table(df, metric=args.metric)
    summary = summarize_over_seeds(df, metric=args.metric)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        gap.to_csv(out / "selection_gap.csv", index=False)
        lift.to_csv(out / "method_lift.csv", index=False)
        summary.to_csv(out / "summary_over_seeds.csv", index=False)
        print(f"Wrote summary tables to {out}")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== Selection-gap table (report error by selection mode) ===")
        print(gap.to_string(index=False))
        print("\n=== Method lift over ERM ===")
        print(lift.to_string(index=False))


if __name__ == "__main__":
    main()
