"""Combine the per-model shard outputs from a parallel run into unified CSVs.

A parallel run launches one process per model (MODEL_SHARD=<model>), each writing to
results_phases/<model>/. This script concatenates those per-model CSVs back into the
top-level results_phases/ so the analysis files look exactly like a single sequential
run. Each shard already computed its own Phase G (TOP_K=1), so this is a plain
concatenation -- no re-aggregation needed.

    uv run python merge_shards.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results_phases"

# Per-run CSVs each shard writes under results_phases/<model>/.
SHARD_CSVS = [
    "phaseB_encoder_search.csv",
    "phaseB_all_trials.csv",
    "phaseC_baseline_reference.csv",
    "phaseC_by_mut_dist.csv",
    "phaseD_top_models.csv",
    "phaseE_method_search.csv",
    "phaseF_all_protocols.csv",
    "phaseF_by_mut_dist.csv",
    "phaseG_analysis.csv",
    "phaseG_by_mut_dist.csv",
]


def main() -> None:
    shard_dirs = sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    if not shard_dirs:
        raise SystemExit(f"No per-model shard folders found under {RESULTS_DIR}.")
    print(f"Merging shards: {[d.name for d in shard_dirs]}")

    for csv in SHARD_CSVS:
        parts = [pd.read_csv(d / csv) for d in shard_dirs if (d / csv).exists()]
        if not parts:
            continue
        combined = pd.concat(parts, ignore_index=True)
        out = RESULTS_DIR / csv
        combined.to_csv(out, index=False)
        print(f"  {csv:<32} {len(parts)} shards -> {len(combined)} rows")

    # Headline, mean over models within each dataset, from the merged analysis table.
    g = RESULTS_DIR / "phaseG_analysis.csv"
    if g.exists():
        df = pd.read_csv(g)
        cols = [c for c in ("lift_close", "lift_far", "G_extrap", "D_shift", "delta_close", "delta_far") if c in df.columns]
        print("\nHEADLINE (mean over models within each dataset)")
        for ds, sub in df.groupby("dataset"):
            print(f"\n  {ds}")
            print(sub.groupby("method")[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
