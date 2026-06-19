import pandas as pd
import pytest

from bioseq_ood.evaluation.aggregate import (
    method_lift_table,
    selection_gap_table,
    summarize_over_seeds,
)


def _synthetic_results() -> pd.DataFrame:
    # report (test) error per (method, selection_mode); oracle is the best.
    table = {
        ("erm", "val_id"): 0.50, ("erm", "val_ood"): 0.45, ("erm", "oracle"): 0.40,
        ("cmd", "val_id"): 0.48, ("cmd", "val_ood"): 0.42, ("cmd", "oracle"): 0.35,
    }
    rows = []
    for (method, mode), base in table.items():
        for seed in (42, 43):
            rows.append(
                {
                    "dataset": "gfp", "model": "cnn", "method": method,
                    "setting": "extrapolative", "selection_mode": mode, "seed": seed,
                    "report_metric": base,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_over_seeds_counts_seeds():
    out = summarize_over_seeds(_synthetic_results())
    assert (out["n_seeds"] == 2).all()
    assert "report_metric_mean" in out.columns


def test_selection_gap_penalty_is_positive_when_realistic_worse_than_oracle():
    gap = selection_gap_table(_synthetic_results())
    cmd_row = gap[gap["method"] == "cmd"].iloc[0]
    assert cmd_row["report@oracle"] == 0.35
    # val_ood selection is worse (higher error) than oracle -> positive penalty.
    assert cmd_row["selection_penalty_val_ood"] == pytest.approx(0.07)
    assert cmd_row["selection_penalty_val_id"] == pytest.approx(0.13)


def test_method_lift_excludes_baseline_and_measures_error_reduction():
    lift = method_lift_table(_synthetic_results())
    assert (lift["method"] != "erm").all()
    oracle = lift[(lift["method"] == "cmd") & (lift["selection_mode"] == "oracle")].iloc[0]
    # cmd (0.35) beats erm (0.40) under oracle selection -> lift = 0.05.
    assert oracle["lift_over_erm"] == pytest.approx(0.05)
