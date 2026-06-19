import itertools

import pandas as pd

from bioseq_ood.data.load_data import _build_mut_dist_split_package, _split_by_mut_dist

# Per-band row counts. The builder requires target_close_full (band 5) to be at
# least as large as target_test (half of the test pool, bands 6-8), so band 5 is
# deliberately large.
_BAND_COUNTS = {1: 40, 2: 40, 3: 40, 4: 40, 5: 200, 6: 20, 7: 20, 8: 20}


def _synthetic_dataframe() -> pd.DataFrame:
    """Unique sequences spread across mutation-distance bands 1..8."""
    rows = []
    for mut_dist, count in _BAND_COUNTS.items():
        for i in range(count):
            rows.append(
                {
                    "sequence": f"SEQ_{mut_dist}_{i}",  # globally unique
                    "label": float(mut_dist) + 0.01 * i,
                    "mut_dist": mut_dist,
                }
            )
    return pd.DataFrame(rows)


SPLIT_DISTANCES = {
    "train": [1, 2, 3],
    "val_ood": [4],
    "target_close": [5],
    "test": [6, 7, 8],
}


def _build(tmp_path):
    return _build_mut_dist_split_package(
        _synthetic_dataframe(),
        dataset_name="gfp",
        split_distances=SPLIT_DISTANCES,
        out_dir=tmp_path,
        source_meta_lines=["synthetic"],
        seed=0,
    )


def test_split_by_mut_dist_filters_band():
    df = _synthetic_dataframe()
    band = _split_by_mut_dist(df, [4])
    assert set(band["mut_dist"].unique()) == {4}


def test_splits_are_pairwise_disjoint_by_sequence(tmp_path):
    (train, val_id, val_ood, tc_full, tc, tt_full, tt, test, _) = _build(tmp_path)
    named = {
        "train": train, "val_id": val_id, "val_ood": val_ood,
        "target_close": tc, "test": test,
    }
    for a, b in itertools.combinations(named, 2):
        shared = set(named[a]["sequence"]) & set(named[b]["sequence"])
        assert not shared, f"{a} and {b} share sequences: {sorted(shared)[:3]}"


def test_oracle_split_disjoint_from_report_split(tmp_path):
    # target_test is the oracle selection signal; it MUST be disjoint from the
    # reported test set, otherwise oracle selection leaks into the headline number.
    (_, _, _, _, _, _, tt, test, _) = _build(tmp_path)
    assert not (set(tt["sequence"]) & set(test["sequence"]))


def test_each_split_stays_in_its_band(tmp_path):
    (train, val_id, val_ood, _, tc, _, tt, test, _) = _build(tmp_path)
    assert set(train["mut_dist"]).issubset({1, 2, 3})
    assert set(val_id["mut_dist"]).issubset({1, 2, 3})  # val_id is held out from the ID pool
    assert set(val_ood["mut_dist"]) == {4}
    assert set(tc["mut_dist"]) == {5}
    assert set(test["mut_dist"]).issubset({6, 7, 8})
    assert set(tt["mut_dist"]).issubset({6, 7, 8})


def test_train_and_val_id_partition_the_id_pool(tmp_path):
    (train, val_id, *_rest) = _build(tmp_path)
    id_pool_size = len(_split_by_mut_dist(_synthetic_dataframe(), [1, 2, 3]))
    assert len(train) + len(val_id) == id_pool_size
    assert len(val_id) > 0
