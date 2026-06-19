"""
load_data.py

Dataset loaders that:
- resolve the project-root data directory robustly, independent of CWD
- build mutation-distance-based domain shift splits
- save CSVs + metadata.txt per dataset
- expose no-input loader functions with consistent return signatures

Split layout:
  train.csv
  val_id.csv
  val_ood.csv
  target_close_full.csv
  target_close.csv
  target_test_full.csv
  target_test.csv
  test.csv
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# -----------------------------
# Data root resolution
# -----------------------------


def _find_repo_root(start: Path) -> Path:
    """
    Walk upward from a starting path until we find repo markers.
    Prefers pyproject.toml, falls back to .git.
    """
    start = start.resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Fallback: assumes src layout and goes up 3 from load_data.py
    return start.parents[3]


def _get_data_root() -> Path:
    """
    Resolve base data directory.
    Order:
      1) env var SSL_FOR_OOD_DATA_DIR
      2) <repo_root>/data
    """
    env = os.getenv("SSL_FOR_OOD_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (_find_repo_root(Path(__file__)) / "data").resolve()


DATA_ROOT = _get_data_root()


def _dataset_splits_dir(dataset_name: str) -> Path:
    d = DATA_ROOT / dataset_name / "splits"
    d.mkdir(parents=True, exist_ok=True)
    return d


# -----------------------------
# Shared helpers
# -----------------------------


def _hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"Hamming distance requires equal lengths, got {len(a)} and {len(b)}")
    return sum(x != y for x, y in zip(a, b))


def _save_files_with_metadata_txt(
    out_dir: Path,
    *,
    files: dict[str, pd.DataFrame],
    meta_lines: list[str],
) -> None:
    """Writes arbitrary CSV files + metadata.txt into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname, df in files.items():
        df.to_csv(out_dir / fname, index=False)

    meta_path = out_dir / "metadata.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        for line in meta_lines:
            f.write(line.rstrip("\n") + "\n")


def _add_mutation_distance(df: pd.DataFrame, *, core: str, seq_col: str = "sequence") -> pd.DataFrame:
    out = df.copy()
    out[seq_col] = out[seq_col].astype(str).str.upper()
    out["mut_dist"] = out[seq_col].apply(lambda s: _hamming(s, core))
    return out


def _finalize_schema(df: pd.DataFrame, *, split_name: str) -> pd.DataFrame:
    """Final column order: sequence,label,mut_dist,split."""
    out = df.copy()
    out["sequence"] = out["sequence"].astype(str)
    out["label"] = out["label"].astype(float)
    out["mut_dist"] = out["mut_dist"].astype(int)
    out["split"] = split_name
    return out[["sequence", "label", "mut_dist", "split"]].reset_index(drop=True)


def _sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic sample without replacement, or all rows if df is smaller than n."""
    if n <= 0:
        return df.iloc[0:0].copy().reset_index(drop=True)
    if len(df) <= n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def _sample_df_keep_index(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic sample without replacement, preserving the original index.

    This is needed when sampled rows must be removed from a larger pool later,
    for example when partitioning TFBind8 mut_dist == 6 into disjoint
    val_ood and target_close subsets.
    """
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, replace=False, random_state=seed).copy()


def _split_id_train_val_id(
    id_df: pd.DataFrame,
    val_id_frac: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create val_id from the ID pool; remaining rows are train."""
    if len(id_df) == 0:
        raise ValueError("Cannot create train/val_id split from an empty ID pool.")

    val_id_df = id_df.sample(frac=val_id_frac, random_state=seed).copy()
    train_df = id_df.drop(index=val_id_df.index).copy()
    return train_df.reset_index(drop=True), val_id_df.reset_index(drop=True)


def _split_by_mut_dist(df: pd.DataFrame, mut_dists: Iterable[int]) -> pd.DataFrame:
    mut_dists = list(mut_dists)
    return df[df["mut_dist"].isin(mut_dists)].copy().reset_index(drop=True)


def _stratified_half_split_by_mut_dist(
    df: pd.DataFrame,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into two disjoint halves, stratified by mutation distance.

    Returns:
      final_test_raw, target_test_full_raw

    For odd group sizes, the extra row stays in final_test_raw. This avoids making
    the final evaluation test set smaller than the target_test_full pool.
    """
    if len(df) == 0:
        raise ValueError("Cannot split an empty test pool into test and target_test_full.")

    rng = np.random.default_rng(seed)
    test_parts: list[pd.DataFrame] = []
    target_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("mut_dist", sort=True):
        group = group.copy()
        permuted_idx = rng.permutation(group.index.to_numpy())

        n_target = len(group) // 2
        target_idx = permuted_idx[:n_target]
        test_idx = permuted_idx[n_target:]

        target_parts.append(group.loc[target_idx])
        test_parts.append(group.loc[test_idx])

    test_raw = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    target_test_full_raw = pd.concat(target_parts, axis=0).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return test_raw, target_test_full_raw


def _build_mut_dist_split_package(
    df: pd.DataFrame,
    *,
    dataset_name: str,
    split_distances: dict[str, list[int]],
    out_dir: Path,
    source_meta_lines: list[str],
    val_id_frac: float = 0.10,
    target_test_cap_n: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    Common split implementation for all datasets.

    Required split_distances keys:
      train, val_ood, target_close, test
    """
    required = {"train", "val_ood", "target_close", "test"}
    missing = required - set(split_distances)
    if missing:
        raise ValueError(f"Missing split distance definitions for {dataset_name}: {sorted(missing)}")

    id_pool = _split_by_mut_dist(df, split_distances["train"])
    val_ood_raw = _split_by_mut_dist(df, split_distances["val_ood"])
    target_close_full_raw = _split_by_mut_dist(df, split_distances["target_close"])
    test_pool = _split_by_mut_dist(df, split_distances["test"])

    if len(id_pool) == 0:
        raise ValueError(f"{dataset_name}: train pool is empty. Check mutation-distance bands.")
    if len(val_ood_raw) == 0:
        raise ValueError(f"{dataset_name}: val_ood pool is empty. Check mutation-distance bands.")
    if len(target_close_full_raw) == 0:
        raise ValueError(f"{dataset_name}: target_close_full pool is empty. Check mutation-distance bands.")
    if len(test_pool) == 0:
        raise ValueError(f"{dataset_name}: test pool is empty. Check mutation-distance bands.")

    train_raw, val_id_raw = _split_id_train_val_id(id_pool, val_id_frac=val_id_frac, seed=seed)

    test_raw, target_test_full_raw = _stratified_half_split_by_mut_dist(test_pool, seed=seed)
    target_test_raw = _sample_df(target_test_full_raw, n=target_test_cap_n, seed=seed)

    if len(target_close_full_raw) < len(target_test_raw):
        raise ValueError(
            f"{dataset_name}: target_close_full has only {len(target_close_full_raw)} rows, "
            f"but target_test has {len(target_test_raw)} rows. Cannot make equal-size target splits."
        )
    target_close_raw = _sample_df(target_close_full_raw, n=len(target_test_raw), seed=seed)

    train_df = _finalize_schema(train_raw, split_name="train")
    val_id_df = _finalize_schema(val_id_raw, split_name="val_id")
    val_ood_df = _finalize_schema(val_ood_raw, split_name="val_ood")
    target_close_full_df = _finalize_schema(target_close_full_raw, split_name="target_close_full")
    target_close_df = _finalize_schema(target_close_raw, split_name="target_close")
    target_test_full_df = _finalize_schema(target_test_full_raw, split_name="target_test_full")
    target_test_df = _finalize_schema(target_test_raw, split_name="target_test")
    test_df = _finalize_schema(test_raw, split_name="test")

    meta_lines = [
        *source_meta_lines,
        f"data_root={DATA_ROOT}",
        f"out_dir={out_dir}",
        "split_strategy=fixed_mut_dist_bands_with_stratified_test_target_split",
        f"train_definition=mut_dist in {split_distances['train']}",
        f"val_id_definition={val_id_frac:.0%} random sample from train band for early stopping",
        f"val_ood_definition=mut_dist in {split_distances['val_ood']}",
        f"target_close_full_definition=mut_dist in {split_distances['target_close']}",
        "target_close_definition=random sample from target_close_full with len(target_close)==len(target_test)",
        f"test_pool_definition=mut_dist in {split_distances['test']}",
        "test_target_split_definition=stratified 50/50 split of test_pool by mut_dist",
        f"target_test_cap_n={target_test_cap_n}",
        "target_test_full_definition=uncapped target half of test_pool",
        "target_test_definition=random sample from target_test_full capped at target_test_cap_n",
        f"split_seed={seed}",
        f"counts_total_after_filtering={len(df)}",
        f"counts_train={len(train_df)}",
        f"counts_val_id={len(val_id_df)}",
        f"counts_val_ood={len(val_ood_df)}",
        f"counts_target_close_full={len(target_close_full_df)}",
        f"counts_target_close={len(target_close_df)}",
        f"counts_target_test_full={len(target_test_full_df)}",
        f"counts_target_test={len(target_test_df)}",
        f"counts_test={len(test_df)}",
    ]

    _save_files_with_metadata_txt(
        out_dir,
        files={
            "train.csv": train_df,
            "val_id.csv": val_id_df,
            "val_ood.csv": val_ood_df,
            "target_close_full.csv": target_close_full_df,
            "target_close.csv": target_close_df,
            "target_test_full.csv": target_test_full_df,
            "target_test.csv": target_test_df,
            "test.csv": test_df,
        },
        meta_lines=meta_lines,
    )

    return (
        train_df,
        val_id_df,
        val_ood_df,
        target_close_full_df,
        target_close_df,
        target_test_full_df,
        target_test_df,
        test_df,
        str(out_dir),
    )


def _build_tfbind8_split_package(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    source_meta_lines: list[str],
    val_id_frac: float = 0.10,
    target_test_cap_n: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    TFBind8-specific split implementation.

    TFBind8 uses mut_dist == 6 for both near-OOD validation and close target
    adaptation. To avoid data leakage, this function partitions the mut_dist == 6
    pool into disjoint val_ood and target_close subsets.
    """
    id_pool = _split_by_mut_dist(df, [1, 2, 3, 4, 5])
    near_pool = df[df["mut_dist"].isin([6])].copy()
    test_pool = _split_by_mut_dist(df, [7, 8])

    if len(id_pool) == 0:
        raise ValueError("tfbind8: train pool is empty. Check mutation-distance bands.")
    if len(near_pool) == 0:
        raise ValueError("tfbind8: near-OOD pool mut_dist == 6 is empty.")
    if len(test_pool) == 0:
        raise ValueError("tfbind8: test pool is empty. Check mutation-distance bands.")

    train_raw, val_id_raw = _split_id_train_val_id(
        id_pool,
        val_id_frac=val_id_frac,
        seed=seed,
    )

    # Keep val_ood capped, but never consume the full near-OOD band because the
    # remaining rows are needed as the close target domain.
    val_ood_n = min(5000, len(near_pool) // 2)
    if val_ood_n <= 0:
        raise ValueError(
            "tfbind8: near-OOD pool is too small to create disjoint val_ood and target_close splits."
        )

    val_ood_raw = _sample_df_keep_index(
        near_pool,
        n=val_ood_n,
        seed=seed,
    )

    target_close_full_raw = (
        near_pool
        .drop(index=val_ood_raw.index)
        .copy()
        .reset_index(drop=True)
    )

    test_raw, target_test_full_raw = _stratified_half_split_by_mut_dist(
        test_pool,
        seed=seed,
    )

    target_test_raw = _sample_df(
        target_test_full_raw,
        n=target_test_cap_n,
        seed=seed,
    )

    if len(target_close_full_raw) < len(target_test_raw):
        raise ValueError(
            f"tfbind8: target_close_full has only {len(target_close_full_raw)} rows, "
            f"but target_test has {len(target_test_raw)} rows. Cannot make equal-size target splits."
        )

    target_close_raw = _sample_df(
        target_close_full_raw,
        n=len(target_test_raw),
        seed=seed,
    )

    train_df = _finalize_schema(train_raw, split_name="train")
    val_id_df = _finalize_schema(val_id_raw, split_name="val_id")
    val_ood_df = _finalize_schema(val_ood_raw, split_name="val_ood")
    target_close_full_df = _finalize_schema(target_close_full_raw, split_name="target_close_full")
    target_close_df = _finalize_schema(target_close_raw, split_name="target_close")
    target_test_full_df = _finalize_schema(target_test_full_raw, split_name="target_test_full")
    target_test_df = _finalize_schema(target_test_raw, split_name="target_test")
    test_df = _finalize_schema(test_raw, split_name="test")

    meta_lines = [
        *source_meta_lines,
        f"data_root={DATA_ROOT}",
        f"out_dir={out_dir}",
        "split_strategy=tfbind8_partitioned_near_ood_band_with_stratified_test_target_split",
        "train_definition=mut_dist in [1, 2, 3, 4, 5]",
        f"val_id_definition={val_id_frac:.0%} random sample from train band for early stopping",
        "near_ood_pool_definition=mut_dist == 6",
        f"val_ood_definition=random sample of {val_ood_n} rows from mut_dist == 6",
        "target_close_full_definition=remaining mut_dist == 6 rows after removing val_ood",
        "target_close_definition=random sample from target_close_full with len(target_close)==len(target_test)",
        "test_pool_definition=mut_dist in [7, 8]",
        "test_target_split_definition=stratified 50/50 split of test_pool by mut_dist",
        f"target_test_cap_n={target_test_cap_n}",
        "target_test_full_definition=uncapped target half of test_pool",
        "target_test_definition=random sample from target_test_full capped at target_test_cap_n",
        f"split_seed={seed}",
        f"counts_total_after_filtering={len(df)}",
        f"counts_train={len(train_df)}",
        f"counts_val_id={len(val_id_df)}",
        f"counts_val_ood={len(val_ood_df)}",
        f"counts_target_close_full={len(target_close_full_df)}",
        f"counts_target_close={len(target_close_df)}",
        f"counts_target_test_full={len(target_test_full_df)}",
        f"counts_target_test={len(target_test_df)}",
        f"counts_test={len(test_df)}",
    ]

    _save_files_with_metadata_txt(
        out_dir,
        files={
            "train.csv": train_df,
            "val_id.csv": val_id_df,
            "val_ood.csv": val_ood_df,
            "target_close_full.csv": target_close_full_df,
            "target_close.csv": target_close_df,
            "target_test_full.csv": target_test_full_df,
            "target_test.csv": target_test_df,
            "test.csv": test_df,
        },
        meta_lines=meta_lines,
    )

    return (
        train_df,
        val_id_df,
        val_ood_df,
        target_close_full_df,
        target_close_df,
        target_test_full_df,
        target_test_df,
        test_df,
        str(out_dir),
    )


# -----------------------------
# Split definitions
# -----------------------------


MUT_DIST_SPLITS = {
    "aav": {
        "train": [1, 2, 3, 4],
        "val_ood": [5],
        "target_close": [6],
        "test": list(range(7, 21)),
    },
    "gfp": {
        "train": [1, 2, 3],
        "val_ood": [4],
        "target_close": [5],
        "test": list(range(6, 16)),
    },
    # TFBind8 requires special handling because val_ood and target_close are
    # both drawn from mut_dist == 6. The loader partitions that band into
    # disjoint subsets instead of reusing the same rows.
    "tfbind8": {
        "train": [1, 2, 3, 4, 5],
        "val_ood": [6],
        "target_close": [6],
        "test": [7, 8],
    },
}


# -----------------------------
# ProteinGym settings for GFP / AAV
# -----------------------------


PROTEINGYM_PARQUET_LINKS = [
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-00000-of-00005.parquet",
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-00001-of-00005.parquet",
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-00002-of-00005.parquet",
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-00003-of-00005.parquet",
    "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1/resolve/main/DMS_substitutions/train-00004-of-00005.parquet",
]


PROTEINGYM_SETTINGS = {
    "gfp": {
        "dms_id": "GFP_AEQVI_Sarkisyan_2016",
        "wt_sequence": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    },
    "aav": {
        "dms_id": "CAPSD_AAV2S_Sinai_2021",
        "wt_sequence": "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEADAAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKTAPGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGSGAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGYSTPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFTDSEYQLPYVLGSAHQGCLPPFPADVFMVPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSYTFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPCYRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTNVDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAKIPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKENSKRWNPEIQYTSNYNKSVNVDFTVDTNGVYSEPRPIGTRYLTRNL",
    },
}


def _load_proteingym_dms_dataset(dms_id: str, wt_sequence: str) -> pd.DataFrame:
    """Load one ProteinGym DMS_substitutions assay and normalize to sequence,label."""
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={"train": PROTEINGYM_PARQUET_LINKS})
    df = ds["train"].to_pandas()

    required_cols = {"mutated_sequence", "target_seq", "DMS_score", "DMS_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"ProteinGym parquet is missing required columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    df = df[df["DMS_id"] == dms_id].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No rows found in ProteinGym for DMS_id='{dms_id}'")

    df["sequence"] = df["mutated_sequence"].astype(str).str.upper()
    df["label"] = pd.to_numeric(df["DMS_score"], errors="coerce")
    df["target_seq"] = df["target_seq"].astype(str).str.upper()
    df = df.dropna(subset=["sequence", "label"]).copy()

    target_seq_values = df["target_seq"].dropna().unique().tolist()
    if len(target_seq_values) == 0:
        raise ValueError(f"No target_seq values available for DMS_id='{dms_id}'")
    if len(set(target_seq_values)) > 1:
        raise ValueError(
            f"Expected a single target_seq for DMS_id='{dms_id}', got {len(set(target_seq_values))}"
        )

    protein_gym_wt = target_seq_values[0]
    if protein_gym_wt != wt_sequence.upper():
        raise ValueError(
            f"Configured WT does not match ProteinGym target_seq for DMS_id='{dms_id}'.\n"
            f"Configured WT length={len(wt_sequence)}\n"
            f"ProteinGym target_seq length={len(protein_gym_wt)}"
        )

    return df[["sequence", "label"]].drop_duplicates(subset="sequence").reset_index(drop=True)


def _load_proteingym_with_mut_dist(dataset_name: str) -> tuple[pd.DataFrame, list[str]]:
    settings = PROTEINGYM_SETTINGS[dataset_name]
    repo_id = "OATML-Markslab/ProteinGym_v1"
    dms_id = settings["dms_id"]
    core = settings["wt_sequence"].upper()

    df = _load_proteingym_dms_dataset(dms_id=dms_id, wt_sequence=core)
    df["seq_len"] = df["sequence"].apply(len)
    df = df[df["seq_len"] == len(core)].copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"{dataset_name.upper()} dataset is empty after filtering to WT length.")

    df = _add_mutation_distance(df, core=core, seq_col="sequence")

    meta_lines = [
        f"dataset={dataset_name}",
        f"repo_id={repo_id}",
        "source_config=DMS_substitutions",
        f"dms_id={dms_id}",
        "source_sequence_column=mutated_sequence",
        "source_label_column=DMS_score",
        "source_wt_column=target_seq",
        f"core_length={len(core)}",
        f"core_sequence={core}",
        "duplicates_removed=True",
        "mutation_distance_definition=Hamming distance to ProteinGym target_seq WT",
    ]
    return df, meta_lines


# -----------------------------
# GFP
# -----------------------------


def load_gfp_data():
    df, source_meta_lines = _load_proteingym_with_mut_dist("gfp")
    out_dir = _dataset_splits_dir("gfp")
    return _build_mut_dist_split_package(
        df,
        dataset_name="gfp",
        split_distances=MUT_DIST_SPLITS["gfp"],
        out_dir=out_dir,
        source_meta_lines=source_meta_lines,
        val_id_frac=0.10,
        target_test_cap_n=5000,
        seed=42,
    )


# -----------------------------
# AAV
# -----------------------------


def load_aav_data():
    df, source_meta_lines = _load_proteingym_with_mut_dist("aav")
    out_dir = _dataset_splits_dir("aav")
    return _build_mut_dist_split_package(
        df,
        dataset_name="aav",
        split_distances=MUT_DIST_SPLITS["aav"],
        out_dir=out_dir,
        source_meta_lines=source_meta_lines,
        val_id_frac=0.10,
        target_test_cap_n=5000,
        seed=42,
    )


# -----------------------------
# TFBind8
# -----------------------------


def load_tfbind8_data():
    """
    TFBind8 loader.

    Keeps the previous mutation-distance bands:
      train: mut_dist in {1,2,3,4,5}
      val_ood: mut_dist == 6
      test pool: mut_dist in {7,8}

    Adds:
      target_close_full / target_close from mut_dist == 6
      target_test_full / target_test from a stratified half of the test pool
    """
    from huggingface_hub import hf_hub_download

    repo_id = "beckhamc/design_bench_data"
    x_file = "tf_bind_8-SIX6_REF_R1/tf_bind_8-x-0.npy"
    y_file = "tf_bind_8-SIX6_REF_R1/tf_bind_8-y-0.npy"

    def _x_to_dna_strings(x: np.ndarray, alphabet=("A", "C", "G", "T")) -> list[str]:
        x = np.asarray(x)

        if x.ndim == 2:
            if x.max() > 3 or x.min() < 0:
                raise ValueError(f"Expected integer tokens in [0,3], got min={x.min()} max={x.max()}")
            idx = x.astype(int)
        elif x.ndim == 3:
            if x.shape[-1] == 4:
                idx = x.argmax(axis=-1)
            elif x.shape[1] == 4:
                idx = x.argmax(axis=1)
            else:
                raise ValueError(f"Unrecognized one-hot shape {x.shape}; expected (...,4) somewhere.")
        else:
            raise ValueError(f"Unsupported x ndim={x.ndim}; shape={x.shape}")

        alpha = np.array(alphabet)
        return ["".join(alpha[row]) for row in idx]

    x_path = hf_hub_download(repo_id=repo_id, filename=x_file, repo_type="dataset")
    y_path = hf_hub_download(repo_id=repo_id, filename=y_file, repo_type="dataset")

    x_raw = np.load(x_path, allow_pickle=False)
    y_raw = np.load(y_path, allow_pickle=False).reshape(-1).astype(float)

    sequences = _x_to_dna_strings(x_raw, alphabet=("A", "C", "G", "T"))
    if not sequences:
        raise ValueError("Empty TFBind8 dataset.")

    seq_len = len(sequences[0])
    if seq_len != 8:
        raise ValueError(f"Expected TFBind8 sequence length 8, got seq_len={seq_len}")

    np.random.seed(42)
    anchor_idx = int(np.random.choice(len(sequences)))
    core = sequences[anchor_idx]

    df = pd.DataFrame({
        "sequence": pd.Series(sequences, dtype="string"),
        "label": y_raw.astype(float),
    })
    df = df.drop_duplicates(subset="sequence", keep="first").reset_index(drop=True)
    df = _add_mutation_distance(df, core=core, seq_col="sequence")

    out_dir = _dataset_splits_dir("tfbind8")
    source_meta_lines = [
        "dataset=tfbind8",
        f"repo_id={repo_id}",
        f"x_source={x_path}",
        f"y_source={y_path}",
        f"seq_len={seq_len}",
        "core_definition=random_anchor_sequence",
        "anchor_seed=42",
        f"anchor_idx={anchor_idx}",
        f"core_anchor_seq={core}",
        "duplicates_removed=True",
        "mutation_distance_definition=Hamming distance to random anchor sequence",
    ]

    return _build_tfbind8_split_package(
        df,
        out_dir=out_dir,
        source_meta_lines=source_meta_lines,
        val_id_frac=0.10,
        target_test_cap_n=5000,
        seed=42,
    )


# -----------------------------
# Convenience
# -----------------------------


def load_all_data():
    """Convenience loader. Returns a dict of dataset_name -> split tuple."""
    return {
        "gfp": load_gfp_data(),
        "aav": load_aav_data(),
        "tfbind8": load_tfbind8_data(),
    }


if __name__ == "__main__":
    load_all_data()
