from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bioseq_ood.data.load_data import _get_data_root

AA_TO_INT = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19,
}

DNA_TO_INT = {
    "A": 0, "C": 1, "G": 2, "T": 3,
}

DATASET_TYPE = {
    "aav": "protein",
    "gfp": "protein",
    "tfbind8": "dna",
    "gb1": "protein",
}

DEFAULT_SPLITS = (
    "train",
    "val_id",
    "val_ood",
    "target_close_full",
    "target_close",
    "target_test_full",
    "target_test",
    "test",
)

# These are the only split files for which labels should be removed during
# preprocessing. Evaluation splits such as target_close, target_test, and test
# must keep labels so they can be scored after adaptation. Target labels are
# ignored later by the UDA dataloader, not deleted from the stored artifacts.
UNLABELED_TARGET_SPLITS = {
    "target_unlabeled",
}


def remove_static_aav_area(data_matrix_np: np.ndarray) -> np.ndarray:
    return np.array([val[560:588] for val in data_matrix_np])


def _validate_dataset(dataset: str) -> None:
    if dataset not in DATASET_TYPE:
        valid = ", ".join(sorted(DATASET_TYPE))
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of: {valid}")


def _splits_dir(dataset: str) -> Path:
    return _get_data_root() / dataset / "splits"


def _preprocessed_dir(dataset: str) -> Path:
    out_dir = _get_data_root() / dataset / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _split_csv_path(dataset: str, split: str) -> Path:
    return _splits_dir(dataset) / f"{split}.csv"


def _preprocessed_npz_path(dataset: str, split: str) -> Path:
    return _preprocessed_dir(dataset) / f"{split}.npz"


def _y_scaler_path(dataset: str) -> Path:
    return _preprocessed_dir(dataset) / "y_scaler.npz"


def encode_protein(seq: str) -> list[int]:
    try:
        return [AA_TO_INT[a] for a in seq]
    except KeyError as e:
        raise ValueError(f"Invalid amino acid '{e.args[0]}' in sequence: {seq}") from e


def encode_dna(seq: str) -> list[int]:
    try:
        return [DNA_TO_INT[a] for a in seq]
    except KeyError as e:
        raise ValueError(f"Invalid DNA base '{e.args[0]}' in sequence: {seq}") from e


def encode_sequence(seq: str, dataset: str) -> list[int]:
    seq = str(seq).strip().upper()
    if DATASET_TYPE[dataset] == "dna":
        return encode_dna(seq)
    return encode_protein(seq)


def encode_dataframe(df: pd.DataFrame, dataset: str) -> np.ndarray:
    if "sequence" not in df.columns:
        raise ValueError("Expected column 'sequence' in dataframe.")

    seqs = df["sequence"].astype(str).str.upper().apply(lambda s: encode_sequence(s, dataset))

    lengths = seqs.apply(len)
    if lengths.nunique() != 1:
        raise ValueError(
            f"Sequences for dataset '{dataset}' do not all have the same length. "
            f"Found lengths: {sorted(lengths.unique().tolist())}"
        )

    return np.asarray(seqs.tolist(), dtype=np.int16)


def fit_y_scaler(y_train: np.ndarray) -> dict[str, float]:
    y_train = np.asarray(y_train, dtype=np.float32)
    if y_train.ndim != 1:
        raise ValueError(f"Expected 1D y_train, got shape={y_train.shape}")

    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0.0:
        std = 1.0

    return {
        "type": "standard",
        "mean": mean,
        "std": std,
    }


def transform_y(y: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    return ((y - scaler["mean"]) / scaler["std"]).astype(np.float32)


def inverse_transform_y(y_scaled: np.ndarray, scaler: dict[str, float]) -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    return (y_scaled * scaler["std"] + scaler["mean"]).astype(np.float32)


def save_y_scaler(dataset: str, scaler: dict[str, float]) -> str:
    path = _y_scaler_path(dataset)
    np.savez_compressed(
        path,
        scaler_type=np.array(scaler["type"]),
        mean=np.array(scaler["mean"], dtype=np.float32),
        std=np.array(scaler["std"], dtype=np.float32),
    )
    return str(path)


def load_y_scaler(dataset: str) -> dict[str, float]:
    path = _y_scaler_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {path}. Run preprocess_data(dataset=...) first."
        )

    data = np.load(path, allow_pickle=False)
    return {
        "type": str(data["scaler_type"]),
        "mean": float(data["mean"]),
        "std": float(data["std"]),
    }


def load_split_dataframe(dataset: str, split: str) -> pd.DataFrame:
    _validate_dataset(dataset)
    path = _split_csv_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. "
            f"Run the data loading/materialization step first."
        )
    return pd.read_csv(path)


def _should_drop_labels(split: str, drop_labels_for_unlabeled: bool) -> bool:
    return drop_labels_for_unlabeled and split in UNLABELED_TARGET_SPLITS


def preprocess_split(
    dataset: str,
    split: str,
    *,
    y_scaler: dict[str, float] | None = None,
    scale_y: bool = True,
    save: bool = True,
    overwrite: bool = False,
    drop_labels_for_unlabeled: bool = True,
) -> dict[str, Any]:
    _validate_dataset(dataset)

    out_path = _preprocessed_npz_path(dataset, split)
    if out_path.exists() and not overwrite and save:
        try:
            loaded = load_preprocessed_split(dataset, split)
            loaded["path"] = str(out_path)
            return loaded
        except (KeyError, ValueError, EOFError):
            pass  # stale or incompatible cache; rebuild below

    df = load_split_dataframe(dataset, split)
    x = encode_dataframe(df, dataset)
    x = remove_static_aav_area(x) if dataset == "aav" else x

    y = None
    y_unscaled = None
    labels_dropped = False
    if "label" in df.columns:
        labels_dropped = _should_drop_labels(split, drop_labels_for_unlabeled)
        if not labels_dropped:
            y_unscaled = df["label"].to_numpy(dtype=np.float32)
            y = transform_y(y_unscaled, y_scaler) if (scale_y and y_scaler is not None) else y_unscaled

    mut_dist = None
    if "mut_dist" in df.columns:
        mut_dist = df["mut_dist"].to_numpy(dtype=np.int16)

    artifact = {
        "x": x,
        "y": y,
        "y_unscaled": y_unscaled,
        "mut_dist": mut_dist,
        "split": split,
        "dataset": dataset,
        "path": str(out_path) if save else None,
        "n_samples": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "y_scaled": bool(y is not None and scale_y and y_scaler is not None),
        "y_scaler": y_scaler,
        "labels_dropped": labels_dropped,
    }

    if save:
        np.savez_compressed(
            out_path,
            x=x,
            y=np.array([] if y is None else y, dtype=np.float32),
            y_unscaled=np.array([] if y_unscaled is None else y_unscaled, dtype=np.float32),
            mut_dist=np.array([] if mut_dist is None else mut_dist, dtype=np.int16),
            dataset=np.array(dataset),
            split=np.array(split),
            y_scaled=np.array(int(y is not None and scale_y and y_scaler is not None), dtype=np.int8),
            labels_dropped=np.array(int(labels_dropped), dtype=np.int8),
            y_scaler_mean=np.array(np.nan if y_scaler is None else y_scaler["mean"], dtype=np.float32),
            y_scaler_std=np.array(np.nan if y_scaler is None else y_scaler["std"], dtype=np.float32),
        )

    return artifact


def preprocess_data(
    dataset: str,
    *,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    save: bool = True,
    overwrite: bool = False,
    drop_labels_for_unlabeled: bool = True,
    scale_y: bool = True,
) -> dict[str, dict[str, Any]]:
    _validate_dataset(dataset)

    y_scaler = None
    scaler_path = None

    if scale_y:
        train_df = load_split_dataframe(dataset, "train")
        if "label" not in train_df.columns:
            raise ValueError(f"Train split for dataset '{dataset}' does not contain a 'label' column.")

        y_train = train_df["label"].to_numpy(dtype=np.float32)
        y_scaler = fit_y_scaler(y_train)

        if save:
            scaler_path = save_y_scaler(dataset, y_scaler)

    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        artifact = preprocess_split(
            dataset,
            split,
            y_scaler=y_scaler,
            scale_y=scale_y,
            save=save,
            overwrite=overwrite,
            drop_labels_for_unlabeled=drop_labels_for_unlabeled,
        )
        artifact["y_scaler_path"] = scaler_path
        out[split] = artifact

    return out


def preprocess_all_data(
    *,
    datasets: tuple[str, ...] = ("aav", "gfp", "tfbind8"),
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    save: bool = True,
    overwrite: bool = False,
    drop_labels_for_unlabeled: bool = True,
    scale_y: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in datasets:
        out[dataset] = preprocess_data(
            dataset,
            splits=splits,
            save=save,
            overwrite=overwrite,
            drop_labels_for_unlabeled=drop_labels_for_unlabeled,
            scale_y=scale_y,
        )
    return out


if __name__ == "__main__":
    preprocess_all_data()


def load_preprocessed_split(dataset: str, split: str) -> dict[str, Any]:
    _validate_dataset(dataset)

    path = _preprocessed_npz_path(dataset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed split not found: {path}. "
            f"Run preprocess_data(...) first."
        )

    data = np.load(path, allow_pickle=False)

    x = data["x"]

    y = data["y"] if "y" in data.files else np.array([])
    if y.size == 0:
        y = None

    y_unscaled = data["y_unscaled"] if "y_unscaled" in data.files else np.array([])
    if y_unscaled.size == 0:
        y_unscaled = None

    mut_dist = data["mut_dist"] if "mut_dist" in data.files else np.array([])
    if mut_dist.size == 0:
        mut_dist = None

    y_scaled = False
    if "y_scaled" in data.files:
        y_scaled = bool(int(data["y_scaled"]))

    labels_dropped = False
    if "labels_dropped" in data.files:
        labels_dropped = bool(int(data["labels_dropped"]))

    scaler = None
    if "y_scaler_mean" in data.files and "y_scaler_std" in data.files:
        mean = float(data["y_scaler_mean"])
        std = float(data["y_scaler_std"])
        if not np.isnan(mean) and not np.isnan(std):
            scaler = {
                "type": "standard",
                "mean": mean,
                "std": std,
            }

    dataset_name = str(data["dataset"]) if "dataset" in data.files else dataset
    split_name = str(data["split"]) if "split" in data.files else split

    return {
        "x": x,
        "y": y,
        "y_unscaled": y_unscaled,
        "mut_dist": mut_dist,
        "dataset": dataset_name,
        "split": split_name,
        "n_samples": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "y_scaled": y_scaled,
        "y_scaler": scaler,
        "labels_dropped": labels_dropped,
    }
