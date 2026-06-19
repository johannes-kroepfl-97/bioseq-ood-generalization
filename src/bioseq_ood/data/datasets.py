from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bioseq_ood.data.load_data import _get_data_root
from bioseq_ood.data.preprocess_data import (
    DATASET_TYPE,
    encode_dataframe,
    load_preprocessed_split,
    load_split_dataframe,
    load_y_scaler,
    remove_static_aav_area,
)

VOCAB_SIZE_BY_DATASET_TYPE = {
    "dna": 4,
    "protein": 20,
}


def _stable_split_seed(base_seed: int, split_name: str) -> int:
    """Create a deterministic split-specific seed independent of Python's hash randomization."""
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(split_name)))
    return int(base_seed) + offset


def _maybe_subsample_artifact(
    artifact: dict[str, Any] | None,
    *,
    split_name: str,
    debug_enabled: bool,
    max_samples_per_split: int | None,
    seed: int,
) -> dict[str, Any] | None:
    """Return a randomly subsampled copy of a split artifact for fast debug runs.
    The function preserves alignment between x, y, and mut_dist and annotates the
    returned artifact with debug metadata. It is intentionally applied in the data
    layer so every orchestrator sees the same behavior.
    """
    if artifact is None or not debug_enabled or max_samples_per_split is None:
        return artifact

    max_samples = int(max_samples_per_split)
    if max_samples <= 0:
        raise ValueError("debug.max_samples_per_split must be a positive integer.")

    x = np.asarray(artifact["x"])
    n_samples = int(x.shape[0])
    if n_samples <= max_samples:
        return artifact

    rng = np.random.default_rng(_stable_split_seed(seed, split_name))
    indices = np.sort(rng.choice(n_samples, size=max_samples, replace=False))

    out = dict(artifact)
    out["x"] = x[indices]
    if artifact.get("y") is not None:
        out["y"] = np.asarray(artifact["y"])[indices]
    if artifact.get("mut_dist") is not None:
        out["mut_dist"] = np.asarray(artifact["mut_dist"])[indices]

    out["n_samples_original"] = n_samples
    out["n_samples"] = max_samples
    out["debug_subsampled"] = True
    out["debug_max_samples_per_split"] = max_samples
    out["debug_seed"] = int(seed)
    out["debug_split_name"] = str(split_name)
    return out


def _encode_raw_sequences_if_needed(x: np.ndarray, dataset_name: str) -> np.ndarray:
    """Return integer-encoded sequence arrays.
    This is deliberately defensive because older cached files may contain raw
    sequence strings instead of integer arrays. New preprocessing should already
    write integer arrays, but the dataset layer should fail gracefully while the
    project is still being refactored.
    """
    arr = np.asarray(x)

    # Already numeric integer encoding, normal path.
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)

    # Already one-hot or floating features, keep as-is.
    if np.issubdtype(arr.dtype, np.floating):
        return arr

    # Object/string arrays from stale caches or direct CSV loading.
    from bioseq_ood.data.preprocess_data import encode_sequence

    if arr.ndim == 1:
        encoded = [encode_sequence(str(seq), dataset_name) for seq in arr.tolist()]
        return np.asarray(encoded, dtype=np.int64)

    if arr.ndim == 2 and arr.shape[1] == 1:
        encoded = [encode_sequence(str(seq[0]), dataset_name) for seq in arr.tolist()]
        return np.asarray(encoded, dtype=np.int64)

    raise ValueError(
        f"Could not interpret x with dtype={arr.dtype} and shape={arr.shape}. "
        "Expected integer-encoded sequences, one-hot tensors, or raw sequence strings."
    )


class SequenceRegressionDataset(Dataset):
    """Tensor dataset for sequence regression.
    `input_encoding="one_hot"` moves deterministic one-hot construction out of
    model.forward(). The one-hot tensor is materialized once when the dataset is
    constructed, not on every forward pass. Use `input_encoding="integer"` if
    memory pressure is more important than speed.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None,
        *,
        dataset_name: str,
        mut_dist: np.ndarray | None = None,
        input_encoding: str = "one_hot",
        vocab_size: int | None = None,
        return_mut_dist: bool = False,
    ) -> None:
        input_encoding = str(input_encoding).lower()
        if input_encoding not in {"integer", "one_hot"}:
            raise ValueError("input_encoding must be either 'integer' or 'one_hot'.")

        x = _encode_raw_sequences_if_needed(x, dataset_name)
        x_tensor = torch.as_tensor(x)

        if x_tensor.ndim == 3:
            # Already one-hot or continuous feature tensor.
            self.x = x_tensor.to(torch.float32)
        elif input_encoding == "one_hot":
            if vocab_size is None:
                raise ValueError("vocab_size is required when input_encoding='one_hot'.")
            x_tensor = x_tensor.to(torch.long)
            self.x = F.one_hot(x_tensor, num_classes=int(vocab_size)).to(torch.float32)
        else:
            self.x = x_tensor.to(torch.long)

        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
        self.mut_dist = None if mut_dist is None else torch.as_tensor(mut_dist, dtype=torch.long).view(-1)
        self.input_encoding = input_encoding
        self.return_mut_dist = bool(return_mut_dist)

        if self.y is not None and self.y.shape[0] != self.x.shape[0]:
            raise ValueError(f"x/y length mismatch: {self.x.shape[0]} vs {self.y.shape[0]}")
        if self.mut_dist is not None and self.mut_dist.shape[0] != self.x.shape[0]:
            raise ValueError(f"x/mut_dist length mismatch: {self.x.shape[0]} vs {self.mut_dist.shape[0]}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        if self.y is None:
            if self.return_mut_dist and self.mut_dist is not None:
                return self.x[index], self.mut_dist[index]
            return self.x[index]

        if self.return_mut_dist and self.mut_dist is not None:
            return self.x[index], self.y[index], self.mut_dist[index]
        return self.x[index], self.y[index]


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_type: str
    vocab_size: int
    seq_len: int
    y_scaler: dict[str, float] | None
    train: dict[str, Any]
    val_id: dict[str, Any]
    val_ood: dict[str, Any]
    target_close: dict[str, Any] | None = None
    target_test: dict[str, Any] | None = None
    test: dict[str, Any] | None = None
    target_unlabeled: dict[str, Any] | None = None
    # Paper-notation pools used by the protocol-based orchestrator
    # (pipeline_phases.py). Populated only when their preprocessed .npz files
    # exist on disk; missing pools are not an error so the legacy notebook
    # pipeline (which uses target_close / target_test / test) still works.
    T_close: dict[str, Any] | None = None    # labelled half of B_close
    U_close: dict[str, Any] | None = None    # unlabelled half of B_close
    T_far:   dict[str, Any] | None = None    # labelled half of B_far  (= old test)
    U_far:   dict[str, Any] | None = None    # unlabelled half of B_far (= old target_test)


class SequenceDataModule:
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = True,
        include_test: bool = False,
        include_target_unlabeled: bool = False,
        include_eval_targets: bool = True,
        target_split_files: list[str] | None = None,
        allow_test_as_target: bool = False,
        target_drop_last: bool = True,
        input_encoding: str = "one_hot",
        debug_enabled: bool = False,
        debug_max_samples_per_split: int | None = None,
        debug_seed: int = 42,
    ) -> None:
        if dataset_name not in DATASET_TYPE:
            valid = ", ".join(sorted(DATASET_TYPE))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of: {valid}")
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.include_test = include_test
        self.include_target_unlabeled = include_target_unlabeled
        self.include_eval_targets = include_eval_targets
        self.target_split_files = target_split_files or ["target_unlabeled.csv"]
        self.allow_test_as_target = allow_test_as_target
        self.target_drop_last = target_drop_last
        self.input_encoding = str(input_encoding).lower()
        self.debug_enabled = bool(debug_enabled)
        self.debug_max_samples_per_split = debug_max_samples_per_split
        self.debug_seed = int(debug_seed)
        self.bundle: DatasetBundle | None = None
        self.train_dataset: SequenceRegressionDataset | None = None
        self.val_id_dataset: SequenceRegressionDataset | None = None
        self.val_ood_dataset: SequenceRegressionDataset | None = None
        self.test_dataset: SequenceRegressionDataset | None = None
        self.target_unlabeled_dataset: SequenceRegressionDataset | None = None

    def _maybe_debug_subsample(self, artifact: dict[str, Any] | None, split_name: str) -> dict[str, Any] | None:
        return _maybe_subsample_artifact(
            artifact,
            split_name=split_name,
            debug_enabled=self.debug_enabled,
            max_samples_per_split=self.debug_max_samples_per_split,
            seed=self.debug_seed,
        )

    def _make_dataset(
        self,
        artifact: dict[str, Any],
        *,
        labels: bool,
        return_mut_dist: bool = False,
    ) -> SequenceRegressionDataset:
        assert self.bundle is not None
        return SequenceRegressionDataset(
            artifact["x"],
            artifact.get("y") if labels else None,
            dataset_name=self.dataset_name,
            mut_dist=artifact.get("mut_dist"),
            input_encoding=self.input_encoding,
            vocab_size=self.bundle.vocab_size,
            return_mut_dist=return_mut_dist,
        )

    def _load_optional_preprocessed_split(self, split: str) -> dict[str, Any] | None:
        try:
            return load_preprocessed_split(self.dataset_name, split)
        except FileNotFoundError:
            return None

    def _load_target_adaptation_arrays(self, *, keep_labels: bool = False) -> dict[str, Any]:
        """Load unlabeled target adaptation inputs from one or more split CSV/NPZ files.

        For the normal training/adaptation path (``keep_labels=False``) labels are
        intentionally discarded, even if available in the CSV or cached NPZ, so the
        adaptation data is genuinely unlabeled. Test leakage is blocked unless
        ``allow_test_as_target=True``.

        ``keep_labels=True`` is used ONLY for post-hoc MC-dropout quality diagnostics.
        """
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        mut_dists: list[np.ndarray] = []
        names: list[str] = []
        all_have_labels = True

        for file_name in self.target_split_files:
            name = Path(file_name).name
            split = name[:-4] if name.endswith(".csv") else name

            if split == "test" and not self.allow_test_as_target:
                raise ValueError(
                    "Refusing to use test.csv as target adaptation data by default. "
                    "Set training.allow_test_as_target=true only for explicit ablation experiments."
                )

            y: np.ndarray | None = None
            try:
                artifact = load_preprocessed_split(self.dataset_name, split)
                x = artifact["x"]
                mut_dist = artifact.get("mut_dist")
                if keep_labels:
                    y = artifact.get("y")
            except FileNotFoundError:
                df = load_split_dataframe(self.dataset_name, split)
                x = encode_dataframe(df, self.dataset_name)
                x = remove_static_aav_area(x) if self.dataset_name == "aav" else x
                mut_dist = df["mut_dist"].to_numpy(dtype=np.int16) if "mut_dist" in df.columns else None
                # CSV fallback does not carry a scaled label column; diagnostics
                # gracefully degrade to "no labels" in that rare path.

            xs.append(np.asarray(x))
            if mut_dist is not None:
                mut_dists.append(np.asarray(mut_dist))
            if keep_labels:
                if y is None:
                    all_have_labels = False
                else:
                    ys.append(np.asarray(y).reshape(-1))
            names.append(name)

        if not xs:
            raise ValueError("No target adaptation split files were provided.")

        x_all = np.concatenate(xs, axis=0) if len(xs) > 1 else xs[0]
        mut_dist_all = None
        if len(mut_dists) == len(xs):
            mut_dist_all = np.concatenate(mut_dists, axis=0) if len(mut_dists) > 1 else mut_dists[0]

        y_all: np.ndarray | None = None
        if keep_labels and all_have_labels and len(ys) == len(xs):
            y_all = np.concatenate(ys, axis=0) if len(ys) > 1 else ys[0]

        return {
            "x": x_all,
            "y": y_all,
            "mut_dist": mut_dist_all,
            "dataset": self.dataset_name,
            "split": "+".join(names),
            "n_samples": int(x_all.shape[0]),
            "seq_len": int(x_all.shape[1]),
            "target_split_files": names,
            "labels_ignored": not keep_labels,
        }

    def target_unlabeled_true_labels(self) -> np.ndarray | None:
        """Return the true (scaled) target labels aligned with `target_unlabeled_dataset`

        FOR POST-HOC DIAGNOSTICS ONLY. These labels are deliberately withheld from
        training and model selection; they are surfaced here purely so MC-dropout
        uncertainty can be checked against realized prediction error.
        """
        if not self.include_target_unlabeled:
            return None
        artifact = self._load_target_adaptation_arrays(keep_labels=True)
        if artifact.get("y") is None:
            return None
        artifact = self._maybe_debug_subsample(artifact, "target_unlabeled")
        y = None if artifact is None else artifact.get("y")
        return None if y is None else np.asarray(y).reshape(-1)

    def setup(self) -> None:
        dataset_type = DATASET_TYPE[self.dataset_name]
        vocab_size = VOCAB_SIZE_BY_DATASET_TYPE[dataset_type]
        train = self._maybe_debug_subsample(load_preprocessed_split(self.dataset_name, "train"), "train")
        val_id = self._maybe_debug_subsample(load_preprocessed_split(self.dataset_name, "val_id"), "val_id")
        val_ood = self._maybe_debug_subsample(load_preprocessed_split(self.dataset_name, "val_ood"), "val_ood")
        test = self._maybe_debug_subsample(load_preprocessed_split(self.dataset_name, "test"), "test") if self.include_test else None
        target_close = self._maybe_debug_subsample(self._load_optional_preprocessed_split("target_close"), "target_close") if self.include_eval_targets else None
        target_test = self._maybe_debug_subsample(self._load_optional_preprocessed_split("target_test"), "target_test") if self.include_eval_targets else None
        # Paper-notation pools used by the protocol-based orchestrator. Loaded if
        # present; missing files are not an error so the legacy pipeline still works.
        T_close = self._maybe_debug_subsample(self._load_optional_preprocessed_split("T_close"), "T_close") if self.include_eval_targets else None
        U_close = self._maybe_debug_subsample(self._load_optional_preprocessed_split("U_close"), "U_close") if self.include_eval_targets else None
        T_far   = self._maybe_debug_subsample(self._load_optional_preprocessed_split("T_far"),   "T_far")   if self.include_eval_targets else None
        U_far   = self._maybe_debug_subsample(self._load_optional_preprocessed_split("U_far"),   "U_far")   if self.include_eval_targets else None
        target_unlabeled = self._maybe_debug_subsample(self._load_target_adaptation_arrays(), "target_unlabeled") if self.include_target_unlabeled else None

        try:
            y_scaler = load_y_scaler(self.dataset_name)
        except FileNotFoundError:
            y_scaler = None

        self.bundle = DatasetBundle(
            dataset_name=self.dataset_name,
            dataset_type=dataset_type,
            vocab_size=vocab_size,
            seq_len=int(train["seq_len"]),
            y_scaler=y_scaler,
            train=train,
            val_id=val_id,
            val_ood=val_ood,
            target_close=target_close,
            target_test=target_test,
            test=test,
            target_unlabeled=target_unlabeled,
            T_close=T_close,
            U_close=U_close,
            T_far=T_far,
            U_far=U_far,
        )

        self.train_dataset = self._make_dataset(train, labels=True)
        self.val_id_dataset = self._make_dataset(val_id, labels=True)
        self.val_ood_dataset = self._make_dataset(val_ood, labels=True)
        self.test_dataset = None if test is None else self._make_dataset(test, labels=True)
        self.target_unlabeled_dataset = None if target_unlabeled is None else self._make_dataset(target_unlabeled, labels=False)

    def get_split_artifact(self, split_name: str) -> dict[str, Any] | None:
        if self.bundle is None:
            raise RuntimeError("Call setup() before requesting split artifacts.")
        return getattr(self.bundle, split_name, None)

    def _loader(self, dataset: SequenceRegressionDataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        # A singleton trailing batch breaks BatchNorm in train mode ("Expected more
        # than 1 value per channel"). The shuffled (training) loaders therefore drop
        # a lone last sample; eval loaders run BN in eval mode and are unaffected.
        if shuffle and (len(dataset) % self.batch_size) == 1:
            drop_last = True
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(
            self.train_dataset,
            shuffle=True,
            drop_last=self.include_target_unlabeled and self.target_drop_last,
        )

    def val_id_dataloader(self) -> DataLoader:
        if self.val_id_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.val_id_dataset, shuffle=False)

    def val_ood_dataloader(self) -> DataLoader:
        if self.val_ood_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return self._loader(self.val_ood_dataset, shuffle=False)

    def labeled_dataloader(self, split_name: str, shuffle: bool = False) -> DataLoader:
        """Build a labeled (x, y) loader for any loaded split.

        Used for oracle model selection, where a labeled far-OOD split (e.g.
        `target_test`) is monitored during fit to pick the checkpoint. The labels
        are read only for the selection metric; the model never trains on them.
        """
        if self.bundle is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        artifact = self.get_split_artifact(split_name)
        if artifact is None or artifact.get("y") is None:
            raise RuntimeError(
                f"Split '{split_name}' is not available with labels. "
                "Ensure include_eval_targets/include_test cover the selection split."
            )
        dataset = self._make_dataset(artifact, labels=True)
        return self._loader(dataset, shuffle=shuffle)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test split was not loaded.")
        return self._loader(self.test_dataset, shuffle=False)

    def target_dataloader(self) -> DataLoader:
        if self.target_unlabeled_dataset is None:
            raise RuntimeError("Target unlabeled split was not loaded.")
        return self._loader(
            self.target_unlabeled_dataset,
            shuffle=True,
            drop_last=self.target_drop_last,
        )


def get_data_root() -> Path:
    return _get_data_root()
