from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bioseq_ood.data.datasets import SequenceDataModule


@dataclass(frozen=True)
class MethodSpec:
    name: str
    target_setting: str
    include_target_unlabeled: bool
    target_split_files: list[str]
    allow_test_as_target: bool = False
    target_drop_last: bool = True

    def build_train_dataloaders(self, data_module: SequenceDataModule):
        return data_module.train_dataloader()

    def to_metadata(self, data_module: SequenceDataModule | None = None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "method_name": self.name,
            "target_setting": self.target_setting,
            "include_target_unlabeled": self.include_target_unlabeled,
            "target_split_files": self.target_split_files,
            "allow_test_as_target": self.allow_test_as_target,
            "target_drop_last": self.target_drop_last,
        }
        if data_module is not None and data_module.bundle is not None and data_module.bundle.target_unlabeled is not None:
            meta["target_n_samples"] = int(data_module.bundle.target_unlabeled["n_samples"])
            meta["target_labels_ignored"] = True
        else:
            meta["target_n_samples"] = None
            meta["target_labels_ignored"] = False
        return meta
