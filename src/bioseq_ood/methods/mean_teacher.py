from __future__ import annotations

from bioseq_ood.data.datasets import SequenceDataModule

from .base import MethodSpec


class MeanTeacherMethod(MethodSpec):
    """Mean Teacher semi-supervised regression method.

    Training uses labeled source batches and unlabeled target batches. The
    student is optimized with a supervised source loss plus a consistency loss
    against an EMA teacher on target inputs. Target labels are never used.
    """

    def __init__(
        self,
        *,
        target_split_files: list[str],
        target_setting: str,
        allow_test_as_target: bool = False,
        target_drop_last: bool = True,
    ) -> None:
        super().__init__(
            name="mean_teacher",
            target_setting=target_setting,
            include_target_unlabeled=True,
            target_split_files=target_split_files,
            allow_test_as_target=allow_test_as_target,
            target_drop_last=target_drop_last,
        )

    def build_train_dataloaders(self, data_module: SequenceDataModule):
        return {
            "source": data_module.train_dataloader(),
            "target": data_module.target_dataloader(),
        }
