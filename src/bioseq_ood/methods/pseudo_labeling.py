from __future__ import annotations

from bioseq_ood.data.datasets import SequenceDataModule

from .base import MethodSpec


class PseudoLabelingMethod(MethodSpec):
    """Two-stage pseudo-labeling method for unlabeled target data.

    Stage 1 is source-only pretraining. Stage 2 generates pseudo labels on the
    configured target split using MC dropout and trains a final model on source
    labels plus filtered pseudo-labeled target samples.
    """

    def __init__(
        self,
        *,
        target_split_files: list[str],
        target_setting: str,
        allow_test_as_target: bool = False,
        target_drop_last: bool = False,
    ) -> None:
        super().__init__(
            name="pseudo_labeling",
            target_setting=target_setting,
            include_target_unlabeled=True,
            target_split_files=target_split_files,
            allow_test_as_target=allow_test_as_target,
            target_drop_last=target_drop_last,
        )

    def build_train_dataloaders(self, data_module: SequenceDataModule):
        # The final pseudo-label training dataloader is constructed inside the
        # trainer after pseudo labels have been generated.
        return data_module.train_dataloader()
