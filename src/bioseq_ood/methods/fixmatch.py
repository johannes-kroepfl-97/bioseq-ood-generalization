from __future__ import annotations

from bioseq_ood.data.datasets import SequenceDataModule

from .base import MethodSpec


class FixMatchMethod(MethodSpec):
    """FixMatch-style semi-supervised regression method.

    Classical FixMatch is a classification method: it creates hard pseudo-labels
    from weakly augmented unlabeled inputs, keeps only confident predictions, and
    trains the model to reproduce those labels under strong augmentation.

    This project predicts scalar biological properties, so we use the analogous
    regression formulation: the weak target prediction is detached and used as a
    pseudo-label, the strong prediction is produced from an input-dropout view,
    and optional MC-dropout uncertainty filtering can mask low-confidence target
    examples. Target labels are never used.
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
            name="fixmatch",
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
