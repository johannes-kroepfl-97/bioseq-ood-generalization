from __future__ import annotations

from bioseq_ood.data.datasets import SequenceDataModule

from .base import MethodSpec


class AdaBNMethod(MethodSpec):
    """Adaptive Batch Normalization (Li et al., 2018, Algorithm 1).

    Parameter-free: there are no hyperparameters to tune. Training is plain
    source-supervised ERM; afterwards the trainer recomputes each BatchNorm layer's
    running mean/variance on the unlabeled target inputs (see _apply_adabn, which
    uses momentum=None for the exact target statistics, not an EMA) while leaving
    the learned scale/shift (gamma, beta) and all other weights unchanged.

    Requires BatchNorm layers to exist, so AdaBN is run with a BN-inducing
    normalization_strategy (see methods.hparams.adabn_norm_strategy). On an
    architecture without BatchNorm it would be a no-op (equivalent to ERM).
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
            name="adabn",
            target_setting=target_setting,
            include_target_unlabeled=True,
            target_split_files=target_split_files,
            allow_test_as_target=allow_test_as_target,
            target_drop_last=target_drop_last,
        )

    def build_train_dataloaders(self, data_module: SequenceDataModule):
        # AdaBN does not change the supervised training loop. Target data is used
        # only after fitting to update BatchNorm statistics.
        return data_module.train_dataloader()
