from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bioseq_ood.training.losses import CMDLoss

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl


def _require_cfg(training_config: dict[str, Any], key: str):
    """Read a required hyperparameter, failing loudly if it was never set.

    Defaults for every method live in config/methods.yaml (applied before training
    by apply_method_hparams); the code reads with no silent fallback so a missing
    or mistyped key stops the run instead of training on a wrong value.
    """
    if key not in training_config:
        raise KeyError(f"training.{key} is required but missing; set it in config/methods.yaml")
    return training_config[key]


def set_dropout_train_bn_eval(model: nn.Module) -> None:
    """Enable stochastic dropout while keeping BatchNorm statistics frozen.

    Paper: Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
    MC dropout requires dropout to remain active at prediction time so each forward
    pass samples a different sub-network.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.train()    # dropout STOCHASTIC (sample a sub-network each pass)
        elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()     # BN FROZEN (use running stats, no extra noise)


class LightningSequenceRegressor(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        training_config: dict[str, Any],
        y_scaler: dict[str, float] | None = None,
        val_stage_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.training_config = training_config
        # val_stage_names[0] is the selection split (val_id): evaluated every epoch and
        # tracked by early stopping / checkpointing. Any further entries (e.g. T_close,
        # T_far) are logged-only OOD diagnostics, evaluated only every diag_every_n epochs.
        self.val_stage_names = list(val_stage_names) if val_stage_names else ["val_id"]
        self.diag_every_n = max(1, int(training_config.get("diag_every_n_epochs", 5)))
        self.loss_name = str(training_config.get("loss", "mse")).lower()
        self.loss_fn = nn.MSELoss() if self.loss_name == "mse" else nn.L1Loss()
        self.y_scaler = y_scaler

        # CMD (Zellinger et al., ICLR 2017), enabled only for method=cmd. The CMD
        # representation is sigmoid-bounded to [a, b] = [0, 1] in every encoder
        # (models/*.py, return_features), which is exactly the paper's bounded-
        # activation requirement and what justifies the 1/|b-a|^k moment
        # normalisation. a, b are therefore fixed to the feature bound, not tuned.
        self.use_cmd = bool(training_config.get("use_cmd", False))
        self.cmd_enabled = self.use_cmd
        self.lambda_cmd = 0.0
        self.cmd_loss_fn = None
        if self.use_cmd:
            self.lambda_cmd = float(_require_cfg(training_config, "lambda_cmd"))
            self.cmd_loss_fn = CMDLoss(n_moments=int(_require_cfg(training_config, "cmd_n_moments")), a=0.0, b=1.0)

        # Pseudo-labeling options: this mode is used only in the second stage of
        # pseudo-label training, after target labels were generated and filtered.
        self.use_pseudo_labeling = bool(training_config.get("use_pseudo_labeling", False))
        self.lambda_pseudo_max = float(training_config.get("lambda_pseudo_max", 0.3))
        self.rampup_epochs = max(1, int(training_config.get("rampup_epochs", 10)))
        self.mae_loss_fn = nn.L1Loss()

        # Mean Teacher options: The student is self.model.
        mt_cfg = training_config.get("mean_teacher", {}) if isinstance(training_config.get("mean_teacher", {}), dict) else {}
        self.use_mean_teacher = bool(training_config.get("use_mean_teacher", False))
        self.teacher_model = deepcopy(model) if self.use_mean_teacher else None
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)
        self.lambda_consistency_max = float(
            training_config.get("lambda_consistency_max", mt_cfg.get("lambda_consistency_max", 1.0))
        )
        self.consistency_rampup_epochs = max(1, int(training_config.get("consistency_rampup_epochs", mt_cfg.get("rampup_epochs", 10))))
        self.ema_decay = float(training_config.get("ema_decay", mt_cfg.get("ema_decay", 0.999)))
        self.ema_decay_warmup = float(training_config.get("ema_decay_warmup", mt_cfg.get("ema_decay_warmup", 0.99)))
        self.teacher_warmup_epochs = int(training_config.get("teacher_warmup_epochs", mt_cfg.get("teacher_warmup_epochs", self.consistency_rampup_epochs)))

        # Default 0.0: random zeroing of one-hot sequence positions is biologically
        # meaningless (test inputs never have missing positions).
        self.mt_input_dropout_p = float(training_config.get("mt_input_dropout_p", mt_cfg.get("input_dropout_p", 0.0)))
        self.mt_use_teacher_for_eval = bool(training_config.get("use_teacher_for_eval", mt_cfg.get("use_teacher_for_eval", True)))

        # the consistency cost applies to both labelled and unlabelled examples,
        # only the classification cost is gated by labels.
        self.mt_consistency_on_source = bool(
            training_config.get("mt_consistency_on_source", mt_cfg.get("consistency_on_source", True))
        )
        self.consistency_loss_fn = nn.MSELoss()

        # FixMatch adapted for regression: there is no softmax confidence, so the
        # pseudo-label is the weak-view MC-dropout mean and the gate keeps the
        # lowest-uncertainty fraction (MC-dropout std). Weak/strong image
        # augmentations are replaced by Gaussian noise on the one-hot input
        # (weak sigma 0 = clean, strong sigma > 0).
        fm_cfg = training_config.get("fixmatch", {}) if isinstance(training_config.get("fixmatch", {}), dict) else {}
        self.use_fixmatch = bool(training_config.get("use_fixmatch", False))
        # Ramped weight (sigmoid, like Mean Teacher). The paper keeps lambda fixed
        # because its confidence threshold provides an implicit curriculum; the
        # fixed-fraction gate used here does not, so the ramp restores it.
        self.lambda_fixmatch_max = float(
            training_config.get("lambda_fixmatch_max", fm_cfg.get("lambda_fixmatch_max", 1.0))
        )
        self.fixmatch_rampup_epochs = max(1, int(
            training_config.get("fixmatch_rampup_epochs", fm_cfg.get("rampup_epochs", 10))
        ))
        # Augmentation using noise replaces input dropout:
        #   weak σ default 0.0 for clean target gives the highest-quality pseudo-label.
        #   strong σ default 0.1 for meaningful perturbation, preserves the qualitative weak/strong gap the paper relies on.
        self.fm_weak_noise_sigma = float(
            training_config.get("fm_weak_noise_sigma", fm_cfg.get("weak_noise_sigma", 0.0))
        )
        self.fm_strong_noise_sigma = float(
            training_config.get("fm_strong_noise_sigma", fm_cfg.get("strong_noise_sigma", 0.1))
        )
        # T stochastic MC-dropout passes for the weak forward (same idea + name as
        # pseudo_labeling's mc_passes).
        self.fm_mc_passes = max(1, int(
            training_config.get("fm_mc_passes", fm_cfg.get("mc_passes", 10))
        ))
        # Scale-free quantile gate: keep the lowest-uncertainty keep_ratio fraction
        # (same idea + name as pseudo_labeling's keep_ratio).
        self.fm_keep_ratio = float(
            training_config.get("fm_keep_ratio", fm_cfg.get("keep_ratio", 0.5))
        )
        if not (0.0 < self.fm_keep_ratio <= 1.0):
            raise ValueError("fixmatch keep_ratio must be in (0, 1].")
        raw_uncertainty_threshold = training_config.get(
            "fm_uncertainty_threshold",
            fm_cfg.get("uncertainty_threshold", None),
        )
        self.fm_uncertainty_threshold = None if raw_uncertainty_threshold is None else float(raw_uncertainty_threshold)
        self.fm_loss_fn = nn.MSELoss(reduction="none")

        # Saved in checkpoints and visible in MLflow hparams through the full config.
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_inference_model()(x)

    def get_inference_model(self) -> nn.Module:
        if self.use_mean_teacher and self.mt_use_teacher_for_eval and self.teacher_model is not None:
            return self.teacher_model
        return self.model

    def _to_model_input_with_dropout_p(self, x: torch.Tensor, dropout_p: float) -> torch.Tensor:
        """Apply explicit input dropout while preserving model-compatible shape."""
        if dropout_p <= 0.0:
            return x
        if x.ndim == 2 and not torch.is_floating_point(x):
            vocab_size = int(getattr(self.model, "vocab_size"))
            x = F.one_hot(x.long(), num_classes=vocab_size).to(torch.float32)
        else:
            x = x.to(torch.float32) if torch.is_floating_point(x) else x
        return F.dropout(x, p=float(dropout_p), training=self.training)

    def _to_model_input_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Backward-compatible Mean Teacher input dropout helper."""
        return self._to_model_input_with_dropout_p(x, self.mt_input_dropout_p)

    def _to_model_input_with_gaussian_noise(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add Gaussian noise to a one-hot input sequence friendly augmentation.
        Used for FixMatch in place of input dropout.
        """
        if sigma <= 0.0:
            return x
        if x.ndim == 2 and not torch.is_floating_point(x):
            vocab_size = int(getattr(self.model, "vocab_size"))
            x = F.one_hot(x.long(), num_classes=vocab_size).to(torch.float32)
        else:
            x = x.to(torch.float32) if torch.is_floating_point(x) else x
        if not self.training:
            return x
        return x + torch.randn_like(x) * float(sigma)

    def _fixmatch_weak_prediction_and_uncertainty(self, target_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the weak-view pseudo target and optional MC-dropout uncertainty. """
        if self.fm_mc_passes <= 1:
            # Single weak-view prediction q_b = f(α(u_b)); no uncertainty → no gate.
            weak_x = self._to_model_input_with_gaussian_noise(target_x, self.fm_weak_noise_sigma)
            return self.model(weak_x), None

        # MC dropout: T = fm_mc_passes stochastic weak-view passes. Freeze BatchNorm
        # (eval, running stats) for the passes so the uncertainty is pure dropout
        # (epistemic) noise rather than BN batch-statistic noise, and target batches
        # do not perturb BN running stats -- matching trainer._predict_mc_dropout.
        # The subsequent strong-view forward needs BN in train mode, so restore it.
        was_training = self.model.training
        set_dropout_train_bn_eval(self.model)
        try:
            preds = []
            for _ in range(self.fm_mc_passes):
                weak_x = self._to_model_input_with_gaussian_noise(target_x, self.fm_weak_noise_sigma)
                preds.append(self.model(weak_x))
        finally:
            self.model.train(was_training)
        stacked = torch.stack(preds, dim=0) # (T, batch, 1)
        return stacked.mean(dim=0), stacked.std(dim=0).view(stacked.shape[1], -1).mean(dim=1)

    def _consistency_lambda(self) -> float:
        t = min(float(self.current_epoch) / float(self.consistency_rampup_epochs), 1.0)
        return self.lambda_consistency_max * float(math.exp(-5.0 * (1.0 - t) ** 2))

    def _fixmatch_lambda(self) -> float:
        # Same sigmoid ramp as Mean Teacher: restores the curriculum the paper gets
        # from its confidence threshold but our fixed-fraction gate does not.
        t = min(float(self.current_epoch) / float(self.fixmatch_rampup_epochs), 1.0)
        return self.lambda_fixmatch_max * float(math.exp(-5.0 * (1.0 - t) ** 2))

    def _current_ema_decay(self) -> float:
        if self.current_epoch < self.teacher_warmup_epochs:
            return self.ema_decay_warmup
        return self.ema_decay

    @torch.no_grad()
    def _update_teacher(self) -> None:
        if not self.use_mean_teacher or self.teacher_model is None:
            return
        decay = self._current_ema_decay()                              # α
        student_params = dict(self.model.named_parameters())
        for name, teacher_param in self.teacher_model.named_parameters():
            student_param = student_params[name]
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)

        student_buffers = dict(self.model.named_buffers())
        for name, teacher_buffer in self.teacher_model.named_buffers():
            if name in student_buffers:
                teacher_buffer.copy_(student_buffers[name])

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self._update_teacher()

    def _compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.y_scaler is not None:
            mean = torch.tensor(self.y_scaler["mean"], dtype=preds.dtype, device=preds.device)
            std = torch.tensor(self.y_scaler["std"], dtype=preds.dtype, device=preds.device)
            preds_eval = preds * std + mean
            targets_eval = targets * std + mean
        else:
            preds_eval = preds
            targets_eval = targets
        mae = torch.mean(torch.abs(preds_eval - targets_eval))
        return preds_eval, mae

    def _shared_eval_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        preds = self.get_inference_model()(x)
        loss = self.loss_fn(preds, y)
        _, mae = self._compute_metrics(preds, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "train"), add_dataloader_idx=False)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        return loss

    def _pseudo_lambda(self) -> float:
        return self.lambda_pseudo_max * min(1.0, float(self.current_epoch) / float(self.rampup_epochs))

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.use_fixmatch:

            source_x, source_y = batch["source"]
            target_x = batch["target"]
            if isinstance(target_x, (tuple, list)):
                target_x = target_x[0]

            source_pred = self.model(source_x)
            source_loss = self.loss_fn(source_pred, source_y)

            with torch.no_grad():
                weak_pred, uncertainty = self._fixmatch_weak_prediction_and_uncertainty(target_x)
                pseudo_target = weak_pred.detach()

            strong_target_x = self._to_model_input_with_gaussian_noise(target_x, self.fm_strong_noise_sigma)
            strong_pred = self.model(strong_target_x)

            per_example_loss = self.fm_loss_fn(strong_pred, pseudo_target).view(strong_pred.shape[0], -1).mean(dim=1)

            n_unlabeled = per_example_loss.numel()
            if uncertainty is None:
                mask = torch.ones_like(per_example_loss)
            elif self.fm_uncertainty_threshold is not None:
                mask = (uncertainty <= self.fm_uncertainty_threshold).to(per_example_loss.dtype)
            elif self.fm_keep_ratio < 1.0:
                n_keep = max(1, int(math.ceil(n_unlabeled * self.fm_keep_ratio)))
                # torch.kthvalue has no deterministic CUDA kernel, so under
                # deterministic=True it raises. `uncertainty` is a small 1-D tensor
                # (one value per unlabeled example), so computing the threshold on
                # CPU is cheap and keeps the run reproducible.
                kth = torch.kthvalue(uncertainty.detach().cpu(), n_keep).values.to(uncertainty.device)
                mask = (uncertainty <= kth).to(per_example_loss.dtype)
            else:
                mask = torch.ones_like(per_example_loss)

            fixmatch_loss = (per_example_loss * mask).sum() / max(float(n_unlabeled), 1.0)

            lambda_fixmatch = self._fixmatch_lambda()   # ramped (see _fixmatch_lambda)
            loss = source_loss + lambda_fixmatch * fixmatch_loss

            self.log("train_source_loss", source_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_fixmatch_loss", fixmatch_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_lambda_fixmatch", lambda_fixmatch, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_fixmatch_mask_fraction", mask.mean(), on_step=False, on_epoch=True, prog_bar=False)
            if uncertainty is not None:
                self.log("train_fixmatch_uncertainty", uncertainty.mean(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_loss_total", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        if self.use_mean_teacher:
            source_x, source_y = batch["source"]
            target_x = batch["target"]
            if isinstance(target_x, (tuple, list)):
                target_x = target_x[0]

            source_pred = self.model(source_x)
            source_loss = self.loss_fn(source_pred, source_y)

            target_student_pred = self.model(
                self._to_model_input_with_dropout(target_x)
            )

            assert self.teacher_model is not None

            set_dropout_train_bn_eval(self.teacher_model)

            with torch.no_grad():
                target_teacher_pred = self.teacher_model(
                    self._to_model_input_with_dropout(target_x)
                )

            target_consistency_loss = self.consistency_loss_fn(
                target_student_pred,
                target_teacher_pred.detach()
            )

            if self.mt_consistency_on_source:
                with torch.no_grad():
                    source_teacher_pred = self.teacher_model(
                        self._to_model_input_with_dropout(source_x)
                    )
                source_consistency_loss = self.consistency_loss_fn(
                    source_pred,
                    source_teacher_pred.detach(),
                )
                consistency_loss = 0.5 * (target_consistency_loss + source_consistency_loss)
            else:
                consistency_loss = target_consistency_loss
                source_consistency_loss = None

            # λ(t): ramped consistency weight (see _consistency_lambda).
            lambda_consistency = self._consistency_lambda()
            loss = source_loss + lambda_consistency * consistency_loss

            self.log("train_source_loss", source_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_consistency_loss", consistency_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_target_consistency_loss", target_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
            if source_consistency_loss is not None:
                self.log("train_source_consistency_loss", source_consistency_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_lambda_consistency", lambda_consistency, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_ema_decay", self._current_ema_decay(), on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_loss_total", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        if self.use_pseudo_labeling:
            source_x, source_y = batch["source"]
            pseudo_x, pseudo_y = batch["pseudo"]

            source_pred = self.model(source_x)
            pseudo_pred = self.model(pseudo_x)

            # Supervised loss uses the shared objective (self.loss_fn) like every other
            # method, so the comparison is not biased by the supervised term. The pseudo
            # term stays MAE, which is robust to noisy pseudo-labels.
            source_loss = self.loss_fn(source_pred, source_y)
            pseudo_loss = self.mae_loss_fn(pseudo_pred, pseudo_y)
            lambda_pseudo = self._pseudo_lambda()
            loss = source_loss + lambda_pseudo * pseudo_loss

            self.log("train_source_loss", source_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_pseudo_mae_loss", pseudo_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_lambda_pseudo", lambda_pseudo, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("train_loss_total", loss, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        if not self.cmd_enabled:
            return self._shared_eval_step(batch, stage="train")

        source_x, source_y = batch["source"]
        target_x = batch["target"]
        if isinstance(target_x, (tuple, list)):
            target_x = target_x[0]

        source_pred, source_z = self.model(source_x, return_features=True)

        # Important: no torch.no_grad() here. CMD gradients must flow through target_z into the shared encoder.
        _, target_z = self.model(target_x, return_features=True) # z_t = features(X_t)

        loss_pred = self.loss_fn(source_pred, source_y) # L_pred: supervised source loss
        loss_cmd = self.cmd_loss_fn(source_z, target_z) # CMD(z_s, z_t): distribution match
        loss = loss_pred + self.lambda_cmd * loss_cmd # total

        self.log("train_loss_pred", loss_pred, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_cmd", loss_cmd, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_total", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor | None:
        # dataloader_idx 0 is the selection split (val_id): always evaluated, so early
        # stopping / checkpointing have a value every epoch. The extra OOD diagnostics
        # (T_close/T_far, idx >= 1) are skipped except every diag_every_n epochs -- this
        # avoids their forward passes on the other epochs while still giving a curve.
        if dataloader_idx != 0 and (self.current_epoch % self.diag_every_n) != 0:
            return None
        if dataloader_idx < len(self.val_stage_names):
            stage = self.val_stage_names[dataloader_idx]
        else:
            stage = f"val_{dataloader_idx}"
        return self._shared_eval_step(batch, stage=stage)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_eval_step(batch, stage="test")

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.training_config["learning_rate"]),
            weight_decay=float(self.training_config.get("weight_decay", 0.0)),
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(self.training_config.get("lr_scheduler_factor", 0.5)),
            patience=int(self.training_config.get("lr_scheduler_patience", 5)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_id_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
