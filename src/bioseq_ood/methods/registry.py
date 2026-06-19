from __future__ import annotations

from typing import Any

from .adabn import AdaBNMethod
from .base import MethodSpec
from .cmd import CMDMethod
from .erm import ERMMethod
from .fixmatch import FixMatchMethod
from .mean_teacher import MeanTeacherMethod
from .pseudo_labeling import PseudoLabelingMethod


def _infer_target_setting(target_split_files: list[str]) -> str:
    joined = "+".join(target_split_files)
    if "target_close" in joined:
        return "target_close"
    if "target_test" in joined:
        return "target_test"
    if "target_unlabeled" in joined:
        return "target_unlabeled"
    if "test" in joined:
        return "test_ablation"
    return joined or "none"


def build_method_from_config(training_cfg: dict[str, Any]) -> MethodSpec:
    explicit_method = str(training_cfg.get("method", "")).strip().lower()
    use_cmd = bool(training_cfg.get("use_cmd", False))
    lambda_cmd = float(training_cfg.get("lambda_cmd", 0.0))

    if explicit_method in {"", "erm", "supervised", "source_only"} and not (use_cmd and lambda_cmd > 0.0):
        return ERMMethod()

    if explicit_method == "cmd" or (use_cmd and lambda_cmd > 0.0):
        target_split_files = list(training_cfg.get("target_split_files", training_cfg.get("cmd_target_split_files", ["target_close.csv"])))
        target_setting = str(training_cfg.get("target_setting", _infer_target_setting(target_split_files)))
        return CMDMethod(
            target_split_files=target_split_files,
            target_setting=target_setting,
            allow_test_as_target=bool(training_cfg.get("allow_test_as_target", training_cfg.get("cmd_allow_test_as_target", False))),
            target_drop_last=bool(training_cfg.get("target_drop_last", training_cfg.get("cmd_drop_last", True))),
        )

    if explicit_method in {"adabn", "ada_bn", "adaptive_batch_norm", "adaptive_batch_normalization"}:
        target_split_files = list(training_cfg.get("target_split_files", training_cfg.get("adabn_target_split_files", ["target_close.csv"])))
        target_setting = str(training_cfg.get("target_setting", _infer_target_setting(target_split_files)))
        return AdaBNMethod(
            target_split_files=target_split_files,
            target_setting=target_setting,
            allow_test_as_target=bool(training_cfg.get("allow_test_as_target", training_cfg.get("adabn_allow_test_as_target", False))),
            target_drop_last=bool(training_cfg.get("target_drop_last", training_cfg.get("adabn_drop_last", False))),
        )

    if explicit_method in {"pseudo_labeling", "pseudo-labeling", "pseudo", "pseudolabel", "pseudo_labels"}:
        pseudo_cfg = training_cfg.get("pseudo_labeling", {}) if isinstance(training_cfg.get("pseudo_labeling", {}), dict) else {}
        target_split_files = list(
            training_cfg.get(
                "target_split_files",
                pseudo_cfg.get("target_split_files", training_cfg.get("pseudo_target_split_files", ["target_close.csv"])),
            )
        )
        target_setting = str(training_cfg.get("target_setting", _infer_target_setting(target_split_files)))
        return PseudoLabelingMethod(
            target_split_files=target_split_files,
            target_setting=target_setting,
            allow_test_as_target=bool(training_cfg.get("allow_test_as_target", pseudo_cfg.get("allow_test_as_target", False))),
            target_drop_last=bool(training_cfg.get("target_drop_last", pseudo_cfg.get("drop_last", False))),
        )

    if explicit_method in {"mean_teacher", "mean-teacher", "mt"}:
        mt_cfg = training_cfg.get("mean_teacher", {}) if isinstance(training_cfg.get("mean_teacher", {}), dict) else {}
        target_split_files = list(
            training_cfg.get(
                "target_split_files",
                mt_cfg.get("target_split_files", training_cfg.get("mean_teacher_target_split_files", ["target_close.csv"])),
            )
        )
        target_setting = str(training_cfg.get("target_setting", _infer_target_setting(target_split_files)))
        return MeanTeacherMethod(
            target_split_files=target_split_files,
            target_setting=target_setting,
            allow_test_as_target=bool(training_cfg.get("allow_test_as_target", mt_cfg.get("allow_test_as_target", False))),
            target_drop_last=bool(training_cfg.get("target_drop_last", mt_cfg.get("drop_last", True))),
        )

    if explicit_method in {"fixmatch", "fix_match", "fix-match", "fm"}:
        fm_cfg = training_cfg.get("fixmatch", {}) if isinstance(training_cfg.get("fixmatch", {}), dict) else {}
        target_split_files = list(
            training_cfg.get(
                "target_split_files",
                fm_cfg.get("target_split_files", training_cfg.get("fixmatch_target_split_files", ["target_close.csv"])),
            )
        )
        target_setting = str(training_cfg.get("target_setting", _infer_target_setting(target_split_files)))
        return FixMatchMethod(
            target_split_files=target_split_files,
            target_setting=target_setting,
            allow_test_as_target=bool(training_cfg.get("allow_test_as_target", fm_cfg.get("allow_test_as_target", False))),
            target_drop_last=bool(training_cfg.get("target_drop_last", fm_cfg.get("drop_last", True))),
        )

    raise ValueError(
        f"Unsupported training.method={explicit_method!r}. Currently supported: erm/source_only, cmd, adabn, pseudo_labeling, mean_teacher, and fixmatch."
    )
