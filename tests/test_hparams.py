"""The method-hparams module: defaults fill non-destructively from methods.yaml,
the AdaBN normalization special case fires, and the search distributions draw the
expected keys."""
import numpy as np
import pytest

from bioseq_ood.methods.hparams import (
    adabn_norm_strategy,
    apply_method_hparams,
    sample_method_hparams,
)


def test_defaults_fill_missing_only():
    cfg = {"training": {}, "model": {}, "model_name": "cnn"}
    apply_method_hparams(cfg, "mean_teacher")
    mt = cfg["training"]["mean_teacher"]
    assert mt["ema_decay"] == 0.999 and mt["rampup_epochs"] == 10


def test_defaults_do_not_overwrite_existing():
    cfg = {"training": {"mean_teacher": {"ema_decay": 0.5}}}
    apply_method_hparams(cfg, "mean_teacher")
    assert cfg["training"]["mean_teacher"]["ema_decay"] == 0.5   # searched value wins
    assert cfg["training"]["mean_teacher"]["rampup_epochs"] == 10  # gap still filled


def test_cmd_flat_defaults_and_pseudo_nested():
    cfg = {"training": {}}
    apply_method_hparams(cfg, "cmd")
    assert cfg["training"]["lambda_cmd"] == 0.1 and cfg["training"]["cmd_n_moments"] == 5
    cfg2 = {"training": {}}
    apply_method_hparams(cfg2, "pseudo_labeling")
    assert cfg2["training"]["pseudo_labeling"]["keep_ratio"] == 0.5


def test_adabn_lstm_always_gets_final_bn_variant():
    # Pure LSTM's native norm is LayerNorm (no running stats), so AdaBN always needs
    # the LN + final-BN variant regardless of the baseline's normalization.
    for current in (None, "architecture_native_norm"):
        cfg = {"training": {}, "model": {"normalization_strategy": current}, "model_name": "lstm"}
        apply_method_hparams(cfg, "adabn")
        assert cfg["model"]["normalization_strategy"] == "native_norm_plus_final_bn"


def test_adabn_adds_bn_when_baseline_has_none():
    cfg = {"training": {}, "model": {"normalization_strategy": None}, "model_name": "cnn"}
    apply_method_hparams(cfg, "adabn")
    assert cfg["model"]["normalization_strategy"] == "architecture_native_norm"


def test_adabn_reuses_baseline_norm_when_already_present():
    # If the runoff already gave the baseline BN, AdaBN runs on that SAME baseline
    # (no override) so the comparison is not confounded by adding BN.
    cfg = {"training": {}, "model": {"normalization_strategy": "architecture_native_norm"}, "model_name": "cnn"}
    apply_method_hparams(cfg, "adabn")
    assert cfg["model"]["normalization_strategy"] == "architecture_native_norm"
    assert adabn_norm_strategy("cnn") == "architecture_native_norm"


@pytest.mark.parametrize("method,keys", [
    ("cmd", {"lambda_cmd", "cmd_n_moments"}),
    ("pseudo_labeling", {"keep_ratio", "lambda_pseudo_max", "rampup_epochs", "mc_passes", "retrain_from_scratch"}),
    ("mean_teacher", {"lambda_consistency_max", "ema_decay", "rampup_epochs", "consistency_on_source"}),
    ("fixmatch", {"lambda_fixmatch_max", "strong_noise_sigma", "keep_ratio", "mc_passes", "rampup_epochs"}),
])
def test_sample_draws_expected_keys(method, keys):
    s = sample_method_hparams(method, np.random.default_rng(0))["training"]
    inner = s if method == "cmd" else s[method]
    assert set(inner) == keys


def test_sample_parameter_free_and_unknown():
    assert sample_method_hparams("erm", np.random.default_rng(0)) == {}
    assert sample_method_hparams("adabn", np.random.default_rng(0)) == {}
    with pytest.raises(ValueError):
        sample_method_hparams("nope", np.random.default_rng(0))
