"""The authored config files (training.yaml, models.yaml, methods.yaml) parse and
have the shape pipeline_phases.py relies on. Catches YAML typos and accidental
reintroduction of the dead keys removed in the config refactor."""
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config"
MODELS = ("cnn", "mlp", "lstm", "transformer", "cnn_lstm", "lstm_cnn")
METHODS = ("erm", "adabn", "cmd", "pseudo_labeling", "mean_teacher", "fixmatch")
# Legacy keys the refactor deleted; they must not creep back into the shared config.
DEAD_TRAINING_KEYS = (
    "use_cmd", "lambda_cmd", "cmd_n_moments", "cmd_a", "cmd_b",
    "cmd_target_split_files", "cmd_allow_test_as_target", "cmd_drop_last",
    "evaluate_test",
)


def _load(name):
    with open(CONFIG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_training_yaml_shape():
    tr = _load("training.yaml")
    assert tr["mlflow"]["enabled"] is False  # off by default
    assert set(tr["search_space"]) == {"batch_size", "learning_rate", "weight_decay"}
    for dead in DEAD_TRAINING_KEYS:
        assert dead not in tr["training"], f"dead key {dead!r} resurfaced in training.yaml"
    # study constants are injected in code, never authored here
    assert "setting" not in tr and "selection" not in tr


def test_models_yaml_shape():
    md = _load("models.yaml")
    assert set(md) == set(MODELS)
    for m, blk in md.items():
        assert set(blk) >= {"defaults", "search_space"}, m
        assert blk["defaults"].get("normalization_strategy", "MISSING") is None, m
        assert blk["search_space"], f"{m} has an empty search space"


def test_methods_yaml_shape():
    me = _load("methods.yaml")
    assert set(me) == set(METHODS)
    assert me["erm"] in (None, {}) and me["adabn"] in (None, {})  # parameter-free
    assert me["cmd"]["lambda_cmd"] is not None
    # SSL methods nest their block under the method name (matches trainer lookup)
    assert "mean_teacher" in me["mean_teacher"]
    assert "pseudo_labeling" in me["pseudo_labeling"]
