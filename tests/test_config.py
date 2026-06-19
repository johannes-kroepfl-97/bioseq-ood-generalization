from bioseq_ood.config.loader import deep_update, load_config, load_yaml


def test_deep_update_merges_nested_without_mutating_base():
    base = {"training": {"lr": 0.1, "epochs": 100}, "model": {"channels": [32]}}
    override = {"training": {"lr": 0.01}, "seed": 7}
    merged = deep_update(base, override)
    assert merged["training"] == {"lr": 0.01, "epochs": 100}
    assert merged["seed"] == 7
    assert merged["model"]["channels"] == [32]
    # base must be untouched
    assert base["training"]["lr"] == 0.1
    assert "seed" not in base


def test_deep_update_replaces_non_dict_values():
    base = {"a": {"b": 1}}
    merged = deep_update(base, {"a": 5})
    assert merged["a"] == 5


def test_load_yaml_and_config_with_override(tmp_path):
    base = tmp_path / "base.yaml"
    override = tmp_path / "aav.yaml"
    base.write_text("dataset:\n  name: gfp\ntraining:\n  lr: 0.1\n", encoding="utf-8")
    override.write_text("dataset:\n  name: aav\n", encoding="utf-8")

    assert load_yaml(base)["training"]["lr"] == 0.1

    config = load_config(base, override)
    assert config["dataset"]["name"] == "aav"  # override wins
    assert config["training"]["lr"] == 0.1     # base preserved
