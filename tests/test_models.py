import pytest

torch = pytest.importorskip("torch")

from bioseq_ood.models.registry import MODEL_REGISTRY, build_model

VOCAB = 4
SEQ_LEN = 8
BATCH = 6

_CONFIGS = {
    "mlp": {"hidden_dims": [16, 16], "num_layers": 2, "dropout_input": 0.0, "dropout_hidden": 0.0},
    "cnn": {"channels": [8, 16], "kernel_size": 3, "fc_dim": 8, "dropout_input": 0.0, "dropout_hidden": 0.0},
    "lstm": {"hidden_dim": 16, "num_layers": 1, "bidirectional": False, "fc_dim": 8, "dropout_input": 0.0, "dropout_hidden": 0.0},
    "cnn_lstm": {"cnn_channels": [8], "kernel_size": 3, "lstm_hidden_dim": 16, "lstm_layers": 1, "bidirectional": False, "fc_dim": 8, "dropout_input": 0.0, "dropout_hidden": 0.0},
    "lstm_cnn": {"lstm_hidden_dim": 16, "lstm_layers": 1, "bidirectional": False, "cnn_channels": [8], "kernel_size": 3, "fc_dim": 8, "dropout_input": 0.0, "dropout_hidden": 0.0},
    "transformer": {"d_model": 16, "nhead": 4, "num_layers": 1, "dim_feedforward": 32, "pooling": "mean", "dropout_input": 0.0, "dropout_hidden": 0.0},
}


def test_registry_matches_configs():
    assert set(MODEL_REGISTRY) == set(_CONFIGS)


@pytest.mark.parametrize("model_name", sorted(_CONFIGS))
def test_forward_accepts_integer_and_one_hot_inputs(model_name):
    model = build_model(model_name, _CONFIGS[model_name], VOCAB, SEQ_LEN).eval()

    x_int = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    y_int = model(x_int)
    assert y_int.shape == (BATCH, 1)

    x_onehot = torch.nn.functional.one_hot(x_int, num_classes=VOCAB).float()
    y_onehot = model(x_onehot)
    assert y_onehot.shape == (BATCH, 1)


@pytest.mark.parametrize("model_name", sorted(_CONFIGS))
def test_return_features_contract_for_cmd(model_name):
    # CMD relies on every encoder returning 2D features bounded to [0, 1] (sigmoid).
    model = build_model(model_name, _CONFIGS[model_name], VOCAB, SEQ_LEN).eval()
    x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    y, feats = model(x, return_features=True)
    assert y.shape == (BATCH, 1)
    assert feats.ndim == 2 and feats.shape[0] == BATCH
    assert float(feats.min()) >= 0.0 and float(feats.max()) <= 1.0
