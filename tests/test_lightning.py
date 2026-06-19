import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")

from torch.utils.data import DataLoader, TensorDataset

from bioseq_ood.models.registry import build_model
from bioseq_ood.training.lightning_module import LightningSequenceRegressor

VOCAB = 4
SEQ_LEN = 8


def _loader(n: int) -> DataLoader:
    x = torch.randint(0, VOCAB, (n, SEQ_LEN))
    y = torch.randn(n, 1)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def _module(val_stage_names):
    model = build_model("cnn", {"channels": [8], "kernel_size": 3, "fc_dim": 8,
                                "dropout_input": 0.0, "dropout_hidden": 0.0}, VOCAB, SEQ_LEN)
    return LightningSequenceRegressor(
        model=model,
        training_config={"learning_rate": 1e-3, "loss": "mse"},
        y_scaler=None,
        val_stage_names=val_stage_names,
    )


def test_erm_fast_dev_run_logs_val_id_and_val_ood():
    module = _module(["val_id", "val_ood"])
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(module, train_dataloaders=_loader(16), val_dataloaders=[_loader(8), _loader(8)])
    assert "val_id_mae" in trainer.callback_metrics
    assert "val_ood_mae" in trainer.callback_metrics


def test_oracle_validation_loader_logs_oracle_metric():
    # Mirrors oracle selection: a third labeled loader logged under "oracle".
    module = _module(["val_id", "val_ood", "oracle"])
    trainer = pl.Trainer(fast_dev_run=True, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(module, train_dataloaders=_loader(16),
                val_dataloaders=[_loader(8), _loader(8), _loader(8)])
    assert "oracle_mae" in trainer.callback_metrics
