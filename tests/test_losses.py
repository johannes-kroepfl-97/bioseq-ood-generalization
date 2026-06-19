import pytest

torch = pytest.importorskip("torch")

from bioseq_ood.training.losses import CMDLoss


def test_cmd_loss_is_zero_for_identical_distributions():
    z = torch.rand(32, 8)
    loss = CMDLoss(n_moments=5)(z, z.clone())
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_cmd_loss_is_positive_for_shifted_distributions():
    source = torch.rand(64, 8)
    target = torch.rand(64, 8) + 1.0  # mean-shifted
    assert float(CMDLoss(n_moments=5)(source, target)) > 0.0


def test_cmd_loss_rejects_non_2d():
    with pytest.raises(ValueError):
        CMDLoss()(torch.rand(8, 4, 2), torch.rand(8, 4, 2))


def test_cmd_loss_rejects_mismatched_feature_dim():
    with pytest.raises(ValueError):
        CMDLoss()(torch.rand(8, 4), torch.rand(8, 5))


def test_cmd_loss_requires_two_samples_per_domain():
    with pytest.raises(ValueError):
        CMDLoss()(torch.rand(1, 4), torch.rand(8, 4))


def test_cmd_loss_validates_construction():
    with pytest.raises(ValueError):
        CMDLoss(n_moments=0)
    with pytest.raises(ValueError):
        CMDLoss(a=1.0, b=0.0)
