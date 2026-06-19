from __future__ import annotations

import torch
import torch.nn as nn


class CMDLoss(nn.Module):
    """Central Moment Discrepancy regulariser (Zellinger et al., ICLR 2017, Def. 2).

        CMD_K(X, Y) = 1/|b-a| * ||E(X)-E(Y)||_2
                      + sum_{k=2..K} 1/|b-a|^k * ||c_k(X)-c_k(Y)||_2

    where c_k is the per-feature k-th central moment. The estimator assumes the
    representations X, Y are bounded on the interval [a, b]; in this project the
    encoders feed CMD a torch.sigmoid feature (models/*.py, return_features), so the
    bound is [0, 1] and a/b are fixed accordingly -- they are not free parameters.
    """

    def __init__(self, n_moments: int = 5, a: float = 0.0, b: float = 1.0) -> None:
        super().__init__()
        if n_moments < 1:
            raise ValueError(f"n_moments must be >= 1, got {n_moments}")
        if b <= a:
            raise ValueError(f"CMD requires b > a, got a={a}, b={b}")
        self.n_moments = int(n_moments)
        self.a = float(a)
        self.b = float(b)
        self.interval_width = float(b - a) # same as abs(b-a) due to sanity check above

    def forward(self, source_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        if source_z.ndim != 2 or target_z.ndim != 2:
            raise ValueError(
                "CMDLoss expects 2D representation tensors with shape "
                f"(batch, features). Got {tuple(source_z.shape)} and {tuple(target_z.shape)}."
            )
        if source_z.shape[1] != target_z.shape[1]:
            raise ValueError(
                "Source and target representations must have the same feature dimension. "
                f"Got {source_z.shape[1]} and {target_z.shape[1]}."
            )
        if source_z.shape[0] < 2 or target_z.shape[0] < 2:
            raise ValueError("CMDLoss needs at least two samples per domain batch.")


        # the paper notation is added as comments below

        source_z = source_z.float()    # X (samples × features)
        target_z = target_z.float()    # Y (samples × features)

        # mean computed per feature over the samples
        source_mean = source_z.mean(dim=0)    # E[X]
        target_mean = target_z.mean(dim=0)    # E[Y]

        # first term with k=1, so  (1/|b-a|) · ||E(X) - E(Y)||_2
        # note the L_2 norm is torch.norm(... , p=2)
        loss = torch.norm(source_mean - target_mean, p=2) / self.interval_width

        # for computations of central moment vectors
        source_centered = source_z - source_mean    # (X_j - E[X_j])
        target_centered = target_z - target_mean    # (Y_j - E[Y_j])

        # higher-order terms starting from k=2, so  (1/|b-a|^k) · ||c_k(X) - c_k(Y)||_2
        for k in range(2, self.n_moments + 1):
            source_moment = (source_centered ** k).mean(dim=0)    # c_k(X)_j = E[(X_j - E[X_j])^k]
            target_moment = (target_centered ** k).mean(dim=0)    # c_k(Y)_j = E[(Y_j - E[Y_j])^k]

            # add ( 1 / (|b-a|^k)) * || c_k[X] - c_k[Y] ||_2
            loss += (1/self.interval_width ** k) * torch.norm(source_moment - target_moment, p=2)

        return loss
