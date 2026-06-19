from __future__ import annotations

from .base import MethodSpec


class ERMMethod(MethodSpec):
    """Empirical risk minimization / source-only supervised baseline."""

    def __init__(self) -> None:
        super().__init__(
            name="erm",
            target_setting="none",
            include_target_unlabeled=False,
            target_split_files=[],
        )


# Backward-compatible alias for earlier naming.
SupervisedMethod = ERMMethod
