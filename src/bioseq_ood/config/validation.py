from __future__ import annotations

from typing import Any


def validate_training_config(config: dict[str, Any]) -> None:
    required = ["dataset", "model", "training"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
