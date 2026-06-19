from __future__ import annotations

import json
from typing import Any


def flatten_dict(d: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            items[new_key] = value
        else:
            items[new_key] = json.dumps(value)
    return items
