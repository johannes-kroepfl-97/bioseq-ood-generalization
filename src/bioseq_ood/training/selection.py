from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SplitPlan:
    """Resolved split roles and the checkpoint-monitor metric for one run.

    The study always selects on in-distribution validation labels (``val_id``),
    following the DomainBed honest-selection rule. The adaptation pool and the
    evaluation pool are chosen per protocol via ``training.target_split_files`` and
    ``evaluation.splits`` (see pipeline_phases.py), not here.
    """

    selection_split: str = "val_id"
    report_split: str = "test"
    monitor_metric: str = "val_id_mae"
    early_stop_metric: str = "val_id_mae"


def plan_from_config(config: dict[str, Any]) -> SplitPlan:
    """Build the (val_id) selection plan for a run.

    Only ``selection.metric`` (default ``mae``) and ``evaluation.report_split``
    (default ``test``) are read; everything else is fixed by the study design.
    """
    selection_cfg = config.get("selection", {}) if isinstance(config.get("selection"), dict) else {}
    metric = str(selection_cfg.get("metric", "mae"))
    report_split = config.get("evaluation", {}).get("report_split", "test")
    return SplitPlan(
        selection_split="val_id",
        report_split=report_split,
        monitor_metric=f"val_id_{metric}",
        early_stop_metric="val_id_mae",
    )
