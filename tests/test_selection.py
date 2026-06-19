from bioseq_ood.training.selection import SplitPlan, plan_from_config


def test_default_plan_monitors_val_id():
    plan = plan_from_config({})
    assert plan.selection_split == "val_id"
    assert plan.monitor_metric == "val_id_mae"
    assert plan.early_stop_metric == "val_id_mae"
    assert plan.report_split == "test"


def test_metric_and_report_split_read_from_config():
    plan = plan_from_config({"selection": {"metric": "rmse"}, "evaluation": {"report_split": "T_far"}})
    assert plan.monitor_metric == "val_id_rmse"
    assert plan.report_split == "T_far"
    # Early stopping always tracks val_id_mae, independent of the selection metric.
    assert plan.early_stop_metric == "val_id_mae"


def test_splitplan_defaults():
    plan = SplitPlan()
    assert plan.selection_split == "val_id"
    assert plan.monitor_metric == "val_id_mae"
