import pytest

from easytsf.task import get_task_spec, validate_task_runtime_conf


@pytest.mark.parametrize(
    "runtime_conf,task_name",
    (
        ({"task": "mtsf", "model": "PCMLP", "val_metric": "val/loss"}, "mtsf"),
        ({"task": "grid3d_forecasting", "model": "unet3d", "val_metric": "val/loss"}, "grid3d_forecasting"),
        (
            {
                "task": "grid3d_forecasting",
                "model": "unet3d",
                "val_metric": "val/loss",
                "val_metric_mode": "max",
            },
            "grid3d_forecasting",
        ),
    ),
)
def test_validate_task_runtime_conf_accepts_current_tasks(runtime_conf, task_name):
    assert validate_task_runtime_conf(runtime_conf) == get_task_spec(task_name)


@pytest.mark.parametrize(
    "runtime_conf,error_match",
    (
        ({"task": "mtsf", "model": "unet3d", "val_metric": "val/loss"}, "not supported for task 'mtsf'"),
        ({"task": "mtsf", "model": "UnknownModel", "val_metric": "val/loss"}, "unknown model 'UnknownModel'"),
        (
            {"task": "grid3d_forecasting", "model": "unet3d", "val_metric": "val/mae"},
            "supported validation metrics",
        ),
        (
            {
                "task": "grid3d_forecasting",
                "model": "unet3d",
                "val_metric": "val/loss",
                "val_metric_mode": "largest",
            },
            "val_metric_mode",
        ),
    ),
)
def test_validate_task_runtime_conf_rejects_mismatches(runtime_conf, error_match):
    with pytest.raises(ValueError, match=error_match):
        validate_task_runtime_conf(runtime_conf)
