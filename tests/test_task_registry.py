import pytest

from easytsf.data import Grid3DDataModule, MTSDataModule
from easytsf.task import get_task_components, get_task_spec
from easytsf.task.grid3d_forecasting import Grid3DForecastingTask
from easytsf.task.mtsf import MTSFTask


def test_get_task_spec_returns_expected_metadata():
    mtsf_spec = get_task_spec("mtsf")
    assert mtsf_spec.name == "mtsf"
    assert mtsf_spec.family == "sequence_prediction"
    assert mtsf_spec.report_group == "mtsf"
    assert mtsf_spec.supported_models == (
        "iTransformer",
        "MixLinear",
        "PCMLP",
        "TQNet",
        "STID",
        "SparseTSF",
        "TimeBase",
    )
    assert mtsf_spec.metric_schema.val_metrics == ("val/loss",)
    assert mtsf_spec.metric_schema.test_metrics == ("test/mae", "test/mse")

    grid_spec = get_task_spec("grid3d_forecasting")
    assert grid_spec.family == "grid_prediction"
    assert grid_spec.report_group == "grid3d_forecasting"
    assert grid_spec.supported_models == (
        "unet3d",
        "unet3d_engram",
        "unet3d_patchcat",
        "patchstg_flat3d",
        "fredn_multivariate3d",
        "fno3d",
        "afno3d",
    )


def test_get_task_components_returns_expected_tuple():
    assert get_task_components("mtsf") == (MTSDataModule, MTSFTask)
    assert get_task_components("grid3d_forecasting") == (Grid3DDataModule, Grid3DForecastingTask)


def test_get_task_spec_rejects_unknown_task():
    with pytest.raises(ValueError, match="unsupported task: unknown_task"):
        get_task_spec("unknown_task")
