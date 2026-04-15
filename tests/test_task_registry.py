from easytsf.data import Grid3DDataModule, MTSDataModule, WeatherDataModule
from easytsf.task import get_task_components, get_task_spec
from easytsf.task.grid3d_risk_prediction import Grid3DRiskPredictionTask
from easytsf.task.mtsf import MTSFTask
from easytsf.task.weatherbench import WeatherBenchTask


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

    weatherbench_spec = get_task_spec("weatherbench")
    assert weatherbench_spec.family == "grid_prediction"
    assert weatherbench_spec.report_group == "weatherbench"
    assert weatherbench_spec.supported_models == ("WeatherBenchPersistence", "ARROW")

    grid_spec = get_task_spec("grid3d_forecasting")
    assert grid_spec.family == "grid_prediction"
    assert grid_spec.report_group == "grid3d_forecasting"
    assert grid_spec.supported_models == (
        "unet3d",
        "unet3d_patchcat",
        "patchstg_flat3d",
        "fredn_multivariate3d",
    )

    risk_spec = get_task_spec("grid3d_risk_prediction")
    assert risk_spec.family == "grid_prediction"
    assert risk_spec.report_group == "grid3d_risk_prediction"
    assert risk_spec.supported_models == (
        "unet3d",
        "unet3d_patchcat",
        "patchstg_flat3d",
        "fredn_multivariate3d",
    )
    assert risk_spec.metric_schema.val_metrics == ("val/loss", "val/macro_f1", "val/high_risk_recall")
    assert risk_spec.metric_schema.test_metrics == ("test/macro_f1", "test/high_risk_recall")


def test_get_task_components_returns_expected_tuple():
    assert get_task_components("weatherbench") == (WeatherDataModule, WeatherBenchTask)
    assert get_task_components("mtsf") == (MTSDataModule, MTSFTask)
    assert get_task_components("grid3d_risk_prediction") == (Grid3DDataModule, Grid3DRiskPredictionTask)
