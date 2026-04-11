from easytsf.data import MTSDataModule, WeatherDataModule
from easytsf.task import get_task_components, get_task_spec
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
    assert weatherbench_spec.supported_models == ("WeatherBenchPersistence",)

    grid_spec = get_task_spec("grid3d_forecasting")
    assert grid_spec.family == "grid_prediction"
    assert grid_spec.report_group == "grid3d_forecasting"
    assert grid_spec.supported_models == ("UNet3D",)


def test_get_task_components_returns_expected_tuple():
    assert get_task_components("weatherbench") == (WeatherDataModule, WeatherBenchTask)
    assert get_task_components("mtsf") == (MTSDataModule, MTSFTask)
