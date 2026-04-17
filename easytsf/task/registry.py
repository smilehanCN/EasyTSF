from __future__ import annotations

from dataclasses import dataclass

from easytsf.data import Grid3DDataModule, MTSDataModule
from easytsf.model import MODEL_REGISTRY

from .grid3d_forecasting import Grid3DForecastingTask
from .mtsf import MTSFTask


@dataclass(frozen=True)
class MetricSchema:
    val_metrics: tuple[str, ...]
    test_metrics: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    family: str
    report_group: str
    datamodule_cls: type
    task_cls: type
    supported_models: tuple[str, ...]
    metric_schema: MetricSchema


TASK_SPECS = {
    "grid3d_forecasting": TaskSpec(
        name="grid3d_forecasting",
        family="grid_prediction",
        report_group="grid3d_forecasting",
        datamodule_cls=Grid3DDataModule,
        task_cls=Grid3DForecastingTask,
        supported_models=(
            "unet3d",
            "unet3d_engram",
            "unet3d_patchcat",
            "patchstg_flat3d",
            "fredn_multivariate3d",
            "fno3d",
            "afno3d",
        ),
        metric_schema=MetricSchema(
            val_metrics=("val/loss",),
            test_metrics=("test/mae", "test/mse"),
        ),
    ),
    "mtsf": TaskSpec(
        name="mtsf",
        family="sequence_prediction",
        report_group="mtsf",
        datamodule_cls=MTSDataModule,
        task_cls=MTSFTask,
        supported_models=("iTransformer", "MixLinear", "PCMLP", "TQNet", "STID", "SparseTSF", "TimeBase"),
        metric_schema=MetricSchema(
            val_metrics=("val/loss",),
            test_metrics=("test/mae", "test/mse"),
        ),
    ),
}


def get_task_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        raise ValueError(
            "unsupported task: {}; supported tasks are {}".format(
                task,
                sorted(TASK_SPECS),
            )
        ) from exc


def get_task_components(task: str) -> tuple[type, type]:
    spec = get_task_spec(task)
    return spec.datamodule_cls, spec.task_cls


def validate_task_runtime_conf(runtime_conf, task_spec: TaskSpec | None = None) -> TaskSpec:
    if task_spec is None:
        task_name = str(runtime_conf.get("task", "mtsf"))
        task_spec = get_task_spec(task_name)
    else:
        task_name = task_spec.name

    model_name = runtime_conf.get("model")
    if model_name is None:
        raise ValueError("runtime config for task '{}' must define 'model'".format(task_name))
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            "unknown model '{}'; registered models are {}".format(
                model_name,
                sorted(MODEL_REGISTRY),
            )
        )
    if model_name not in task_spec.supported_models:
        raise ValueError(
            "model '{}' is not supported for task '{}'; supported models are {}".format(
                model_name,
                task_name,
                list(task_spec.supported_models),
            )
        )

    val_metric = runtime_conf.get("val_metric")
    if val_metric is not None and val_metric not in task_spec.metric_schema.val_metrics:
        raise ValueError(
            "val_metric '{}' is invalid for task '{}'; supported validation metrics are {}".format(
                val_metric,
                task_name,
                list(task_spec.metric_schema.val_metrics),
            )
        )
    val_metric_mode = runtime_conf.get("val_metric_mode", "min")
    if val_metric_mode not in {"min", "max"}:
        raise ValueError("val_metric_mode '{}' is invalid; expected 'min' or 'max'".format(val_metric_mode))
    return task_spec
