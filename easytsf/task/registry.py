from __future__ import annotations

from easytsf.data import MTSDataModule, WeatherDataModule

from .mtsf import MTSFTask
from .weatherbench import WeatherBenchTask


TASK_REGISTRY = {
    "mtsf": (MTSDataModule, MTSFTask),
    "weatherbench": (WeatherDataModule, WeatherBenchTask),
}


def get_task_components(task: str) -> tuple[type, type]:
    try:
        return TASK_REGISTRY[task]
    except KeyError as exc:
        raise ValueError(
            "unsupported task: {}; supported tasks are {}".format(
                task,
                sorted(TASK_REGISTRY),
            )
        ) from exc
