from __future__ import annotations

from dataclasses import dataclass

from easytsf.data import MTSDataModule
from .mtsf import MTSFTask


@dataclass(frozen=True)
class TaskRegistryEntry:
    task: str
    datamodule_cls: type
    task_cls: type
    stability: str = "maintained"


TASK_REGISTRY = {
    "mtsf": TaskRegistryEntry(
        task="mtsf",
        datamodule_cls=MTSDataModule,
        task_cls=MTSFTask,
    ),
}


def get_task_entry(task):
    try:
        return TASK_REGISTRY[task]
    except KeyError as exc:
        raise ValueError(
            "unsupported task: {}; supported tasks are {}".format(
                task,
                sorted(TASK_REGISTRY),
            )
        ) from exc
