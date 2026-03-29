from __future__ import annotations

from dataclasses import dataclass

from easytsf.data import MTSDataModule
from .mtsf import MTSFTask


@dataclass(frozen=True)
class TaskRegistryEntry:
    task_name: str
    datamodule_cls: type
    task_cls: type
    stability: str = "maintained"


TASK_REGISTRY = {
    "mtsf": TaskRegistryEntry(
        task_name="mtsf",
        datamodule_cls=MTSDataModule,
        task_cls=MTSFTask,
    ),
}


def get_task_registry_entry(task_name):
    try:
        return TASK_REGISTRY[task_name]
    except KeyError as exc:
        raise ValueError(
            "unsupported task_name: {}; supported task names are {}".format(
                task_name,
                sorted(TASK_REGISTRY),
            )
        ) from exc
