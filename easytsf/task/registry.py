from __future__ import annotations

from easytsf.data import MTSDataModule

from .mtsf import MTSFTask


TASK_REGISTRY = {
    "mtsf": (MTSDataModule, MTSFTask),
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
