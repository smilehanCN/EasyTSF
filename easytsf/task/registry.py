from __future__ import annotations

from dataclasses import dataclass

from easytsf.data import DataInterface, GridDataInterface

from .gridstf import Grid2DTSFTask, Grid3DTSFTask, GridSTFTask
from .mtsf import MTSFTask
from .stf import STFTask


_SIDE_INPUT_FLAGS = {
    "graph": "has_graph",
    "grid_mask": "has_grid_mask",
    "coord": "has_coord",
}


@dataclass(frozen=True)
class TaskRegistryEntry:
    task_name: str
    datamodule_cls: type
    task_cls: type
    expected_layout_kind: str
    expected_spatial_ndim: int | None = None
    task_init_side_inputs: tuple[str, ...] = ()
    required_side_inputs: tuple[str, ...] = ()
    stability: str = "maintained"

    def build_task_kwargs(self, datamodule):
        return {name: getattr(datamodule, name) for name in self.task_init_side_inputs}

    def validate_data_spec(self, dataset_name, data_spec):
        if data_spec.layout_kind != self.expected_layout_kind:
            raise ValueError(
                "task '{}' requires {} data, but dataset '{}' resolved to layout '{}'".format(
                    self.task_name,
                    self.expected_layout_kind,
                    dataset_name,
                    data_spec.layout_kind,
                )
            )

        if self.expected_spatial_ndim is not None and data_spec.spatial_ndim != self.expected_spatial_ndim:
            expected_shape = "[L, C, H, W]" if self.expected_spatial_ndim == 2 else "[L, C, X, Y, Z]"
            raise ValueError(
                "{} experiment requires {} data but dataset '{}' has spatial_ndim={}".format(
                    self.task_name,
                    expected_shape,
                    dataset_name,
                    data_spec.spatial_ndim,
                )
            )

        missing_side_inputs = []
        for side_input in self.required_side_inputs:
            flag_name = _SIDE_INPUT_FLAGS[side_input]
            if not getattr(data_spec, flag_name):
                missing_side_inputs.append(side_input)
        if missing_side_inputs:
            raise ValueError(
                "task '{}' requires dataset side inputs {}".format(
                    self.task_name,
                    missing_side_inputs,
                )
            )


TASK_REGISTRY = {
    "mtsf": TaskRegistryEntry(
        task_name="mtsf",
        datamodule_cls=DataInterface,
        task_cls=MTSFTask,
        expected_layout_kind="sequence",
    ),
    "stf": TaskRegistryEntry(
        task_name="stf",
        datamodule_cls=DataInterface,
        task_cls=STFTask,
        expected_layout_kind="sequence",
        task_init_side_inputs=("graph",),
        required_side_inputs=("graph",),
    ),
    "grid2dtsf": TaskRegistryEntry(
        task_name="grid2dtsf",
        datamodule_cls=GridDataInterface,
        task_cls=Grid2DTSFTask,
        expected_layout_kind="grid",
        expected_spatial_ndim=2,
        task_init_side_inputs=("grid_mask", "coord"),
    ),
    "grid3dtsf": TaskRegistryEntry(
        task_name="grid3dtsf",
        datamodule_cls=GridDataInterface,
        task_cls=Grid3DTSFTask,
        expected_layout_kind="grid",
        expected_spatial_ndim=3,
        task_init_side_inputs=("grid_mask", "coord"),
    ),
    "gridstf": TaskRegistryEntry(
        task_name="gridstf",
        datamodule_cls=GridDataInterface,
        task_cls=GridSTFTask,
        expected_layout_kind="grid",
        task_init_side_inputs=("grid_mask", "coord"),
        stability="experimental",
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
