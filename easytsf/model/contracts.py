from __future__ import annotations

from dataclasses import dataclass


_SIDE_INPUT_FLAGS = {
    "graph": "has_graph",
    "grid_mask": "has_grid_mask",
    "coord": "has_coord",
}


@dataclass(frozen=True)
class ModelContract:
    model_name: str
    supported_task_names: tuple[str, ...]
    required_side_inputs: tuple[str, ...] = ()
    maintenance_tier: str = "maintained"
    note: str = ""

    @property
    def module_name(self):
        return self.model_name.lower()

    @property
    def is_legacy(self):
        return self.maintenance_tier == "legacy"

    def validate(self, task_name, data_spec):
        if task_name not in self.supported_task_names:
            raise ValueError(
                "model '{}' does not support task '{}'; supported tasks are {}".format(
                    self.model_name,
                    task_name,
                    list(self.supported_task_names),
                )
            )

        missing_side_inputs = []
        for side_input in self.required_side_inputs:
            flag_name = _SIDE_INPUT_FLAGS[side_input]
            if data_spec is None or not getattr(data_spec, flag_name):
                missing_side_inputs.append(side_input)
        if missing_side_inputs:
            raise ValueError(
                "model '{}' requires side inputs {} for task '{}'".format(
                    self.model_name,
                    missing_side_inputs,
                    task_name,
                )
            )


MODEL_CONTRACTS = {
    "SimpleMLP": ModelContract("SimpleMLP", ("mtsf",)),
    "iTransformer": ModelContract("iTransformer", ("mtsf",)),
    "MOMENT": ModelContract("MOMENT", ("mtsf",)),
    "CoRA": ModelContract("CoRA", ("mtsf",)),
    "SimpleGraphMLP": ModelContract("SimpleGraphMLP", ("stf",), required_side_inputs=("graph",)),
    "STGCN": ModelContract("STGCN", ("stf",), required_side_inputs=("graph",)),
    "CoRAGraph": ModelContract("CoRAGraph", ("stf",), required_side_inputs=("graph",)),
    "SimpleGridMLP": ModelContract("SimpleGridMLP", ("grid2dtsf", "grid3dtsf", "gridstf")),
    "CoRAGrid": ModelContract("CoRAGrid", ("grid2dtsf", "grid3dtsf", "gridstf")),
    "MLP": ModelContract("MLP", ("mtsf",), maintenance_tier="legacy", note="Legacy baseline without preset/smoke coverage."),
    "STID": ModelContract("STID", ("mtsf",), maintenance_tier="legacy", note="Legacy baseline without preset/smoke coverage."),
    "SparseTSF": ModelContract(
        "SparseTSF",
        ("mtsf",),
        maintenance_tier="legacy",
        note="Legacy baseline without preset/smoke coverage.",
    ),
    "TimeLLM": ModelContract(
        "TimeLLM",
        ("mtsf",),
        maintenance_tier="legacy",
        note="Legacy baseline without preset/smoke coverage.",
    ),
    "RLinear": ModelContract(
        "RLinear",
        (),
        maintenance_tier="legacy",
        note="Legacy implementation that no longer matches the maintained mtsf tensor contract.",
    ),
}


def get_model_contract(model_name):
    try:
        return MODEL_CONTRACTS[model_name]
    except KeyError as exc:
        raise ValueError(
            "unknown model '{}'; registered models are {}".format(
                model_name,
                sorted(MODEL_CONTRACTS),
            )
        ) from exc


def get_maintained_model_names():
    return sorted(name for name, contract in MODEL_CONTRACTS.items() if not contract.is_legacy)
