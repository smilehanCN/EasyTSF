from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContract:
    model_name: str
    supported_task_names: tuple[str, ...]
    maintenance_tier: str = "maintained"
    note: str = ""

    @property
    def module_name(self):
        return self.model_name.lower()

    @property
    def is_legacy(self):
        return self.maintenance_tier == "legacy"

    def validate(self, task_name):
        if task_name not in self.supported_task_names:
            raise ValueError(
                "model '{}' does not support task '{}'; supported tasks are {}".format(
                    self.model_name,
                    task_name,
                    list(self.supported_task_names),
                )
            )


MODEL_CONTRACTS = {
    "iTransformer": ModelContract("iTransformer", ("mtsf",)),
    "TQNet": ModelContract("TQNet", ("mtsf",)),
    "STGCN": ModelContract("STGCN", ("mtsf",), maintenance_tier="legacy", note="Legacy baseline without preset/smoke coverage."),
    "STID": ModelContract("STID", ("mtsf",), maintenance_tier="legacy", note="Legacy baseline without preset/smoke coverage."),
    "SparseTSF": ModelContract(
        "SparseTSF",
        ("mtsf",),
        maintenance_tier="legacy",
        note="Legacy baseline without preset/smoke coverage.",
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
