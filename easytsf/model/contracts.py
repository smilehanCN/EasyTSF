from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelContract:
    model_name: str
    supported_tasks: tuple[str, ...]
    maintenance_tier: str = "maintained"
    note: str = ""

    @property
    def module_name(self):
        return self.model_name.lower()

    @property
    def is_legacy(self):
        return self.maintenance_tier == "legacy"

    def validate(self, task):
        if task not in self.supported_tasks:
            raise ValueError(
                "model '{}' does not support task '{}'; supported tasks are {}".format(
                    self.model_name,
                    task,
                    list(self.supported_tasks),
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
