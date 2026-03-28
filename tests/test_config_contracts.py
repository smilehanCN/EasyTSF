import unittest
from pathlib import Path

from easytsf.data import DataInterface, DataSpec, GridDataInterface
from easytsf.model import get_maintained_model_names, get_model_contract
from easytsf.task import get_task_registry_entry
from easytsf.workflow.experiment import load_config


class ConfigContractsTestCase(unittest.TestCase):
    def test_all_experiment_presets_use_maintained_model_contracts(self):
        experiment_root = Path(__file__).resolve().parents[1] / "config" / "experiments"

        for path in sorted(experiment_root.glob("*/*.yaml")):
            config_ref = str(path.relative_to(experiment_root).with_suffix(""))
            conf = load_config(config_ref)
            contract = get_model_contract(conf["model_name"])
            task_entry = get_task_registry_entry(conf["task_name"])

            with self.subTest(config_ref=config_ref):
                self.assertFalse(contract.is_legacy)
                self.assertIn(conf["task_name"], contract.supported_task_names)

    def test_maintained_model_names_match_experiment_matrix(self):
        experiment_root = Path(__file__).resolve().parents[1] / "config" / "experiments"
        configured_models = set()

        for path in sorted(experiment_root.glob("*/*.yaml")):
            config_ref = str(path.relative_to(experiment_root).with_suffix(""))
            conf = load_config(config_ref)
            configured_models.add(conf["model_name"])

        self.assertEqual(configured_models, set(get_maintained_model_names()))

    def test_gridstf_registry_entry_is_marked_experimental(self):
        self.assertEqual(get_task_registry_entry("gridstf").stability, "experimental")

    def test_stf_registry_entry_no_longer_requires_graph(self):
        self.assertEqual(get_task_registry_entry("stf").required_side_inputs, ())
        self.assertIn("stf", get_model_contract("iTransformer").supported_task_names)

    def test_easytsf_data_public_exports_remain_available(self):
        self.assertIsNotNone(DataInterface)
        self.assertIsNotNone(GridDataInterface)
        self.assertIsNotNone(DataSpec)


if __name__ == "__main__":
    unittest.main()
