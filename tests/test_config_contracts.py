import tempfile
import unittest
from pathlib import Path

from easytsf.data import MTSDataModule
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

        self.assertTrue(configured_models)
        self.assertTrue(configured_models.issubset(set(get_maintained_model_names())))

    def test_experiment_presets_no_longer_depend_on_config_tasks(self):
        task_config_dir = Path(__file__).resolve().parents[1] / "config" / "tasks"
        self.assertFalse(any(task_config_dir.glob("*.yaml")))

        experiment_root = Path(__file__).resolve().parents[1] / "config" / "experiments"
        for path in sorted(experiment_root.glob("*/*.yaml")):
            config_ref = str(path.relative_to(experiment_root).with_suffix(""))
            conf = load_config(config_ref)
            with self.subTest(config_ref=config_ref):
                self.assertEqual(conf["task_name"], "mtsf")

    def test_experiment_config_requires_complete_runtime_and_train_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "incomplete.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  model_name: TQNet",
                        "  cycle: 24",
                        "  cycle_feature_name: 'time of day'",
                        "  d_model: 8",
                        "  dropout: 0.0",
                        "  use_revin: false",
                        "  use_tq: true",
                        "  channel_aggre: true",
                        "  channel_aggre_heads: 2",
                        "data:",
                        "  dataset_name: Demo",
                        "  hist_len: 96",
                        "  pred_len: 12",
                        "  var_num: 7",
                        "  precompute_window_index: false",
                        "  time_feature_descriptions:",
                        "    - time of day",
                        "train:",
                        "  batch_size: 32",
                        "  max_epochs: 30",
                        "  lr: 0.001",
                        "  lr_scheduler: CycleNetLRS",
                        "  optimizer: Adam",
                        "  es_patience: 5",
                        "  gradient_clip_val: 0.0",
                        "  gradient_clip_algorithm: norm",
                        "  val_metric: val/loss",
                        "  use_mix_loss: false",
                        "runtime:",
                        "  task_name: mtsf",
                        "  num_workers: 2",
                        "  pin_memory: null",
                        "  persistent_workers: null",
                        "  prefetch_factor: 2",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "runtime.use_mmap"):
                load_config(str(config_path))

    def test_removed_task_names_are_unsupported(self):
        for task_name in ("stf", "grid2dtsf", "grid3dtsf", "gridstf"):
            with self.subTest(task_name=task_name):
                with self.assertRaisesRegex(ValueError, "unsupported task_name"):
                    get_task_registry_entry(task_name)

    def test_easytsf_data_public_exports_only_keep_sequence_interface(self):
        self.assertIsNotNone(MTSDataModule)


if __name__ == "__main__":
    unittest.main()
