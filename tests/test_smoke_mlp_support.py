import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from easytsf.data import MTSDataModule
from easytsf.task import get_task_registry_entry
from easytsf.workflow.config import finalize_runtime_conf
from easytsf.workflow.experiment import build_experiment


def _build_basic_ts_timestamps(length, freq, descriptions, offset=0):
    steps_per_day = int((24 * 60) / int(freq))
    global_index = offset + np.arange(length, dtype=np.int32)
    features = []
    for description in descriptions:
        if description == "time of day":
            feature = (global_index % steps_per_day) / steps_per_day
        elif description == "day of week":
            feature = ((global_index // steps_per_day) % 7) / 7.0
        else:
            raise ValueError("unsupported timestamp description in test helper: {}".format(description))
        features.append(feature.astype(np.float32))
    if len(features) == 0:
        return np.empty((length, 0), dtype=np.float32)
    return np.stack(features, axis=-1).astype(np.float32)


class SequenceSmokeTestCase(unittest.TestCase):
    def _make_dataset_dir(self, root, dataset_name):
        dataset_dir = Path(root) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _write_meta(self, dataset_dir, dataset_name, freq=60, descriptions=()):
        meta = {
            "name": dataset_name,
            "frequency (minutes)": int(freq),
            "timestamps_description": list(descriptions),
        }
        with (dataset_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    def _write_sequence_dataset(
        self,
        root,
        dataset_name,
        variable,
        split_lengths=(12, 8, 8),
        freq=60,
        descriptions=("time of day",),
    ):
        dataset_dir = self._make_dataset_dir(root, dataset_name)
        offset = 0
        for split_name, split_length in zip(("train", "val", "test"), split_lengths):
            split_data = variable[offset:offset + split_length].astype(np.float32)
            np.save(dataset_dir / "{}_data.npy".format(split_name), split_data)
            timestamps = _build_basic_ts_timestamps(split_length, freq, descriptions, offset=offset)
            np.save(dataset_dir / "{}_timestamps.npy".format(split_name), timestamps)
            offset += split_length
        self._write_meta(dataset_dir, dataset_name, freq=freq, descriptions=descriptions)
        return dataset_dir

    def _make_conf(self, root, dataset_name, model_name="iTransformer", extra_model_kwargs=None):
        base_conf = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "hist_len": 4,
            "pred_len": 2,
            "batch_size": 2,
            "num_workers": 0,
            "max_epochs": 1,
            "lr": 0.001,
            "lr_scheduler": "OneCycleLR",
            "lrs_pct_start": 0.3,
            "es_patience": 1,
            "optimizer": "Adam",
            "gradient_clip_val": 0.0,
            "gradient_clip_algorithm": "norm",
            "val_metric": "val/loss",
            "use_mix_loss": False,
            "task_name": "mtsf",
            "data_root": root,
            "save_root": root,
            "accelerator": "cpu",
            "devices": 1,
            "use_wandb": 0,
            "seed": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "use_mmap": False,
            "precompute_window_index": False,
            "var_num": 2,
            "output_attention": False,
            "d_model": 8,
            "dropout": 0.0,
            "factor": 1,
            "n_heads": 1,
            "d_ff": 16,
            "activation": "gelu",
            "e_layers": 1,
        }
        if extra_model_kwargs:
            base_conf.update(extra_model_kwargs)
        return finalize_runtime_conf(base_conf)

    def _run_training_step_without_trainer(self, task, batch):
        task.log = lambda *args, **kwargs: None
        return task.training_step(batch, 0)

    def test_sequence_datamodule_builds_all_three_loaders_with_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(tmpdir, "plain_case", variable, descriptions=("time of day",))

            datamodule = MTSDataModule(**self._make_conf(tmpdir, "plain_case"))
            train_batch = next(iter(datamodule.train_dataloader()))
            val_batch = next(iter(datamodule.val_dataloader()))
            test_batch = next(iter(datamodule.test_dataloader()))

            self.assertEqual(set(train_batch.keys()), {"inputs", "inputs_timestamps", "targets", "targets_timestamps"})
            self.assertEqual(tuple(train_batch["inputs"].shape), (2, 4, 2))
            self.assertEqual(tuple(train_batch["inputs_timestamps"].shape), (2, 4, 1))
            self.assertEqual(tuple(val_batch["targets"].shape), (2, 2, 2))
            self.assertEqual(tuple(test_batch["inputs"].shape), (1, 4, 2))

    def test_sequence_datamodule_restores_basic_ts_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(
                tmpdir,
                "timestamp_case",
                variable,
                descriptions=("time of day", "day of week"),
            )

            datamodule = MTSDataModule(**self._make_conf(tmpdir, "timestamp_case"))
            np.testing.assert_array_equal(
                datamodule.split_time_feature["train"][:5, 0],
                np.arange(5, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                datamodule.split_time_feature["train"][:5, 1],
                np.zeros(5, dtype=np.float32),
            )

            batch = next(iter(datamodule.val_dataloader()))
            self.assertIn("inputs_timestamps", batch)
            self.assertEqual(tuple(batch["inputs_timestamps"].shape), (2, 4, 2))
            self.assertEqual(tuple(batch["targets_timestamps"].shape), (2, 2, 2))

    def test_sequence_datamodule_rejects_timestamp_description_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(
                tmpdir,
                "timestamp_description_mismatch",
                variable,
                descriptions=("time of day", "day of week"),
            )

            with self.assertRaisesRegex(ValueError, "time_feature_descriptions"):
                MTSDataModule(
                    **self._make_conf(
                        tmpdir,
                        "timestamp_description_mismatch",
                        extra_model_kwargs={
                            "time_feature_descriptions": ("day of week", "time of day"),
                        },
                    )
                )

    def test_mtsf_task_smoke_with_itransformer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(tmpdir, "itransformer_case", variable, descriptions=("time of day",))

            experiment = build_experiment(self._make_conf(tmpdir, "itransformer_case"), training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))
            prediction, label = experiment.task.forward(batch, 0)
            loss = self._run_training_step_without_trainer(experiment.task, batch)

            self.assertEqual(prediction.shape, label.shape)
            self.assertEqual(loss.dim(), 0)
            self.assertTrue(torch.isfinite(loss).item())

    def test_mtsf_tqnet_smoke_with_explicit_time_feature_descriptions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(
                tmpdir,
                "tqnet_case",
                variable,
                descriptions=("time of day", "day of week"),
            )

            conf = self._make_conf(
                tmpdir,
                "tqnet_case",
                model_name="TQNet",
                extra_model_kwargs={
                    "cycle": 24,
                    "cycle_feature_name": "time of day",
                    "time_feature_descriptions": ("time of day", "day of week"),
                    "d_model": 8,
                    "dropout": 0.0,
                    "use_revin": False,
                    "use_tq": True,
                    "channel_aggre": True,
                    "channel_aggre_heads": 2,
                },
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))
            prediction, label = experiment.task.forward(batch, 0)
            loss = self._run_training_step_without_trainer(experiment.task, batch)

            self.assertEqual(prediction.shape, label.shape)
            self.assertTrue(torch.isfinite(loss).item())

    def test_stf_task_name_is_unsupported(self):
        with self.assertRaisesRegex(ValueError, "unsupported task_name"):
            get_task_registry_entry("stf")


if __name__ == "__main__":
    unittest.main()
