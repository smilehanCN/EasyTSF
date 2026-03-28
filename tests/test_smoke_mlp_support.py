import importlib.util
import json
import pickle
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch

from easytsf.model.tqnet import Model as TQNetModel
from easytsf.workflow.experiment import build_experiment, finalize_runtime_conf, load_config, run_training
from easytsf.workflow.study import run_study


def _load_migration_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_sequence_dataset_to_basicts.py"
    spec = importlib.util.spec_from_file_location("easytsf_migrate_sequence_dataset_to_basicts", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_timestamp(length, unit="h", offset=0):
    base = np.datetime64("2024-01-01T00:00")
    return base + (offset + np.arange(length)) * np.timedelta64(1, unit)


def _build_basic_ts_timestamps(length, freq, descriptions, offset=0):
    steps_per_day = int((24 * 60) / int(freq))
    if steps_per_day <= 0:
        raise ValueError("frequency must produce a positive steps_per_day")

    global_index = offset + np.arange(length, dtype=np.int32)
    features = []
    for description in descriptions:
        if description == "time of day":
            feature = (global_index % steps_per_day) / steps_per_day
        elif description == "day of week":
            feature = ((global_index // steps_per_day) % 7) / 7.0
        elif description == "day of month":
            feature = ((global_index // steps_per_day) % 31) / 31.0
        elif description == "day of year":
            feature = ((global_index // steps_per_day) % 366) / 366.0
        else:
            raise ValueError("unsupported timestamp description in test helper: {}".format(description))
        features.append(feature.astype(np.float32))
    if len(features) == 0:
        return np.empty((length, 0), dtype=np.float32)
    return np.stack(features, axis=-1).astype(np.float32)


class SmokeMLPSupportTestCase(unittest.TestCase):
    @staticmethod
    def _default_regular_settings(norm_each_channel=True, rescale=False, null_val=np.nan):
        return {
            "norm_each_channel": bool(norm_each_channel),
            "rescale": bool(rescale),
            "null_val": null_val,
        }

    def _make_dataset_dir(self, root, dataset_name):
        dataset_dir = Path(root) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _write_meta(
        self,
        dataset_dir,
        dataset_name,
        freq,
        num_vars=None,
        split_lengths=None,
        descriptions=(),
        has_graph=False,
        regular_settings=None,
        extra_meta=None,
    ):
        total_steps = int(sum(split_lengths or []))
        meta = {
            "name": dataset_name,
            "frequency (minutes)": int(freq),
            "has_graph": bool(has_graph),
            "split_lengths": [int(item) for item in (split_lengths or [])],
            "timestamps_description": list(descriptions),
            "num_time_steps": total_steps,
            "regular_settings": {} if regular_settings is None else dict(regular_settings),
        }
        if num_vars is not None:
            meta["num_vars"] = int(num_vars)
            meta["shape"] = [total_steps, int(num_vars)]
        if descriptions:
            meta["timestamps_shape"] = [total_steps, len(descriptions)]
        if extra_meta:
            meta.update(dict(extra_meta))
        with (dataset_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    def _load_meta(self, dataset_dir):
        with (Path(dataset_dir) / "meta.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save_meta(self, dataset_dir, meta):
        with (Path(dataset_dir) / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, sort_keys=True)

    def _write_sequence_dataset(
        self,
        root,
        dataset_name,
        variable,
        split_lengths=(12, 8, 8),
        freq=60,
        descriptions=("time of day", "day of week"),
        include_timestamps=True,
        meta_name=None,
        regular_settings=None,
        extra_meta=None,
    ):
        split_lengths = [int(item) for item in split_lengths]
        self.assertEqual(sum(split_lengths), int(len(variable)))
        dataset_dir = self._make_dataset_dir(root, dataset_name)
        offset = 0
        for split_name, split_length in zip(("train", "val", "test"), split_lengths):
            split_data = variable[offset:offset + split_length]
            np.save(dataset_dir / "{}_data.npy".format(split_name), split_data.astype(np.float32))
            if include_timestamps:
                split_timestamps = _build_basic_ts_timestamps(split_length, freq, descriptions, offset=offset)
                np.save(dataset_dir / "{}_timestamps.npy".format(split_name), split_timestamps)
            offset += split_length

        self._write_meta(
            dataset_dir,
            meta_name or dataset_name,
            freq=freq,
            num_vars=None if variable.ndim == 1 else variable.shape[1],
            split_lengths=split_lengths,
            descriptions=descriptions if include_timestamps else (),
            has_graph=False,
            regular_settings=self._default_regular_settings() if regular_settings is None else regular_settings,
            extra_meta=extra_meta,
        )
        return dataset_dir

    def _write_grid_dataset(self, root, dataset_name, variable, grid_mask=None, coord=None, split_lengths=(8, 4, 4), freq=60):
        if variable.ndim == 5:
            raise ValueError("use _write_grid3d_sharded_dataset for 3D grid cases")
        dataset_dir = self._make_dataset_dir(root, dataset_name)
        np.savez(
            dataset_dir / "data.npz",
            scaled_variable=variable.astype(np.float32),
            timestamp=_make_timestamp(variable.shape[0]),
        )
        if grid_mask is not None:
            np.save(dataset_dir / "grid_mask.npy", grid_mask.astype(np.float32))
        if coord is not None:
            np.save(dataset_dir / "coord.npy", coord.astype(np.float32))

        self._write_meta(
            dataset_dir,
            dataset_name,
            freq=freq,
            num_vars=variable.shape[1],
            split_lengths=split_lengths,
            descriptions=(),
            has_graph=False,
        )
        return dataset_dir

    def _write_grid3d_sharded_dataset(
        self,
        root,
        dataset_name,
        variable,
        grid_mask=None,
        coord=None,
        split_lengths=(6, 5, 5),
        freq=60,
        shard_len=5,
    ):
        if variable.ndim != 5:
            raise ValueError("3D sharded dataset writer expects variable shaped [L, C, X, Y, Z]")
        split_lengths = [int(item) for item in split_lengths]
        self.assertEqual(sum(split_lengths), int(variable.shape[0]))
        dataset_dir = self._make_dataset_dir(root, dataset_name)

        if grid_mask is not None:
            np.save(dataset_dir / "grid_mask.npy", grid_mask.astype(np.float32))
        if coord is not None:
            np.save(dataset_dir / "coord.npy", coord.astype(np.float32))

        offset = 0
        for split_name, split_length in zip(("train", "val", "test"), split_lengths):
            split_dir = dataset_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_variable = variable[offset:offset + split_length].astype(np.float32)
            files = []
            shard_index = 0
            for shard_start in range(0, split_length, int(shard_len)):
                shard_stop = min(split_length, shard_start + int(shard_len))
                shard_name = "{:06d}.npy".format(shard_index)
                np.save(split_dir / shard_name, split_variable[shard_start:shard_stop])
                files.append({"path": shard_name, "length": int(shard_stop - shard_start)})
                shard_index += 1

            with (split_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "split": split_name,
                        "num_steps": int(split_length),
                        "num_shards": len(files),
                        "shard_len": int(shard_len),
                        "files": files,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            np.save(
                split_dir / "timestamps.npy",
                _make_timestamp(split_length, unit="m", offset=offset),
            )
            offset += split_length

        self._write_meta(
            dataset_dir,
            dataset_name,
            freq=freq,
            num_vars=variable.shape[1],
            split_lengths=split_lengths,
            descriptions=(),
            has_graph=False,
            extra_meta={
                "storage_format": "grid3d_split_sharded_npy_v1",
                "spatial_shape": [int(size) for size in variable.shape[2:]],
                "channel_num": int(variable.shape[1]),
                "dtype": "float32",
                "shard_len": int(shard_len),
            },
        )
        return dataset_dir

    def _write_legacy_flat_dataset(self, root, dataset_name, variable):
        dataset_path = Path(root) / "{}.npz".format(dataset_name)
        np.savez(
            dataset_path,
            scaled_variable=variable.astype(np.float32),
            timestamp=_make_timestamp(variable.shape[0]),
        )
        return dataset_path

    def _write_graph(self, root, dataset_name, graph, tuple_format=False):
        graph_path = self._make_dataset_dir(root, dataset_name) / "adj_mx.pkl"
        payload = graph.astype(np.float32)
        if tuple_format:
            sensor_ids = [str(index) for index in range(payload.shape[0])]
            sensor_id_to_ind = {sensor_id: index for index, sensor_id in enumerate(sensor_ids)}
            payload = (sensor_ids, sensor_id_to_ind, payload)
        with graph_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return graph_path

    def _make_conf(
        self,
        root,
        dataset_name,
        model_name,
        task_name="mtsf",
        var_num=None,
        batch_size=2,
        hist_len=3,
        pred_len=2,
        max_epochs=1,
        extra_model_kwargs=None,
    ):
        base_conf = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "hist_len": hist_len,
            "pred_len": pred_len,
            "batch_size": batch_size,
            "num_workers": 0,
            "max_epochs": max_epochs,
            "lr": 0.001,
            "lr_scheduler": "OneCycleLR",
            "lrs_pct_start": 0.3,
            "es_patience": 2,
            "optimizer": "Adam",
            "gradient_clip_val": 0.0,
            "gradient_clip_algorithm": "norm",
            "val_metric": "val/loss",
            "use_mix_loss": False,
            "task_name": task_name,
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
            "cache_npz_as_npy": False,
            "precompute_window_index": False,
            "hidden_dim": 8,
            "dropout": 0.1,
        }
        if var_num is not None:
            base_conf["var_num"] = int(var_num)
        if extra_model_kwargs:
            base_conf.update(extra_model_kwargs)
        return finalize_runtime_conf(base_conf)

    def _assert_scalar_loss(self, loss):
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).item())

    def _run_training_step_without_trainer(self, task, batch):
        task.log = lambda *args, **kwargs: None
        return task.training_step(batch, 0)

    def test_mtsf_path_rejects_raw_univariate_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28, dtype=np.float32)
            self._write_sequence_dataset(tmpdir, "raw_univariate", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="raw_univariate",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=1,
            )
            with self.assertRaisesRegex(ValueError, r"\[L, 1\]"):
                build_experiment(conf, training=False)

    def test_stf_path_rejects_raw_univariate_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28, dtype=np.float32)
            self._write_sequence_dataset(tmpdir, "raw_univariate_graph", variable)
            self._write_graph(tmpdir, "raw_univariate_graph", np.eye(1, dtype=np.float32))
            dataset_dir = Path(tmpdir) / "raw_univariate_graph"
            meta = self._load_meta(dataset_dir)
            meta["has_graph"] = True
            self._save_meta(dataset_dir, meta)

            conf = self._make_conf(
                tmpdir,
                dataset_name="raw_univariate_graph",
                model_name="STGCN",
                task_name="stf",
                var_num=1,
            )
            with self.assertRaisesRegex(ValueError, r"\[L, 1\]"):
                build_experiment(conf, training=False)

    def test_mtsf_path_rejects_legacy_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_legacy_flat_dataset(tmpdir, "legacy_flat", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="legacy_flat",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=2,
            )
            with self.assertRaisesRegex(FileNotFoundError, "legacy flat dataset layout is no longer supported"):
                build_experiment(conf, training=False)

    def test_mtsf_path_requires_meta_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._make_dataset_dir(tmpdir, "missing_meta")
            np.save(dataset_dir / "train_data.npy", np.zeros((12, 2), dtype=np.float32))
            np.save(dataset_dir / "val_data.npy", np.zeros((8, 2), dtype=np.float32))
            np.save(dataset_dir / "test_data.npy", np.zeros((8, 2), dtype=np.float32))

            conf = self._make_conf(tmpdir, "missing_meta", "SimpleMLP", var_num=2)
            with self.assertRaisesRegex(FileNotFoundError, "meta.json"):
                build_experiment(conf, training=False)

    def test_mtsf_path_rejects_meta_name_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(tmpdir, "meta_name_case", variable, meta_name="OtherDataset")

            conf = self._make_conf(tmpdir, "meta_name_case", "SimpleMLP", var_num=2)
            with self.assertRaisesRegex(ValueError, "does not match dataset directory/config"):
                build_experiment(conf, training=False)

    def test_mtsf_path_rejects_timestamp_length_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            dataset_dir = self._write_sequence_dataset(tmpdir, "timestamp_mismatch", variable)
            np.save(dataset_dir / "val_timestamps.npy", np.zeros((7, 2), dtype=np.float32))

            conf = self._make_conf(tmpdir, "timestamp_mismatch", "SimpleMLP", var_num=2)
            with self.assertRaisesRegex(ValueError, "timestamp length"):
                build_experiment(conf, training=False)

    def test_mtsf_path_requires_regular_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(
                tmpdir,
                "missing_regular_settings",
                variable,
                regular_settings={},
            )

            conf = self._make_conf(tmpdir, "missing_regular_settings", "SimpleMLP", var_num=2)
            with self.assertRaisesRegex(ValueError, "regular_settings must define"):
                build_experiment(conf, training=False)

    def test_sequence_datamodule_records_per_channel_scaler_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.stack(
                [
                    np.arange(1, 57, 2, dtype=np.float32),
                    np.arange(10, 122, 4, dtype=np.float32),
                ],
                axis=-1,
            )
            self._write_sequence_dataset(tmpdir, "per_channel_stats_case", variable)

            conf = self._make_conf(
                tmpdir,
                "per_channel_stats_case",
                "iTransformer",
                var_num=2,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)

            expected_train = variable[:12]
            expected_mean = np.mean(expected_train, axis=0, keepdims=True)
            expected_std = np.std(expected_train, axis=0, keepdims=True)
            np.testing.assert_allclose(experiment.datamodule.scaler_stats["mean"], expected_mean.astype(np.float32))
            np.testing.assert_allclose(experiment.datamodule.scaler_stats["std"], expected_std.astype(np.float32))
            np.testing.assert_allclose(
                np.asarray(experiment.conf["scaler_stats"]["mean"], dtype=np.float32),
                expected_mean.astype(np.float32),
            )

    def test_sequence_datamodule_ignores_meta_embedded_scaler_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(1, 57, dtype=np.float32).reshape(28, 2)
            self._write_sequence_dataset(
                tmpdir,
                "ignore_meta_scaler_stats_case",
                variable,
                extra_meta={
                    "scaler_stats": {
                        "mean": [999.0, 999.0],
                        "std": [888.0, 888.0],
                    }
                },
            )

            conf = self._make_conf(
                tmpdir,
                "ignore_meta_scaler_stats_case",
                "iTransformer",
                var_num=2,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)

            expected_train = variable[:12]
            expected_mean = np.mean(expected_train, axis=0, keepdims=True).astype(np.float32)
            expected_std = np.std(expected_train, axis=0, keepdims=True).astype(np.float32)
            np.testing.assert_allclose(experiment.datamodule.scaler_stats["mean"], expected_mean)
            np.testing.assert_allclose(experiment.datamodule.scaler_stats["std"], expected_std)
            self.assertNotEqual(float(np.asarray(experiment.datamodule.scaler_stats["mean"]).reshape(-1)[0]), 999.0)

    def test_sequence_datamodule_clamps_zero_std_for_global_scaler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.full((28, 2), 7.0, dtype=np.float32)
            self._write_sequence_dataset(
                tmpdir,
                "global_stats_case",
                variable,
                regular_settings=self._default_regular_settings(
                    norm_each_channel=False,
                    rescale=True,
                    null_val=0.0,
                ),
            )

            conf = self._make_conf(
                tmpdir,
                "global_stats_case",
                "iTransformer",
                var_num=2,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)

            self.assertFalse(experiment.conf["norm_each_channel"])
            self.assertTrue(experiment.conf["rescale"])
            self.assertEqual(float(experiment.datamodule.scaler_stats["mean"]), 7.0)
            self.assertEqual(float(experiment.datamodule.scaler_stats["std"]), 1.0)
            self.assertEqual(float(experiment.conf["scaler_stats"]["mean"]), 7.0)
            self.assertEqual(float(experiment.conf["scaler_stats"]["std"]), 1.0)

    def test_migration_script_overwrite_removes_stale_optional_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            migration = _load_migration_module()
            dataset_dir = self._make_dataset_dir(tmpdir, "migration_overwrite_case")
            variable = np.arange(28 * 2, dtype=np.float32).reshape(28, 2)
            raw_timestamp = _make_timestamp(28)
            split_lengths = [12, 8, 8]

            np.savez(
                dataset_dir / "data.npz",
                scaled_variable=variable.astype(np.float32),
                timestamp=raw_timestamp,
            )
            np.save(dataset_dir / "graph.npy", np.eye(2, dtype=np.float32))

            timestamps, descriptions = migration.build_basic_ts_timestamps(raw_timestamp, ["tod", "dow"], 60)
            migration.save_split_files(dataset_dir, variable, timestamps, split_lengths, overwrite=True)
            has_graph = migration.maybe_save_graph(dataset_dir, None, overwrite=True)
            migration.save_meta(
                dataset_dir,
                dataset_name="migration_overwrite_case",
                freq=60,
                variable=variable,
                split_lengths=split_lengths,
                timestamps=timestamps,
                descriptions=descriptions,
                has_graph=has_graph,
                overwrite=True,
            )

            self.assertTrue((dataset_dir / "train_timestamps.npy").exists())
            self.assertTrue((dataset_dir / "adj_mx.pkl").exists())

            (dataset_dir / "graph.npy").unlink()
            migration.save_split_files(dataset_dir, variable, None, split_lengths, overwrite=True)
            has_graph = migration.maybe_save_graph(dataset_dir, None, overwrite=True)
            migration.save_meta(
                dataset_dir,
                dataset_name="migration_overwrite_case",
                freq=60,
                variable=variable,
                split_lengths=split_lengths,
                timestamps=None,
                descriptions=[],
                has_graph=has_graph,
                overwrite=True,
            )

            self.assertFalse((dataset_dir / "train_timestamps.npy").exists())
            self.assertFalse((dataset_dir / "val_timestamps.npy").exists())
            self.assertFalse((dataset_dir / "test_timestamps.npy").exists())
            self.assertFalse((dataset_dir / "adj_mx.pkl").exists())

            conf = self._make_conf(tmpdir, "migration_overwrite_case", "SimpleMLP", var_num=2)
            experiment = build_experiment(conf, training=False)
            self.assertEqual(experiment.conf["time_feature_dim"], 0)
            self.assertFalse(experiment.conf["has_graph"])

    def test_simple_mlp_supports_univariate_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28, dtype=np.float32).reshape(28, 1)
            self._write_sequence_dataset(tmpdir, "univariate_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="univariate_case",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=1,
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 1))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_simple_mlp_supports_multivariate_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 3, dtype=np.float32).reshape(28, 3)
            self._write_sequence_dataset(tmpdir, "multivariate_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="multivariate_case",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=3,
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 3))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_preprocess_batch_uses_train_split_stats_and_nan_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(1, 29, dtype=np.float32).reshape(28, 1)
            variable[15, 0] = np.nan
            self._write_sequence_dataset(tmpdir, "nan_mask_case", variable)

            conf = self._make_conf(
                tmpdir,
                "nan_mask_case",
                "iTransformer",
                var_num=1,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))
            processed = experiment.task.preprocess_batch(batch)

            mean = float(np.asarray(experiment.datamodule.scaler_stats["mean"]).item())
            std = float(np.asarray(experiment.datamodule.scaler_stats["std"]).item())
            raw_inputs = batch["inputs"].float()
            raw_targets = batch["targets"].float()
            expected_inputs = torch.where(
                torch.isnan(raw_inputs),
                torch.zeros_like(raw_inputs),
                (raw_inputs - mean) / std,
            )
            expected_targets = torch.where(
                torch.isnan(raw_targets),
                torch.zeros_like(raw_targets),
                (raw_targets - mean) / std,
            )

            self.assertFalse(processed["targets_mask"][0, 0, 0].item())
            self.assertFalse(torch.isnan(processed["targets"]).any().item())
            torch.testing.assert_close(processed["inputs"], expected_inputs)
            torch.testing.assert_close(processed["targets"], expected_targets)

    def test_postprocess_outputs_rescale_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(1, 29, dtype=np.float32).reshape(28, 1)
            self._write_sequence_dataset(
                tmpdir,
                "rescale_case",
                variable,
                regular_settings=self._default_regular_settings(
                    norm_each_channel=False,
                    rescale=True,
                    null_val=0.0,
                ),
            )

            conf = self._make_conf(
                tmpdir,
                "rescale_case",
                "iTransformer",
                var_num=1,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)

            prediction = torch.tensor([[[1.0]]], dtype=torch.float32)
            label = torch.tensor([[[0.0]]], dtype=torch.float32)
            targets_mask = torch.ones_like(prediction, dtype=torch.bool)
            mean = float(experiment.datamodule.scaler_stats["mean"])
            std = float(experiment.datamodule.scaler_stats["std"])

            loss = experiment.task.loss_function(prediction, label, targets_mask)
            post_prediction, post_label = experiment.task.postprocess_outputs(prediction, label, targets_mask)
            post_mse = experiment.task.mse_loss_func(post_prediction, post_label, targets_mask)

            self.assertEqual(float(loss), 1.0)
            self.assertAlmostEqual(float(post_label), mean, places=5)
            self.assertAlmostEqual(float(post_prediction), mean + std, places=5)
            self.assertAlmostEqual(float(post_mse), std * std, places=5)

    def test_postprocess_outputs_keeps_normalized_scale_when_rescale_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(1, 29, dtype=np.float32).reshape(28, 1)
            self._write_sequence_dataset(
                tmpdir,
                "no_rescale_case",
                variable,
                regular_settings=self._default_regular_settings(
                    norm_each_channel=False,
                    rescale=False,
                    null_val=0.0,
                ),
            )

            conf = self._make_conf(
                tmpdir,
                "no_rescale_case",
                "iTransformer",
                var_num=1,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)

            prediction = torch.tensor([[[1.0]]], dtype=torch.float32)
            label = torch.tensor([[[0.0]]], dtype=torch.float32)
            targets_mask = torch.ones_like(prediction, dtype=torch.bool)

            post_prediction, post_label = experiment.task.postprocess_outputs(prediction, label, targets_mask)
            post_mse = experiment.task.mse_loss_func(post_prediction, post_label, targets_mask)

            self.assertFalse(experiment.conf["rescale"])
            self.assertEqual(float(post_label), 0.0)
            self.assertEqual(float(post_prediction), 1.0)
            self.assertEqual(float(post_mse), 1.0)

    def test_basic_ts_markers_drive_itransformer_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "itransformer_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="itransformer_case",
                model_name="iTransformer",
                task_name="mtsf",
                var_num=4,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)
            self.assertEqual(experiment.conf["time_feature_dim"], 2)
            np.testing.assert_array_equal(
                experiment.datamodule.split_time_feature["train"][:5, 0],
                np.array([0, 1, 2, 3, 4], dtype=np.float32),
            )
            batch = next(iter(experiment.datamodule.val_dataloader()))
            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            var_x, marker_x, _, marker_y = experiment.task._prepare_batch(batch)
            direct_prediction = experiment.task.model(var_x, marker_x, marker_y)
            self.assertEqual(tuple(direct_prediction.shape), tuple(label.shape))

    def test_stf_path_allows_itransformer_without_graph_side_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "stf_itransformer_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="stf_itransformer_case",
                model_name="iTransformer",
                task_name="stf",
                var_num=4,
                extra_model_kwargs={
                    "output_attention": False,
                    "d_model": 8,
                    "d_ff": 16,
                    "dropout": 0.0,
                    "factor": 1,
                    "n_heads": 1,
                    "activation": "gelu",
                    "e_layers": 1,
                },
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))

            self.assertFalse(experiment.conf["has_graph"])
            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_basic_ts_markers_drive_stid_discrete_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "stid_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="stid_case",
                model_name="STID",
                task_name="mtsf",
                var_num=4,
                extra_model_kwargs={
                    "block_num": 1,
                    "ts_emb_dim": 4,
                    "node_emb_dim": 2,
                    "tod_emb_dim": 2,
                    "dow_emb_dim": 2,
                },
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                experiment = build_experiment(conf, training=False)
            np.testing.assert_array_equal(
                experiment.datamodule.split_time_feature["train"][:5, 0],
                np.array([0, 1, 2, 3, 4], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                experiment.datamodule.split_time_feature["train"][:5, 1],
                np.zeros(5, dtype=np.float32),
            )
            batch = next(iter(experiment.datamodule.val_dataloader()))
            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))

    def test_tqnet_supports_forward_and_training_step_with_future_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "tqnet_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="tqnet_case",
                model_name="TQNet",
                task_name="mtsf",
                var_num=4,
                hist_len=4,
                pred_len=2,
                extra_model_kwargs={
                    "cycle": 24,
                    "cycle_feature_name": "time of day",
                    "d_model": 8,
                    "dropout": 0.0,
                    "use_revin": True,
                    "use_tq": True,
                    "channel_aggre": True,
                    "channel_aggre_heads": 4,
                },
            )
            experiment = build_experiment(conf, training=False)
            self.assertEqual(experiment.conf["time_feature_descriptions"], ("time of day", "day of week"))
            batch = next(iter(experiment.datamodule.val_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_tqnet_uses_future_marker_anchor_for_cycle_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "tqnet_cycle_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="tqnet_cycle_case",
                model_name="TQNet",
                task_name="mtsf",
                var_num=4,
                hist_len=4,
                pred_len=2,
                extra_model_kwargs={
                    "cycle": 24,
                    "cycle_feature_name": "time of day",
                    "d_model": 8,
                    "dropout": 0.0,
                    "use_revin": True,
                    "use_tq": True,
                    "channel_aggre": True,
                    "channel_aggre_heads": 4,
                },
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))
            _, marker_x, _, marker_y = experiment.task._prepare_batch(batch)

            cycle_index = experiment.task.model._extract_cycle_index(marker_y)
            self.assertTrue(torch.equal(cycle_index, marker_y[:, 0, 0].round().long()))
            self.assertTrue(torch.any(cycle_index != marker_x[:, -1, 0].round().long()).item())

    def test_tqnet_supports_stf_without_graph_and_uses_future_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "stf_tqnet_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="stf_tqnet_case",
                model_name="TQNet",
                task_name="stf",
                var_num=4,
                hist_len=4,
                pred_len=2,
                extra_model_kwargs={
                    "cycle": 24,
                    "cycle_feature_name": "time of day",
                    "d_model": 8,
                    "dropout": 0.0,
                    "use_revin": True,
                    "use_tq": True,
                    "channel_aggre": True,
                    "channel_aggre_heads": 4,
                },
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))
            prediction, label = experiment.task.forward(batch, 0)
            _, marker_x, _, marker_y = experiment.task._prepare_batch(batch)
            cycle_index = experiment.task.model._extract_cycle_index(marker_y)

            self.assertFalse(experiment.conf["has_graph"])
            self.assertEqual(tuple(prediction.shape), (2, 2, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            self.assertTrue(torch.equal(cycle_index, marker_y[:, 0, 0].round().long()))
            self.assertTrue(torch.any(cycle_index != marker_x[:, -1, 0].round().long()).item())

    def test_tqnet_requires_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "tqnet_no_ts", variable, include_timestamps=False)

            conf = self._make_conf(
                tmpdir,
                dataset_name="tqnet_no_ts",
                model_name="TQNet",
                task_name="mtsf",
                var_num=4,
                hist_len=4,
                pred_len=2,
                extra_model_kwargs={
                    "cycle": 24,
                    "cycle_feature_name": "time of day",
                    "d_model": 8,
                    "dropout": 0.0,
                    "use_revin": True,
                    "use_tq": True,
                    "channel_aggre": True,
                    "channel_aggre_heads": 4,
                },
            )
            with self.assertRaisesRegex(ValueError, "requires time features"):
                build_experiment(conf, training=False)

    def test_tqnet_requires_time_of_day_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(
                tmpdir,
                "tqnet_missing_tod",
                variable,
                descriptions=("day of week",),
            )

            conf = self._make_conf(
                tmpdir,
                dataset_name="tqnet_missing_tod",
                model_name="TQNet",
                task_name="mtsf",
                var_num=4,
                hist_len=4,
                pred_len=2,
                extra_model_kwargs={
                    "cycle": 24,
                    "cycle_feature_name": "time of day",
                    "d_model": 8,
                    "dropout": 0.0,
                    "use_revin": True,
                    "use_tq": True,
                    "channel_aggre": True,
                    "channel_aggre_heads": 4,
                },
            )
            with self.assertRaisesRegex(ValueError, "time of day"):
                build_experiment(conf, training=False)

    def test_tqnet_time_of_week_extracts_weekly_phase(self):
        model = TQNetModel(
            hist_len=4,
            pred_len=2,
            var_num=4,
            cycle=168,
            cycle_feature_name="time of week",
            time_feature_descriptions=("time of day", "day of week"),
            d_model=8,
            dropout=0.0,
            use_revin=True,
            use_tq=True,
            channel_aggre=True,
            channel_aggre_heads=4,
        )
        marker_y = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[23.0, 6.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        cycle_index = model._extract_cycle_index(marker_y)
        self.assertEqual(cycle_index.tolist(), [0, 167])

    def test_tqnet_time_of_week_requires_day_of_week_feature(self):
        with self.assertRaisesRegex(ValueError, "day of week"):
            TQNetModel(
                hist_len=4,
                pred_len=2,
                var_num=4,
                cycle=168,
                cycle_feature_name="time of week",
                time_feature_descriptions=("time of day",),
                d_model=8,
                dropout=0.0,
                use_revin=True,
                use_tq=True,
                channel_aggre=True,
                channel_aggre_heads=4,
            )

    def test_tqnet_time_of_week_requires_cycle_divisible_by_seven(self):
        with self.assertRaisesRegex(ValueError, "divisible by 7"):
            TQNetModel(
                hist_len=4,
                pred_len=2,
                var_num=4,
                cycle=24,
                cycle_feature_name="time of week",
                time_feature_descriptions=("time of day", "day of week"),
                d_model=8,
                dropout=0.0,
                use_revin=True,
                use_tq=True,
                channel_aggre=True,
                channel_aggre_heads=4,
            )

    def test_tqnet_time_of_day_extracts_cycle_index_for_pems_style_timestamps(self):
        model = TQNetModel(
            hist_len=4,
            pred_len=2,
            var_num=4,
            cycle=288,
            cycle_feature_name="time of day",
            time_feature_descriptions=("time of day", "day of week"),
            d_model=8,
            dropout=0.0,
            use_revin=False,
            use_tq=True,
            channel_aggre=True,
            channel_aggre_heads=4,
        )
        marker_y = torch.tensor(
            [
                [[172.0, 4.0], [173.0, 4.0]],
                [[229.0, 5.0], [230.0, 5.0]],
            ],
            dtype=torch.float32,
        )
        cycle_index = model._extract_cycle_index(marker_y)
        self.assertEqual(cycle_index.tolist(), [172, 229])

    def test_tqnet_experiment_presets_load(self):
        expected = {
            "tqnet/etth1": ("TQNet", "mtsf", "ETTh1"),
            "tqnet/weather": ("TQNet", "mtsf", "Weather"),
            "tqnet/traffic": ("TQNet", "mtsf", "Traffic"),
            "tqnet/ecl": ("TQNet", "mtsf", "ECL"),
            "tqnet/electricity": ("TQNet", "mtsf", "Electricity"),
            "tqnet/pems03": ("TQNet", "mtsf", "PEMS03"),
        }

        for config_ref, (model_name, task_name, dataset_name) in expected.items():
            conf = load_config(config_ref)
            self.assertEqual(conf["model_name"], model_name)
            self.assertEqual(conf["task_name"], task_name)
            self.assertEqual(conf["dataset_name"], dataset_name)

        electricity_conf = load_config("tqnet/electricity")
        self.assertEqual(electricity_conf["cycle"], 168)
        self.assertEqual(electricity_conf["cycle_feature_name"], "time of week")
        self.assertEqual(electricity_conf["lr_scheduler"], "CycleNetLRS")
        self.assertEqual(electricity_conf["batch_size"], 32)
        self.assertFalse(electricity_conf["use_mix_loss"])

        pems03_conf = load_config("tqnet/pems03")
        self.assertEqual(pems03_conf["cycle"], 288)
        self.assertEqual(pems03_conf["cycle_feature_name"], "time of day")
        self.assertFalse(pems03_conf["use_revin"])
        self.assertEqual(pems03_conf["lr_scheduler"], "CycleNetLRS")

    def test_tqnet_study_dry_run_expands_cases(self):
        result = run_study(
            "tqnet/core",
            runtime_overrides={
                "data_root": "dataset",
                "save_root": "save",
                "accelerator": "cpu",
                "devices": 1,
                "use_wandb": 0,
            },
            dry_run=True,
        )

        self.assertEqual(result["study_name"], "tqnet_core")
        self.assertEqual(result["run_count"], 15)
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["runs_path"])
        self.assertIsNone(result["summary_path"])

    def test_tqnet_alignment_study_dry_run_expands_cases(self):
        result = run_study(
            "tqnet/alignment",
            runtime_overrides={
                "data_root": "dataset",
                "save_root": "save",
                "accelerator": "cpu",
                "devices": 1,
                "use_wandb": 0,
            },
            dry_run=True,
        )

        self.assertEqual(result["study_name"], "tqnet_alignment")
        self.assertEqual(result["run_count"], 8)
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["runs_path"])
        self.assertIsNone(result["summary_path"])

    def test_stgcn_supports_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            graph = np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            dataset_dir = self._write_sequence_dataset(tmpdir, "graph_case", variable)
            self._write_graph(tmpdir, "graph_case", graph)
            meta = self._load_meta(dataset_dir)
            meta["has_graph"] = True
            self._save_meta(dataset_dir, meta)

            conf = self._make_conf(
                tmpdir,
                dataset_name="graph_case",
                model_name="STGCN",
                task_name="stf",
                var_num=4,
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            var_x, marker_x, _, marker_y = experiment.task._prepare_batch(batch)
            direct_prediction = experiment.task.model(var_x, marker_x, marker_y)
            self.assertEqual(tuple(direct_prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_stf_path_uses_global_scaler_settings_from_regular_settings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            variable[13, 0] = 0.0
            graph = np.eye(4, dtype=np.float32)
            dataset_dir = self._write_sequence_dataset(
                tmpdir,
                "graph_scaler_case",
                variable,
                regular_settings=self._default_regular_settings(
                    norm_each_channel=False,
                    rescale=True,
                    null_val=0.0,
                ),
            )
            self._write_graph(tmpdir, "graph_scaler_case", graph)
            meta = self._load_meta(dataset_dir)
            meta["has_graph"] = True
            self._save_meta(dataset_dir, meta)

            conf = self._make_conf(
                tmpdir,
                dataset_name="graph_scaler_case",
                model_name="STGCN",
                task_name="stf",
                var_num=4,
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.val_dataloader()))
            processed = experiment.task.preprocess_batch(batch)
            var_x, marker_x, _, marker_y = experiment.task._prepare_batch(batch)
            direct_prediction = experiment.task.model(var_x, marker_x, marker_y)

            self.assertFalse(experiment.conf["norm_each_channel"])
            self.assertTrue(experiment.conf["rescale"])
            self.assertTrue(np.isscalar(experiment.conf["scaler_stats"]["mean"]))
            self.assertTrue(torch.equal(processed["inputs_mask"], (batch["inputs"] != 0).to(dtype=torch.bool)))
            self.assertEqual(tuple(direct_prediction.shape), tuple(batch["targets"].shape))

    def test_stf_path_supports_basic_ts_graph_tuple_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 3, dtype=np.float32).reshape(28, 3)
            graph = np.eye(3, dtype=np.float32)
            dataset_dir = self._write_sequence_dataset(tmpdir, "graph_tuple_case", variable)
            self._write_graph(tmpdir, "graph_tuple_case", graph, tuple_format=True)
            meta = self._load_meta(dataset_dir)
            meta["has_graph"] = True
            self._save_meta(dataset_dir, meta)

            conf = self._make_conf(tmpdir, "graph_tuple_case", "STGCN", task_name="stf", var_num=3)
            experiment = build_experiment(conf, training=False)
            np.testing.assert_array_equal(experiment.datamodule.graph, graph)

    def test_stf_path_requires_graph_side_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "graph_missing", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="graph_missing",
                model_name="STGCN",
                task_name="stf",
                var_num=4,
            )
            with self.assertRaisesRegex(ValueError, "requires side inputs"):
                build_experiment(conf, training=False)

    def test_stf_path_rejects_graph_node_count_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            dataset_dir = self._write_sequence_dataset(tmpdir, "graph_mismatch", variable)
            self._write_graph(tmpdir, "graph_mismatch", np.eye(3, dtype=np.float32))
            meta = self._load_meta(dataset_dir)
            meta["has_graph"] = True
            self._save_meta(dataset_dir, meta)

            conf = self._make_conf(
                tmpdir,
                dataset_name="graph_mismatch",
                model_name="STGCN",
                task_name="stf",
                var_num=4,
            )
            with self.assertRaisesRegex(ValueError, "graph node count"):
                build_experiment(conf, training=False)

    def test_model_contract_rejects_unsupported_task_combination(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            self._write_sequence_dataset(tmpdir, "unsupported_combo", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="unsupported_combo",
                model_name="RLinear",
                task_name="stf",
                var_num=4,
            )
            with self.assertRaisesRegex(ValueError, "does not support task 'stf'"):
                build_experiment(conf, training=False)

    def test_simple_grid_mlp_supports_grid2d_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 2 * 3 * 4, dtype=np.float32).reshape(16, 2, 3, 4)
            grid_mask = np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            coord = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), axis=0).astype(np.float32)
            self._write_grid_dataset(tmpdir, "grid2d_case", variable, grid_mask=grid_mask, coord=coord)

            conf = self._make_conf(
                tmpdir,
                dataset_name="grid2d_case",
                model_name="SimpleGridMLP",
                task_name="grid2dtsf",
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.train_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_simple_grid_mlp_supports_grid3d_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(16, 2, 2, 3, 4)
            grid_mask = np.ones((2, 3, 4), dtype=np.float32)
            coord = np.stack(
                np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij"),
                axis=0,
            ).astype(np.float32)
            self._write_grid3d_sharded_dataset(
                tmpdir,
                "grid3d_case",
                variable,
                grid_mask=grid_mask,
                coord=coord,
                split_lengths=(6, 5, 5),
            )

            conf = self._make_conf(
                tmpdir,
                dataset_name="grid3d_case",
                model_name="SimpleGridMLP",
                task_name="grid3dtsf",
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.train_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 2, 2, 3, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_simple_mlp_experiment_presets_load(self):
        expected = {
            "simplemlp/pseudo": ("SimpleMLP", "mtsf", "Pseudo"),
            "simplemlp/etth1": ("SimpleMLP", "mtsf", "ETTh1"),
            "simplegridmlp/grid2d_demo": ("SimpleGridMLP", "grid2dtsf", "Grid2DDemo"),
            "simplegridmlp/grid3d_demo": ("SimpleGridMLP", "grid3dtsf", "Grid3DDemo"),
            "simplegridmlp/windfield3d_demo": ("SimpleGridMLP", "grid3dtsf", "WindField3DDemo"),
        }

        for config_ref, (model_name, task_name, dataset_name) in expected.items():
            conf = load_config(config_ref)
            self.assertEqual(conf["model_name"], model_name)
            self.assertEqual(conf["task_name"], task_name)
            self.assertEqual(conf["dataset_name"], dataset_name)

    def test_smoke_training_runs_simple_mlp_univariate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28, dtype=np.float32).reshape(28, 1)
            self._write_sequence_dataset(tmpdir, "train_univariate", variable)
            conf = self._make_conf(
                tmpdir,
                dataset_name="train_univariate",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=1,
            )

            metrics = run_training(conf)
            self.assertEqual(metrics["status"], "success")
            self.assertTrue(Path(metrics["ckpt_path"]).exists())

    def test_smoke_training_runs_simple_mlp_multivariate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 3, dtype=np.float32).reshape(28, 3)
            self._write_sequence_dataset(tmpdir, "train_multivariate", variable)
            conf = self._make_conf(
                tmpdir,
                dataset_name="train_multivariate",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=3,
            )

            metrics = run_training(conf)
            self.assertEqual(metrics["status"], "success")
            self.assertTrue(Path(metrics["ckpt_path"]).exists())

    def test_smoke_training_runs_stgcn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(28 * 4, dtype=np.float32).reshape(28, 4)
            graph = np.eye(4, dtype=np.float32)
            dataset_dir = self._write_sequence_dataset(tmpdir, "train_graph", variable)
            self._write_graph(tmpdir, "train_graph", graph)
            meta = self._load_meta(dataset_dir)
            meta["has_graph"] = True
            self._save_meta(dataset_dir, meta)
            conf = self._make_conf(
                tmpdir,
                dataset_name="train_graph",
                model_name="STGCN",
                task_name="stf",
                var_num=4,
            )

            metrics = run_training(conf)
            self.assertEqual(metrics["status"], "success")
            self.assertTrue(Path(metrics["ckpt_path"]).exists())

    def test_simplegridmlp_study_dry_run_expands_cases(self):
        result = run_study(
            "simplegridmlp/core",
            runtime_overrides={
                "data_root": "dataset",
                "save_root": "save",
                "accelerator": "cpu",
                "devices": 1,
                "use_wandb": 0,
            },
            dry_run=True,
        )

        self.assertEqual(result["study_name"], "simplegridmlp_core")
        self.assertEqual(result["run_count"], 2)
        self.assertEqual(result["rows"], [])
        self.assertIsNone(result["runs_path"])
        self.assertIsNone(result["summary_path"])

    def test_smoke_training_runs_simple_grid2d_mlp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 2 * 3 * 4, dtype=np.float32).reshape(16, 2, 3, 4)
            grid_mask = np.ones((3, 4), dtype=np.float32)
            coord = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), axis=0).astype(np.float32)
            self._write_grid_dataset(tmpdir, "train_grid2d", variable, grid_mask=grid_mask, coord=coord)
            conf = self._make_conf(
                tmpdir,
                dataset_name="train_grid2d",
                model_name="SimpleGridMLP",
                task_name="grid2dtsf",
            )

            metrics = run_training(conf)
            self.assertEqual(metrics["status"], "success")
            self.assertTrue(Path(metrics["ckpt_path"]).exists())

    def test_smoke_training_runs_simple_grid3d_mlp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(16, 2, 2, 3, 4)
            grid_mask = np.ones((2, 3, 4), dtype=np.float32)
            coord = np.stack(
                np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij"),
                axis=0,
            ).astype(np.float32)
            self._write_grid3d_sharded_dataset(
                tmpdir,
                "train_grid3d",
                variable,
                grid_mask=grid_mask,
                coord=coord,
                split_lengths=(6, 5, 5),
            )
            conf = self._make_conf(
                tmpdir,
                dataset_name="train_grid3d",
                model_name="SimpleGridMLP",
                task_name="grid3dtsf",
            )

            metrics = run_training(conf)
            self.assertEqual(metrics["status"], "success")
            self.assertTrue(Path(metrics["ckpt_path"]).exists())


if __name__ == "__main__":
    unittest.main()
