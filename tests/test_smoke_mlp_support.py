import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from easytsf.workflow.experiment import build_experiment, finalize_runtime_conf, load_config, run_training


def _make_timestamp(length, unit="h"):
    base = np.datetime64("2024-01-01T00:00")
    return base + np.arange(length) * np.timedelta64(1, unit)


class SmokeMLPSupportTestCase(unittest.TestCase):
    def _write_sequence_dataset(self, root, dataset_name, variable):
        dataset_path = Path(root) / "{}.npz".format(dataset_name)
        np.savez(
            dataset_path,
            scaled_variable=variable.astype(np.float32),
            timestamp=_make_timestamp(variable.shape[0]),
        )
        return dataset_path

    def _write_grid_dataset(self, root, dataset_name, variable, grid_mask=None, coord=None):
        dataset_path = Path(root) / "{}.npz".format(dataset_name)
        payload = {
            "scaled_variable": variable.astype(np.float32),
            "timestamp": _make_timestamp(variable.shape[0]),
        }
        if grid_mask is not None:
            payload["grid_mask"] = grid_mask.astype(np.float32)
        if coord is not None:
            payload["coord"] = coord.astype(np.float32)
        np.savez(dataset_path, **payload)
        return dataset_path

    def _write_graph(self, root, graph_name, graph):
        graph_path = Path(root) / graph_name
        np.save(graph_path, graph.astype(np.float32))
        return graph_path

    def _make_conf(
        self,
        root,
        dataset_name,
        model_name,
        task_name="mtsf",
        var_num=None,
        graph_path=None,
        batch_size=2,
        hist_len=3,
        pred_len=2,
        max_epochs=1,
    ):
        time_feature_cls = ["tod"] if task_name == "mtsf" else []
        base_conf = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "hist_len": hist_len,
            "pred_len": pred_len,
            "batch_size": batch_size,
            "num_workers": 0,
            "norm_time_feature": False,
            "data_split": [8, 4, 4],
            "time_feature_cls": time_feature_cls,
            "freq": 60,
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
        if graph_path is not None:
            base_conf["graph_path"] = graph_path
        return finalize_runtime_conf(base_conf)

    def _assert_scalar_loss(self, loss):
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss).item())

    def _run_training_step_without_trainer(self, task, batch):
        task.log = lambda *args, **kwargs: None
        return task.training_step(batch, 0)

    def test_mtsf_path_rejects_raw_univariate_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16, dtype=np.float32)
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
            variable = np.arange(16, dtype=np.float32)
            self._write_sequence_dataset(tmpdir, "raw_univariate_graph", variable)
            self._write_graph(tmpdir, "graph.npy", np.eye(1, dtype=np.float32))

            conf = self._make_conf(
                tmpdir,
                dataset_name="raw_univariate_graph",
                model_name="SimpleGraphMLP",
                task_name="stf",
                var_num=1,
                graph_path="graph.npy",
            )
            with self.assertRaisesRegex(ValueError, r"\[L, 1\]"):
                build_experiment(conf, training=False)

    def test_simple_mlp_supports_univariate_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16, dtype=np.float32).reshape(16, 1)
            self._write_sequence_dataset(tmpdir, "univariate_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="univariate_case",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=1,
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.train_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 1))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_simple_mlp_supports_multivariate_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
            self._write_sequence_dataset(tmpdir, "multivariate_case", variable)

            conf = self._make_conf(
                tmpdir,
                dataset_name="multivariate_case",
                model_name="SimpleMLP",
                task_name="mtsf",
                var_num=3,
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.train_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 3))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

    def test_simple_graph_mlp_supports_forward_and_training_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
            graph = np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
            self._write_sequence_dataset(tmpdir, "graph_case", variable)
            self._write_graph(tmpdir, "graph.npy", graph)

            conf = self._make_conf(
                tmpdir,
                dataset_name="graph_case",
                model_name="SimpleGraphMLP",
                task_name="stf",
                var_num=4,
                graph_path="graph.npy",
            )
            experiment = build_experiment(conf, training=False)
            batch = next(iter(experiment.datamodule.train_dataloader()))

            prediction, label = experiment.task.forward(batch, 0)
            self.assertEqual(tuple(prediction.shape), (2, 2, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))
            loss = self._run_training_step_without_trainer(experiment.task, batch)
            self._assert_scalar_loss(loss)

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
            self._write_grid_dataset(tmpdir, "grid3d_case", variable, grid_mask=grid_mask, coord=coord)

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
            "simplegraphmlp/pems03": ("SimpleGraphMLP", "stf", "PEMS03"),
            "simplegridmlp/grid2d_demo": ("SimpleGridMLP", "grid2dtsf", "Grid2DDemo"),
            "simplegridmlp/grid3d_demo": ("SimpleGridMLP", "grid3dtsf", "Grid3DDemo"),
        }

        for config_ref, (model_name, task_name, dataset_name) in expected.items():
            conf = load_config(config_ref)
            self.assertEqual(conf["model_name"], model_name)
            self.assertEqual(conf["task_name"], task_name)
            self.assertEqual(conf["dataset_name"], dataset_name)

    def test_smoke_training_runs_simple_mlp_univariate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16, dtype=np.float32).reshape(16, 1)
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
            variable = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
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

    def test_smoke_training_runs_simple_graph_mlp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(16 * 4, dtype=np.float32).reshape(16, 4)
            graph = np.eye(4, dtype=np.float32)
            self._write_sequence_dataset(tmpdir, "train_graph", variable)
            self._write_graph(tmpdir, "graph.npy", graph)
            conf = self._make_conf(
                tmpdir,
                dataset_name="train_graph",
                model_name="SimpleGraphMLP",
                task_name="stf",
                var_num=4,
                graph_path="graph.npy",
            )

            metrics = run_training(conf)
            self.assertEqual(metrics["status"], "success")
            self.assertTrue(Path(metrics["ckpt_path"]).exists())

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
            self._write_grid_dataset(tmpdir, "train_grid3d", variable, grid_mask=grid_mask, coord=coord)
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
