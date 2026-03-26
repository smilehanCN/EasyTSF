import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from easytsf.data import GridDataInterface
from easytsf.task import Grid2DTSFTask, Grid3DTSFTask, GridSTFTask
from easytsf.workflow.experiment import build_experiment, finalize_runtime_conf


def _make_timestamp(length, unit="h"):
    base = np.datetime64("2024-01-01T00:00")
    return base + np.arange(length) * np.timedelta64(1, unit)


class DummyGridModel(torch.nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = int(pred_len)

    def forward(self, var_x, marker_x, grid_mask=None, coord=None):
        del marker_x, grid_mask, coord
        return var_x[:, -self.pred_len:, ...].contiguous()


def _make_common_conf(tmpdir, dataset_name, task_name):
    base_conf = {
        "model_name": "DummyGridModel",
        "dataset_name": dataset_name,
        "hist_len": 3,
        "pred_len": 2,
        "batch_size": 2,
        "num_workers": 0,
        "norm_time_feature": False,
        "data_split": [7, 3, 2],
        "time_feature_cls": ["tod"],
        "freq": 60,
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
        "task_name": task_name,
        "data_root": tmpdir,
        "save_root": tmpdir,
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
    }
    return finalize_runtime_conf(base_conf)


class GridSupportTestCase(unittest.TestCase):
    def _make_dataset_dir(self, root, dataset_name):
        dataset_dir = Path(root) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _write_grid_dataset(self, root, dataset_name, variable, grid_mask=None, coord=None):
        dataset_dir = self._make_dataset_dir(root, dataset_name)
        dataset_path = dataset_dir / "data.npz"
        np.savez(
            dataset_path,
            scaled_variable=variable.astype(np.float32),
            timestamp=_make_timestamp(variable.shape[0]),
        )
        if grid_mask is not None:
            np.save(dataset_dir / "grid_mask.npy", grid_mask.astype(np.float32))
        if coord is not None:
            np.save(dataset_dir / "coord.npy", coord.astype(np.float32))
        return dataset_path

    def _write_legacy_flat_grid_dataset(self, root, dataset_name, variable):
        dataset_path = Path(root) / "{}.npz".format(dataset_name)
        np.savez(
            dataset_path,
            scaled_variable=variable.astype(np.float32),
            timestamp=_make_timestamp(variable.shape[0]),
        )
        return dataset_path

    def _build_grid_datamodule(self, root, dataset_name):
        return GridDataInterface(
            num_workers=0,
            batch_size=2,
            hist_len=3,
            pred_len=2,
            norm_time_feature=False,
            data_split=[7, 3, 2],
            time_feature_cls=["tod"],
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2,
            use_mmap=False,
            cache_npz_as_npy=False,
            precompute_window_index=False,
            data_root=root,
            dataset_name=dataset_name,
            freq=60,
        )

    def test_grid_data_interface_loads_2d_side_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
            grid_mask = np.ones((3, 4), dtype=np.float32)
            coord = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), axis=0).astype(np.float32)
            self._write_grid_dataset(tmpdir, "grid2d_case", variable, grid_mask=grid_mask, coord=coord)

            datamodule = self._build_grid_datamodule(tmpdir, "grid2d_case")
            batch = next(iter(datamodule.train_dataloader()))

            self.assertEqual(datamodule.channel_num, 2)
            self.assertEqual(datamodule.spatial_shape, (3, 4))
            self.assertEqual(datamodule.spatial_ndim, 2)
            self.assertEqual(tuple(datamodule.grid_mask.shape), (3, 4))
            self.assertEqual(tuple(datamodule.coord.shape), (2, 3, 4))
            self.assertEqual(tuple(batch[0].shape), (2, 3, 2, 3, 4))
            self.assertEqual(tuple(batch[1].shape), (2, 3, 1))
            self.assertEqual(tuple(batch[2].shape), (2, 2, 2, 3, 4))

    def test_grid_data_interface_validates_3d_coord_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 2, 3, 4)
            grid_mask = np.ones((2, 3, 4), dtype=np.float32)
            bad_coord = np.zeros((2, 2, 3, 4), dtype=np.float32)
            self._write_grid_dataset(tmpdir, "grid3d_bad_coord", variable, grid_mask=grid_mask, coord=bad_coord)

            with self.assertRaisesRegex(ValueError, "coord shape"):
                self._build_grid_datamodule(tmpdir, "grid3d_bad_coord")

    def test_grid_data_interface_rejects_legacy_flat_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
            self._write_legacy_flat_grid_dataset(tmpdir, "legacy_grid", variable)

            with self.assertRaisesRegex(FileNotFoundError, "legacy flat dataset layout is no longer supported"):
                self._build_grid_datamodule(tmpdir, "legacy_grid")

    def test_windfield3d_directory_contract_loads_expected_shapes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = self._make_dataset_dir(tmpdir, "WindField3DDemo")
            variable = np.zeros((16, 3, 64, 64, 64), dtype=np.float32)
            timestamp = _make_timestamp(variable.shape[0], unit="m")
            axes = [
                np.linspace(-320.0, 310.0, 64, dtype=np.float32),
                np.linspace(-320.0, 310.0, 64, dtype=np.float32),
                np.linspace(750.0, 1380.0, 64, dtype=np.float32),
            ]
            coord = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0).astype(np.float32)
            np.savez(dataset_dir / "data.npz", scaled_variable=variable, timestamp=timestamp)
            np.save(dataset_dir / "coord.npy", coord)

            datamodule = GridDataInterface(
                num_workers=0,
                batch_size=1,
                hist_len=2,
                pred_len=2,
                norm_time_feature=False,
                data_split=[10, 3, 3],
                time_feature_cls=[],
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=2,
                use_mmap=False,
                cache_npz_as_npy=False,
                precompute_window_index=False,
                data_root=tmpdir,
                dataset_name="WindField3DDemo",
                freq=1,
            )
            batch = next(iter(datamodule.train_dataloader()))

            self.assertEqual(datamodule.channel_num, 3)
            self.assertEqual(datamodule.spatial_shape, (64, 64, 64))
            self.assertEqual(datamodule.spatial_ndim, 3)
            self.assertEqual(tuple(datamodule.coord.shape), (3, 64, 64, 64))
            self.assertIsNone(datamodule.grid_mask)
            self.assertEqual(tuple(batch[0].shape), (1, 2, 3, 64, 64, 64))
            self.assertEqual(tuple(batch[2].shape), (1, 2, 3, 64, 64, 64))

    def test_build_experiment_dispatches_grid2d_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
            grid_mask = np.ones((3, 4), dtype=np.float32)
            coord = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), axis=0).astype(np.float32)
            self._write_grid_dataset(tmpdir, "grid2d_case", variable, grid_mask=grid_mask, coord=coord)
            conf = _make_common_conf(tmpdir, "grid2d_case", "grid2dtsf")

            def build_dummy_model(task):
                return DummyGridModel(pred_len=task.hparams.pred_len)

            with mock.patch.object(Grid2DTSFTask, "_build_model", build_dummy_model):
                experiment = build_experiment(conf, training=False)
                batch = next(iter(experiment.datamodule.train_dataloader()))
                prediction, label = experiment.task.forward(batch, 0)

            self.assertIsInstance(experiment.task, Grid2DTSFTask)
            self.assertEqual(experiment.conf["spatial_ndim"], 2)
            self.assertEqual(experiment.conf["channel_num"], 2)
            self.assertEqual(tuple(experiment.task.grid_mask.shape), (3, 4))
            self.assertEqual(tuple(experiment.task.coord.shape), (2, 3, 4))
            self.assertEqual(tuple(prediction.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))

    def test_build_experiment_dispatches_grid3d_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 2, 3, 4)
            grid_mask = np.ones((2, 3, 4), dtype=np.float32)
            coord = np.stack(
                np.meshgrid(np.arange(2), np.arange(3), np.arange(4), indexing="ij"),
                axis=0,
            ).astype(np.float32)
            self._write_grid_dataset(tmpdir, "grid3d_case", variable, grid_mask=grid_mask, coord=coord)
            conf = _make_common_conf(tmpdir, "grid3d_case", "grid3dtsf")

            def build_dummy_model(task):
                return DummyGridModel(pred_len=task.hparams.pred_len)

            with mock.patch.object(Grid3DTSFTask, "_build_model", build_dummy_model):
                experiment = build_experiment(conf, training=False)
                batch = next(iter(experiment.datamodule.train_dataloader()))
                prediction, label = experiment.task.forward(batch, 0)

            self.assertIsInstance(experiment.task, Grid3DTSFTask)
            self.assertEqual(experiment.conf["spatial_ndim"], 3)
            self.assertEqual(experiment.conf["channel_num"], 2)
            self.assertEqual(tuple(experiment.task.grid_mask.shape), (2, 3, 4))
            self.assertEqual(tuple(experiment.task.coord.shape), (3, 2, 3, 4))
            self.assertEqual(tuple(prediction.shape), (2, 2, 2, 2, 3, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))

    def test_build_experiment_dispatches_gridstf_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
            grid_mask = np.ones((3, 4), dtype=np.float32)
            coord = np.stack(np.meshgrid(np.arange(3), np.arange(4), indexing="ij"), axis=0).astype(np.float32)
            self._write_grid_dataset(tmpdir, "gridstf_case", variable, grid_mask=grid_mask, coord=coord)
            conf = _make_common_conf(tmpdir, "gridstf_case", "gridstf")

            def build_dummy_model(task):
                return DummyGridModel(pred_len=task.hparams.pred_len)

            with mock.patch.object(GridSTFTask, "_build_model", build_dummy_model):
                experiment = build_experiment(conf, training=False)
                batch = next(iter(experiment.datamodule.train_dataloader()))
                prediction, label = experiment.task.forward(batch, 0)

            self.assertIsInstance(experiment.task, GridSTFTask)
            self.assertEqual(tuple(prediction.shape), (2, 2, 2, 3, 4))
            self.assertEqual(tuple(prediction.shape), tuple(label.shape))

    def test_build_experiment_rejects_task_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            variable = np.arange(12 * 2 * 3 * 4, dtype=np.float32).reshape(12, 2, 3, 4)
            self._write_grid_dataset(tmpdir, "grid2d_case", variable)
            conf = _make_common_conf(tmpdir, "grid2d_case", "grid3dtsf")

            with self.assertRaisesRegex(ValueError, "grid3dtsf experiment requires"):
                build_experiment(conf, training=False)


if __name__ == "__main__":
    unittest.main()
