import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from easytsf.data import Grid3DDataModule
from easytsf.task import get_task_components
from easytsf.task.grid3d_forecasting import Grid3DForecastingTask
from scripts.grid3d_import import import_grid3d_dataset


GRID_SHAPE = (4, 4, 3)
SPLIT_SPEC = {
    "train": [0, 5],
    "val": [5, 8],
    "test": [8, 11],
}


def _write_grid3d_step(path: Path, step_index: int, *, irregular_z: bool = False) -> None:
    y_size, x_size, z_size = GRID_SHAPE
    y = np.arange(y_size, dtype=np.float32)
    x = np.arange(x_size, dtype=np.float32)
    z = np.arange(z_size, dtype=np.float32)
    if irregular_z:
        z = z.copy()
        z[-1] += 0.5
    yy, xx, zz = np.meshgrid(y, x, z, indexing="ij")

    base = float(step_index)
    u = (base + yy + 0.1 * xx + 0.01 * zz).astype(np.float32)
    v = (base * 0.5 - 0.2 * yy + xx + 0.02 * zz).astype(np.float32)
    w = (-base * 0.25 + 0.3 * yy - 0.1 * xx + zz).astype(np.float32)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("U", data=u)
        handle.create_dataset("V", data=v)
        handle.create_dataset("W", data=w)
        handle.create_dataset("x", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("z", data=z)
        handle.create_dataset("time_s", data=np.asarray(step_index * 60.0, dtype=np.float64))


def _build_raw_dataset(root: Path, *, irregular_z: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for step_index in range(11):
        _write_grid3d_step(
            root / "wind_grid_t{:04d}.nc".format(step_index * 60),
            step_index,
            irregular_z=irregular_z,
        )
    return root


def _import_tiny_grid3d_dataset(tmp_path: Path) -> Path:
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_tiny"
    import_grid3d_dataset(
        input_dir=raw_dir,
        out_dir=dataset_dir,
        split_spec=SPLIT_SPEC,
        z_slice_end=None,
    )
    return dataset_dir


def _runtime_conf(dataset_root: Path, dataset_name: str) -> dict:
    return {
        "model": "unet3d",
        "dataset": dataset_name,
        "hist_len": 2,
        "pred_len": 1,
        "batch_size": 1,
        "num_workers": 0,
        "lr": 1e-3,
        "lr_scheduler": "StepLR",
        "lr_step_size": 1,
        "lr_gamma": 1.0,
        "optimizer": "Adam",
        "max_epochs": 1,
        "es_patience": 1,
        "val_metric": "val/loss",
        "val_metric_mode": "min",
        "gradient_clip_val": 0.0,
        "gradient_clip_algorithm": "norm",
        "test_metric_space": "original",
        "task": "grid3d_forecasting",
        "accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "use_mmap": False,
        "use_coords": True,
        "base_channels": 2,
        "patch_size": [1, 1, 1],
        "downsample_scale": [1, 1, 1],
        "downsample_scales": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        "kernel_size": [3, 3, 3],
        "expansion": 2,
        "data_root": str(dataset_root),
        "save_root": str(dataset_root / "checkpoints"),
        "seed": 42,
    }


def _prepare_runtime_conf(runtime_conf: dict) -> tuple[Grid3DDataModule, dict]:
    datamodule = Grid3DDataModule(**runtime_conf)
    prepared_conf = dict(runtime_conf)
    prepared_conf.update(datamodule.export_task_hparams())
    return datamodule, prepared_conf


def test_grid3d_importer_datamodule_and_task_smoke(tmp_path):
    dataset_dir = _import_tiny_grid3d_dataset(tmp_path)

    with (dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    assert meta["storage_format"] == "grid3d_split_npy_v1"
    assert meta["data_layout"] == "T,C,Y,X,Z"
    assert meta["grid_shape"] == list(GRID_SHAPE)
    assert meta["channel_names"] == ["U", "V", "W", "shear_x", "shear_y", "shear_z"]
    assert np.load(dataset_dir / "train_data.npy").shape == (5, 6, *GRID_SHAPE)
    assert np.load(dataset_dir / "coord.npy").shape == (3, *GRID_SHAPE)

    runtime_conf = _runtime_conf(tmp_path, dataset_dir.name)
    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.train_dataloader()))

    assert batch["inputs"].shape == (1, 2, 6, *GRID_SHAPE)
    assert batch["targets"].shape == (1, 1, 6, *GRID_SHAPE)
    assert batch["coords"].shape == (1, 3, *GRID_SHAPE)

    datamodule_cls, task_cls = get_task_components("grid3d_forecasting")
    assert datamodule_cls is Grid3DDataModule
    assert task_cls is Grid3DForecastingTask

    task = Grid3DForecastingTask(**prepared_conf)
    prediction, label = task._forward(batch)
    loss = task.loss_function(prediction, label)

    assert prediction.shape == (1, 1, 6, *GRID_SHAPE)
    assert label.shape == (1, 1, 6, *GRID_SHAPE)
    assert torch.isfinite(loss)


def test_grid3d_importer_rejects_irregular_spacing(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw", irregular_z=True)

    with pytest.raises(ValueError, match="axis 'z' must be evenly spaced"):
        import_grid3d_dataset(
            input_dir=raw_dir,
            out_dir=tmp_path / "grid3d_tiny",
            split_spec=SPLIT_SPEC,
            z_slice_end=None,
        )


def test_grid3d_datamodule_rejects_removed_patch_args(tmp_path):
    dataset_dir = _import_tiny_grid3d_dataset(tmp_path)
    runtime_conf = _runtime_conf(tmp_path, dataset_dir.name)
    runtime_conf["train_patch_shape"] = [2, 2, 2]

    with pytest.raises(ValueError, match="train_patch_shape is no longer supported"):
        Grid3DDataModule(**runtime_conf)
