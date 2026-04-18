import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from easytsf.data import Grid3DDataModule
from easytsf.task import get_task_components
from easytsf.task.grid3d_forecasting import Grid3DForecastingTask
from easytsf.workflow.experiment import prepare_runtime_conf_for_task
from scripts.grid3d_import import import_grid3d_dataset


GRID_SHAPE = (16, 16, 30)
TRAIN_SPLIT_SPEC = {
    "train": [0, 24],
    "val": [24, 44],
    "test": [44, 64],
}


def _write_grid3d_step(
    path: Path,
    step_index: int,
    grid_shape: tuple[int, int, int] = GRID_SHAPE,
    *,
    irregular_z: bool = False,
) -> None:
    y_size, x_size, z_size = grid_shape
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


def _build_raw_dataset(root: Path, num_steps: int = 64, *, irregular_z: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for step_index in range(num_steps):
        _write_grid3d_step(
            root / "wind_grid_t{:04d}.nc".format(step_index * 60),
            step_index,
            irregular_z=irregular_z,
        )
    return root


def _build_runtime_conf(
    dataset_root: Path,
    dataset_name: str,
    *,
    hist_len: int,
    pred_len: int,
) -> dict:
    runtime_conf = {
        "model": "unet3d",
        "dataset": dataset_name,
        "hist_len": hist_len,
        "pred_len": pred_len,
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
        "output_mode": "regression",
        "accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "use_mmap": False,
        "use_coords": True,
        "base_channels": 8,
        "patch_size": [4, 4, 2],
        "downsample_scale": [2, 2, 2],
        "downsample_scales": [[2, 2, 2], [2, 2, 2], [1, 1, 1]],
        "kernel_size": [3, 3, 3],
        "expansion": 2,
        "data_root": str(dataset_root),
        "save_root": str(dataset_root / "checkpoints"),
        "seed": 42,
    }
    return runtime_conf


def _prepare_runtime_conf(runtime_conf: dict) -> tuple[Grid3DDataModule, dict]:
    datamodule = Grid3DDataModule(**runtime_conf)
    prepared_conf = dict(runtime_conf)
    prepared_conf.update(datamodule.export_task_hparams())
    return datamodule, prepared_conf


def _rewrite_grid3d_storage_as_raw(dataset_dir: Path) -> None:
    with np.load(dataset_dir / "stats.npz", allow_pickle=False) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)

    for split_name in ("train", "val", "test"):
        split_path = dataset_dir / "{}_data.npy".format(split_name)
        standardized = np.load(split_path, allow_pickle=False)
        raw = standardized * std + mean
        np.save(split_path, raw.astype(np.float32, copy=False))

    meta_path = dataset_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    meta["data_is_standardized"] = False
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True)


def _rewrite_grid3d_stats_as_legacy_vector(dataset_dir: Path) -> None:
    with np.load(dataset_dir / "stats.npz", allow_pickle=False) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(stats["std"], dtype=np.float32).reshape(-1)
    np.savez(dataset_dir / "stats.npz", mean=mean, std=std)


def test_grid3d_importer_writes_split_arrays(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    out_dir = tmp_path / "grid3d_demo"

    meta = import_grid3d_dataset(
        input_dir=raw_dir,
        out_dir=out_dir,
        split_spec=TRAIN_SPLIT_SPEC,
    )

    assert meta["storage_format"] == "grid3d_split_npy_v1"
    assert meta["data_layout"] == "T,C,Y,X,Z"
    assert meta["data_is_standardized"] is True
    assert meta["grid_shape"] == list(GRID_SHAPE)
    assert meta["split_lengths"] == {"train": 24, "val": 20, "test": 20}
    assert (out_dir / "coord.npy").exists()
    assert (out_dir / "axes.npz").exists()
    assert (out_dir / "stats.npz").exists()
    assert (out_dir / "train_timestamps.npy").exists()
    assert (out_dir / "train_data.npy").exists()
    assert (out_dir / "test_data.npy").exists()
    assert meta["grid_spacing_m"] == [1.0, 1.0, 1.0]
    assert meta["axis_layout"] == ["y", "x", "z"]
    assert meta["channel_names"] == ["U", "V", "W", "shear_x", "shear_y", "shear_z"]
    assert meta["velocity_channel_names"] == ["U", "V", "W"]
    assert meta["derived_channel_names"] == ["shear_x", "shear_y", "shear_z"]
    assert meta["coord_min"] == [0.0, 0.0, 0.0]
    assert meta["coord_max"] == [float(GRID_SHAPE[0] - 1), float(GRID_SHAPE[1] - 1), float(GRID_SHAPE[2] - 1)]

    with np.load(out_dir / "stats.npz") as stats:
        assert stats["mean"].shape == (1, 6, 1, 1, 1)
        assert stats["std"].shape == (1, 6, 1, 1, 1)

    with np.load(out_dir / "axes.npz") as axes:
        assert axes["x"].shape == (GRID_SHAPE[1],)
        assert axes["y"].shape == (GRID_SHAPE[0],)
        assert axes["z"].shape == (GRID_SHAPE[2],)

    coord = np.load(out_dir / "coord.npy")
    assert coord.shape == (3, *GRID_SHAPE)
    assert coord.dtype == np.float32
    assert np.isclose(coord[0, 0, 0, 0], -1.0)
    assert np.isclose(coord[0, -1, 0, 0], 1.0)
    assert np.isclose(coord[1, 0, 0, 0], -1.0)
    assert np.isclose(coord[1, 0, -1, 0], 1.0)
    assert np.isclose(coord[2, 0, 0, 0], -1.0)
    assert np.isclose(coord[2, 0, 0, -1], 1.0)

    train_data = np.load(out_dir / "train_data.npy")
    assert train_data.shape == (24, 6, *GRID_SHAPE)
    assert train_data.dtype == np.float32


def test_grid3d_importer_rejects_irregular_spacing(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw", irregular_z=True)
    out_dir = tmp_path / "grid3d_demo"

    with pytest.raises(ValueError, match="axis 'z' must be evenly spaced"):
        import_grid3d_dataset(
            input_dir=raw_dir,
            out_dir=out_dir,
            split_spec=TRAIN_SPLIT_SPEC,
        )


def test_grid3d_datamodule_and_task_forward(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))

    assert train_batch["inputs"].shape == (1, 5, 6, *GRID_SHAPE)
    assert train_batch["targets"].shape == (1, 1, 6, *GRID_SHAPE)
    assert train_batch["coords"].shape == (1, 3, *GRID_SHAPE)
    assert val_batch["inputs"].shape == (1, 5, 6, *GRID_SHAPE)
    assert val_batch["targets"].shape == (1, 1, 6, *GRID_SHAPE)

    datamodule_cls, task_cls = get_task_components("grid3d_forecasting")
    assert datamodule_cls is Grid3DDataModule
    assert task_cls is Grid3DForecastingTask

    task = Grid3DForecastingTask(**prepared_conf)
    processed_batch = task.preprocess_batch(train_batch)
    prediction, label = task._forward(train_batch)
    loss = task.loss_function(prediction, label)
    assert prediction.shape == (1, 1, 6, *GRID_SHAPE)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert task.scaler_policy.data_is_standardized is True
    assert torch.allclose(processed_batch["inputs"], train_batch["inputs"])
    assert torch.allclose(processed_batch["targets"], train_batch["targets"])
    assert torch.allclose(processed_batch["coords"], train_batch["coords"])


def test_grid3d_forecasting_raw_storage_ignores_stats_without_standardized_flag(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)
    _rewrite_grid3d_storage_as_raw(dataset_dir)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.train_dataloader()))
    task = Grid3DForecastingTask(**prepared_conf)
    train_data = np.load(dataset_dir / "train_data.npy", allow_pickle=False)
    expected_mean = torch.as_tensor(train_data.mean(axis=(0, 2, 3, 4))[None, :, None, None, None], dtype=torch.float32)
    expected_std = torch.as_tensor(train_data.std(axis=(0, 2, 3, 4))[None, :, None, None, None], dtype=torch.float32)

    processed_batch = task.preprocess_batch(batch)

    assert task.scaler_policy.data_is_standardized is False
    assert torch.allclose(task.scaler.mean, expected_mean, atol=1e-5, rtol=1e-5)
    assert torch.allclose(task.scaler.std, expected_std, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(processed_batch["inputs"], batch["inputs"].float())
    assert not torch.allclose(processed_batch["targets"], batch["targets"].float())


def test_grid3d_forecasting_standardized_storage_requires_stats_file(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)
    (dataset_dir / "stats.npz").unlink()

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )

    with pytest.raises(ValueError, match="data_is_standardized=true"):
        Grid3DForecastingTask(**_prepare_runtime_conf(runtime_conf)[1])


def test_grid3d_forecasting_legacy_vector_stats_fail_on_postprocess(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)
    _rewrite_grid3d_stats_as_legacy_vector(dataset_dir)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.test_dataloader()))
    task = Grid3DForecastingTask(**prepared_conf)

    assert tuple(task.scaler.mean.shape) == (6,)
    with pytest.raises(RuntimeError, match="must match"):
        task.postprocess_outputs(batch["targets"], batch["targets"])


def test_grid3d_forecasting_shear_metric_uses_shear_channels_without_spacing(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    runtime_conf["test_metric_space"] = "shear"
    runtime_conf.pop("grid_spacing_m", None)

    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.train_dataloader()))
    task = Grid3DForecastingTask(**prepared_conf)
    prediction, label = task._forward(batch)

    metric_pairs = list(task._iter_test_metric_pairs(batch, prediction, label))
    assert len(metric_pairs) == 1
    pred_shear, label_shear = metric_pairs[0]
    physical_prediction, physical_label = task.postprocess_outputs(prediction, label)

    assert pred_shear.shape[2] == 3
    assert label_shear.shape[2] == 3
    assert torch.allclose(pred_shear, physical_prediction[:, :, 3:6, ...], atol=1e-5, rtol=1e-5)
    assert torch.allclose(label_shear, physical_label[:, :, 3:6, ...], atol=1e-5, rtol=1e-5)


def test_grid3d_forecasting_original_metric_uses_full_physical_channels(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    runtime_conf["test_metric_space"] = "original"

    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.train_dataloader()))
    task = Grid3DForecastingTask(**prepared_conf)
    prediction, label = task._forward(batch)

    metric_pairs = list(task._iter_test_metric_pairs(batch, prediction, label))
    assert len(metric_pairs) == 1
    pred_original, label_original = metric_pairs[0]
    physical_prediction, physical_label = task.postprocess_outputs(prediction, label)

    assert pred_original.shape[2] == 6
    assert label_original.shape[2] == 6
    assert torch.allclose(pred_original, physical_prediction, atol=1e-5, rtol=1e-5)
    assert torch.allclose(label_original, physical_label, atol=1e-5, rtol=1e-5)


def test_grid3d_forecasting_loss_uses_shear_channels_directly(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    runtime_conf.update(
        {
            "shear_loss_weight": 1.0,
        }
    )
    task = Grid3DForecastingTask(**_prepare_runtime_conf(runtime_conf)[1])
    task.scaler.set_stats(
        torch.zeros((1, 6, 1, 1, 1), dtype=torch.float32),
        torch.ones((1, 6, 1, 1, 1), dtype=torch.float32),
    )

    prediction = torch.zeros((1, 1, 6, 2, 2, 2), dtype=torch.float32)
    label = torch.zeros_like(prediction)
    label[:, :, 3:6, ...] = 1.0

    loss = task._compute_total_loss(prediction, label)

    expected_flow = torch.nn.functional.mse_loss(prediction, label)
    expected_shear = torch.nn.functional.mse_loss(prediction[:, :, 3:6, ...], label[:, :, 3:6, ...])
    expected_total = expected_flow + expected_shear

    assert torch.allclose(loss, expected_total, atol=1e-6, rtol=1e-6)


def test_grid3d_forecasting_preprocess_batch_keeps_integer_tensors_without_scaling(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    task = Grid3DForecastingTask(**_prepare_runtime_conf(runtime_conf)[1])
    batch = {
        "inputs": torch.ones((1, 2, 6, 2, 2, 2), dtype=torch.int16),
        "targets": torch.ones((1, 1, 6, 2, 2, 2), dtype=torch.int32),
        "coords": torch.ones((1, 3, 2, 2, 2), dtype=torch.int64),
    }

    processed_batch = task.preprocess_batch(batch)

    assert processed_batch["inputs"].dtype == torch.int16
    assert processed_batch["targets"].dtype == torch.int32
    assert processed_batch["coords"].dtype == torch.int64


def test_grid3d_workflow_prepares_runtime_conf_before_task_init(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    task_spec, datamodule, prepared_conf = prepare_runtime_conf_for_task(
        _build_runtime_conf(
            tmp_path,
            dataset_dir.name,
            hist_len=5,
            pred_len=1,
        )
    )

    assert isinstance(datamodule, Grid3DDataModule)
    assert prepared_conf["history_len"] == 5
    assert prepared_conf["in_channels"] == 6
    assert prepared_conf["coord_channels"] == 3
    assert prepared_conf["data_is_standardized"] is True
    assert prepared_conf["steps_per_epoch"] > 0

    task = task_spec.task_cls(**prepared_conf)
    assert task.grid_shape == GRID_SHAPE
    assert task.channel_names == ["U", "V", "W", "shear_x", "shear_y", "shear_z"]


def test_grid3d_full_volume_train_path(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=10,
        pred_len=10,
    )
    datamodule = Grid3DDataModule(**runtime_conf)
    train_batch = next(iter(datamodule.train_dataloader()))
    assert train_batch["inputs"].shape == (1, 10, 6, *GRID_SHAPE)
    assert train_batch["targets"].shape == (1, 10, 6, *GRID_SHAPE)


def test_grid3d_split_requires_enough_steps(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw", num_steps=40)
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(
        input_dir=raw_dir,
        out_dir=dataset_dir,
        split_spec={"train": [0, 20], "val": [20, 30], "test": [30, 40]},
    )

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=10,
        pred_len=10,
    )
    datamodule = Grid3DDataModule(**runtime_conf)

    with pytest.raises(ValueError, match="invalid dataset split for sliding window"):
        datamodule.val_dataloader()


def test_grid3d_datamodule_rejects_legacy_patch_args(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
    )
    runtime_conf["train_patch_shape"] = [32, 32, 16]

    with pytest.raises(ValueError, match="train_patch_shape is no longer supported"):
        Grid3DDataModule(**runtime_conf)
