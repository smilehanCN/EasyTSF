from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from easytsf.data import Grid3DDataModule
from easytsf.data.grid3d_data_module import build_tile_bboxes, build_valid_crop_slices
from easytsf.task import get_task_components
from easytsf.task.grid3d_forecasting import Grid3DForecastingTask
from easytsf.task.grid3d_risk_prediction import Grid3DRiskPredictionTask
from scripts.analyze_grid3d_risk_bins import analyze as analyze_grid3d_risk_bins
from scripts.grid3d_import import import_grid3d_dataset


GRID_SHAPE = (48, 48, 24)
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
    train_patch_shape,
    task: str = "grid3d_forecasting",
    output_mode: str = "regression",
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
        "task": task,
        "output_mode": output_mode,
        "accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "use_mmap": False,
        "use_coords": True,
        "train_patch_shape": list(train_patch_shape),
        "eval_tile_shape": [32, 32, 16],
        "eval_tile_overlap": [16, 16, 8],
        "base_channels": 8,
        "patch_size": [4, 4, 2],
        "downsample_scale": [2, 2, 2],
        "kernel_size": [3, 3, 3],
        "expansion": 2,
        "data_root": str(dataset_root),
        "save_root": str(dataset_root / "checkpoints"),
        "seed": 42,
    }
    if task == "grid3d_risk_prediction":
        runtime_conf.update(
            {
                "risk_bins": {
                    "speed": [4.0, 6.0],
                    "shear": [0.05, 0.08],
                },
                "risk_num_classes": 3,
                "risk_num_heads": 4,
                "head_loss_weights": [1.0, 1.0, 1.0, 1.0],
                "high_risk_class_index": 2,
                "val_metric": "val/macro_f1",
                "val_metric_mode": "max",
            }
        )
    return runtime_conf


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
    assert meta["coord_min"] == [0.0, 0.0, 0.0]
    assert meta["coord_max"] == [float(GRID_SHAPE[0] - 1), float(GRID_SHAPE[1] - 1), float(GRID_SHAPE[2] - 1)]

    with np.load(out_dir / "stats.npz") as stats:
        assert stats["mean"].shape == (3,)
        assert stats["std"].shape == (3,)

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
    assert train_data.shape == (24, 3, *GRID_SHAPE)
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
        train_patch_shape=(32, 32, 16),
    )
    datamodule = Grid3DDataModule(**runtime_conf)
    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))

    assert train_batch["inputs"].shape == (1, 5, 3, 32, 32, 16)
    assert train_batch["targets"].shape == (1, 1, 3, 32, 32, 16)
    assert train_batch["coords"].shape == (1, 3, 32, 32, 16)
    assert train_batch["tile_bbox"].shape == (1, 6)
    assert val_batch["inputs"].shape == (1, 5, 3, 32, 32, 16)
    assert val_batch["targets"].shape == (1, 1, 3, 32, 32, 16)

    datamodule_cls, task_cls = get_task_components("grid3d_forecasting")
    assert datamodule_cls is Grid3DDataModule
    assert task_cls is Grid3DForecastingTask

    task = Grid3DForecastingTask(**runtime_conf)
    prediction, label = task._forward(train_batch)
    loss = task.loss_function(prediction, label)
    assert prediction.shape == (1, 1, 3, 32, 32, 16)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_grid3d_risk_task_forward_loss_and_metrics(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
        train_patch_shape=(32, 32, 16),
        task="grid3d_risk_prediction",
        output_mode="classification",
    )
    datamodule = Grid3DDataModule(**runtime_conf)
    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))

    datamodule_cls, task_cls = get_task_components("grid3d_risk_prediction")
    assert datamodule_cls is Grid3DDataModule
    assert task_cls is Grid3DRiskPredictionTask

    task = Grid3DRiskPredictionTask(**runtime_conf)
    prediction, label = task._forward(train_batch)
    loss = task.loss_function(prediction, label)
    assert prediction.shape == (1, 1, 12, 32, 32, 16)
    assert label.shape == (1, 1, 4, 32, 32, 16)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)

    val_loss = task.validation_step(val_batch, 0)
    assert isinstance(val_loss, torch.Tensor)
    assert torch.isfinite(val_loss)
    task.on_validation_epoch_end()
    assert int(task.val_confusion.sum().item()) > 0

    task.on_test_epoch_start()
    task.test_step(val_batch, 0)
    assert int(task.test_confusion.sum().item()) > 0
    metrics = task._compute_confusion_metrics(task.test_confusion)
    assert torch.isfinite(metrics["macro_f1"])
    assert torch.isfinite(metrics["high_risk_recall"])


def test_grid3d_risk_task_rejects_runtime_spacing_mismatch(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
        train_patch_shape=(32, 32, 16),
        task="grid3d_risk_prediction",
        output_mode="classification",
    )
    runtime_conf["grid_spacing_m"] = [2.0, 1.0, 1.0]

    with pytest.raises(ValueError, match="does not match dataset meta"):
        Grid3DRiskPredictionTask(**runtime_conf)


def test_grid3d_risk_label_construction_matches_expected_bins(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
        train_patch_shape=(32, 32, 16),
        task="grid3d_risk_prediction",
        output_mode="classification",
    )
    runtime_conf["risk_bins"] = {
        "speed": [0.5, 1.5],
        "shear": [0.5, 1.5],
    }

    task = Grid3DRiskPredictionTask(**runtime_conf)
    task.scaler.set_stats(
        torch.zeros((1, 1, 3, 1, 1, 1), dtype=torch.float32),
        torch.ones((1, 1, 3, 1, 1, 1), dtype=torch.float32),
    )

    u_component = torch.tensor(
        [[[[0.0], [1.0], [3.0]], [[0.0], [2.0], [2.0]]]],
        dtype=torch.float32,
    )
    v_component = torch.zeros_like(u_component)
    w_component = torch.zeros_like(u_component)
    physical_targets = torch.stack([u_component, v_component, w_component], dim=2)
    labels = task._build_risk_labels(physical_targets)

    expected_shear_x = torch.tensor(
        [[[[1], [2], [2]], [[2], [0], [0]]]],
        dtype=torch.long,
    ).unsqueeze(0)
    expected_shear_y = torch.tensor(
        [[[[0], [1], [1]], [[0], [1], [1]]]],
        dtype=torch.long,
    ).unsqueeze(0)
    expected_shear_z = torch.zeros((1, 1, 2, 3, 1), dtype=torch.long)
    expected_speed = torch.tensor(
        [[[[0], [1], [2]], [[0], [2], [2]]]],
        dtype=torch.long,
    ).unsqueeze(0)

    assert labels.shape == (1, 1, 4, 2, 3, 1)
    assert torch.equal(labels[:, :, 0, ...], expected_shear_x)
    assert torch.equal(labels[:, :, 1, ...], expected_shear_y)
    assert torch.equal(labels[:, :, 2, ...], expected_shear_z)
    assert torch.equal(labels[:, :, 3, ...], expected_speed)


def test_grid3d_full_size_train_path(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=10,
        pred_len=10,
        train_patch_shape=GRID_SHAPE,
    )
    datamodule = Grid3DDataModule(**runtime_conf)
    train_batch = next(iter(datamodule.train_dataloader()))
    assert train_batch["inputs"].shape == (1, 10, 3, *GRID_SHAPE)
    assert train_batch["targets"].shape == (1, 10, 3, *GRID_SHAPE)


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
        train_patch_shape=(32, 32, 16),
    )
    datamodule = Grid3DDataModule(**runtime_conf)

    with pytest.raises(ValueError, match="invalid dataset split for sliding window"):
        datamodule.val_dataloader()


def test_grid3d_eval_tiles_cover_volume_once():
    grid_shape = GRID_SHAPE
    tile_shape = (32, 32, 16)
    tile_overlap = (16, 16, 8)
    coverage = np.zeros(grid_shape, dtype=np.int32)

    for tile_bbox in build_tile_bboxes(grid_shape, tile_shape, tile_overlap):
        y_start, _, x_start, _, z_start, _ = tile_bbox
        crop_slices = build_valid_crop_slices(tile_bbox, grid_shape, tile_overlap)
        coverage[
            y_start + crop_slices[0].start : y_start + crop_slices[0].stop,
            x_start + crop_slices[1].start : x_start + crop_slices[1].stop,
            z_start + crop_slices[2].start : z_start + crop_slices[2].stop,
        ] += 1

    assert coverage.min() == 1
    assert coverage.max() == 1


def test_grid3d_risk_bin_analysis_reports_shared_weights(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
        train_patch_shape=(32, 32, 16),
        task="grid3d_risk_prediction",
        output_mode="classification",
    )
    summary = analyze_grid3d_risk_bins(runtime_conf, "train", tmp_path, y_chunk=8)

    assert summary["grid_spacing_m"] == [1.0, 1.0, 1.0]
    assert len(summary["recommended_risk_class_weights"]) == 3
    assert all(weight >= 1.0 for weight in summary["recommended_risk_class_weights"])


def test_grid3d_risk_bin_analysis_rejects_zero_count_weights(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
        train_patch_shape=(32, 32, 16),
        task="grid3d_risk_prediction",
        output_mode="classification",
    )
    runtime_conf["risk_bins"] = {
        "speed": [1e9, 2e9],
        "shear": [1e9, 2e9],
    }

    with pytest.raises(ValueError, match="aggregate class counts contain zeros"):
        analyze_grid3d_risk_bins(runtime_conf, "train", tmp_path, y_chunk=8)


def test_grid3d_risk_prediction_rejects_legacy_axis_specific_shear_bins(tmp_path):
    raw_dir = _build_raw_dataset(tmp_path / "raw")
    dataset_dir = tmp_path / "grid3d_demo"
    import_grid3d_dataset(input_dir=raw_dir, out_dir=dataset_dir, split_spec=TRAIN_SPLIT_SPEC)

    runtime_conf = _build_runtime_conf(
        tmp_path,
        dataset_dir.name,
        hist_len=5,
        pred_len=1,
        train_patch_shape=(32, 32, 16),
        task="grid3d_risk_prediction",
        output_mode="classification",
    )
    runtime_conf["risk_bins"] = {
        "speed": [4.0, 6.0],
        "shear_x": [0.05, 0.08],
        "shear_y": [0.05, 0.08],
        "shear_z": [0.05, 0.08],
    }

    with pytest.raises(ValueError, match="risk_bins must define unified 'shear'"):
        Grid3DRiskPredictionTask(**runtime_conf)
