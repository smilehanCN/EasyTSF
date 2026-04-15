from pathlib import Path

import numpy as np
import pytest
import torch

from easytsf.data import WeatherDataModule
from easytsf.task import get_task_components
from easytsf.task.weatherbench import WeatherBenchTask
from scripts.weather_import import write_canonical_weather_dataset

xr = pytest.importorskip("xarray")


def _build_dynamic_dataset():
    valid_time = np.arange(
        np.datetime64("2020-01-01T00:00", "h"),
        np.datetime64("2020-01-04T00:00", "h"),
        np.timedelta64(6, "h"),
    )
    latitude = np.array([10.0, 20.0], dtype=np.float32)
    longitude = np.array([100.0, 110.0, 120.0], dtype=np.float32)

    t2m = np.arange(valid_time.size * latitude.size * longitude.size, dtype=np.float32).reshape(
        valid_time.size,
        latitude.size,
        longitude.size,
    )
    z500 = t2m + 100.0
    z850 = t2m + 200.0
    geopotential = np.stack([z500, z850], axis=1)

    return xr.Dataset(
        data_vars={
            "t2m": (("valid_time", "latitude", "longitude"), t2m),
            "z": (("valid_time", "level", "latitude", "longitude"), geopotential),
        },
        coords={
            "valid_time": valid_time,
            "latitude": latitude,
            "longitude": longitude,
            "level": np.array([500, 850], dtype=np.int32),
        },
    )


def _build_static_dataset():
    latitude = np.array([10.0, 20.0], dtype=np.float32)
    longitude = np.array([100.0, 110.0, 120.0], dtype=np.float32)
    land_sea_mask = np.linspace(0.0, 1.0, latitude.size * longitude.size, dtype=np.float32).reshape(
        latitude.size,
        longitude.size,
    )
    return xr.Dataset(
        data_vars={"land_sea_mask": (("latitude", "longitude"), land_sea_mask)},
        coords={"latitude": latitude, "longitude": longitude},
    )


def _build_runtime_conf(dataset_root: Path, dataset_name: str) -> dict:
    return {
        "model": "WeatherBenchPersistence",
        "dataset": dataset_name,
        "hist_len": 2,
        "pred_len": 1,
        "batch_size": 2,
        "num_workers": 0,
        "lr": 1e-3,
        "lr_scheduler": "StepLR",
        "lr_step_size": 1,
        "lr_gamma": 1.0,
        "optimizer": "Adam",
        "max_epochs": 1,
        "es_patience": 1,
        "val_metric": "val/loss",
        "gradient_clip_val": 0.0,
        "gradient_clip_algorithm": "norm",
        "test_metric_space": "original",
        "task": "weatherbench",
        "accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "use_mmap": False,
        "shard_cache_size": 2,
        "data_root": str(dataset_root),
        "save_root": str(dataset_root / "checkpoints"),
        "seed": 42,
    }


def test_weatherbench_canonical_path_with_static_fields(tmp_path):
    dataset_dir = tmp_path / "weatherbench_tiny"
    meta = write_canonical_weather_dataset(
        ds=_build_dynamic_dataset(),
        static_ds=_build_static_dataset(),
        out_dir=dataset_dir,
        input_variables=["t2m", "z"],
        target_variables=["t2m"],
        static_variables=["land_sea_mask"],
        levels={"z": [500]},
        split_spec={
            "train": ["2020-01-01T00:00", "2020-01-01T18:00"],
            "val": ["2020-01-02T00:00", "2020-01-02T18:00"],
            "test": ["2020-01-03T00:00", "2020-01-03T18:00"],
        },
        shard_len=2,
        source_format="weatherbench_netcdf",
        regrid_shape=[3, 4],
    )

    assert meta["grid_shape"] == [3, 4]
    assert meta["input_channels"] == ["t2m", "z_500"]
    assert meta["target_channels"] == ["t2m"]
    assert meta["static_channels"] == ["land_sea_mask"]
    assert (dataset_dir / "static.npy").exists()
    assert (dataset_dir / "static_channels.json").exists()
    assert (dataset_dir / "train" / "climatology.npz").exists()

    with np.load(dataset_dir / "stats.npz") as stats:
        assert stats["mean"].shape == (1, 2, 1, 1)
        assert stats["std"].shape == (1, 2, 1, 1)

    datamodule = WeatherDataModule(**_build_runtime_conf(tmp_path, dataset_dir.name))
    batch = next(iter(datamodule.train_dataloader()))
    assert batch["inputs"].shape == (2, 2, 2, 3, 4)
    assert batch["targets"].shape == (2, 1, 1, 3, 4)
    assert batch["static_inputs"].shape == (2, 1, 3, 4)
    assert datamodule.split_climatology["train"].shape == (1, 3, 4)

    datamodule_cls, task_cls = get_task_components("weatherbench")
    assert datamodule_cls is WeatherDataModule
    assert task_cls is WeatherBenchTask

    task = WeatherBenchTask(**_build_runtime_conf(tmp_path, dataset_dir.name))
    prediction, label = task._forward(batch)
    loss = task.loss_function(prediction, label)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)


def test_weatherbench_loader_without_static_fields(tmp_path):
    dataset_dir = tmp_path / "weatherbench_no_static"
    write_canonical_weather_dataset(
        ds=_build_dynamic_dataset(),
        out_dir=dataset_dir,
        input_variables=["t2m"],
        target_variables=["t2m"],
        split_spec={
            "train": ["2020-01-01T00:00", "2020-01-01T18:00"],
            "val": ["2020-01-02T00:00", "2020-01-02T18:00"],
            "test": ["2020-01-03T00:00", "2020-01-03T18:00"],
        },
        shard_len=2,
        source_format="weatherbench_netcdf",
    )

    datamodule = WeatherDataModule(**_build_runtime_conf(tmp_path, dataset_dir.name))
    batch = next(iter(datamodule.train_dataloader()))
    assert "static_inputs" not in batch


def test_weatherbench_invalid_target_input_channel_combo_still_fails(tmp_path):
    dataset_dir = tmp_path / "weatherbench_bad_channels"
    write_canonical_weather_dataset(
        ds=_build_dynamic_dataset(),
        out_dir=dataset_dir,
        input_variables=["t2m", "z"],
        target_variables=["t2m"],
        levels={"z": [500]},
        split_spec={
            "train": ["2020-01-01T00:00", "2020-01-01T18:00"],
            "val": ["2020-01-02T00:00", "2020-01-02T18:00"],
            "test": ["2020-01-03T00:00", "2020-01-03T18:00"],
        },
        shard_len=2,
        source_format="weatherbench_netcdf",
    )

    runtime_conf = _build_runtime_conf(tmp_path, dataset_dir.name)
    runtime_conf["input_channel_names"] = ["t2m"]
    runtime_conf["target_channel_names"] = ["z_500"]

    with pytest.raises(KeyError, match="z_500"):
        WeatherBenchTask(**runtime_conf)
