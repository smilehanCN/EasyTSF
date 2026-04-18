import json
from pathlib import Path

import numpy as np
import torch

from easytsf.data import MTSDataModule
from easytsf.task import get_task_components
from easytsf.task.mtsf import MTSFTask


def _write_split(dataset_dir: Path, split: str, values: np.ndarray, timestamps: np.ndarray) -> None:
    np.save(dataset_dir / "{}_data.npy".format(split), values.astype(np.float32, copy=False))
    np.save(dataset_dir / "{}_timestamps.npy".format(split), timestamps.astype(np.float32, copy=False))


def _write_meta(dataset_dir: Path, *, data_is_standardized: bool | None = None) -> None:
    meta = {
        "timestamps_description": ["time of day"],
        "frequency (minutes)": 60,
    }
    if data_is_standardized is not None:
        meta["data_is_standardized"] = bool(data_is_standardized)
    with (dataset_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True)


def _write_stats(dataset_dir: Path, mean: np.ndarray, std: np.ndarray) -> None:
    np.savez(dataset_dir / "stats.npz", mean=np.asarray(mean, dtype=np.float32), std=np.asarray(std, dtype=np.float32))


def _build_runtime_conf(dataset_root: Path, dataset_name: str) -> dict:
    return {
        "model": "SparseTSF",
        "dataset": dataset_name,
        "hist_len": 4,
        "pred_len": 2,
        "var_num": 3,
        "period_len": 2,
        "d_model": 8,
        "model_type": "linear",
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
        "test_metric_space": "scaled",
        "task": "mtsf",
        "accelerator": "cpu",
        "devices": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": 2,
        "use_mmap": False,
        "data_root": str(dataset_root),
        "save_root": str(dataset_root / "checkpoints"),
        "seed": 42,
    }


def _prepare_runtime_conf(runtime_conf: dict) -> dict:
    datamodule = MTSDataModule(**runtime_conf)
    prepared_conf = dict(runtime_conf)
    prepared_conf.update(datamodule.export_task_hparams())
    return prepared_conf


def test_mtsf_datamodule_and_task_forward(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _write_meta(dataset_dir)
    train_data = np.arange(36, dtype=np.float32).reshape(12, 3)
    val_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 100.0
    test_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 200.0
    train_timestamps = (np.arange(12, dtype=np.float32) % 24).reshape(-1, 1)
    val_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)
    test_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)

    _write_split(dataset_dir, "train", train_data, train_timestamps)
    _write_split(dataset_dir, "val", val_data, val_timestamps)
    _write_split(dataset_dir, "test", test_data, test_timestamps)

    runtime_conf = _build_runtime_conf(tmp_path, dataset_dir.name)
    datamodule = MTSDataModule(**runtime_conf)
    train_batch = next(iter(datamodule.train_dataloader()))

    assert train_batch["inputs"].shape == (2, 4, 3)
    assert train_batch["targets"].shape == (2, 2, 3)
    assert train_batch["inputs_timestamps"].shape == (2, 4, 1)
    assert train_batch["targets_timestamps"].shape == (2, 2, 1)

    datamodule_cls, task_cls = get_task_components("mtsf")
    assert datamodule_cls is MTSDataModule
    assert task_cls is MTSFTask

    prepared_conf = dict(runtime_conf)
    prepared_conf.update(datamodule.export_task_hparams())
    task = MTSFTask(**prepared_conf)
    prediction, label = task._forward(train_batch)
    loss = task.loss_function(prediction, label)
    expected_mean = torch.as_tensor(train_data.mean(axis=0, keepdims=True), dtype=torch.float32)
    expected_std = torch.as_tensor(train_data.std(axis=0, keepdims=True), dtype=torch.float32)

    assert prediction.shape == (2, 2, 3)
    assert label.shape == (2, 2, 3)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
    assert task.scaler_policy.data_is_standardized is False
    assert torch.allclose(task.scaler.mean, expected_mean)
    assert torch.allclose(task.scaler.std, expected_std)


def test_mtsf_ignores_stats_without_standardized_flag(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    _write_meta(dataset_dir)
    train_data = np.arange(36, dtype=np.float32).reshape(12, 3)
    val_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 100.0
    test_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 200.0
    train_timestamps = (np.arange(12, dtype=np.float32) % 24).reshape(-1, 1)
    val_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)
    test_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)

    _write_split(dataset_dir, "train", train_data, train_timestamps)
    _write_split(dataset_dir, "val", val_data, val_timestamps)
    _write_split(dataset_dir, "test", test_data, test_timestamps)
    _write_stats(dataset_dir, np.full((3,), 999.0, dtype=np.float32), np.full((3,), 7.0, dtype=np.float32))

    task = MTSFTask(**_prepare_runtime_conf(_build_runtime_conf(tmp_path, dataset_dir.name)))
    expected_mean = torch.as_tensor(train_data.mean(axis=0, keepdims=True), dtype=torch.float32)
    expected_std = torch.as_tensor(train_data.std(axis=0, keepdims=True), dtype=torch.float32)

    assert task.scaler_policy.data_is_standardized is False
    assert torch.allclose(task.scaler.mean, expected_mean)
    assert torch.allclose(task.scaler.std, expected_std)


def test_mtsf_standardized_dataset_passthroughs_preprocess_and_inverse_restores_raw(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_train_data = np.arange(36, dtype=np.float32).reshape(12, 3)
    raw_val_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 100.0
    raw_test_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 200.0
    train_timestamps = (np.arange(12, dtype=np.float32) % 24).reshape(-1, 1)
    val_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)
    test_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)

    mean = raw_train_data.mean(axis=0, keepdims=True)
    std = raw_train_data.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)

    _write_meta(dataset_dir, data_is_standardized=True)
    _write_stats(dataset_dir, mean, std)
    _write_split(dataset_dir, "train", (raw_train_data - mean) / std, train_timestamps)
    _write_split(dataset_dir, "val", (raw_val_data - mean) / std, val_timestamps)
    _write_split(dataset_dir, "test", (raw_test_data - mean) / std, test_timestamps)

    runtime_conf = _build_runtime_conf(tmp_path, dataset_dir.name)
    datamodule = MTSDataModule(**runtime_conf)
    test_batch = next(iter(datamodule.test_dataloader()))
    prepared_conf = dict(runtime_conf)
    prepared_conf.update(datamodule.export_task_hparams())
    task = MTSFTask(**prepared_conf)

    processed_batch = task.preprocess_batch(test_batch)
    _, restored_label = task.postprocess_outputs(test_batch["targets"], test_batch["targets"])
    expected_label = torch.as_tensor(raw_test_data[4:6][None, ...], dtype=torch.float32)

    assert task.scaler_policy.data_is_standardized is True
    assert torch.allclose(processed_batch["inputs"], test_batch["inputs"])
    assert torch.allclose(processed_batch["targets"], test_batch["targets"])
    assert torch.allclose(restored_label, expected_label)


def test_mtsf_standardized_dataset_accepts_vector_stats_without_prevalidation(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_train_data = np.arange(36, dtype=np.float32).reshape(12, 3)
    raw_val_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 100.0
    raw_test_data = np.arange(24, dtype=np.float32).reshape(8, 3) + 200.0
    train_timestamps = (np.arange(12, dtype=np.float32) % 24).reshape(-1, 1)
    val_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)
    test_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)

    mean = raw_train_data.mean(axis=0)
    std = raw_train_data.std(axis=0)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)

    _write_meta(dataset_dir, data_is_standardized=True)
    _write_stats(dataset_dir, mean, std)
    _write_split(dataset_dir, "train", (raw_train_data - mean) / std, train_timestamps)
    _write_split(dataset_dir, "val", (raw_val_data - mean) / std, val_timestamps)
    _write_split(dataset_dir, "test", (raw_test_data - mean) / std, test_timestamps)

    runtime_conf = _build_runtime_conf(tmp_path, dataset_dir.name)
    datamodule = MTSDataModule(**runtime_conf)
    test_batch = next(iter(datamodule.test_dataloader()))
    task = MTSFTask(**_prepare_runtime_conf(runtime_conf))
    _, restored_label = task.postprocess_outputs(test_batch["targets"], test_batch["targets"])
    expected_label = torch.as_tensor(raw_test_data[4:6][None, ...], dtype=torch.float32)

    assert task.scaler_policy.data_is_standardized is True
    assert tuple(task.scaler.mean.shape) == (3,)
    assert torch.allclose(restored_label, expected_label)
