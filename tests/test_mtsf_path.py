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


def _write_tiny_mtsf_dataset(
    dataset_dir: Path,
    *,
    data_is_standardized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    raw_train = np.arange(36, dtype=np.float32).reshape(12, 3)
    raw_val = np.arange(24, dtype=np.float32).reshape(8, 3) + 100.0
    raw_test = np.arange(24, dtype=np.float32).reshape(8, 3) + 200.0
    train_timestamps = (np.arange(12, dtype=np.float32) % 24).reshape(-1, 1)
    val_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)
    test_timestamps = (np.arange(8, dtype=np.float32) % 24).reshape(-1, 1)

    meta = {
        "timestamps_description": ["time of day"],
        "frequency (minutes)": 60,
        "data_is_standardized": bool(data_is_standardized),
    }
    with (dataset_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=True)

    mean = raw_train.mean(axis=0, keepdims=True)
    std = raw_train.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)
    if data_is_standardized:
        np.savez(dataset_dir / "stats.npz", mean=mean, std=std)
        _write_split(dataset_dir, "train", (raw_train - mean) / std, train_timestamps)
        _write_split(dataset_dir, "val", (raw_val - mean) / std, val_timestamps)
        _write_split(dataset_dir, "test", (raw_test - mean) / std, test_timestamps)
    else:
        _write_split(dataset_dir, "train", raw_train, train_timestamps)
        _write_split(dataset_dir, "val", raw_val, val_timestamps)
        _write_split(dataset_dir, "test", raw_test, test_timestamps)

    return raw_train, raw_test


def _runtime_conf(dataset_root: Path, dataset_name: str) -> dict:
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


def _prepare_runtime_conf(runtime_conf: dict) -> tuple[MTSDataModule, dict]:
    datamodule = MTSDataModule(**runtime_conf)
    prepared_conf = dict(runtime_conf)
    prepared_conf.update(datamodule.export_task_hparams())
    return datamodule, prepared_conf


def test_mtsf_datamodule_and_task_smoke(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    raw_train, _ = _write_tiny_mtsf_dataset(dataset_dir)

    runtime_conf = _runtime_conf(tmp_path, dataset_dir.name)
    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.train_dataloader()))

    assert batch["inputs"].shape == (2, 4, 3)
    assert batch["targets"].shape == (2, 2, 3)
    assert batch["inputs_timestamps"].shape == (2, 4, 1)
    assert batch["targets_timestamps"].shape == (2, 2, 1)

    datamodule_cls, task_cls = get_task_components("mtsf")
    assert datamodule_cls is MTSDataModule
    assert task_cls is MTSFTask

    task = MTSFTask(**prepared_conf)
    prediction, label = task._forward(batch)
    loss = task.loss_function(prediction, label)

    expected_mean = torch.as_tensor(raw_train.mean(axis=0, keepdims=True), dtype=torch.float32)
    expected_std = torch.as_tensor(raw_train.std(axis=0, keepdims=True), dtype=torch.float32)
    assert prediction.shape == (2, 2, 3)
    assert label.shape == (2, 2, 3)
    assert torch.isfinite(loss)
    assert task.scaler_policy.data_is_standardized is False
    assert torch.allclose(task.scaler.mean, expected_mean)
    assert torch.allclose(task.scaler.std, expected_std)


def test_mtsf_standardized_dataset_passthrough_and_inverse(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    _, raw_test = _write_tiny_mtsf_dataset(dataset_dir, data_is_standardized=True)

    runtime_conf = _runtime_conf(tmp_path, dataset_dir.name)
    datamodule, prepared_conf = _prepare_runtime_conf(runtime_conf)
    batch = next(iter(datamodule.test_dataloader()))
    task = MTSFTask(**prepared_conf)

    processed_batch = task.preprocess_batch(batch)
    _, restored_label = task.postprocess_outputs(batch["targets"], batch["targets"])

    assert task.scaler_policy.data_is_standardized is True
    assert torch.allclose(processed_batch["inputs"], batch["inputs"])
    assert torch.allclose(processed_batch["targets"], batch["targets"])
    assert torch.allclose(restored_label, torch.as_tensor(raw_test[4:6][None, ...], dtype=torch.float32))
