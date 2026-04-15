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


def test_mtsf_datamodule_and_task_forward(tmp_path):
    dataset_dir = tmp_path / "mtsf_tiny"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with (dataset_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamps_description": ["time of day"],
                "frequency (minutes)": 60,
            },
            handle,
            ensure_ascii=True,
        )

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

    task = MTSFTask(**runtime_conf)
    prediction, label = task._forward(train_batch)
    loss = task.loss_function(prediction, label)

    assert prediction.shape == (2, 2, 3)
    assert label.shape == (2, 2, 3)
    assert isinstance(loss, torch.Tensor)
    assert torch.isfinite(loss)
