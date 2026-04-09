from pathlib import Path

import numpy as np
import torch

from easytsf.data.scaler import StandardScaler
from easytsf.task.base import BaseForecastTask


class MTSFTask(BaseForecastTask):
    def _setup_task_state(self):
        self.scaler = self._build_scaler()

    def _iter_standard_scalers(self):
        return (self.scaler,)

    def _build_scaler(self):
        dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        mmap_mode = "r" if bool(getattr(self.hparams, "use_mmap", False)) else None
        train_variable = np.load(dataset_dir / "train_data.npy", mmap_mode=mmap_mode, allow_pickle=False)
        return StandardScaler.fit(train_variable)

    def preprocess_batch(self, batch):
        var_x = batch["inputs"]
        marker_x = batch.get("inputs_timestamps")
        var_y = batch["targets"]
        marker_y = batch.get("targets_timestamps")
        tensors = (var_x, marker_x, var_y, marker_y)
        var_x, marker_x, var_y, marker_y = tuple(
            tensor if tensor.dtype == torch.float32 else tensor.float() for tensor in tensors
        )
        return (
            self.scaler.transform(var_x),
            marker_x,
            self.scaler.transform(var_y),
            marker_y,
        )

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        return (
            self.scaler.inverse_transform(prediction, mask=targets_mask),
            self.scaler.inverse_transform(label, mask=targets_mask),
        )

    def _forward(self, batch):
        var_x, marker_x, var_y, marker_y = self.preprocess_batch(batch)
        label = var_y[:, -self.hparams.pred_len :, ...]

        prediction = self.model(var_x, marker_x, marker_y)
        return prediction, label
