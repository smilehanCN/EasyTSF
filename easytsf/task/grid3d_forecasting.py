import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from easytsf.data.grid3d_data_module import build_valid_crop_slices
from easytsf.data.scaler import StandardScaler
from easytsf.task.base import BaseForecastTask


class Grid3DForecastingTask(BaseForecastTask):
    def _setup_task_state(self):
        self.dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        self.grid_shape = tuple(int(size) for size in self.meta["grid_shape"])
        self.channel_names = list(self.meta["channel_names"])
        self.eval_tile_overlap = tuple(int(size) for size in getattr(self.hparams, "eval_tile_overlap", (0, 0, 0)))

        self.scaler = self._build_scaler()

    def _get_model_derived_args(self):
        return {
            "hist_len": int(self.hparams.hist_len),
            "pred_len": int(self.hparams.pred_len),
            "in_channels": len(self.channel_names),
            "coord_channels": 3,
            "grid_shape": self.grid_shape,
            "height": int(self.grid_shape[0]),
            "width": int(self.grid_shape[1]),
            "depth": int(self.grid_shape[2]),
        }

    def _build_scaler(self):
        with np.load(self.dataset_dir / "stats.npz") as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        return StandardScaler(mean[None, None, :, None, None, None], std[None, None, :, None, None, None])

    def preprocess_batch(self, batch):
        var_x = batch["inputs"].float()
        var_y = batch["targets"].float()
        coords = batch.get("coords")
        if coords is not None:
            coords = coords.float()
        return var_x, var_y, coords

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        del targets_mask
        return (
            self.scaler.inverse_transform(prediction),
            self.scaler.inverse_transform(label),
        )

    def _forward(self, batch):
        var_x, var_y, coords = self.preprocess_batch(batch)
        prediction = self.model(var_x, coords=coords)
        return prediction, var_y

    def _iter_eval_crops(self, batch, prediction, label):
        tile_bboxes = batch["tile_bbox"]
        if tile_bboxes.ndim == 1:
            tile_bboxes = tile_bboxes.unsqueeze(0)
        for batch_index in range(prediction.size(0)):
            tile_bbox = tuple(int(value) for value in tile_bboxes[batch_index].tolist())
            crop_slices = build_valid_crop_slices(
                tile_bbox=tile_bbox,
                grid_shape=self.grid_shape,
                tile_overlap=self.eval_tile_overlap,
            )
            pred_crop = prediction[batch_index : batch_index + 1, :, :, crop_slices[0], crop_slices[1], crop_slices[2]]
            label_crop = label[batch_index : batch_index + 1, :, :, crop_slices[0], crop_slices[1], crop_slices[2]]
            yield pred_crop, label_crop

    def _compute_validation_loss(self, batch, prediction, label):
        loss_sum = prediction.new_zeros(())
        value_count = 0
        for pred_crop, label_crop in self._iter_eval_crops(batch, prediction, label):
            loss_sum = loss_sum + F.mse_loss(pred_crop, label_crop, reduction="sum")
            value_count += pred_crop.numel()
        loss = loss_sum / max(value_count, 1)
        return loss, {"batch_size": value_count}

    def _iter_test_metric_pairs(self, batch, prediction, label):
        metric_space = getattr(self.hparams, "test_metric_space", "original")
        if metric_space == "original":
            prediction, label = self.postprocess_outputs(prediction, label)
        for pred_crop, label_crop in self._iter_eval_crops(batch, prediction, label):
            yield pred_crop, label_crop
