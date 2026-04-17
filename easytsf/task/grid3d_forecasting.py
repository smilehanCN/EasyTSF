import numpy as np
import torch.nn.functional as F

from easytsf.task.base import BaseForecastTask


class Grid3DForecastingTask(BaseForecastTask):
    _SHEAR_CHANNEL_SLICE = slice(3, 6)

    def _enable_base_single_scaler(self):
        return True

    def _setup_additional_task_state(self):
        self.grid_shape = tuple(int(size) for size in self.meta["grid_shape"])
        self.channel_names = list(self.meta["channel_names"])
        self.shear_loss_weight = float(getattr(self.hparams, "shear_loss_weight", 0.0))

    def _get_model_derived_args(self):
        return {
            "hist_len": int(self.hparams.hist_len),
            "history_len": int(self.hparams.hist_len),
            "pred_len": int(self.hparams.pred_len),
            "in_channels": len(self.channel_names),
            "coord_channels": 3,
            "grid_shape": self.grid_shape,
            "height": int(self.grid_shape[0]),
            "width": int(self.grid_shape[1]),
            "depth": int(self.grid_shape[2]),
        }

    def _build_loss_function(self):
        return self._compute_total_loss

    def _get_scaler_fit_reduce_dims(self):
        return (0, 2, 3, 4)

    def _format_single_scaler_stats(self, mean, std):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        return mean[None, None, :, None, None, None], std[None, None, :, None, None, None]

    def _get_preprocess_float_keys(self):
        return ("inputs", "targets", "coords")

    def _get_preprocess_scale_keys(self):
        return ("inputs", "targets")

    def _compute_total_loss(self, prediction, label):
        total_loss = F.mse_loss(prediction, label)
        if self.shear_loss_weight <= 0.0:
            return total_loss

        physical_prediction = self.scaler.inverse_transform(prediction)
        physical_label = self.scaler.inverse_transform(label)
        shear_prediction = physical_prediction[:, :, self._SHEAR_CHANNEL_SLICE, ...]
        shear_label = physical_label[:, :, self._SHEAR_CHANNEL_SLICE, ...]

        return total_loss + self.shear_loss_weight * F.mse_loss(shear_prediction, shear_label)

    def _iter_test_metric_pairs(self, batch, prediction, label):
        del batch
        metric_space = getattr(self.hparams, "test_metric_space", "original")
        if metric_space == "shear":
            prediction, label = self.postprocess_outputs(prediction, label)
            yield (
                prediction[:, :, self._SHEAR_CHANNEL_SLICE, ...],
                label[:, :, self._SHEAR_CHANNEL_SLICE, ...],
            )
            return
        prediction, label = self.postprocess_outputs(prediction, label)
        yield prediction, label

    def _forward(self, batch):
        batch = self.preprocess_batch(batch)
        var_x = batch["inputs"]
        var_y = batch["targets"]
        coords = batch.get("coords")
        prediction = self.model(var_x, coords=coords)
        return prediction, var_y
