import numpy as np
from easytsf.task.base import BaseForecastTask


class MTSFTask(BaseForecastTask):
    def _enable_base_single_scaler(self):
        return True

    def _get_scaler_fit_reduce_dims(self):
        return 0

    def _format_single_scaler_stats(self, mean, std):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        if mean.ndim == 0:
            mean = mean.reshape(1, 1)
            std = std.reshape(1, 1)
        elif mean.ndim == 1:
            mean = mean[None, :]
            std = std[None, :]
        elif mean.ndim != 2:
            raise ValueError("unsupported mtsf statistics shape {}".format(tuple(mean.shape)))
        return mean, std

    def _get_preprocess_float_keys(self):
        return ("inputs", "inputs_timestamps", "targets", "targets_timestamps")

    def _get_preprocess_scale_keys(self):
        return ("inputs", "targets")

    def _forward(self, batch):
        batch = self.preprocess_batch(batch)
        var_x = batch["inputs"]
        marker_x = batch.get("inputs_timestamps")
        var_y = batch["targets"]
        marker_y = batch.get("targets_timestamps")
        label = var_y[:, -self.hparams.pred_len :, ...]

        prediction = self.model(var_x, marker_x, marker_y)
        return prediction, label
