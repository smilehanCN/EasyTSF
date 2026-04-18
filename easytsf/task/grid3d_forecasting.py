import torch.nn.functional as F

from easytsf.task.base import BaseForecastTask


class Grid3DForecastingTask(BaseForecastTask):
    _SHEAR_CHANNEL_SLICE = slice(3, 6)

    def _build_loss_function(self):
        return self._compute_total_loss

    def _compute_total_loss(self, prediction, label):
        total_loss = F.mse_loss(prediction, label)
        shear_loss_weight = float(getattr(self.hparams, "shear_loss_weight", 0.0))
        if shear_loss_weight <= 0.0:
            return total_loss

        physical_prediction = self.scaler.inverse_transform(prediction)
        physical_label = self.scaler.inverse_transform(label)
        shear_prediction = physical_prediction[:, :, self._SHEAR_CHANNEL_SLICE, ...]
        shear_label = physical_label[:, :, self._SHEAR_CHANNEL_SLICE, ...]

        return total_loss + shear_loss_weight * F.mse_loss(shear_prediction, shear_label)

    def _iter_test_metric_pairs(self, batch, prediction, label):
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
