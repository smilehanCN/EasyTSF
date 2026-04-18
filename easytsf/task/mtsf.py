from easytsf.task.base import BaseForecastTask


class MTSFTask(BaseForecastTask):
    def _forward(self, batch):
        batch = self.preprocess_batch(batch)
        var_x = batch["inputs"]
        marker_x = batch.get("inputs_timestamps")
        var_y = batch["targets"]
        marker_y = batch.get("targets_timestamps")
        label = var_y[:, -self.hparams.pred_len :, ...]

        prediction = self.model(var_x, marker_x, marker_y)
        return prediction, label
