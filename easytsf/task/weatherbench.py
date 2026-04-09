import inspect
import json
from pathlib import Path

import numpy as np
import torch

from easytsf.data.scaler import StandardScaler
from easytsf.task.base import BaseForecastTask


class WeatherBenchTask(BaseForecastTask):
    def _setup_task_state(self):
        self.dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        with (self.dataset_dir / "channels.json").open("r", encoding="utf-8") as handle:
            self.channels = json.load(handle)
        with (self.dataset_dir / "static_channels.json").open("r", encoding="utf-8") as handle:
            self.static_channels = json.load(handle)

        self.input_channel_names = list(getattr(self.hparams, "input_channel_names", None) or self.meta["input_channels"])
        self.target_channel_names = list(getattr(self.hparams, "target_channel_names", None) or self.meta["target_channels"])

        channel_name_to_index = {channel["name"]: int(channel["index"]) for channel in self.channels}
        self.input_channel_indices = [channel_name_to_index[name] for name in self.input_channel_names]
        self.target_channel_indices = [channel_name_to_index[name] for name in self.target_channel_names]
        input_channel_positions = {name: index for index, name in enumerate(self.input_channel_names)}
        if not set(self.target_channel_names).issubset(set(self.input_channel_names)):
            raise ValueError("target_channel_names must be a subset of input_channel_names")

        self.register_buffer(
            "target_input_index_tensor",
            torch.as_tensor(
                [input_channel_positions[name] for name in self.target_channel_names],
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.input_scaler, self.target_scaler = self._build_scalers()

    def _iter_standard_scalers(self):
        return (self.input_scaler, self.target_scaler)

    def _get_model_derived_args(self):
        return {
            "hist_len": int(self.hparams.hist_len),
            "pred_len": int(self.hparams.pred_len),
            "var_num": len(self.input_channel_names),
            "input_var_num": len(self.input_channel_names),
            "target_var_num": len(self.target_channel_names),
            "static_var_num": len(self.static_channels),
            "grid_shape": tuple(self.meta["grid_shape"]),
            "height": int(self.meta["grid_shape"][0]),
            "width": int(self.meta["grid_shape"][1]),
        }

    def _build_scalers(self):
        with np.load(self.dataset_dir / "stats.npz") as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        if mean.ndim == 1:
            mean = mean[None, :, None, None]
        if std.ndim == 1:
            std = std[None, :, None, None]

        input_mean = mean[:, self.input_channel_indices, :, :]
        input_std = std[:, self.input_channel_indices, :, :]
        target_mean = mean[:, self.target_channel_indices, :, :]
        target_std = std[:, self.target_channel_indices, :, :]
        return StandardScaler(input_mean, input_std), StandardScaler(target_mean, target_std)

    def preprocess_batch(self, batch):
        var_x = batch["inputs"].float()
        marker_x = batch.get("inputs_timestamps")
        var_y = batch["targets"].float()
        marker_y = batch.get("targets_timestamps")
        static_inputs = batch.get("static_inputs")
        if static_inputs is not None:
            static_inputs = static_inputs.float()
        return (
            self.input_scaler.transform(var_x),
            marker_x,
            self.target_scaler.transform(var_y),
            marker_y,
            static_inputs,
        )

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        return (
            self.target_scaler.inverse_transform(prediction, mask=targets_mask),
            self.target_scaler.inverse_transform(label, mask=targets_mask),
        )

    def _forward(self, batch):
        var_x, marker_x, var_y, marker_y, static_inputs = self.preprocess_batch(batch)
        label = var_y[:, -self.hparams.pred_len :, ...]

        model_forward_parameters = inspect.signature(self.model.forward).parameters
        model_kwargs = {}
        if "static_inputs" in model_forward_parameters and static_inputs is not None:
            model_kwargs["static_inputs"] = static_inputs
        if "target_input_indices" in model_forward_parameters:
            model_kwargs["target_input_indices"] = self.target_input_index_tensor

        prediction = self.model(var_x, marker_x, marker_y, **model_kwargs)
        return prediction, label
