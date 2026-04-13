import inspect
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from easytsf.data.scaler import StandardScaler
from easytsf.task.base import BaseForecastTask


class LatitudeWeightedMSELoss(nn.Module):
    def __init__(self, latitude):
        super().__init__()
        weights = np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float32)))
        weights = weights / np.maximum(weights.mean(), 1e-12)
        self.register_buffer("latitude_weights", torch.as_tensor(weights, dtype=torch.float32))

    def forward(self, prediction, label):
        error = (prediction - label) ** 2
        if error.ndim == 5:
            weights = self.latitude_weights.view(1, 1, 1, -1, 1)
        elif error.ndim == 4:
            weights = self.latitude_weights.view(1, 1, -1, 1)
        else:
            raise ValueError(
                "LatitudeWeightedMSELoss expects a 4D or 5D tensor, but received {}".format(tuple(error.shape))
            )
        return (error * weights).mean()


def _ensure_stat_layout(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :, None, None]
    if array.ndim == 3:
        return array[None, :, :, :]
    if array.ndim == 4:
        return array
    raise ValueError("unsupported statistics shape {}; expected 1D, 3D, or 4D arrays".format(tuple(array.shape)))


def _normalize_named_channel_stat(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1, 1)
    if array.ndim == 1:
        if array.size == 1:
            return array.reshape(1, 1, 1)
        return array.reshape(1, array.shape[0], 1)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim == 3:
        return array
    raise ValueError("unsupported per-channel stat shape {}".format(tuple(array.shape)))


class WeatherBenchTask(BaseForecastTask):
    def _setup_task_state(self):
        self.dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        with (self.dataset_dir / "channels.json").open("r", encoding="utf-8") as handle:
            self.channels = json.load(handle)
        with (self.dataset_dir / "static_channels.json").open("r", encoding="utf-8") as handle:
            self.static_channels = json.load(handle)

        self.latitude = np.load(self.dataset_dir / "latitude.npy", allow_pickle=False)
        self.longitude = np.load(self.dataset_dir / "longitude.npy", allow_pickle=False)
        self.static_channel_names = [str(channel["name"]) for channel in self.static_channels]
        self.input_channel_names = list(getattr(self.hparams, "input_channel_names", None) or self.meta["input_channels"])
        self.target_channel_names = list(getattr(self.hparams, "target_channel_names", None) or self.meta["target_channels"])

        channel_name_to_index = {channel["name"]: int(channel["index"]) for channel in self.channels}
        self.input_channel_indices = [channel_name_to_index[name] for name in self.input_channel_names]
        self.target_channel_indices = [channel_name_to_index[name] for name in self.target_channel_names]
        input_channel_positions = {name: index for index, name in enumerate(self.input_channel_names)}

        self.register_buffer(
            "target_input_index_tensor",
            torch.as_tensor([input_channel_positions[name] for name in self.target_channel_names], dtype=torch.long),
            persistent=False,
        )

        self.input_scaler, self.target_scaler = self._build_scalers()
        self.interval_diff_scalers = self._build_interval_diff_scalers()

    def _iter_standard_scalers(self):
        for scaler in (self.input_scaler, self.target_scaler):
            yield scaler
        for interval in sorted(self.interval_diff_scalers):
            yield self.interval_diff_scalers[interval]

    def _get_model_derived_args(self):
        return {
            "hist_len": int(self.hparams.hist_len),
            "pred_len": int(self.hparams.pred_len),
            "var_num": len(self.input_channel_names),
            "input_var_num": len(self.input_channel_names),
            "target_var_num": len(self.target_channel_names),
            "static_var_num": len(self.static_channels),
            "input_channel_names": tuple(self.input_channel_names),
            "target_channel_names": tuple(self.target_channel_names),
            "static_channel_names": tuple(self.static_channel_names),
            "grid_shape": tuple(self.meta["grid_shape"]),
            "height": int(self.meta["grid_shape"][0]),
            "width": int(self.meta["grid_shape"][1]),
        }

    def _build_loss_function(self):
        weather_loss = str(getattr(self.hparams, "weather_loss", "mse"))
        if weather_loss == "mse":
            return nn.MSELoss()
        if weather_loss == "lat_weighted_mse":
            return LatitudeWeightedMSELoss(self.latitude)
        raise ValueError("unsupported weather_loss '{}'; expected one of ['mse', 'lat_weighted_mse']".format(weather_loss))

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

    def _select_input_channel_stats(self, stat_array: np.ndarray) -> np.ndarray:
        stat_array = _ensure_stat_layout(stat_array)
        if stat_array.shape[1] == len(self.input_channel_names):
            return stat_array
        if stat_array.shape[1] == len(self.channels):
            return stat_array[:, self.input_channel_indices, ...]
        raise ValueError(
            "diff stats channel dimension {} does not match input channels {} or all channels {}".format(
                stat_array.shape[1],
                len(self.input_channel_names),
                len(self.channels),
            )
        )

    def _load_named_diff_stats(self, stats_path: Path) -> np.ndarray:
        with np.load(stats_path, allow_pickle=False) as stats:
            pieces = []
            for channel_name in self.input_channel_names:
                if channel_name not in stats.files:
                    raise ValueError(
                        "diff stats file '{}' is missing channel '{}'; available keys are {}".format(
                            stats_path,
                            channel_name,
                            sorted(stats.files),
                        )
                    )
                pieces.append(_normalize_named_channel_stat(np.asarray(stats[channel_name], dtype=np.float32)))
        return _ensure_stat_layout(np.concatenate(pieces, axis=0))

    def _load_interval_diff_stats(self, interval_hours: int) -> tuple[np.ndarray, np.ndarray]:
        canonical_path = self.dataset_dir / "diff_stats_{}.npz".format(interval_hours)
        if canonical_path.exists():
            with np.load(canonical_path, allow_pickle=False) as stats:
                if "mean" not in stats or "std" not in stats:
                    raise ValueError("canonical diff stats '{}' must define 'mean' and 'std' arrays".format(canonical_path))
                mean = self._select_input_channel_stats(np.asarray(stats["mean"], dtype=np.float32))
                std = self._select_input_channel_stats(np.asarray(stats["std"], dtype=np.float32))
            return mean, std

        mean_npz_path = self.dataset_dir / "normalize_diff_mean_{}.npz".format(interval_hours)
        std_npz_path = self.dataset_dir / "normalize_diff_std_{}.npz".format(interval_hours)
        if mean_npz_path.exists() and std_npz_path.exists():
            return self._load_named_diff_stats(mean_npz_path), self._load_named_diff_stats(std_npz_path)

        mean_npy_path = self.dataset_dir / "normalize_diff_mean_{}.npy".format(interval_hours)
        std_npy_path = self.dataset_dir / "normalize_diff_std_{}.npy".format(interval_hours)
        if mean_npy_path.exists() and std_npy_path.exists():
            mean = self._select_input_channel_stats(np.load(mean_npy_path, allow_pickle=False))
            std = self._select_input_channel_stats(np.load(std_npy_path, allow_pickle=False))
            return mean, std

        raise ValueError(
            "missing diff stats for interval {} under '{}'; expected either '{}' or paired normalize_diff_mean/std files".format(
                interval_hours,
                self.dataset_dir,
                canonical_path.name,
            )
        )

    def _build_interval_diff_scalers(self):
        if str(getattr(self.hparams, "model", "")) != "ARROW":
            return {}
        intervals = list(getattr(self.hparams, "arrow_train_intervals", (6, 12, 24)))
        scalers = {}
        for interval in intervals:
            interval_hours = int(interval)
            if interval_hours in scalers:
                continue
            mean, std = self._load_interval_diff_stats(interval_hours)
            std = np.where(std == 0, 1.0, std)
            scalers[interval_hours] = StandardScaler(mean, std)
        return scalers

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
        if "interval_diff_scalers" in model_forward_parameters:
            model_kwargs["interval_diff_scalers"] = self.interval_diff_scalers
        if "input_state_scaler" in model_forward_parameters:
            model_kwargs["input_state_scaler"] = self.input_scaler

        prediction = self.model(var_x, marker_x, marker_y, **model_kwargs)
        return prediction, label
