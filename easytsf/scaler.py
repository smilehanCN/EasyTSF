import math

import numpy as np
import torch


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = self._coerce_stats(mean)
        self.std = self._coerce_stats(std)
        if self.mean.shape != self.std.shape:
            raise ValueError("scaler mean/std shape mismatch: {} vs {}".format(tuple(self.mean.shape), tuple(self.std.shape)))
        if self._any_non_positive(self.std):
            raise ValueError("scaler std must be strictly positive")

    @classmethod
    def fit(cls, data, null_val=None, norm_each_channel=True):
        mean, std = fit_zscore_stats(data, null_val=null_val, norm_each_channel=norm_each_channel)
        return cls(mean, std)

    @staticmethod
    def _coerce_stats(stats):
        if isinstance(stats, torch.Tensor):
            return stats.detach()
        return np.asarray(stats, dtype=np.float32)

    @staticmethod
    def _any_non_positive(stats):
        if isinstance(stats, torch.Tensor):
            return bool(torch.any(stats <= 0).item())
        return bool(np.any(stats <= 0))

    @property
    def shape(self):
        return tuple(self.mean.shape)

    def _stats_like(self, input_data):
        if isinstance(input_data, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=input_data.dtype, device=input_data.device)
            std = torch.as_tensor(self.std, dtype=input_data.dtype, device=input_data.device)
            return mean, std
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return mean, std

    def transform(self, input_data, mask=None):
        mean, std = self._stats_like(input_data)
        return transform_by_stats(input_data, mean, std, mask=mask)

    def inverse_transform(self, input_data, mask=None):
        mean, std = self._stats_like(input_data)
        return inverse_transform_by_stats(input_data, mean, std, mask=mask)

    def export_numpy(self):
        return np.asarray(self.mean, dtype=np.float32).copy(), np.asarray(self.std, dtype=np.float32).copy()


def _is_nan_null_value(null_val):
    if null_val is None:
        return False
    try:
        return math.isnan(float(null_val))
    except (TypeError, ValueError):
        return False


def build_valid_mask(data, null_val):
    if isinstance(data, torch.Tensor):
        if null_val is None:
            return torch.ones_like(data, dtype=torch.bool)
        if _is_nan_null_value(null_val):
            return ~torch.isnan(data)
        return data != null_val

    array = np.asarray(data)
    if null_val is None:
        return np.ones_like(array, dtype=bool)
    if _is_nan_null_value(null_val):
        return ~np.isnan(array)
    return array != null_val


def fit_zscore_stats(data, null_val, norm_each_channel):
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("sequence scaler stats require train data shaped [L, N], but received {}".format(tuple(array.shape)))

    valid_mask = build_valid_mask(array, null_val)
    masked_array = np.where(valid_mask, array, 0.0).astype(np.float32, copy=False)

    if norm_each_channel:
        valid_count = valid_mask.sum(axis=0, keepdims=True).astype(np.float32, copy=False)
        if np.any(valid_count <= 0):
            raise ValueError("cannot compute per-channel scaler stats because at least one variable has no valid training values")
        mean = masked_array.sum(axis=0, keepdims=True) / valid_count
        centered = np.where(valid_mask, array - mean, 0.0).astype(np.float32, copy=False)
        std = np.sqrt((centered * centered).sum(axis=0, keepdims=True) / valid_count).astype(np.float32, copy=False)
        std[std == 0] = 1.0
        return mean.astype(np.float32, copy=False), std

    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        raise ValueError("cannot compute global scaler stats because the training split has no valid values")
    mean = np.float32(masked_array.sum() / valid_count)
    centered = np.where(valid_mask, array - mean, 0.0).astype(np.float32, copy=False)
    std = np.float32(np.sqrt((centered * centered).sum() / valid_count))
    if float(std) == 0.0:
        std = np.float32(1.0)
    return mean, std


def transform_by_stats(input_data, mean, std, mask=None):
    normed_data = (input_data - mean) / std
    if mask is not None:
        normed_data = torch.where(mask, normed_data, input_data)
    return normed_data


def inverse_transform_by_stats(input_data, mean, std, mask=None):
    denormed_data = input_data * std + mean
    if mask is not None:
        denormed_data = torch.where(mask, denormed_data, input_data)
    return denormed_data


def fill_invalid_values(input_data, mask, fill_value=0.0):
    fill_tensor = torch.full_like(input_data, float(fill_value))
    return torch.where(mask, input_data, fill_tensor)


def _masked_reduce(error, mask):
    if mask is None:
        return error.mean()

    valid_mask = mask.to(dtype=torch.bool)
    valid_count = int(valid_mask.sum().item())
    if valid_count <= 0:
        raise ValueError("masked reduction requires at least one valid target value")

    masked_error = torch.where(valid_mask, error, torch.zeros_like(error))
    return masked_error.sum() / valid_mask.to(dtype=error.dtype).sum()


def masked_mae(prediction, targets, mask=None):
    return _masked_reduce(torch.abs(prediction - targets), mask)


def masked_mse(prediction, targets, mask=None):
    return _masked_reduce((prediction - targets) ** 2, mask)
