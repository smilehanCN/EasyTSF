from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from ._arrow_backbone import ArrowBackbone, ensure_arrow_dependencies_available


def _sum_off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    return matrix.sum() - torch.diagonal(matrix, dim1=1, dim2=2).sum()


def _pairwise_cross_entropy(p: torch.Tensor, q: torch.Tensor | None = None, eps: float = 1e-9) -> torch.Tensor:
    if q is None:
        q = p
    p = p.unsqueeze(2)
    q = q.unsqueeze(1)
    log_q = torch.log(q + eps)
    return torch.sum(p * log_q, dim=-1)


def _uniform_cross_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p / p.sum(dim=1, keepdim=True)
    expert_count = p.size(1)
    return -torch.sum(torch.log(p + 1e-8), dim=1) / expert_count


class SparseMoEAuxLoss(nn.Module):
    def __init__(self, eps=1e-6, alpha=1e-2, beta=1.0, w_aux_1=True, w_aux_2=False):
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.w_aux_1 = bool(w_aux_1)
        self.w_aux_2 = bool(w_aux_2)

    def forward(self, noises: torch.Tensor):
        del self.eps
        layers = noises.shape[0]
        noises = torch.softmax(noises, dim=-1)
        aux_loss_1 = _sum_off_diagonal(_pairwise_cross_entropy(noises)) / (2 * layers)
        noises = noises.mean(dim=1)
        aux_loss_2 = _uniform_cross_entropy(noises).mean(dim=0)
        if self.w_aux_1 and self.w_aux_2:
            return self.alpha * (self.beta * aux_loss_1 + aux_loss_2)
        if self.w_aux_1 and not self.w_aux_2:
            return self.alpha * self.beta * aux_loss_1
        if not self.w_aux_1 and self.w_aux_2:
            return self.alpha * aux_loss_2
        return noises.new_zeros(())


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        height,
        width,
        input_channel_names,
        target_channel_names,
        static_channel_names=(),
        static_var_num=0,
        arrow_patch_size=4,
        arrow_hidden_size=1024,
        arrow_depth=16,
        arrow_num_heads=16,
        arrow_mlp_ratio=2.0,
        arrow_rope_type="mixed",
        arrow_rope_theta=100.0,
        arrow_use_mla=False,
        arrow_use_moe=True,
        arrow_routed_num_experts=9,
        arrow_shared_num_experts=1,
        arrow_selected_experts=3,
        arrow_train_intervals=(6, 12, 24),
        arrow_ring_pos_embed=True,
    ):
        super().__init__()
        ensure_arrow_dependencies_available()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.height = int(height)
        self.width = int(width)
        if self.hist_len != 1:
            raise ValueError("ARROW requires hist_len == 1, but received {}".format(self.hist_len))
        if self.pred_len <= 0:
            raise ValueError("ARROW requires pred_len > 0, but received {}".format(self.pred_len))
        if self.height <= 0 or self.width <= 0:
            raise ValueError("ARROW requires positive height/width, but received ({}, {})".format(self.height, self.width))

        self.input_channel_names = tuple(str(name) for name in input_channel_names)
        self.target_channel_names = tuple(str(name) for name in target_channel_names)
        if len(self.input_channel_names) == 0:
            raise ValueError("ARROW requires at least one input channel")
        if not set(self.target_channel_names).issubset(set(self.input_channel_names)):
            raise ValueError("target_channel_names must be a subset of input_channel_names")

        self.static_channel_names = tuple(str(name) for name in static_channel_names)
        if int(static_var_num) != len(self.static_channel_names):
            raise ValueError(
                "static_var_num {} does not match the number of static_channel_names {}".format(
                    static_var_num,
                    len(self.static_channel_names),
                )
            )

        unique_intervals = []
        for interval in arrow_train_intervals:
            interval_int = int(interval)
            if interval_int <= 0:
                raise ValueError("ARROW train intervals must be positive hours")
            if interval_int not in unique_intervals:
                unique_intervals.append(interval_int)
        self.arrow_train_intervals = tuple(unique_intervals)
        self._allowed_interval_set = set(self.arrow_train_intervals)
        self._target_input_indices = tuple(self.input_channel_names.index(name) for name in self.target_channel_names)

        self.backbone = ArrowBackbone(
            in_img_size=(self.height, self.width),
            variables=list(self.input_channel_names),
            static_channel_names=list(self.static_channel_names),
            patch_size=arrow_patch_size,
            hidden_size=arrow_hidden_size,
            depth=arrow_depth,
            num_heads=arrow_num_heads,
            mlp_ratio=arrow_mlp_ratio,
            rope_type=arrow_rope_type,
            rope_theta=arrow_rope_theta,
            use_mla=arrow_use_mla,
            use_moe=arrow_use_moe,
            routed_num_experts=arrow_routed_num_experts,
            shared_num_experts=arrow_shared_num_experts,
            selected_experts=arrow_selected_experts,
            list_time_intervals=self.arrow_train_intervals,
            ring_pos_embed=arrow_ring_pos_embed,
        )
        self._aux_loss_module = SparseMoEAuxLoss()
        self._aux_loss = None

    def get_aux_loss(self):
        return self._aux_loss

    def _ensure_timestamp_tensor(self, tensor, name, batch_size, seq_len, device):
        if tensor is None:
            raise ValueError("ARROW requires {} to infer rollout intervals".format(name))
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor, device=device)
        else:
            tensor = tensor.to(device=device)
        if tensor.ndim != 2:
            raise ValueError("ARROW expects {} as [B, T], but received {}".format(name, tuple(tensor.shape)))
        if tensor.shape != (batch_size, seq_len):
            raise ValueError(
                "ARROW expected {} shape ({}, {}), but received {}".format(
                    name,
                    batch_size,
                    seq_len,
                    tuple(tensor.shape),
                )
            )
        return tensor.to(dtype=torch.int64)

    def _infer_rollout_intervals(self, marker_x, marker_y, batch_size, device):
        marker_x = self._ensure_timestamp_tensor(marker_x, "marker_x", batch_size, self.hist_len, device)
        marker_y = self._ensure_timestamp_tensor(marker_y, "marker_y", batch_size, self.pred_len, device)
        previous = torch.cat([marker_x[:, -1:].contiguous(), marker_y[:, :-1].contiguous()], dim=1)
        delta_ns = marker_y - previous
        if (delta_ns <= 0).any():
            raise ValueError("ARROW requires strictly increasing target timestamps")
        hour_ns = torch.as_tensor(3600 * 10**9, device=device, dtype=torch.int64)
        if torch.any(delta_ns % hour_ns != 0):
            raise ValueError("ARROW requires timestamp intervals that are exact multiples of one hour")
        intervals = delta_ns // hour_ns
        invalid = sorted({int(value) for value in intervals.flatten().tolist() if int(value) not in self._allowed_interval_set})
        if invalid:
            raise ValueError(
                "ARROW received rollout intervals {} but only {} are configured".format(
                    invalid,
                    list(self.arrow_train_intervals),
                )
            )
        return intervals

    def _select_target_indices(self, target_input_indices, device):
        if target_input_indices is None:
            return torch.as_tensor(self._target_input_indices, device=device, dtype=torch.long)
        if not torch.is_tensor(target_input_indices):
            target_input_indices = torch.as_tensor(target_input_indices, device=device, dtype=torch.long)
        else:
            target_input_indices = target_input_indices.to(device=device, dtype=torch.long)
        return target_input_indices

    def _require_scaler_mapping(self, interval_diff_scalers):
        if interval_diff_scalers is None:
            raise ValueError("ARROW requires interval_diff_scalers for denormalized delta rollout")
        if not isinstance(interval_diff_scalers, Mapping):
            raise ValueError("interval_diff_scalers must be a mapping from interval hours to StandardScaler")
        return interval_diff_scalers

    def _is_scaler_like(self, scaler) -> bool:
        return hasattr(scaler, "transform") and hasattr(scaler, "inverse_transform")

    def _get_interval_scaler(self, interval_diff_scalers, interval_hours: int):
        try:
            scaler = interval_diff_scalers[int(interval_hours)]
        except KeyError as exc:
            raise ValueError(
                "missing diff scaler for interval {}; available intervals are {}".format(
                    interval_hours,
                    sorted(int(key) for key in interval_diff_scalers),
                )
            ) from exc
        if not self._is_scaler_like(scaler):
            raise ValueError("interval_diff_scalers must contain scaler-like values with transform/inverse_transform")
        return scaler

    def forward(
        self,
        var_x,
        marker_x,
        marker_y,
        static_inputs=None,
        target_input_indices=None,
        interval_diff_scalers=None,
        input_state_scaler=None,
    ):
        if var_x.ndim != 5:
            raise ValueError("ARROW expects var_x as [B, hist_len, C, H, W], but received {}".format(tuple(var_x.shape)))
        batch_size, hist_len, channel_count, height, width = var_x.shape
        if hist_len != self.hist_len:
            raise ValueError("ARROW requires hist_len {} at runtime, but received {}".format(self.hist_len, hist_len))
        if channel_count != len(self.input_channel_names):
            raise ValueError(
                "ARROW requires {} dynamic channels at runtime, but received {}".format(
                    len(self.input_channel_names),
                    channel_count,
                )
            )
        if (height, width) != (self.height, self.width):
            raise ValueError(
                "ARROW requires grid shape ({}, {}), but received ({}, {})".format(
                    self.height,
                    self.width,
                    height,
                    width,
                )
            )
        if input_state_scaler is None or not self._is_scaler_like(input_state_scaler):
            raise ValueError("ARROW requires input_state_scaler with transform/inverse_transform methods")

        interval_diff_scalers = self._require_scaler_mapping(interval_diff_scalers)
        rollout_intervals = self._infer_rollout_intervals(marker_x, marker_y, batch_size, var_x.device)
        selected_target_indices = self._select_target_indices(target_input_indices, var_x.device)

        state = var_x[:, -1, :, :, :]
        predictions = []
        aux_losses = []
        for step_idx in range(self.pred_len):
            step_hours = rollout_intervals[:, step_idx]
            normalized_interval = step_hours.to(dtype=state.dtype) / 10.0
            normalized_diff = self.backbone(
                state,
                variables=list(self.input_channel_names),
                time_interval=normalized_interval,
                static_inputs=static_inputs,
            )
            if hasattr(self.backbone, "moe_noises") and isinstance(getattr(self.backbone, "moe_noises"), torch.Tensor):
                aux_losses.append(self._aux_loss_module(self.backbone.moe_noises))

            raw_diff = torch.empty_like(normalized_diff)
            for interval_value in torch.unique(step_hours).detach().cpu().tolist():
                mask = step_hours == int(interval_value)
                scaler = self._get_interval_scaler(interval_diff_scalers, int(interval_value))
                raw_diff[mask] = scaler.inverse_transform(normalized_diff[mask])

            raw_state = input_state_scaler.inverse_transform(state)
            raw_state = raw_state + raw_diff
            state = input_state_scaler.transform(raw_state)
            predictions.append(state.index_select(dim=1, index=selected_target_indices))

        self._aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return torch.stack(predictions, dim=1)
