from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _as_tuple3(value: object, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        parsed = tuple(int(v) for v in value)
        if any(v <= 0 for v in parsed):
            raise ValueError(f"Expected {field_name} entries to be positive, got {value!r}")
        return parsed
    raise ValueError(f"Expected {field_name} to be a length-3 tuple/list, got {value!r}")


@dataclass
class FreDNMultivariate3DModelConfig:
    model_name: str = "fredn_multivariate3d"
    in_channels: int = 3
    embed_size: int = 64
    hidden_size: int = 128
    hidden_layers: int = 2
    dropout: float = 0.1
    use_revin: bool = True
    revin_affine: bool = True
    revin_subtract_last: bool = False
    voxel_chunk_size: int = 32768
    spatial_downsample_factor_3d: tuple[int, int, int] = (4, 4, 4)
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> FreDNMultivariate3DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "fredn_multivariate3d"))
        if "spatial_downsample_factor_3d" in values:
            values["spatial_downsample_factor_3d"] = _as_tuple3(
                values["spatial_downsample_factor_3d"],
                "spatial_downsample_factor_3d",
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class RevIN(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self._center: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unsupported RevIN mode: {mode}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        if self.subtract_last:
            self._center = x[:, -1:, :].detach()
        else:
            self._center = torch.mean(x, dim=1, keepdim=True).detach()
        variance = torch.var(x, dim=1, keepdim=True, unbiased=False)
        self._stdev = torch.sqrt(variance + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._center is None or self._stdev is None:
            raise RuntimeError("Call RevIN in 'norm' mode before normalization.")
        x = (x - self._center.to(x.device)) / self._stdev.to(x.device)
        if self.affine:
            weight = self.affine_weight.view(1, 1, -1).to(x.device)
            bias = self.affine_bias.view(1, 1, -1).to(x.device)
            x = x * weight + bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._center is None or self._stdev is None:
            raise RuntimeError("Call RevIN in 'norm' mode before denormalization.")
        if self.affine:
            weight = self.affine_weight.view(1, 1, -1).to(x.device)
            bias = self.affine_bias.view(1, 1, -1).to(x.device)
            x = (x - bias) / (weight + self.eps * self.eps)
        return x * self._stdev.to(x.device) + self._center.to(x.device)


class FreqDecomp(nn.Module):
    def __init__(self, seq_len: int, num_features: int, embed_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.embed_size = embed_size
        self.num_freq = seq_len // 2 + 1

        freq_idx = torch.arange(self.num_freq, dtype=torch.float32)
        base_val = 1.0 / torch.pow(freq_idx + 1.0, 0.5)
        base_val = (base_val / base_val.min() * 5.0).view(self.num_freq, 1, 1)
        self.mask = nn.Parameter(base_val.repeat(1, num_features, embed_size))
        with torch.no_grad():
            self.mask[..., 0] = 0.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
        trend_fft = x_fft * torch.sigmoid(self.mask).unsqueeze(0)
        trend = torch.fft.irfft(trend_fft, n=self.seq_len, dim=1, norm="ortho")
        season = x - trend
        return season, trend


class Model(nn.Module):
    def __init__(
        self,
        history_len: int,
        pred_len: int = 1,
        in_channels: int = 3,
        embed_size: int = 64,
        hidden_size: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.1,
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_subtract_last: bool = False,
        voxel_chunk_size: int = 32768,
        spatial_downsample_factor_3d: tuple[int, int, int] = (4, 4, 4),
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
    ) -> None:
        super().__init__()
        if voxel_chunk_size <= 0:
            raise ValueError(f"voxel_chunk_size must be positive, got {voxel_chunk_size}.")
        if any(factor <= 0 for factor in spatial_downsample_factor_3d):
            raise ValueError(
                "spatial_downsample_factor_3d must contain positive integers, "
                f"got {spatial_downsample_factor_3d}."
            )

        self.history_len = history_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.use_revin = use_revin
        self.voxel_chunk_size = voxel_chunk_size
        self.spatial_downsample_factor_3d = spatial_downsample_factor_3d
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "fredn_multivariate3d output_mode must be one of ['regression', 'classification'], got {}".format(
                    self.output_mode
                )
            )
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")
        self.output_channels = (
            self.in_channels if self.output_mode == "regression" else self.risk_num_classes * self.risk_num_heads
        )

        self.emb = nn.Parameter(torch.empty(history_len, embed_size))
        nn.init.xavier_uniform_(self.emb)
        self.decomp = FreqDecomp(seq_len=history_len, num_features=in_channels, embed_size=embed_size)
        self.revin = (
            RevIN(
                num_features=in_channels,
                affine=revin_affine,
                subtract_last=revin_subtract_last,
            )
            if use_revin
            else None
        )
        self.freq_learner = self._create_learner(history_len // 2 + 1, pred_len // 2 + 1)
        self.trend_learner = self._create_learner(history_len, pred_len)
        self.emb_proj = nn.Linear(embed_size, 1)
        self.spatial_downsample = nn.AvgPool3d(
            kernel_size=spatial_downsample_factor_3d,
            stride=spatial_downsample_factor_3d,
            ceil_mode=True,
        )
        self.risk_head = None
        if self.output_mode == "classification":
            self.risk_head = nn.Linear(self.in_channels, self.output_channels)

    def _calc_hidden_dims(self, output_dim: int) -> list[int]:
        ratio = (self.hidden_size / output_dim) ** (1 / (self.hidden_layers + 1))
        return [max(1, int(self.hidden_size / (ratio**i))) for i in range(self.hidden_layers + 1)]

    def _create_learner(self, input_size: int, output_size: int) -> nn.ModuleDict:
        hidden_dims = self._calc_hidden_dims(output_size)
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_dims[0])]
        for idx in range(self.hidden_layers):
            layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            if idx % 2 == 0:
                layers.append(nn.LayerNorm(hidden_dims[idx + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
        shared = nn.ModuleDict(
            {
                "res_proj": nn.Linear(input_size, hidden_dims[-1]),
                "proj_layers": nn.Sequential(*layers),
                "pred_layers": nn.Linear(hidden_dims[-1], output_size),
            }
        )
        return nn.ModuleDict({"real_part": shared, "imag_part": shared})

    def _freq_forward(self, season: torch.Tensor) -> torch.Tensor:
        season = season.float()
        x_freq = torch.fft.rfft(season.permute(0, 2, 3, 1), dim=3, norm="ortho")
        res_real = self.freq_learner["real_part"]["res_proj"](x_freq.real)
        res_imag = self.freq_learner["imag_part"]["res_proj"](x_freq.imag)
        proj_real = self.freq_learner["real_part"]["proj_layers"](x_freq.real)
        proj_imag = self.freq_learner["imag_part"]["proj_layers"](x_freq.imag)
        pred_real = self.freq_learner["real_part"]["pred_layers"](proj_real + res_real)
        pred_imag = self.freq_learner["imag_part"]["pred_layers"](proj_imag + res_imag)
        pred_freq = torch.complex(pred_real.float(), pred_imag.float())
        pred = torch.fft.irfft(pred_freq, n=self.pred_len, dim=3, norm="ortho")
        return self.emb_proj(pred.permute(0, 3, 1, 2)).squeeze(-1)

    def _trend_forward(self, trend: torch.Tensor) -> torch.Tensor:
        x = trend.permute(0, 2, 3, 1)
        res = self.trend_learner["real_part"]["res_proj"](x)
        proj = self.trend_learner["real_part"]["proj_layers"](x)
        pred = self.trend_learner["real_part"]["pred_layers"](proj + res)
        return self.emb_proj(pred.permute(0, 3, 1, 2)).squeeze(-1)

    def _forward_sequence_batch(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin is not None:
            x = self.revin(x, "norm")
        x_embed = x.unsqueeze(-1) * self.emb.view(1, self.history_len, 1, self.embed_size)
        season, trend = self.decomp(x_embed)
        prediction = self._freq_forward(season) + self._trend_forward(trend)
        if self.revin is not None:
            prediction = self.revin(prediction, "denorm")
        return prediction

    def _resample_spatial_tensor(self, tensor: torch.Tensor, *, output_shape: tuple[int, int, int] | None = None) -> torch.Tensor:
        original_dtype = tensor.dtype
        needs_promotion = tensor.device.type == "cpu" and tensor.dtype in {torch.float16, torch.bfloat16}
        if needs_promotion:
            tensor = tensor.float()
        if output_shape is None:
            tensor = self.spatial_downsample(tensor)
        else:
            tensor = F.interpolate(tensor, size=output_shape, mode="trilinear", align_corners=False)
        if needs_promotion:
            tensor = tensor.to(dtype=original_dtype)
        return tensor

    def _downsample_spatial(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, _, _, _ = x.shape
        flattened = x.reshape(batch_size * steps, channels, *x.shape[-3:])
        downsampled = self._resample_spatial_tensor(flattened)
        return downsampled.reshape(batch_size, steps, channels, *downsampled.shape[-3:])

    def _upsample_spatial(self, x: torch.Tensor, output_shape: tuple[int, int, int]) -> torch.Tensor:
        if x.shape[-3:] == output_shape:
            return x
        batch_size, steps, channels, _, _, _ = x.shape
        flattened = x.reshape(batch_size * steps, channels, *x.shape[-3:])
        upsampled = self._resample_spatial_tensor(flattened, output_shape=output_shape)
        return upsampled.reshape(batch_size, steps, channels, *output_shape)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del coords, mask
        batch_size, history_len, channels, size_y, size_x, size_z = x.shape
        if history_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {history_len}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected channels={self.in_channels}, got {channels}.")

        original_shape = (size_y, size_x, size_z)
        x = self._downsample_spatial(x)
        reduced_shape = x.shape[-3:]
        voxel_series = x.permute(0, 3, 4, 5, 1, 2).reshape(batch_size * reduced_shape[0] * reduced_shape[1] * reduced_shape[2], history_len, channels)
        predictions = []
        for start in range(0, voxel_series.size(0), self.voxel_chunk_size):
            end = min(start + self.voxel_chunk_size, voxel_series.size(0))
            predictions.append(self._forward_sequence_batch(voxel_series[start:end]))
        forecast = torch.cat(predictions, dim=0)
        forecast = (
            forecast.reshape(batch_size, reduced_shape[0], reduced_shape[1], reduced_shape[2], self.pred_len, channels)
            .permute(0, 4, 5, 1, 2, 3)
            .contiguous()
        )
        forecast = self._upsample_spatial(forecast, output_shape=original_shape)
        if self.output_mode == "regression":
            return forecast

        logits = self.risk_head(forecast.permute(0, 1, 3, 4, 5, 2))
        return logits.permute(0, 1, 5, 2, 3, 4).contiguous()
