from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _as_tuple3(value: object, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)
    raise ValueError("Expected {} to be a length-3 tuple/list, got {!r}".format(field_name, value))


def _as_int_tuple(
    value: object,
    field_name: str,
    *,
    expected_len: int | None = None,
    min_len: int = 1,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected {} to be a list/tuple, got {!r}".format(field_name, value))
    parsed = tuple(int(v) for v in value)
    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError("Expected {} to contain {} values, got {}".format(field_name, expected_len, len(parsed)))
    if len(parsed) < min_len:
        raise ValueError("Expected {} to contain at least {} values, got {}".format(field_name, min_len, len(parsed)))
    return parsed


def _as_downsample_scales(
    value: object | None,
    fallback: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if value is None:
        return (fallback, fallback, fallback)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(
            _as_tuple3(scale, "downsample_scales[{}]".format(index))
            for index, scale in enumerate(value)
        )
    raise ValueError("Expected downsample_scales to contain three length-3 scales, got {!r}".format(value))


def _as_offset_tuples(value: object, field_name: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("Expected {} to be a non-empty list/tuple of 3D offsets, got {!r}".format(field_name, value))
    return tuple(_as_tuple3(item, "{}[{}]".format(field_name, index)) for index, item in enumerate(value))


@dataclass
class UNet3DModelConfig:
    model_name: str = "unet3d"
    in_channels: int = 6
    out_channels: int | None = None
    coord_channels: int = 3
    base_channels: int = 16
    patch_size: tuple[int, int, int] = (1, 1, 1)
    io_downsample_scale: tuple[int, int, int] = (1, 1, 1)
    downsample_scale: tuple[int, int, int] = (2, 2, 2)
    downsample_scales: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    use_coords: bool = True
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> UNet3DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "unet3d"))
        for key in ("patch_size", "io_downsample_scale", "downsample_scale", "kernel_size"):
            if key in values:
                values[key] = _as_tuple3(values[key], key)
        if "downsample_scales" in values:
            values["downsample_scales"] = _as_downsample_scales(
                values["downsample_scales"],
                values.get("downsample_scale", cls.downsample_scale),
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


def make_norm3d(num_channels: int) -> nn.Module:
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class DoubleConv3D(nn.Module):
    """Paper-aligned 3D U-Net block: two 3x3x3 convs with BN and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    """3D U-Net analysis step: 2x2x2 max-pooling followed by DoubleConv3D."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_scale: tuple[int, int, int],
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(kernel_size=downsample_scale, stride=downsample_scale),
            DoubleConv3D(in_channels, out_channels, kernel_size=kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _center_crop_or_pad_3d(x: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Match spatial shape for U-Net skip concatenation with odd-sized grids."""

    target_y, target_x, target_z = target_shape
    size_y, size_x, size_z = tuple(int(size) for size in x.shape[-3:])

    crop_y = max(size_y - target_y, 0)
    crop_x = max(size_x - target_x, 0)
    crop_z = max(size_z - target_z, 0)
    if crop_y or crop_x or crop_z:
        start_y = crop_y // 2
        start_x = crop_x // 2
        start_z = crop_z // 2
        x = x[
            ...,
            start_y : start_y + min(size_y, target_y),
            start_x : start_x + min(size_x, target_x),
            start_z : start_z + min(size_z, target_z),
        ]

    size_y, size_x, size_z = tuple(int(size) for size in x.shape[-3:])
    diff_y = target_y - size_y
    diff_x = target_x - size_x
    diff_z = target_z - size_z
    if diff_y or diff_x or diff_z:
        x = F.pad(
            x,
            [
                diff_z // 2,
                diff_z - diff_z // 2,
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
    return x


def _upsample_prediction_3d(
    prediction: torch.Tensor,
    target_shape: tuple[int, int, int],
) -> torch.Tensor:
    batch, time_steps, channels, ydim, xdim, zdim = prediction.shape
    reshaped = prediction.reshape(batch, time_steps * channels, ydim, xdim, zdim)
    resized = F.interpolate(reshaped, size=target_shape, mode="trilinear", align_corners=False)
    return resized.reshape(batch, time_steps, channels, *target_shape)


class Up3D(nn.Module):
    """3D U-Net synthesis step: 2x2x2 up-conv, skip concatenation, DoubleConv3D."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_scale: tuple[int, int, int],
        kernel_size: tuple[int, int, int] = (3, 3, 3),
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_scale,
            stride=upsample_scale,
            bias=False,
        )
        self.conv = DoubleConv3D(out_channels + skip_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _center_crop_or_pad_3d(x, tuple(int(size) for size in skip.shape[-3:]))
        return self.conv(torch.cat([skip, x], dim=1))


class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DSConvBlock3D(nn.Module):
    """Compatibility helper used by unet3d_patchcat; not used by the paper baseline Model."""

    def __init__(
        self,
        channels: int,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        padding = tuple(size // 2 for size in kernel_size)
        self.block = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            make_norm3d(channels),
            nn.GELU(),
            nn.Conv3d(channels, hidden_channels, kernel_size=1, bias=False),
            make_norm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, channels, kernel_size=1, bias=False),
            make_norm3d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class PatchEmbed3D(nn.Module):
    """Compatibility helper used by unet3d_patchcat; not used by the paper baseline Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
            make_norm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Downsample3D(nn.Module):
    """Compatibility helper used by unet3d_patchcat; not used by the paper baseline Model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_scale: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=downsample_scale,
                stride=downsample_scale,
                bias=False,
            ),
            make_norm3d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpsampleAdd3D(nn.Module):
    """Compatibility helper used by unet3d_patchcat; not used by the paper baseline Model."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        upsample_scale: tuple[int, int, int],
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=upsample_scale,
            stride=upsample_scale,
            bias=False,
        )
        self.skip_proj = nn.Conv3d(skip_channels, out_channels, kernel_size=1, bias=False)
        self.block = DSConvBlock3D(out_channels, kernel_size=kernel_size, expansion=expansion)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = _center_crop_or_pad_3d(x, tuple(int(size) for size in skip.shape[-3:]))
        x = x + self.skip_proj(skip)
        return self.block(x)


class Model(nn.Module):
    """3D U-Net baseline adapted from Cicek et al. for equal-resolution forecasting."""

    def __init__(
        self,
        history_len: int | None = None,
        pred_len: int = 1,
        in_channels: int = 6,
        coord_channels: int = 3,
        base_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        io_downsample_scale: tuple[int, int, int] = (1, 1, 1),
        downsample_scale: tuple[int, int, int] = (2, 2, 2),
        downsample_scales: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        expansion: int = 2,
        use_coords: bool = True,
        out_channels: int | None = None,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
        hist_len: int | None = None,
    ) -> None:
        super().__init__()
        if history_len is None:
            if hist_len is None:
                raise ValueError("unet3d requires history_len or hist_len")
            history_len = hist_len
        elif hist_len is not None and int(hist_len) != int(history_len):
            raise ValueError("history_len {} does not match hist_len {}".format(history_len, hist_len))

        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.output_channels = self.in_channels if out_channels is None else int(out_channels)
        self.coord_channels = int(coord_channels)
        # Kept only so old configs remain loadable; pure 3D U-Net does not patch-embed the input.
        self.patch_size = _as_tuple3(patch_size, "patch_size")
        self.io_downsample_scale = _as_tuple3(io_downsample_scale, "io_downsample_scale")
        if any(scale <= 0 for scale in self.io_downsample_scale):
            raise ValueError("io_downsample_scale values must be > 0, got {}".format(self.io_downsample_scale))
        self.enable_io_downsample = any(scale > 1 for scale in self.io_downsample_scale)
        self.input_downsample = (
            nn.AvgPool3d(kernel_size=self.io_downsample_scale, stride=self.io_downsample_scale)
            if self.enable_io_downsample
            else nn.Identity()
        )
        self.downsample_scales = _as_downsample_scales(downsample_scales, _as_tuple3(downsample_scale, "downsample_scale"))
        self.kernel_size = _as_tuple3(kernel_size, "kernel_size")
        self.use_coords = bool(use_coords)
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "unet3d output_mode must be one of ['regression', 'classification'], got {}".format(self.output_mode)
            )
        if self.output_channels <= 0:
            raise ValueError("out_channels must be > 0")
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")
        self.classification_channels = self.risk_num_classes * self.risk_num_heads

        total_in_channels = self.history_len * self.in_channels
        if self.use_coords:
            total_in_channels += self.coord_channels

        self.inc = DoubleConv3D(total_in_channels, base_channels, kernel_size=self.kernel_size)
        self.down1 = Down3D(base_channels, base_channels * 2, self.downsample_scales[0], kernel_size=self.kernel_size)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, self.downsample_scales[1], kernel_size=self.kernel_size)
        self.down3 = Down3D(base_channels * 4, base_channels * 8, self.downsample_scales[2], kernel_size=self.kernel_size)

        self.up1 = Up3D(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            self.downsample_scales[2],
            kernel_size=self.kernel_size,
        )
        self.up2 = Up3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            self.downsample_scales[1],
            kernel_size=self.kernel_size,
        )
        self.up3 = Up3D(
            base_channels * 2,
            base_channels,
            base_channels,
            self.downsample_scales[0],
            kernel_size=self.kernel_size,
        )
        self.regression_head = OutConv3D(base_channels, self.pred_len * self.output_channels)
        self.classification_head = OutConv3D(base_channels, self.pred_len * self.classification_channels)

    def _normalize_coords(
        self,
        coords: torch.Tensor | None,
        batch_size: int,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor | None:
        if coords is None:
            return None
        if coords.dim() == 4:
            coords = coords.unsqueeze(0)
        if coords.dim() != 5:
            raise ValueError("UNet3D expects coords with 4 or 5 dims, but received {}".format(coords.dim()))
        if coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1, -1, -1)
        if coords.size(0) != batch_size:
            raise ValueError("UNet3D expects coords batch size {}, but received {}".format(batch_size, coords.size(0)))
        if coords.size(1) != self.coord_channels:
            raise ValueError("UNet3D expects coord_channels {}, but received {}".format(self.coord_channels, coords.size(1)))
        if tuple(coords.shape[-3:]) != spatial_shape:
            raise ValueError(
                "UNet3D expects coords spatial shape {}, but received {}".format(spatial_shape, tuple(coords.shape[-3:]))
            )
        return coords

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError("UNet3D expects x as [B, history_len, C, Y, X, Z], got {}".format(tuple(x.shape)))
        batch, time_steps, channels, ydim, xdim, zdim = x.shape
        if time_steps != self.history_len:
            raise ValueError("UNet3D expects history_len {}, got {}".format(self.history_len, time_steps))
        if channels != self.in_channels:
            raise ValueError("UNet3D expects in_channels {}, got {}".format(self.in_channels, channels))

        coords = self._normalize_coords(coords, batch_size=batch, spatial_shape=(ydim, xdim, zdim))
        if self.use_coords and coords is None:
            raise ValueError("coords are required when use_coords=True")

        output_spatial_shape = (ydim, xdim, zdim)
        x = x.reshape(batch, time_steps * channels, ydim, xdim, zdim)
        x = self.input_downsample(x)
        model_spatial_shape = tuple(int(size) for size in x.shape[-3:])

        if self.use_coords and coords is not None:
            if self.enable_io_downsample:
                coords = F.interpolate(coords, size=model_spatial_shape, mode="trilinear", align_corners=False)
            x = torch.cat([x, coords], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = _center_crop_or_pad_3d(x, model_spatial_shape)

        if self.output_mode == "regression":
            x = self.regression_head(x)
            x = x.view(batch, self.pred_len, self.output_channels, *model_spatial_shape)
            if self.enable_io_downsample:
                x = _upsample_prediction_3d(x, output_spatial_shape)
            return x

        x = self.classification_head(x)
        x = x.view(batch, self.pred_len, self.classification_channels, *model_spatial_shape)
        if self.enable_io_downsample:
            x = _upsample_prediction_3d(x, output_spatial_shape)
        return x

    def get_aux_loss(self) -> torch.Tensor | None:
        return None
