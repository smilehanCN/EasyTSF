from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _as_tuple3(value: object, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(v) for v in value)
    raise ValueError(f"Expected {field_name} to be a length-3 tuple/list, got {value!r}")


@dataclass
class UNet3DModelConfig:
    model_name: str = "unet3d"
    in_channels: int = 3
    coord_channels: int = 3
    base_channels: int = 16
    patch_size: tuple[int, int, int] = (4, 4, 2)
    downsample_scale: tuple[int, int, int] = (2, 2, 2)
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    expansion: int = 2
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> UNet3DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "unet3d"))
        for key in ("patch_size", "downsample_scale", "kernel_size"):
            if key in values:
                values[key] = _as_tuple3(values[key], key)
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


def make_norm3d(num_channels: int) -> nn.Module:
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class DSConvBlock3D(nn.Module):
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
        diff_y = skip.size(-3) - x.size(-3)
        diff_x = skip.size(-2) - x.size(-2)
        diff_z = skip.size(-1) - x.size(-1)
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
        x = x + self.skip_proj(skip)
        return self.block(x)


class Model(nn.Module):
    def __init__(
        self,
        history_len: int,
        pred_len: int = 1,
        in_channels: int = 3,
        coord_channels: int = 3,
        base_channels: int = 16,
        patch_size: tuple[int, int, int] = (4, 4, 2),
        downsample_scale: tuple[int, int, int] = (2, 2, 2),
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        expansion: int = 2,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "unet3d output_mode must be one of ['regression', 'classification'], got {}".format(self.output_mode)
            )
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")
        self.classification_channels = self.risk_num_classes * self.risk_num_heads

        total_in_channels = history_len * in_channels + coord_channels

        self.patch_embed = PatchEmbed3D(total_in_channels, base_channels, patch_size)
        self.enc1 = DSConvBlock3D(base_channels, kernel_size=kernel_size, expansion=expansion)
        self.down1 = Downsample3D(base_channels, base_channels * 2, downsample_scale)
        self.enc2 = DSConvBlock3D(base_channels * 2, kernel_size=kernel_size, expansion=expansion)
        self.down2 = Downsample3D(base_channels * 2, base_channels * 4, downsample_scale)
        self.enc3 = DSConvBlock3D(base_channels * 4, kernel_size=kernel_size, expansion=expansion)
        self.down3 = Downsample3D(base_channels * 4, base_channels * 8, downsample_scale)
        self.bottleneck = DSConvBlock3D(base_channels * 8, kernel_size=kernel_size, expansion=expansion)

        self.up3 = UpsampleAdd3D(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            downsample_scale,
            kernel_size=kernel_size,
            expansion=expansion,
        )
        self.up2 = UpsampleAdd3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            downsample_scale,
            kernel_size=kernel_size,
            expansion=expansion,
        )
        self.up1 = UpsampleAdd3D(
            base_channels * 2,
            base_channels,
            base_channels,
            downsample_scale,
            kernel_size=kernel_size,
            expansion=expansion,
        )
        self.final_up = nn.ConvTranspose3d(
            base_channels,
            base_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.regression_head = nn.Conv3d(base_channels, pred_len * self.in_channels, kernel_size=1)
        self.classification_head = nn.Conv3d(base_channels, pred_len * self.classification_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        batch, time_steps, channels, ydim, xdim, zdim = x.shape
        x = x.reshape(batch, time_steps * channels, ydim, xdim, zdim)
        if coords is not None:
            x = torch.cat([x, coords], dim=1)

        s1 = self.enc1(self.patch_embed(x))
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        bottleneck = self.bottleneck(self.down3(s3))

        x = self.up3(bottleneck, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.final_up(x)

        diff_y = ydim - x.size(-3)
        diff_x = xdim - x.size(-2)
        diff_z = zdim - x.size(-1)
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

        if self.output_mode == "regression":
            x = self.regression_head(x)
            return x.view(batch, self.pred_len, self.in_channels, ydim, xdim, zdim)

        x = self.classification_head(x)
        return x.view(batch, self.pred_len, self.classification_channels, ydim, xdim, zdim)
