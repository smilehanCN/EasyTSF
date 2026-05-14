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
class SimVPv23DModelConfig:
    model_name: str = "simvpv2_3d"
    in_channels: int = 3
    out_channels: int | None = None
    coord_channels: int = 3
    embed_dim: int = 64
    hidden_dim: int = 128
    depth: int = 4
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    spatial_downsample_factor_3d: tuple[int, int, int] = (2, 2, 1)
    dropout: float = 0.0
    use_coords: bool = True
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> SimVPv23DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "simvpv2_3d"))
        if "kernel_size" in values:
            values["kernel_size"] = _as_tuple3(values["kernel_size"], "kernel_size")
        if "spatial_downsample_factor_3d" in values:
            values["spatial_downsample_factor_3d"] = _as_tuple3(
                values["spatial_downsample_factor_3d"],
                "spatial_downsample_factor_3d",
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class GatedConv3DBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        kernel_size: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        padding = tuple(size // 2 for size in kernel_size)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.expand = nn.Conv3d(channels, hidden_dim * 2, kernel_size=1)
        self.mix = nn.Conv3d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_dim,
        )
        self.project = nn.Conv3d(hidden_dim, channels, kernel_size=1)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value, gate = self.expand(self.norm(x)).chunk(2, dim=1)
        x = self.mix(value) * torch.sigmoid(gate)
        x = self.project(F.gelu(x))
        return residual + self.dropout(x)


class Model(nn.Module):
    """SimVPv2-style encoder-translator-decoder adapted to dense 3D wind volumes."""

    def __init__(
        self,
        history_len: int,
        pred_len: int = 1,
        in_channels: int = 3,
        out_channels: int | None = None,
        coord_channels: int = 3,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        depth: int = 4,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        spatial_downsample_factor_3d: tuple[int, int, int] = (2, 2, 1),
        dropout: float = 0.0,
        use_coords: bool = True,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.regression_out_channels = self.in_channels if out_channels is None else int(out_channels)
        self.coord_channels = int(coord_channels)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.kernel_size = _as_tuple3(kernel_size, "kernel_size")
        self.spatial_downsample_factor_3d = _as_tuple3(
            spatial_downsample_factor_3d,
            "spatial_downsample_factor_3d",
        )
        self.dropout = float(dropout)
        self.use_coords = bool(use_coords)
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)

        if self.history_len <= 0 or self.pred_len <= 0:
            raise ValueError("history_len and pred_len must be > 0")
        if self.embed_dim <= 0 or self.hidden_dim <= 0 or self.depth <= 0:
            raise ValueError("embed_dim, hidden_dim, and depth must be > 0")
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "simvpv2_3d output_mode must be one of ['regression', 'classification'], got {}".format(
                    self.output_mode
                )
            )
        if self.regression_out_channels <= 0:
            raise ValueError("out_channels must be > 0")
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")

        self.classification_channels = self.risk_num_classes * self.risk_num_heads
        self.output_channels = self.regression_out_channels if self.output_mode == "regression" else self.classification_channels
        input_channels = self.in_channels + (self.coord_channels if self.use_coords else 0)

        self.spatial_downsample = nn.AvgPool3d(
            kernel_size=self.spatial_downsample_factor_3d,
            stride=self.spatial_downsample_factor_3d,
            ceil_mode=True,
        )
        self.frame_encoder = nn.Sequential(
            nn.Conv3d(input_channels, self.embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=self.embed_dim),
            nn.GELU(),
        )
        translator_channels = self.history_len * self.embed_dim
        self.translator = nn.Sequential(
            *[
                GatedConv3DBlock(
                    channels=translator_channels,
                    hidden_dim=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.temporal_projector = nn.Conv3d(translator_channels, self.pred_len * self.embed_dim, kernel_size=1)
        self.frame_decoder = nn.Sequential(
            nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=self.embed_dim),
            nn.GELU(),
            nn.Conv3d(self.embed_dim, self.output_channels, kernel_size=1),
        )

    def _normalize_coords(self, coords: torch.Tensor | None, batch_size: int, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        if coords is None:
            grid_y = torch.linspace(0.0, 1.0, spatial_shape[0], device=self.frame_encoder[0].weight.device)
            grid_x = torch.linspace(0.0, 1.0, spatial_shape[1], device=self.frame_encoder[0].weight.device)
            grid_z = torch.linspace(0.0, 1.0, spatial_shape[2], device=self.frame_encoder[0].weight.device)
            mesh_y, mesh_x, mesh_z = torch.meshgrid(grid_y, grid_x, grid_z, indexing="ij")
            coords = torch.stack([mesh_y, mesh_x, mesh_z], dim=0).unsqueeze(0)
        if coords.dim() == 4:
            coords = coords.unsqueeze(0)
        if coords.dim() != 5:
            raise ValueError(f"simvpv2_3d expects coords as [B,C,Y,X,Z] or [C,Y,X,Z], got {tuple(coords.shape)}")
        if coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1, -1, -1)
        if coords.size(0) != batch_size:
            raise ValueError(f"simvpv2_3d expects coords batch size {batch_size}, got {coords.size(0)}")
        if coords.size(1) != self.coord_channels:
            raise ValueError(f"simvpv2_3d expects coord_channels={self.coord_channels}, got {coords.size(1)}")
        if tuple(coords.shape[-3:]) != spatial_shape:
            raise ValueError(
                "simvpv2_3d expects coords spatial shape {}, got {}".format(spatial_shape, tuple(coords.shape[-3:]))
            )
        return coords

    def _resample_spatial_tensor(
        self,
        tensor: torch.Tensor,
        *,
        output_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
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

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError(f"simvpv2_3d expects x as [B,H,C,Y,X,Z], got {tuple(x.shape)}")
        batch_size, history_len, channels, size_y, size_x, size_z = x.shape
        if history_len != self.history_len:
            raise ValueError(f"simvpv2_3d expects history_len={self.history_len}, got {history_len}")
        if channels != self.in_channels:
            raise ValueError(f"simvpv2_3d expects in_channels={self.in_channels}, got {channels}")

        original_shape = (int(size_y), int(size_x), int(size_z))
        frames = x.reshape(batch_size * history_len, channels, size_y, size_x, size_z)
        frames = self._resample_spatial_tensor(frames)
        down_shape = tuple(int(size) for size in frames.shape[-3:])
        if self.use_coords:
            coords = self._normalize_coords(coords, batch_size=batch_size, spatial_shape=original_shape)
            coords = self._resample_spatial_tensor(coords)
            coords = coords.unsqueeze(1).expand(-1, history_len, -1, -1, -1, -1)
            coords = coords.reshape(batch_size * history_len, self.coord_channels, *down_shape)
            frames = torch.cat([frames, coords.to(dtype=frames.dtype, device=frames.device)], dim=1)

        encoded = self.frame_encoder(frames)
        encoded = encoded.reshape(batch_size, history_len * self.embed_dim, *down_shape)
        translated = self.translator(encoded)
        latent = self.temporal_projector(translated)
        latent = latent.reshape(batch_size * self.pred_len, self.embed_dim, *down_shape)
        decoded = self.frame_decoder(latent)
        decoded = self._resample_spatial_tensor(decoded, output_shape=original_shape)
        return decoded.reshape(batch_size, self.pred_len, self.output_channels, *original_shape)

    def get_aux_loss(self) -> torch.Tensor | None:
        return None
