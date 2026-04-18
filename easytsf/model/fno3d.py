from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _as_tuple3(value: object, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        parsed = tuple(int(v) for v in value)
        if any(v <= 0 for v in parsed):
            raise ValueError("Expected {} entries to be positive, got {!r}".format(field_name, value))
        return parsed
    raise ValueError("Expected {} to be a length-3 tuple/list, got {!r}".format(field_name, value))


@dataclass
class FNO3DModelConfig:
    model_name: str = "fno3d"
    in_channels: int = 6
    coord_channels: int = 3
    fno_width: int = 20
    fno_layers: int = 4
    fno_modes: tuple[int, int, int] = (12, 12, 12)
    fno_padding: int = 6
    fno_projection_dim: int = 128
    use_coords: bool = True
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> FNO3DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "fno3d"))
        if "fno_modes" in values:
            values["fno_modes"] = _as_tuple3(values["fno_modes"], "fno_modes")
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class SpectralConv3D(nn.Module):
    """3D Fourier layer from the original FNO3d implementation."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int, int]) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_y, self.modes_x, self.modes_z = _as_tuple3(modes, "modes")

        self.scale = 1.0 / max(1, self.in_channels * self.out_channels)
        weight_shape = (
            self.in_channels,
            self.out_channels,
            self.modes_y,
            self.modes_x,
            self.modes_z,
        )
        self.weights1 = nn.Parameter(self.scale * torch.rand(*weight_shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(*weight_shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(*weight_shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(*weight_shape, dtype=torch.cfloat))

    def compl_mul3d(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", inputs, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        size_y, size_x, size_z = tuple(int(size) for size in x.shape[-3:])
        modes_y = min(self.modes_y, max(1, size_y // 2))
        modes_x = min(self.modes_x, max(1, size_x // 2))
        modes_z = min(self.modes_z, size_z // 2 + 1)

        x_ft = torch.fft.rfftn(x.float(), dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            size_y,
            size_x,
            size_z // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, :modes_y, :modes_x, :modes_z] = self.compl_mul3d(
            x_ft[:, :, :modes_y, :modes_x, :modes_z],
            self.weights1[:, :, :modes_y, :modes_x, :modes_z],
        )
        out_ft[:, :, -modes_y:, :modes_x, :modes_z] = self.compl_mul3d(
            x_ft[:, :, -modes_y:, :modes_x, :modes_z],
            self.weights2[:, :, :modes_y, :modes_x, :modes_z],
        )
        out_ft[:, :, :modes_y, -modes_x:, :modes_z] = self.compl_mul3d(
            x_ft[:, :, :modes_y, -modes_x:, :modes_z],
            self.weights3[:, :, :modes_y, :modes_x, :modes_z],
        )
        out_ft[:, :, -modes_y:, -modes_x:, :modes_z] = self.compl_mul3d(
            x_ft[:, :, -modes_y:, -modes_x:, :modes_z],
            self.weights4[:, :, :modes_y, :modes_x, :modes_z],
        )

        return torch.fft.irfftn(out_ft, s=(size_y, size_x, size_z), dim=(-3, -2, -1))


class Model(nn.Module):
    """FNO3d baseline adapted from Li et al. for WindShear volumes.

    The topology follows the official FNO3d code: lift with fc0, apply Fourier
    layers u'=(W+K)(u), then project with fc1/fc2.
    """

    def __init__(
        self,
        history_len: int | None = None,
        pred_len: int = 1,
        in_channels: int = 6,
        coord_channels: int = 3,
        fno_width: int = 20,
        fno_layers: int = 4,
        fno_modes: tuple[int, int, int] = (12, 12, 12),
        fno_padding: int = 6,
        fno_projection_dim: int = 128,
        use_coords: bool = True,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
        hist_len: int | None = None,
    ) -> None:
        super().__init__()
        if history_len is None:
            if hist_len is None:
                raise ValueError("fno3d requires history_len or hist_len")
            history_len = hist_len
        elif hist_len is not None and int(hist_len) != int(history_len):
            raise ValueError("history_len {} does not match hist_len {}".format(history_len, hist_len))

        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.coord_channels = int(coord_channels)
        self.fno_width = int(fno_width)
        self.fno_layers = int(fno_layers)
        self.fno_modes = _as_tuple3(fno_modes, "fno_modes")
        self.fno_padding = int(fno_padding)
        self.fno_projection_dim = int(fno_projection_dim)
        self.use_coords = bool(use_coords)
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.fno_width <= 0:
            raise ValueError("fno_width must be > 0")
        if self.fno_layers <= 0:
            raise ValueError("fno_layers must be > 0")
        if self.fno_padding < 0:
            raise ValueError("fno_padding must be >= 0")
        if self.fno_projection_dim <= 0:
            raise ValueError("fno_projection_dim must be > 0")
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "fno3d output_mode must be one of ['regression', 'classification'], got {}".format(self.output_mode)
            )
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")

        self.classification_channels = self.risk_num_classes * self.risk_num_heads
        self.output_channels = self.in_channels if self.output_mode == "regression" else self.classification_channels
        total_in_features = self.history_len * self.in_channels + (self.coord_channels if self.use_coords else 0)

        self.fc0 = nn.Linear(total_in_features, self.fno_width)
        self.spectral_convs = nn.ModuleList(
            SpectralConv3D(self.fno_width, self.fno_width, self.fno_modes) for _ in range(self.fno_layers)
        )
        self.ws = nn.ModuleList(nn.Conv3d(self.fno_width, self.fno_width, kernel_size=1) for _ in range(self.fno_layers))
        self.fc1 = nn.Linear(self.fno_width, self.fno_projection_dim)
        self.fc2 = nn.Linear(self.fno_projection_dim, self.pred_len * self.output_channels)

    def _normalize_coords(
        self,
        coords: torch.Tensor | None,
        batch_size: int,
        spatial_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor | None:
        if coords is None:
            grid_y = torch.linspace(0.0, 1.0, spatial_shape[0], device=device)
            grid_x = torch.linspace(0.0, 1.0, spatial_shape[1], device=device)
            grid_z = torch.linspace(0.0, 1.0, spatial_shape[2], device=device)
            mesh_y, mesh_x, mesh_z = torch.meshgrid(grid_y, grid_x, grid_z, indexing="ij")
            coords = torch.stack([mesh_y, mesh_x, mesh_z], dim=0).unsqueeze(0)
        if coords.dim() == 4:
            coords = coords.unsqueeze(0)
        if coords.dim() != 5:
            raise ValueError("FNO3D expects coords with 4 or 5 dims, but received {}".format(coords.dim()))
        if coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1, -1, -1)
        if coords.size(0) != batch_size:
            raise ValueError("FNO3D expects coords batch size {}, but received {}".format(batch_size, coords.size(0)))
        if coords.size(1) != self.coord_channels:
            raise ValueError(
                "FNO3D expects coord_channels {}, but received {}".format(self.coord_channels, coords.size(1))
            )
        if tuple(coords.shape[-3:]) != spatial_shape:
            raise ValueError(
                "FNO3D expects coords spatial shape {}, but received {}".format(spatial_shape, tuple(coords.shape[-3:]))
            )
        return coords

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError("FNO3D expects x as [B, history_len, C, Y, X, Z], got {}".format(tuple(x.shape)))
        batch_size, history_len, channels, size_y, size_x, size_z = x.shape
        if history_len != self.history_len:
            raise ValueError("FNO3D expects history_len {}, got {}".format(self.history_len, history_len))
        if channels != self.in_channels:
            raise ValueError("FNO3D expects in_channels {}, got {}".format(self.in_channels, channels))

        x = x.reshape(batch_size, history_len * channels, size_y, size_x, size_z)
        if self.use_coords:
            coords = self._normalize_coords(
                coords,
                batch_size=batch_size,
                spatial_shape=(size_y, size_x, size_z),
                device=x.device,
            )
            x = torch.cat([x, coords], dim=1)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        if self.fno_padding > 0:
            x = F.pad(x, [0, self.fno_padding])

        for layer_index, (spectral_conv, pointwise_conv) in enumerate(zip(self.spectral_convs, self.ws)):
            x = spectral_conv(x) + pointwise_conv(x)
            if layer_index != self.fno_layers - 1:
                x = F.gelu(x)

        if self.fno_padding > 0:
            x = x[..., : -self.fno_padding]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x.view(batch_size, self.pred_len, self.output_channels, size_y, size_x, size_z)

    def get_aux_loss(self) -> torch.Tensor | None:
        return None
