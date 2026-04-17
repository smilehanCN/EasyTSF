from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .unet3d import (
    DSConvBlock3D,
    Downsample3D,
    PatchEmbed3D,
    UpsampleAdd3D,
    _as_downsample_scales,
    _as_tuple3,
)


_WIND_COMPONENTS = ("u", "v", "w")


@dataclass
class UNet3DPatchCatModelConfig:
    model_name: str = "unet3d_patchcat"
    in_channels: int = 3
    coord_channels: int = 3
    base_channels: int = 16
    patch_size: tuple[int, int, int] = (4, 4, 2)
    downsample_scale: tuple[int, int, int] = (2, 2, 2)
    downsample_scales: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    expansion: int = 2
    input_embed_dim: int | None = None
    use_coords: bool = True
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> UNet3DPatchCatModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "unet3d_patchcat"))
        for key in ("patch_size", "downsample_scale", "kernel_size"):
            if key in values:
                values[key] = _as_tuple3(values[key], key)
        if "downsample_scales" in values:
            values["downsample_scales"] = _as_downsample_scales(
                values["downsample_scales"],
                values.get("downsample_scale", cls.downsample_scale),
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class PatchCatInputEmbedding(nn.Module):
    def __init__(self, history_len: int, embed_dim: int, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"unet3d_patchcat expects in_channels=3 for U/V/W, got {in_channels}.")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}.")

        self.history_len = history_len
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.early_len = history_len // 2
        self.early_dim = 0 if self.early_len == 0 else embed_dim // 4
        self.late_len = history_len - self.early_len
        self.late_dim = embed_dim - self.early_dim

        if self.early_len > 0 and embed_dim % 4 != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by 4 when history_len >= 2.")

        self.early_projs = self._build_projection_heads(self.early_len, self.early_dim)
        self.late_projs = self._build_projection_heads(self.late_len, self.late_dim)

    def _build_projection_heads(self, seq_len: int, out_dim: int) -> nn.ModuleDict:
        if seq_len == 0 or out_dim == 0:
            return nn.ModuleDict()
        return nn.ModuleDict({name: nn.Linear(seq_len, out_dim) for name in _WIND_COMPONENTS})

    def _project_group(
        self,
        group_series: torch.Tensor | None,
        projection_heads: nn.ModuleDict,
        out_dim: int,
    ) -> torch.Tensor | None:
        if group_series is None or out_dim == 0:
            return None
        component_embeddings = [projection_heads[name](group_series[..., channel_idx]) for channel_idx, name in enumerate(_WIND_COMPONENTS)]
        return torch.stack(component_embeddings, dim=0).mean(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, history_len, channels, size_y, size_x, size_z = x.shape
        if history_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {history_len}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected channels={self.in_channels}, got {channels}.")

        voxel_series = x.permute(0, 3, 4, 5, 1, 2)
        early_series = None if self.early_len == 0 else voxel_series[..., : self.early_len, :]
        late_series = voxel_series[..., self.early_len :, :]

        embeddings = []
        early_embedding = self._project_group(early_series, self.early_projs, self.early_dim)
        if early_embedding is not None:
            embeddings.append(early_embedding)
        late_embedding = self._project_group(late_series, self.late_projs, self.late_dim)
        if late_embedding is not None:
            embeddings.append(late_embedding)
        if not embeddings:
            raise RuntimeError("PatchCatInputEmbedding produced no embeddings.")
        voxel_embedding = torch.cat(embeddings, dim=-1)
        return voxel_embedding.permute(0, 4, 1, 2, 3).contiguous()


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
        downsample_scales: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        expansion: int = 2,
        input_embed_dim: int | None = None,
        use_coords: bool = True,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.coord_channels = coord_channels
        self.patch_size = patch_size
        self.use_coords = use_coords
        self.downsample_scales = _as_downsample_scales(downsample_scales, downsample_scale)
        self.input_embed_dim = base_channels if input_embed_dim is None else int(input_embed_dim)
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "unet3d_patchcat output_mode must be one of ['regression', 'classification'], got {}".format(
                    self.output_mode
                )
            )
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")
        self.classification_channels = self.risk_num_classes * self.risk_num_heads

        total_in_channels = self.input_embed_dim + (coord_channels if use_coords else 0)

        self.input_embed = PatchCatInputEmbedding(history_len=history_len, embed_dim=self.input_embed_dim, in_channels=in_channels)
        self.patch_embed = PatchEmbed3D(total_in_channels, base_channels, patch_size)
        self.enc1 = DSConvBlock3D(base_channels, kernel_size=kernel_size, expansion=expansion)
        self.down1 = Downsample3D(base_channels, base_channels * 2, self.downsample_scales[0])
        self.enc2 = DSConvBlock3D(base_channels * 2, kernel_size=kernel_size, expansion=expansion)
        self.down2 = Downsample3D(base_channels * 2, base_channels * 4, self.downsample_scales[1])
        self.enc3 = DSConvBlock3D(base_channels * 4, kernel_size=kernel_size, expansion=expansion)
        self.down3 = Downsample3D(base_channels * 4, base_channels * 8, self.downsample_scales[2])
        self.bottleneck = DSConvBlock3D(base_channels * 8, kernel_size=kernel_size, expansion=expansion)

        self.up3 = UpsampleAdd3D(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4,
            self.downsample_scales[2],
            kernel_size=kernel_size,
            expansion=expansion,
        )
        self.up2 = UpsampleAdd3D(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2,
            self.downsample_scales[1],
            kernel_size=kernel_size,
            expansion=expansion,
        )
        self.up1 = UpsampleAdd3D(
            base_channels * 2,
            base_channels,
            base_channels,
            self.downsample_scales[0],
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
            raise ValueError(f"Expected coords to have 4 or 5 dims, got {coords.dim()}.")
        if coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1, -1, -1)
        if coords.size(0) != batch_size:
            raise ValueError(f"Expected coords batch size {batch_size}, got {coords.size(0)}.")
        if coords.size(1) != self.coord_channels:
            raise ValueError(f"Expected coord channels={self.coord_channels}, got {coords.size(1)}.")
        if tuple(coords.shape[-3:]) != spatial_shape:
            raise ValueError(f"Expected coords spatial shape {spatial_shape}, got {tuple(coords.shape[-3:])}.")
        return coords

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        batch_size, _, _, size_y, size_x, size_z = x.shape

        coords = self._normalize_coords(coords, batch_size=batch_size, spatial_shape=(size_y, size_x, size_z))
        if self.use_coords and coords is None:
            raise ValueError("coords are required when use_coords=True.")

        x = self.input_embed(x)
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

        diff_y = size_y - x.size(-3)
        diff_x = size_x - x.size(-2)
        diff_z = size_z - x.size(-1)
        if diff_y or diff_x or diff_z:
            x = nn.functional.pad(
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
            return x.view(batch_size, self.pred_len, self.in_channels, size_y, size_x, size_z)

        x = self.classification_head(x)
        return x.view(batch_size, self.pred_len, self.classification_channels, size_y, size_x, size_z)
