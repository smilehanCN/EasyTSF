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
class PatchSTGFlat3DModelConfig:
    model_name: str = "patchstg_flat3d"
    in_channels: int = 3
    out_channels: int | None = None
    coord_channels: int = 3
    patch_size_3d: tuple[int, int, int] = (4, 4, 4)
    embed_dim: int = 96
    depth: int = 3
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    use_coords: bool = True
    spatial_downsample_factor_3d: tuple[int, int, int] = (4, 4, 4)
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> PatchSTGFlat3DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "patchstg_flat3d"))
        if "patch_size_3d" in values:
            values["patch_size_3d"] = _as_tuple3(values["patch_size_3d"], "patch_size_3d")
        if "spatial_downsample_factor_3d" in values:
            values["spatial_downsample_factor_3d"] = _as_tuple3(
                values["spatial_downsample_factor_3d"],
                "spatial_downsample_factor_3d",
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(embed_dim, int(embed_dim * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.local_norm1 = nn.LayerNorm(embed_dim)
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.local_norm2 = nn.LayerNorm(embed_dim)
        self.local_mlp = FeedForward(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

        self.global_norm1 = nn.LayerNorm(embed_dim)
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.global_norm2 = nn.LayerNorm(embed_dim)
        self.global_mlp = FeedForward(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, patch_count, patch_volume, embed_dim = x.shape

        local = x.reshape(batch_size * patch_count, patch_volume, embed_dim)
        local_input = self.local_norm1(local)
        local_attn, _ = self.local_attn(local_input, local_input, local_input, need_weights=False)
        local = local + local_attn
        local = local + self.local_mlp(self.local_norm2(local))
        x = local.reshape(batch_size, patch_count, patch_volume, embed_dim)

        global_tokens = x.transpose(1, 2).reshape(batch_size * patch_volume, patch_count, embed_dim)
        global_input = self.global_norm1(global_tokens)
        global_attn, _ = self.global_attn(global_input, global_input, global_input, need_weights=False)
        global_tokens = global_tokens + global_attn
        global_tokens = global_tokens + self.global_mlp(self.global_norm2(global_tokens))
        return global_tokens.reshape(batch_size, patch_volume, patch_count, embed_dim).transpose(1, 2).contiguous()


class Model(nn.Module):
    def __init__(
        self,
        history_len: int,
        pred_len: int = 1,
        in_channels: int = 3,
        out_channels: int | None = None,
        coord_channels: int = 3,
        patch_size_3d: tuple[int, int, int] = (4, 4, 4),
        embed_dim: int = 96,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        use_coords: bool = True,
        spatial_downsample_factor_3d: tuple[int, int, int] = (4, 4, 4),
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")
        if any(factor <= 0 for factor in spatial_downsample_factor_3d):
            raise ValueError(
                "spatial_downsample_factor_3d must contain positive integers, "
                f"got {spatial_downsample_factor_3d}."
            )

        self.history_len = history_len
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.output_channels = self.in_channels if out_channels is None else int(out_channels)
        self.coord_channels = coord_channels
        self.patch_size_3d = patch_size_3d
        self.embed_dim = embed_dim
        self.use_coords = use_coords
        self.spatial_downsample_factor_3d = spatial_downsample_factor_3d
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "patchstg_flat3d output_mode must be one of ['regression', 'classification'], got {}".format(
                    self.output_mode
                )
            )
        if self.output_channels <= 0:
            raise ValueError("out_channels must be > 0")
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")
        self.classification_channels = self.risk_num_classes * self.risk_num_heads

        input_dim = history_len * in_channels + (coord_channels if use_coords else 0)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            [DualAttentionBlock(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.regression_output_proj = nn.Linear(embed_dim, pred_len * self.output_channels)
        self.classification_output_proj = nn.Linear(embed_dim, pred_len * self.classification_channels)
        self.spatial_downsample = nn.AvgPool3d(
            kernel_size=spatial_downsample_factor_3d,
            stride=spatial_downsample_factor_3d,
            ceil_mode=True,
        )

    @property
    def patch_volume(self) -> int:
        return self.patch_size_3d[0] * self.patch_size_3d[1] * self.patch_size_3d[2]

    def _padding_for_shape(self, spatial_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple(
            (patch - (size % patch)) % patch for size, patch in zip(spatial_shape, self.patch_size_3d, strict=True)
        )

    def _pad_spatial(self, tensor: torch.Tensor, padding: tuple[int, int, int]) -> torch.Tensor:
        pad_y, pad_x, pad_z = padding
        if pad_y == 0 and pad_x == 0 and pad_z == 0:
            return tensor
        return F.pad(tensor, (0, pad_z, 0, pad_x, 0, pad_y))

    def _normalize_coords(self, coords: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
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
        return coords

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

    def _downsample_sequence(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, steps, channels, _, _, _ = x.shape
        flattened = x.reshape(batch_size * steps, channels, *x.shape[-3:])
        downsampled = self._resample_spatial_tensor(flattened)
        return downsampled.reshape(batch_size, steps, channels, *downsampled.shape[-3:])

    def _downsample_coords(self, coords: torch.Tensor | None) -> torch.Tensor | None:
        if coords is None:
            return None
        return self._resample_spatial_tensor(coords)

    def _upsample_sequence(self, x: torch.Tensor, output_shape: tuple[int, int, int]) -> torch.Tensor:
        if x.shape[-3:] == output_shape:
            return x
        batch_size, steps, channels, _, _, _ = x.shape
        flattened = x.reshape(batch_size * steps, channels, *x.shape[-3:])
        upsampled = self._resample_spatial_tensor(flattened, output_shape=output_shape)
        return upsampled.reshape(batch_size, steps, channels, *output_shape)

    def _patchify_nodes(self, tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        batch_size, size_y, size_x, size_z, feature_dim = tensor.shape
        patch_y, patch_x, patch_z = self.patch_size_3d
        grid_y = size_y // patch_y
        grid_x = size_x // patch_x
        grid_z = size_z // patch_z
        patches = (
            tensor.view(batch_size, grid_y, patch_y, grid_x, patch_x, grid_z, patch_z, feature_dim)
            .permute(0, 1, 3, 5, 2, 4, 6, 7)
            .reshape(batch_size, grid_y * grid_x * grid_z, self.patch_volume, feature_dim)
        )
        return patches, (grid_y, grid_x, grid_z)

    def _unpatchify_nodes(
        self,
        patches: torch.Tensor,
        grid_shape: tuple[int, int, int],
        padded_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        batch_size, _, _, feature_dim = patches.shape
        grid_y, grid_x, grid_z = grid_shape
        patch_y, patch_x, patch_z = self.patch_size_3d
        size_y, size_x, size_z = padded_shape
        return (
            patches.view(batch_size, grid_y, grid_x, grid_z, patch_y, patch_x, patch_z, feature_dim)
            .permute(0, 1, 4, 2, 5, 3, 6, 7)
            .reshape(batch_size, size_y, size_x, size_z, feature_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, history_len, channels, size_y, size_x, size_z = x.shape
        if history_len != self.history_len:
            raise ValueError(f"Expected history_len={self.history_len}, got {history_len}.")
        if channels != self.in_channels:
            raise ValueError(f"Expected channels={self.in_channels}, got {channels}.")

        original_shape = (size_y, size_x, size_z)
        coords = self._normalize_coords(coords, batch_size)
        if self.use_coords and coords is None:
            raise ValueError("coords are required when use_coords=True.")

        x = self._downsample_sequence(x)
        coords = self._downsample_coords(coords)
        size_y, size_x, size_z = x.shape[-3:]
        padding = self._padding_for_shape((size_y, size_x, size_z))
        padded_x = self._pad_spatial(x, padding)
        padded_coords = None if coords is None else self._pad_spatial(coords, padding)

        padded_shape = padded_x.shape[-3:]
        node_features = padded_x.permute(0, 3, 4, 5, 1, 2).reshape(batch_size, *padded_shape, history_len * channels)
        if self.use_coords and padded_coords is not None:
            coord_features = padded_coords.permute(0, 2, 3, 4, 1)
            node_features = torch.cat([node_features, coord_features], dim=-1)

        patches, grid_shape = self._patchify_nodes(node_features)
        patches = self.input_proj(patches)
        for block in self.blocks:
            patches = block(patches)
        normalized_patches = self.output_norm(patches)
        if self.output_mode == "regression":
            patches = self.regression_output_proj(normalized_patches)
            output_channels = self.output_channels
        else:
            patches = self.classification_output_proj(normalized_patches)
            output_channels = self.classification_channels

        restored = self._unpatchify_nodes(patches, grid_shape=grid_shape, padded_shape=padded_shape)
        restored = restored[:, :size_y, :size_x, :size_z, :]
        forecast = restored.reshape(batch_size, size_y, size_x, size_z, self.pred_len, output_channels).permute(
            0, 4, 5, 1, 2, 3
        )
        return self._upsample_sequence(forecast, output_shape=original_shape)
