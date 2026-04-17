from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _as_tuple3(value: object, field_name: str) -> tuple[int, int, int]:
    if value is None:
        raise ValueError("Expected {} to be a length-3 tuple/list, got None".format(field_name))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        parsed = tuple(int(v) for v in value)
        if any(v <= 0 for v in parsed):
            raise ValueError("Expected {} entries to be positive, got {!r}".format(field_name, value))
        return parsed
    raise ValueError("Expected {} to be a length-3 tuple/list, got {!r}".format(field_name, value))


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


@dataclass
class AFNO3DModelConfig:
    model_name: str = "afno3d"
    in_channels: int = 6
    coord_channels: int = 3
    patch_size: tuple[int, int, int] = (7, 7, 2)
    afno_embed_dim: int = 64
    afno_depth: int = 4
    afno_num_blocks: int = 8
    afno_hidden_size_factor: int = 1
    afno_mlp_ratio: float = 4.0
    afno_dropout: float = 0.0
    afno_drop_path_rate: float = 0.0
    afno_sparsity_threshold: float = 0.01
    afno_hard_thresholding_fraction: float = 1.0
    afno_double_skip: bool = True
    use_coords: bool = True
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> AFNO3DModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "afno3d"))
        if "patch_size" in values:
            values["patch_size"] = _as_tuple3(values["patch_size"], "patch_size")
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = int(out_features or in_features)
        hidden_features = int(hidden_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO3D(nn.Module):
    """3D extension of the official AFNO2D token mixer used by FourCastNet."""

    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
    ) -> None:
        super().__init__()
        if hidden_size % num_blocks != 0:
            raise ValueError("hidden_size {} should be divisible by num_blocks {}".format(hidden_size, num_blocks))
        self.hidden_size = int(hidden_size)
        self.sparsity_threshold = float(sparsity_threshold)
        self.num_blocks = int(num_blocks)
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = float(hard_thresholding_fraction)
        self.hidden_size_factor = int(hidden_size_factor)
        if self.hidden_size_factor <= 0:
            raise ValueError("hidden_size_factor must be > 0")
        if not (0.0 < self.hard_thresholding_fraction <= 1.0):
            raise ValueError("hard_thresholding_fraction must be in (0, 1]")
        self.scale = 0.02

        hidden_block_size = self.block_size * self.hidden_size_factor
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, hidden_block_size))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, hidden_block_size))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, hidden_block_size, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def _kept_slices(self, size_y: int, size_x: int, size_z: int) -> tuple[slice, slice, slice]:
        total_y_modes = size_y // 2 + 1
        total_x_modes = size_x // 2 + 1
        total_z_modes = size_z // 2 + 1
        kept_y_modes = max(1, int(total_y_modes * self.hard_thresholding_fraction))
        kept_x_modes = max(1, int(total_x_modes * self.hard_thresholding_fraction))
        kept_z_modes = max(1, int(total_z_modes * self.hard_thresholding_fraction))
        return (
            slice(total_y_modes - kept_y_modes, total_y_modes + kept_y_modes),
            slice(total_x_modes - kept_x_modes, total_x_modes + kept_x_modes),
            slice(0, kept_z_modes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = x
        original_dtype = x.dtype
        x = x.float()
        batch_size, size_y, size_x, size_z, hidden_size = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(batch_size, size_y, size_x, size_z // 2 + 1, self.num_blocks, self.block_size)

        hidden_block_size = self.block_size * self.hidden_size_factor
        o1_real = torch.zeros(
            batch_size,
            size_y,
            size_x,
            size_z // 2 + 1,
            self.num_blocks,
            hidden_block_size,
            device=x.device,
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros_like(x.real)
        o2_imag = torch.zeros_like(x.imag)
        kept_y, kept_x, kept_z = self._kept_slices(size_y, size_x, size_z)

        x_kept = x[:, kept_y, kept_x, kept_z]
        o1_real[:, kept_y, kept_x, kept_z] = F.relu(
            torch.einsum("...bi,bio->...bo", x_kept.real, self.w1[0])
            - torch.einsum("...bi,bio->...bo", x_kept.imag, self.w1[1])
            + self.b1[0]
        )
        o1_imag[:, kept_y, kept_x, kept_z] = F.relu(
            torch.einsum("...bi,bio->...bo", x_kept.imag, self.w1[0])
            + torch.einsum("...bi,bio->...bo", x_kept.real, self.w1[1])
            + self.b1[1]
        )

        o1_real_kept = o1_real[:, kept_y, kept_x, kept_z]
        o1_imag_kept = o1_imag[:, kept_y, kept_x, kept_z]
        o2_real[:, kept_y, kept_x, kept_z] = (
            torch.einsum("...bi,bio->...bo", o1_real_kept, self.w2[0])
            - torch.einsum("...bi,bio->...bo", o1_imag_kept, self.w2[1])
            + self.b2[0]
        )
        o2_imag[:, kept_y, kept_x, kept_z] = (
            torch.einsum("...bi,bio->...bo", o1_imag_kept, self.w2[0])
            + torch.einsum("...bi,bio->...bo", o1_real_kept, self.w2[1])
            + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(batch_size, size_y, size_x, size_z // 2 + 1, hidden_size)
        x = torch.fft.irfftn(x, s=(size_y, size_x, size_z), dim=(1, 2, 3), norm="ortho")
        return x.to(dtype=original_dtype) + bias


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        double_skip: bool = True,
        num_blocks: int = 8,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
        hidden_size_factor: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = AFNO3D(
            dim,
            num_blocks=num_blocks,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=hard_thresholding_fraction,
            hidden_size_factor=hidden_size_factor,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.double_skip = bool(double_skip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.filter(x)
        if self.double_skip:
            x = x + residual
            residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        return x + residual


class Model(nn.Module):
    """FourCastNet-style AFNO baseline adapted from 2D weather grids to 3D volumes."""

    def __init__(
        self,
        history_len: int | None = None,
        pred_len: int = 1,
        in_channels: int = 6,
        coord_channels: int = 3,
        patch_size: tuple[int, int, int] = (7, 7, 2),
        afno_embed_dim: int = 64,
        afno_depth: int = 4,
        afno_num_blocks: int = 8,
        afno_hidden_size_factor: int = 1,
        afno_hidden_factor: float | None = None,
        afno_mlp_ratio: float = 4.0,
        afno_dropout: float = 0.0,
        afno_drop_path_rate: float = 0.0,
        afno_sparsity_threshold: float = 0.01,
        afno_hard_thresholding_fraction: float = 1.0,
        afno_double_skip: bool = True,
        use_coords: bool = True,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
        grid_shape: tuple[int, int, int] | None = None,
        hist_len: int | None = None,
    ) -> None:
        super().__init__()
        if history_len is None:
            if hist_len is None:
                raise ValueError("afno3d requires history_len or hist_len")
            history_len = hist_len
        elif hist_len is not None and int(hist_len) != int(history_len):
            raise ValueError("history_len {} does not match hist_len {}".format(history_len, hist_len))
        if afno_hidden_factor is not None:
            afno_hidden_size_factor = int(round(float(afno_hidden_factor)))

        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.coord_channels = int(coord_channels)
        self.patch_size = _as_tuple3(patch_size, "patch_size")
        self.afno_embed_dim = int(afno_embed_dim)
        self.afno_depth = int(afno_depth)
        self.use_coords = bool(use_coords)
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        if self.afno_embed_dim <= 0:
            raise ValueError("afno_embed_dim must be > 0")
        if self.afno_depth <= 0:
            raise ValueError("afno_depth must be > 0")
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "afno3d output_mode must be one of ['regression', 'classification'], got {}".format(self.output_mode)
            )
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")

        self.classification_channels = self.risk_num_classes * self.risk_num_heads
        self.output_channels = self.in_channels if self.output_mode == "regression" else self.classification_channels
        self.patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        total_in_channels = self.history_len * self.in_channels + (self.coord_channels if self.use_coords else 0)

        self.patch_embed = nn.Conv3d(
            total_in_channels,
            self.afno_embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = None
        if grid_shape is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, *self._token_shape(_as_tuple3(grid_shape, "grid_shape")), self.afno_embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=afno_dropout)

        drop_path_rates = torch.linspace(0, float(afno_drop_path_rate), self.afno_depth).tolist()
        self.blocks = nn.ModuleList(
            Block(
                dim=self.afno_embed_dim,
                mlp_ratio=afno_mlp_ratio,
                drop=afno_dropout,
                drop_path=drop_path_rates[layer_index],
                double_skip=afno_double_skip,
                num_blocks=afno_num_blocks,
                sparsity_threshold=afno_sparsity_threshold,
                hard_thresholding_fraction=afno_hard_thresholding_fraction,
                hidden_size_factor=afno_hidden_size_factor,
            )
            for layer_index in range(self.afno_depth)
        )
        self.norm = nn.LayerNorm(self.afno_embed_dim)
        self.head = nn.Linear(self.afno_embed_dim, self.patch_volume * self.pred_len * self.output_channels)

    def _token_shape(self, spatial_shape: tuple[int, int, int]) -> tuple[int, int, int]:
        return tuple((int(size) + patch - 1) // patch for size, patch in zip(spatial_shape, self.patch_size))

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
            raise ValueError("AFNO3D expects coords with 4 or 5 dims, but received {}".format(coords.dim()))
        if coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1, -1, -1)
        if coords.size(0) != batch_size:
            raise ValueError("AFNO3D expects coords batch size {}, but received {}".format(batch_size, coords.size(0)))
        if coords.size(1) != self.coord_channels:
            raise ValueError(
                "AFNO3D expects coord_channels {}, but received {}".format(self.coord_channels, coords.size(1))
            )
        if tuple(coords.shape[-3:]) != spatial_shape:
            raise ValueError(
                "AFNO3D expects coords spatial shape {}, but received {}".format(
                    spatial_shape,
                    tuple(coords.shape[-3:]),
                )
            )
        return coords

    def _pad_to_patch_multiple(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        size_y, size_x, size_z = tuple(int(size) for size in x.shape[-3:])
        pad_y = (-size_y) % self.patch_size[0]
        pad_x = (-size_x) % self.patch_size[1]
        pad_z = (-size_z) % self.patch_size[2]
        if pad_y or pad_x or pad_z:
            x = F.pad(x, [0, pad_z, 0, pad_x, 0, pad_y])
        return x, (size_y, size_x, size_z)

    def _get_pos_embed(self, token_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor | None:
        if self.pos_embed is None:
            return None
        if tuple(int(size) for size in self.pos_embed.shape[1:4]) == token_shape:
            return self.pos_embed.to(device=device)
        pos = self.pos_embed.permute(0, 4, 1, 2, 3)
        pos = F.interpolate(pos, size=token_shape, mode="trilinear", align_corners=False)
        return pos.permute(0, 2, 3, 4, 1).contiguous().to(device=device)

    def _unpatchify(
        self,
        x: torch.Tensor,
        batch_size: int,
        token_shape: tuple[int, int, int],
        original_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        token_y, token_x, token_z = token_shape
        patch_y, patch_x, patch_z = self.patch_size
        x = x.view(
            batch_size,
            token_y,
            token_x,
            token_z,
            patch_y,
            patch_x,
            patch_z,
            self.pred_len * self.output_channels,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(
            batch_size,
            self.pred_len * self.output_channels,
            token_y * patch_y,
            token_x * patch_x,
            token_z * patch_z,
        )
        x = x[..., : original_shape[0], : original_shape[1], : original_shape[2]]
        return x.view(
            batch_size,
            self.pred_len,
            self.output_channels,
            original_shape[0],
            original_shape[1],
            original_shape[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        if x.ndim != 6:
            raise ValueError("AFNO3D expects x as [B, history_len, C, Y, X, Z], got {}".format(tuple(x.shape)))
        batch_size, history_len, channels, size_y, size_x, size_z = x.shape
        if history_len != self.history_len:
            raise ValueError("AFNO3D expects history_len {}, got {}".format(self.history_len, history_len))
        if channels != self.in_channels:
            raise ValueError("AFNO3D expects in_channels {}, got {}".format(self.in_channels, channels))

        coords = self._normalize_coords(coords, batch_size=batch_size, spatial_shape=(size_y, size_x, size_z))
        if self.use_coords and coords is None:
            raise ValueError("coords are required when use_coords=True")

        x = x.reshape(batch_size, history_len * channels, size_y, size_x, size_z)
        if self.use_coords and coords is not None:
            x = torch.cat([x, coords], dim=1)
        x, original_shape = self._pad_to_patch_multiple(x)
        x = self.patch_embed(x)
        token_shape = tuple(int(size) for size in x.shape[-3:])
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        pos_embed = self._get_pos_embed(token_shape, x.device)
        if pos_embed is not None:
            x = x + pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return self._unpatchify(x, batch_size=batch_size, token_shape=token_shape, original_shape=original_shape)

    def get_aux_loss(self) -> torch.Tensor | None:
        return None
