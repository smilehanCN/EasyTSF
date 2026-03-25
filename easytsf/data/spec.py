from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSpec:
    layout_kind: str
    spatial_ndim: int
    spatial_shape: tuple[int, ...]
    channel_num: int | None
    has_graph: bool
    has_grid_mask: bool
    has_coord: bool
    time_feature_dim: int
