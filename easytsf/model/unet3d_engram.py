from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .unet3d import (
    DoubleConv3D,
    Down3D,
    OutConv3D,
    Up3D,
    _as_downsample_scales,
    _as_int_tuple,
    _as_offset_tuples,
    _as_tuple3,
    _center_crop_or_pad_3d,
)

"""
Engram3DBlock core design ideas
===============================

1) Deterministic sparse lookup for local patterns:
   - Convert dense voxel features into integer tokens via a learnable tokenizer.
   - Build local context tuples (2-gram / 3-gram / ...).
   - Hash contexts into fixed embedding tables with O(1) index access.

2) Multi-head hashing to reduce collision risk:
   - Each n-gram order uses multiple hash heads.
   - Each head uses an independent prime-sized table.
   - Retrieved embeddings are concatenated as memory evidence.

3) Context-aware gating:
   - Query from current voxel hidden state.
   - Key/Value from retrieved memory embedding.
   - Sigmoid gate suppresses noisy collisions and keeps useful memory.

4) Lightweight local refinement:
   - Depthwise Conv3D + residual path on memory branch.
   - Keeps additional compute small while adding local smoothing.

Potential discussion points (interactive)
-----------------------------------------
- Context directions: current {self, -x, -y, -z} vs richer directional set.
- Hash collision policy: static prime tables vs trainable/hash-adaptive options.
- Injection depth strategy: early-only vs layered early+mid placement.
- Whether to share one memory table across levels or keep per-level tables.
"""


def _make_norm3d(num_channels: int) -> nn.Module:
    num_groups = min(8, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    upper = int(math.sqrt(value)) + 1
    for factor in range(3, upper, 2):
        if value % factor == 0:
            return False
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = max(2, int(start) + 1)
    if candidate % 2 == 0 and candidate != 2:
        candidate += 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 2
    return candidate


class MultiHeadHashEmbedding(nn.Module):
    def __init__(self, vocab_sizes: list[int], embedding_dim: int) -> None:
        super().__init__()
        if not vocab_sizes:
            raise ValueError("vocab_sizes must be non-empty")
        if any(size <= 0 for size in vocab_sizes):
            raise ValueError("all vocab_sizes must be > 0")
        self.num_heads = len(vocab_sizes)
        self.embedding_dim = int(embedding_dim)

        offsets = [0]
        for size in vocab_sizes[:-1]:
            offsets.append(offsets[-1] + int(size))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.embedding = nn.Embedding(sum(int(size) for size in vocab_sizes), self.embedding_dim)

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        shifted_ids = hash_ids + self.offsets
        return self.embedding(shifted_ids)


class LearnableVoxelTokenizer3D(nn.Module):
    """
    Learnable voxel tokenizer via vector quantization.

    - Encoder maps voxel feature -> latent.
    - Codebook nearest-neighbor lookup yields discrete token IDs.
    - VQ losses train encoder + codebook.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        num_codes: int = 4096,
        embed_dim: int = 16,
        commitment_weight: float = 0.25,
        codebook_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if num_codes <= 1:
            raise ValueError(f"num_codes must be > 1, got {num_codes}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if commitment_weight < 0.0:
            raise ValueError(f"commitment_weight must be >= 0, got {commitment_weight}")
        if codebook_weight < 0.0:
            raise ValueError(f"codebook_weight must be >= 0, got {codebook_weight}")

        self.num_codes = int(num_codes)
        self.embed_dim = int(embed_dim)
        self.commitment_weight = float(commitment_weight)
        self.codebook_weight = float(codebook_weight)

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, self.embed_dim, kernel_size=1, bias=False),
            _make_norm3d(self.embed_dim),
            nn.SiLU(),
            nn.Conv3d(self.embed_dim, self.embed_dim, kernel_size=1, bias=False),
        )
        self.codebook = nn.Embedding(self.num_codes, self.embed_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)
        self._latest_aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        batch_size, embed_dim, size_y, size_x, size_z = latent.shape

        flat_latent = latent.permute(0, 2, 3, 4, 1).reshape(-1, embed_dim)
        flat_latent_fp32 = flat_latent.float()
        codebook_fp32 = self.codebook.weight.float()

        latent_sq = flat_latent_fp32.square().sum(dim=1, keepdim=True)
        codebook_sq = codebook_fp32.square().sum(dim=1, keepdim=True).t()
        distances = latent_sq + codebook_sq - 2.0 * torch.matmul(flat_latent_fp32, codebook_fp32.t())
        token_ids = torch.argmin(distances, dim=1).long()

        quantized_flat = F.embedding(token_ids, self.codebook.weight)
        quantized = quantized_flat.view(batch_size, size_y, size_x, size_z, embed_dim).permute(0, 4, 1, 2, 3).contiguous()

        commitment_loss = F.mse_loss(latent, quantized.detach(), reduction="mean")
        codebook_loss = F.mse_loss(quantized, latent.detach(), reduction="mean")
        self._latest_aux_loss = self.commitment_weight * commitment_loss + self.codebook_weight * codebook_loss

        return token_ids.view(batch_size, size_y, size_x, size_z)

    def get_aux_loss(self) -> torch.Tensor | None:
        return self._latest_aux_loss


def _finite_difference_3d(x: torch.Tensor, dim: int) -> torch.Tensor:
    if dim not in {2, 3, 4}:
        raise ValueError(f"expected dim in {{2, 3, 4}}, got {dim}")
    out = x.new_zeros(x.shape)
    axis_length = int(x.size(dim))
    if axis_length <= 1:
        return out

    lhs_index = [slice(None)] * x.ndim
    rhs_index = [slice(None)] * x.ndim
    lhs_index[dim] = slice(1, None)
    rhs_index[dim] = slice(0, -1)
    delta = x[tuple(lhs_index)] - x[tuple(rhs_index)]

    fill_index = [slice(None)] * x.ndim
    fill_index[dim] = slice(0, -1)
    out[tuple(fill_index)] = delta

    last_index = [slice(None)] * x.ndim
    last_index[dim] = -1
    out[tuple(last_index)] = delta[tuple(last_index)]
    return out


class StructureStem3D(nn.Module):
    """Encode local 3D shear structure from latent features and finite differences."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        hidden_channels = int(in_channels if hidden_channels is None else hidden_channels)
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden_channels}")

        self.in_channels = int(in_channels)
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.input_proj = nn.Conv3d(self.in_channels * 4, self.hidden_channels, kernel_size=1, bias=False)
        self.input_norm = _make_norm3d(self.hidden_channels)
        self.depthwise = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=self.hidden_channels,
            bias=False,
        )
        self.depthwise_norm = _make_norm3d(self.hidden_channels)
        self.output_proj = nn.Conv3d(self.hidden_channels, self.in_channels, kernel_size=1, bias=False)
        self.output_norm = _make_norm3d(self.in_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = _finite_difference_3d(x, dim=3)
        dy = _finite_difference_3d(x, dim=2)
        dz = _finite_difference_3d(x, dim=4)
        features = torch.cat([x, dx, dy, dz], dim=1)
        hidden = self.act(self.input_norm(self.input_proj(features)))
        hidden = self.act(self.depthwise_norm(self.depthwise(hidden)))
        structure = self.output_proj(hidden)
        return self.act(self.output_norm(structure + x))


class Engram3DBlock(nn.Module):
    """Hash-based conditional memory block for 3D feature maps."""

    def __init__(
        self,
        in_channels: int,
        *,
        max_ngram_size: int = 3,
        num_heads: int = 4,
        head_dim: int = 8,
        vocab_sizes: tuple[int, ...] | None = None,
        context_offsets: tuple[tuple[int, int, int], ...] | None = None,
        hash_dim: int = 8,
        tokenizer_num_codes: int = 4096,
        tokenizer_embed_dim: int | None = None,
        tokenizer_commitment_weight: float = 0.25,
        tokenizer_codebook_weight: float = 1.0,
        use_context_rotation: bool = True,
        use_gating: bool = True,
        use_short_conv: bool = True,
        token_scale: float = 32.0,
        conv_kernel_size: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if max_ngram_size < 2:
            raise ValueError(f"max_ngram_size must be >= 2, got {max_ngram_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if hash_dim <= 0:
            raise ValueError(f"hash_dim must be > 0, got {hash_dim}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be a positive odd integer, got {conv_kernel_size}")

        self.in_channels = int(in_channels)
        self.max_ngram_size = int(max_ngram_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.hash_dim = int(hash_dim)
        self.num_hash_tables = (self.max_ngram_size - 1) * self.num_heads
        self.token_scale = float(token_scale)
        self.pad_token_id = 0
        self.use_context_rotation = bool(use_context_rotation)
        self.use_gating = bool(use_gating)
        self.use_short_conv = bool(use_short_conv)
        if context_offsets is None:
            context_offsets = (
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, -1),
                (-1, -1, 0),
                (-1, 0, -1),
                (0, -1, -1),
                (-1, -1, -1),
            )
        if len(context_offsets) == 0:
            raise ValueError("context_offsets must be non-empty")
        self.context_offsets = tuple((int(dy), int(dx), int(dz)) for dy, dx, dz in context_offsets)

        if vocab_sizes is None:
            vocab_sizes = tuple(2048 + 1024 * idx for idx in range(self.max_ngram_size - 1))
        if len(vocab_sizes) != self.max_ngram_size - 1:
            raise ValueError(
                "vocab_sizes length {} does not match max_ngram_size-1 ({})".format(
                    len(vocab_sizes),
                    self.max_ngram_size - 1,
                )
            )
        if any(size <= 0 for size in vocab_sizes):
            raise ValueError("all vocab_sizes must be > 0")

        seen_primes: set[int] = set()
        flat_vocab_sizes: list[int] = []
        for order_index, base_size in enumerate(vocab_sizes):
            start = int(base_size) - 1
            order_primes = []
            for _ in range(self.num_heads):
                prime = _next_prime(start, seen_primes)
                seen_primes.add(prime)
                order_primes.append(prime)
                start = prime
            self.register_buffer(
                f"hash_mods_{order_index}",
                torch.tensor(order_primes, dtype=torch.long),
                persistent=False,
            )
            flat_vocab_sizes.extend(order_primes)

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        for order_index in range(self.max_ngram_size - 1):
            order_size = order_index + 2
            multipliers = torch.randint(
                low=1,
                high=(1 << 30) - 1,
                size=(order_size,),
                generator=generator,
                dtype=torch.long,
            )
            multipliers = multipliers | 1
            self.register_buffer(f"hash_multipliers_{order_index}", multipliers, persistent=False)

        tokenizer_embed_dim = self.hash_dim if tokenizer_embed_dim is None else int(tokenizer_embed_dim)
        self.tokenizer = LearnableVoxelTokenizer3D(
            in_channels=self.in_channels,
            num_codes=tokenizer_num_codes,
            embed_dim=tokenizer_embed_dim,
            commitment_weight=tokenizer_commitment_weight,
            codebook_weight=tokenizer_codebook_weight,
        )
        self.hash_embedding = MultiHeadHashEmbedding(flat_vocab_sizes, self.head_dim)
        self._latest_aux_loss: torch.Tensor | None = None

        engram_hidden_size = self.num_hash_tables * self.head_dim
        self.value_proj = nn.Linear(engram_hidden_size, self.in_channels)
        self.key_proj = nn.Linear(engram_hidden_size, self.in_channels)
        self.query_norm = nn.LayerNorm(self.in_channels)
        self.key_norm = nn.LayerNorm(self.in_channels)

        self.memory_norm = _make_norm3d(self.in_channels)
        self.short_conv = nn.Conv3d(
            self.in_channels,
            self.in_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=self.in_channels,
            bias=False,
        )
        nn.init.zeros_(self.short_conv.weight)
        self.act = nn.SiLU()

    def _shift_with_pad(self, tokens: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
        shift = tuple(-value for value in offset)
        shifted = torch.roll(tokens, shifts=shift, dims=(1, 2, 3))
        for dim, amount in zip((1, 2, 3), shift):
            if amount == 0:
                continue
            index = [slice(None)] * shifted.dim()
            if amount > 0:
                index[dim] = slice(0, amount)
            else:
                index[dim] = slice(amount, None)
            shifted[tuple(index)] = self.pad_token_id
        return shifted

    def _build_context_tokens(self, token_volume: torch.Tensor) -> list[torch.Tensor]:
        contexts = [token_volume]
        for offset in self.context_offsets:
            contexts.append(self._shift_with_pad(token_volume, offset))
        return contexts

    def _hash_tokens(self, token_volume: torch.Tensor) -> torch.Tensor:
        contexts = self._build_context_tokens(token_volume)
        num_context_dirs = len(contexts) - 1
        all_hashes: list[torch.Tensor] = []
        for order_index in range(self.max_ngram_size - 1):
            order_size = order_index + 2
            multipliers = getattr(self, f"hash_multipliers_{order_index}")
            mods = getattr(self, f"hash_mods_{order_index}")
            for head_index in range(self.num_heads):
                selected_contexts = [contexts[0]]
                for token_index in range(1, order_size):
                    if self.use_context_rotation:
                        context_pick = (head_index + token_index - 1) % num_context_dirs
                    else:
                        context_pick = (token_index - 1) % num_context_dirs
                    selected_contexts.append(contexts[context_pick + 1])
                mixed = selected_contexts[0] * multipliers[0]
                for token_index in range(1, order_size):
                    mixed = torch.bitwise_xor(mixed, selected_contexts[token_index] * multipliers[token_index])
                hash_id = torch.remainder(mixed + (head_index + 1) * 1315423911, mods[head_index])
                all_hashes.append(hash_id)
        return torch.stack(all_hashes, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, size_y, size_x, size_z = x.shape
        flat_features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)

        token_volume = self.tokenizer(x)
        self._latest_aux_loss = self.tokenizer.get_aux_loss()
        hash_ids = self._hash_tokens(token_volume).reshape(-1, self.num_hash_tables)

        embedding = self.hash_embedding(hash_ids).flatten(start_dim=-2)
        key = self.key_norm(self.key_proj(embedding))
        query = self.query_norm(flat_features)
        if self.use_gating:
            gate = torch.sigmoid((query * key).sum(dim=-1, keepdim=True) / math.sqrt(channels))
        else:
            gate = torch.ones((flat_features.size(0), 1), dtype=flat_features.dtype, device=flat_features.device)
        gated_value = gate * self.value_proj(embedding)

        memory = gated_value.view(batch_size, size_y, size_x, size_z, channels).permute(0, 4, 1, 2, 3).contiguous()
        if self.use_short_conv:
            memory = memory + self.short_conv(self.act(self.memory_norm(memory)))
        return x + memory

    def get_aux_loss(self) -> torch.Tensor | None:
        return self._latest_aux_loss


class WindEngram3DBlock(nn.Module):
    """Structure-aware residual memory for 3D wind shear forecasting."""

    def __init__(
        self,
        in_channels: int,
        *,
        max_ngram_size: int = 3,
        num_heads: int = 4,
        head_dim: int = 8,
        vocab_sizes: tuple[int, ...] | None = None,
        context_offsets: tuple[tuple[int, int, int], ...] | None = None,
        flow_tokenizer_num_codes: int = 4096,
        structure_tokenizer_num_codes: int = 2048,
        tokenizer_embed_dim: int | None = None,
        tokenizer_commitment_weight: float = 0.25,
        tokenizer_codebook_weight: float = 1.0,
        use_context_rotation: bool = True,
        use_gating: bool = True,
        use_short_conv: bool = True,
        conv_kernel_size: int = 3,
        structure_kernel_size: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if max_ngram_size < 2:
            raise ValueError(f"max_ngram_size must be >= 2, got {max_ngram_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be a positive odd integer, got {conv_kernel_size}")

        self.in_channels = int(in_channels)
        self.max_ngram_size = int(max_ngram_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_hash_tables = (self.max_ngram_size - 1) * self.num_heads
        self.pad_token_id = 0
        self.use_context_rotation = bool(use_context_rotation)
        self.use_gating = bool(use_gating)
        self.use_short_conv = bool(use_short_conv)

        if context_offsets is None:
            context_offsets = (
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, -1),
                (-1, -1, 0),
                (-1, 0, -1),
                (0, -1, -1),
                (-1, -1, -1),
            )
        if len(context_offsets) == 0:
            raise ValueError("context_offsets must be non-empty")
        self.context_offsets = tuple((int(dy), int(dx), int(dz)) for dy, dx, dz in context_offsets)

        if vocab_sizes is None:
            vocab_sizes = tuple(2048 + 1024 * idx for idx in range(self.max_ngram_size - 1))
        if len(vocab_sizes) != self.max_ngram_size - 1:
            raise ValueError(
                "vocab_sizes length {} does not match max_ngram_size-1 ({})".format(
                    len(vocab_sizes),
                    self.max_ngram_size - 1,
                )
            )
        if any(size <= 0 for size in vocab_sizes):
            raise ValueError("all vocab_sizes must be > 0")

        seen_primes: set[int] = set()
        flat_vocab_sizes: list[int] = []
        for order_index, base_size in enumerate(vocab_sizes):
            start = int(base_size) - 1
            order_primes = []
            for _ in range(self.num_heads):
                prime = _next_prime(start, seen_primes)
                seen_primes.add(prime)
                order_primes.append(prime)
                start = prime
            self.register_buffer(
                f"wind_hash_mods_{order_index}",
                torch.tensor(order_primes, dtype=torch.long),
                persistent=False,
            )
            flat_vocab_sizes.extend(order_primes)

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        for order_index in range(self.max_ngram_size - 1):
            order_size = order_index + 2
            multipliers = torch.randint(
                low=1,
                high=(1 << 30) - 1,
                size=(order_size,),
                generator=generator,
                dtype=torch.long,
            )
            multipliers = multipliers | 1
            self.register_buffer(f"wind_hash_multipliers_{order_index}", multipliers, persistent=False)

        token_embed_dim = self.head_dim if tokenizer_embed_dim is None else int(tokenizer_embed_dim)
        self.structure_stem = StructureStem3D(
            in_channels=self.in_channels,
            hidden_channels=self.in_channels,
            kernel_size=structure_kernel_size,
        )
        self.flow_tokenizer = LearnableVoxelTokenizer3D(
            in_channels=self.in_channels,
            num_codes=flow_tokenizer_num_codes,
            embed_dim=token_embed_dim,
            commitment_weight=tokenizer_commitment_weight,
            codebook_weight=tokenizer_codebook_weight,
        )
        self.structure_tokenizer = LearnableVoxelTokenizer3D(
            in_channels=self.in_channels,
            num_codes=structure_tokenizer_num_codes,
            embed_dim=token_embed_dim,
            commitment_weight=tokenizer_commitment_weight,
            codebook_weight=tokenizer_codebook_weight,
        )
        self.hash_embedding = MultiHeadHashEmbedding(flat_vocab_sizes, self.head_dim)
        self._latest_aux_loss: torch.Tensor | None = None

        hidden_size = self.num_hash_tables * self.head_dim
        self.value_proj = nn.Linear(hidden_size, self.in_channels)
        self.key_proj = nn.Linear(hidden_size, self.in_channels)
        self.query_norm = nn.LayerNorm(self.in_channels)
        self.key_norm = nn.LayerNorm(self.in_channels)

        self.memory_norm = _make_norm3d(self.in_channels)
        self.short_conv = nn.Conv3d(
            self.in_channels,
            self.in_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=self.in_channels,
            bias=False,
        )
        nn.init.zeros_(self.short_conv.weight)
        self.act = nn.SiLU()

    def _shift_with_pad(self, tokens: torch.Tensor, offset: tuple[int, int, int]) -> torch.Tensor:
        shift = tuple(-value for value in offset)
        shifted = torch.roll(tokens, shifts=shift, dims=(1, 2, 3))
        for dim, amount in zip((1, 2, 3), shift):
            if amount == 0:
                continue
            index = [slice(None)] * shifted.dim()
            if amount > 0:
                index[dim] = slice(0, amount)
            else:
                index[dim] = slice(amount, None)
            shifted[tuple(index)] = self.pad_token_id
        return shifted

    def _build_structure_contexts(self, structure_tokens: torch.Tensor) -> list[torch.Tensor]:
        return [self._shift_with_pad(structure_tokens, offset) for offset in self.context_offsets]

    def _hash_tokens(self, flow_tokens: torch.Tensor, structure_tokens: torch.Tensor) -> torch.Tensor:
        contexts = self._build_structure_contexts(structure_tokens)
        num_context_dirs = len(contexts)
        all_hashes: list[torch.Tensor] = []
        for order_index in range(self.max_ngram_size - 1):
            order_size = order_index + 2
            multipliers = getattr(self, f"wind_hash_multipliers_{order_index}")
            mods = getattr(self, f"wind_hash_mods_{order_index}")
            for head_index in range(self.num_heads):
                selected_tokens = [flow_tokens, structure_tokens]
                extra_context_count = order_size - 2
                for token_index in range(extra_context_count):
                    if self.use_context_rotation:
                        context_pick = (head_index + token_index) % num_context_dirs
                    else:
                        context_pick = token_index % num_context_dirs
                    selected_tokens.append(contexts[context_pick])

                mixed = selected_tokens[0] * multipliers[0]
                for token_index in range(1, len(selected_tokens)):
                    mixed = torch.bitwise_xor(mixed, selected_tokens[token_index] * multipliers[token_index])
                hash_id = torch.remainder(mixed + (head_index + 1) * 1315423911, mods[head_index])
                all_hashes.append(hash_id)
        return torch.stack(all_hashes, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, size_y, size_x, size_z = x.shape
        flat_features = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)

        structure_features = self.structure_stem(x)
        flow_tokens = self.flow_tokenizer(x)
        structure_tokens = self.structure_tokenizer(structure_features)

        aux_losses = []
        flow_aux = self.flow_tokenizer.get_aux_loss()
        if flow_aux is not None:
            aux_losses.append(flow_aux)
        structure_aux = self.structure_tokenizer.get_aux_loss()
        if structure_aux is not None:
            aux_losses.append(structure_aux)
        self._latest_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        hash_ids = self._hash_tokens(flow_tokens, structure_tokens).reshape(-1, self.num_hash_tables)
        embedding = self.hash_embedding(hash_ids).flatten(start_dim=-2)
        key = self.key_norm(self.key_proj(embedding))
        query = self.query_norm(flat_features)
        if self.use_gating:
            gate = torch.sigmoid((query * key).sum(dim=-1, keepdim=True) / math.sqrt(channels))
        else:
            gate = torch.ones((flat_features.size(0), 1), dtype=flat_features.dtype, device=flat_features.device)
        residual = gate * self.value_proj(embedding)

        memory = residual.view(batch_size, size_y, size_x, size_z, channels).permute(0, 4, 1, 2, 3).contiguous()
        if self.use_short_conv:
            memory = memory + self.short_conv(self.act(self.memory_norm(memory)))
        return x + memory

    def get_aux_loss(self) -> torch.Tensor | None:
        return self._latest_aux_loss


@dataclass
class UNet3DEngramModelConfig:
    model_name: str = "unet3d_engram"
    in_channels: int = 6
    coord_channels: int = 3
    base_channels: int = 16
    patch_size: tuple[int, int, int] = (1, 1, 1)
    downsample_scale: tuple[int, int, int] = (2, 2, 2)
    downsample_scales: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    use_coords: bool = True
    output_mode: str = "regression"
    risk_num_classes: int = 3
    risk_num_heads: int = 4
    use_engram: bool = False
    engram_layer_ids: tuple[int, ...] = (3, 4)
    engram_max_ngram_size: int = 3
    engram_num_heads: int = 4
    engram_head_dim: int = 8
    engram_vocab_sizes: tuple[int, ...] | None = None
    engram_context_offsets: tuple[tuple[int, int, int], ...] | None = None
    engram_hash_dim: int = 8
    engram_tokenizer_num_codes: int = 4096
    engram_tokenizer_embed_dim: int | None = None
    engram_tokenizer_commitment_weight: float = 0.25
    engram_tokenizer_codebook_weight: float = 1.0
    engram_use_context_rotation: bool = True
    engram_use_gating: bool = True
    engram_use_short_conv: bool = True
    engram_token_scale: float = 32.0
    engram_conv_kernel_size: int = 3
    engram_seed: int = 0
    use_windengram: bool = False
    windengram_layer_ids: tuple[int, ...] = (3, 4)
    windengram_max_ngram_size: int = 3
    windengram_num_heads: int = 4
    windengram_head_dim: int = 8
    windengram_vocab_sizes: tuple[int, ...] | None = None
    windengram_context_offsets: tuple[tuple[int, int, int], ...] | None = None
    windengram_flow_tokenizer_num_codes: int = 4096
    windengram_structure_tokenizer_num_codes: int = 2048
    windengram_tokenizer_embed_dim: int | None = None
    windengram_tokenizer_commitment_weight: float = 0.25
    windengram_tokenizer_codebook_weight: float = 1.0
    windengram_use_context_rotation: bool = True
    windengram_use_gating: bool = True
    windengram_use_short_conv: bool = True
    windengram_conv_kernel_size: int = 3
    windengram_structure_kernel_size: int = 3
    windengram_seed: int = 0

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> UNet3DEngramModelConfig:
        values = dict(raw)
        values["model_name"] = str(values.get("model_name", "unet3d_engram"))
        for key in ("patch_size", "downsample_scale", "kernel_size"):
            if key in values:
                values[key] = _as_tuple3(values[key], key)
        if "downsample_scales" in values:
            values["downsample_scales"] = _as_downsample_scales(
                values["downsample_scales"],
                values.get("downsample_scale", cls.downsample_scale),
            )
        if "engram_layer_ids" in values:
            values["engram_layer_ids"] = _as_int_tuple(values["engram_layer_ids"], "engram_layer_ids", min_len=1)
        if "engram_vocab_sizes" in values and values["engram_vocab_sizes"] is not None:
            values["engram_vocab_sizes"] = _as_int_tuple(values["engram_vocab_sizes"], "engram_vocab_sizes", min_len=1)
        if "engram_context_offsets" in values and values["engram_context_offsets"] is not None:
            values["engram_context_offsets"] = _as_offset_tuples(values["engram_context_offsets"], "engram_context_offsets")
        if "windengram_layer_ids" in values:
            values["windengram_layer_ids"] = _as_int_tuple(values["windengram_layer_ids"], "windengram_layer_ids", min_len=1)
        if "windengram_vocab_sizes" in values and values["windengram_vocab_sizes"] is not None:
            values["windengram_vocab_sizes"] = _as_int_tuple(
                values["windengram_vocab_sizes"],
                "windengram_vocab_sizes",
                min_len=1,
            )
        if "windengram_context_offsets" in values and values["windengram_context_offsets"] is not None:
            values["windengram_context_offsets"] = _as_offset_tuples(
                values["windengram_context_offsets"],
                "windengram_context_offsets",
            )
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


class Model(nn.Module):
    """3D U-Net backbone with optional Engram/WindEngram memory blocks."""

    def __init__(
        self,
        history_len: int | None = None,
        pred_len: int = 1,
        in_channels: int = 6,
        coord_channels: int = 3,
        base_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 1, 1),
        downsample_scale: tuple[int, int, int] = (2, 2, 2),
        downsample_scales: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None,
        kernel_size: tuple[int, int, int] = (3, 3, 3),
        expansion: int = 2,
        use_coords: bool = True,
        output_mode: str = "regression",
        risk_num_classes: int = 3,
        risk_num_heads: int = 4,
        use_engram: bool = False,
        engram_layer_ids: tuple[int, ...] = (3, 4),
        engram_max_ngram_size: int = 3,
        engram_num_heads: int = 4,
        engram_head_dim: int = 8,
        engram_vocab_sizes: tuple[int, ...] | None = None,
        engram_context_offsets: tuple[tuple[int, int, int], ...] | None = None,
        engram_hash_dim: int = 8,
        engram_tokenizer_num_codes: int = 4096,
        engram_tokenizer_embed_dim: int | None = None,
        engram_tokenizer_commitment_weight: float = 0.25,
        engram_tokenizer_codebook_weight: float = 1.0,
        engram_use_context_rotation: bool = True,
        engram_use_gating: bool = True,
        engram_use_short_conv: bool = True,
        engram_token_scale: float = 32.0,
        engram_conv_kernel_size: int = 3,
        engram_seed: int = 0,
        use_windengram: bool = False,
        windengram_layer_ids: tuple[int, ...] = (3, 4),
        windengram_max_ngram_size: int = 3,
        windengram_num_heads: int = 4,
        windengram_head_dim: int = 8,
        windengram_vocab_sizes: tuple[int, ...] | None = None,
        windengram_context_offsets: tuple[tuple[int, int, int], ...] | None = None,
        windengram_flow_tokenizer_num_codes: int = 4096,
        windengram_structure_tokenizer_num_codes: int = 2048,
        windengram_tokenizer_embed_dim: int | None = None,
        windengram_tokenizer_commitment_weight: float = 0.25,
        windengram_tokenizer_codebook_weight: float = 1.0,
        windengram_use_context_rotation: bool = True,
        windengram_use_gating: bool = True,
        windengram_use_short_conv: bool = True,
        windengram_conv_kernel_size: int = 3,
        windengram_structure_kernel_size: int = 3,
        windengram_seed: int = 0,
        hist_len: int | None = None,
    ) -> None:
        super().__init__()
        if history_len is None:
            if hist_len is None:
                raise ValueError("unet3d_engram requires history_len or hist_len")
            history_len = hist_len
        elif hist_len is not None and int(hist_len) != int(history_len):
            raise ValueError("history_len {} does not match hist_len {}".format(history_len, hist_len))

        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.coord_channels = int(coord_channels)
        self.patch_size = _as_tuple3(patch_size, "patch_size")
        self.downsample_scales = _as_downsample_scales(downsample_scales, _as_tuple3(downsample_scale, "downsample_scale"))
        self.kernel_size = _as_tuple3(kernel_size, "kernel_size")
        self.use_coords = bool(use_coords)
        self.output_mode = str(output_mode)
        self.risk_num_classes = int(risk_num_classes)
        self.risk_num_heads = int(risk_num_heads)
        self.use_engram = bool(use_engram)
        self.use_windengram = bool(use_windengram)
        if self.use_engram and self.use_windengram:
            raise ValueError("use_engram and use_windengram are mutually exclusive")
        if self.output_mode not in {"regression", "classification"}:
            raise ValueError(
                "unet3d_engram output_mode must be one of ['regression', 'classification'], got {}".format(
                    self.output_mode
                )
            )
        if self.risk_num_classes <= 0 or self.risk_num_heads <= 0:
            raise ValueError("risk_num_classes and risk_num_heads must be > 0")

        active_layer_ids = windengram_layer_ids if self.use_windengram else engram_layer_ids
        self.engram_layer_ids = tuple(sorted({int(layer_id) for layer_id in active_layer_ids}))
        if self.use_engram or self.use_windengram:
            invalid_layer_ids = [layer_id for layer_id in self.engram_layer_ids if layer_id not in {1, 2, 3, 4}]
            if invalid_layer_ids:
                raise ValueError("engram_layer_ids must be chosen from [1,2,3,4], got invalid {}".format(invalid_layer_ids))
        self.classification_channels = self.risk_num_classes * self.risk_num_heads

        total_in_channels = self.history_len * self.in_channels
        if self.use_coords:
            total_in_channels += self.coord_channels

        self.inc = DoubleConv3D(total_in_channels, base_channels, kernel_size=self.kernel_size)
        self.down1 = Down3D(base_channels, base_channels * 2, self.downsample_scales[0], kernel_size=self.kernel_size)
        self.down2 = Down3D(base_channels * 2, base_channels * 4, self.downsample_scales[1], kernel_size=self.kernel_size)
        self.down3 = Down3D(base_channels * 4, base_channels * 8, self.downsample_scales[2], kernel_size=self.kernel_size)

        self.engram_blocks = nn.ModuleDict()
        if self.use_engram or self.use_windengram:
            layer_channels = {
                1: base_channels,
                2: base_channels * 2,
                3: base_channels * 4,
                4: base_channels * 8,
            }
            for layer_id in self.engram_layer_ids:
                if self.use_windengram:
                    self.engram_blocks[str(layer_id)] = WindEngram3DBlock(
                        in_channels=layer_channels[layer_id],
                        max_ngram_size=windengram_max_ngram_size,
                        num_heads=windengram_num_heads,
                        head_dim=windengram_head_dim,
                        vocab_sizes=windengram_vocab_sizes,
                        context_offsets=windengram_context_offsets,
                        flow_tokenizer_num_codes=windengram_flow_tokenizer_num_codes,
                        structure_tokenizer_num_codes=windengram_structure_tokenizer_num_codes,
                        tokenizer_embed_dim=windengram_tokenizer_embed_dim,
                        tokenizer_commitment_weight=windengram_tokenizer_commitment_weight,
                        tokenizer_codebook_weight=windengram_tokenizer_codebook_weight,
                        use_context_rotation=windengram_use_context_rotation,
                        use_gating=windengram_use_gating,
                        use_short_conv=windengram_use_short_conv,
                        conv_kernel_size=windengram_conv_kernel_size,
                        structure_kernel_size=windengram_structure_kernel_size,
                        seed=windengram_seed + layer_id * 997,
                    )
                else:
                    self.engram_blocks[str(layer_id)] = Engram3DBlock(
                        in_channels=layer_channels[layer_id],
                        max_ngram_size=engram_max_ngram_size,
                        num_heads=engram_num_heads,
                        head_dim=engram_head_dim,
                        vocab_sizes=engram_vocab_sizes,
                        context_offsets=engram_context_offsets,
                        hash_dim=engram_hash_dim,
                        tokenizer_num_codes=engram_tokenizer_num_codes,
                        tokenizer_embed_dim=engram_tokenizer_embed_dim,
                        tokenizer_commitment_weight=engram_tokenizer_commitment_weight,
                        tokenizer_codebook_weight=engram_tokenizer_codebook_weight,
                        use_context_rotation=engram_use_context_rotation,
                        use_gating=engram_use_gating,
                        use_short_conv=engram_use_short_conv,
                        token_scale=engram_token_scale,
                        conv_kernel_size=engram_conv_kernel_size,
                        seed=engram_seed + layer_id * 997,
                    )

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
        self.regression_head = OutConv3D(base_channels, self.pred_len * self.in_channels)
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
            raise ValueError("UNet3DEngram expects coords with 4 or 5 dims, but received {}".format(coords.dim()))
        if coords.size(0) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1, -1, -1, -1)
        if coords.size(0) != batch_size:
            raise ValueError("UNet3DEngram expects coords batch size {}, but received {}".format(batch_size, coords.size(0)))
        if coords.size(1) != self.coord_channels:
            raise ValueError("UNet3DEngram expects coord_channels {}, but received {}".format(self.coord_channels, coords.size(1)))
        if tuple(coords.shape[-3:]) != spatial_shape:
            raise ValueError(
                "UNet3DEngram expects coords spatial shape {}, but received {}".format(
                    spatial_shape,
                    tuple(coords.shape[-3:]),
                )
            )
        return coords

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError("UNet3DEngram expects x as [B, history_len, C, Y, X, Z], got {}".format(tuple(x.shape)))
        batch, time_steps, channels, ydim, xdim, zdim = x.shape
        if time_steps != self.history_len:
            raise ValueError("UNet3DEngram expects history_len {}, got {}".format(self.history_len, time_steps))
        if channels != self.in_channels:
            raise ValueError("UNet3DEngram expects in_channels {}, got {}".format(self.in_channels, channels))

        coords = self._normalize_coords(coords, batch_size=batch, spatial_shape=(ydim, xdim, zdim))
        if self.use_coords and coords is None:
            raise ValueError("coords are required when use_coords=True")

        x = x.reshape(batch, time_steps * channels, ydim, xdim, zdim)
        if self.use_coords and coords is not None:
            x = torch.cat([x, coords], dim=1)

        x1 = self.inc(x)
        if "1" in self.engram_blocks:
            x1 = self.engram_blocks["1"](x1)
        x2 = self.down1(x1)
        if "2" in self.engram_blocks:
            x2 = self.engram_blocks["2"](x2)
        x3 = self.down2(x2)
        if "3" in self.engram_blocks:
            x3 = self.engram_blocks["3"](x3)
        x4 = self.down3(x3)
        if "4" in self.engram_blocks:
            x4 = self.engram_blocks["4"](x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = _center_crop_or_pad_3d(x, (ydim, xdim, zdim))

        if self.output_mode == "regression":
            x = self.regression_head(x)
            return x.view(batch, self.pred_len, self.in_channels, ydim, xdim, zdim)

        x = self.classification_head(x)
        return x.view(batch, self.pred_len, self.classification_channels, ydim, xdim, zdim)

    def get_aux_loss(self) -> torch.Tensor | None:
        if len(self.engram_blocks) == 0:
            return None
        losses = []
        for block in self.engram_blocks.values():
            if hasattr(block, "get_aux_loss"):
                block_aux = block.get_aux_loss()
                if block_aux is not None:
                    losses.append(block_aux)
        if not losses:
            return None
        total = losses[0].new_zeros(())
        for item in losses:
            total = total + item
        return total / len(losses)


def get_engram3d_design_notes() -> dict[str, Any]:
    return {
        "principles": [
            "learnable voxel tokenization with VQ discrete codes",
            "deterministic sparse lookup for local reusable patterns",
            "multi-head hashing with prime table sizes for collision robustness",
            "3D-structured context offsets with per-head directional coverage",
            "context-aware gating to suppress noisy retrievals",
            "optional lightweight depthwise refinement for stable integration",
        ],
        "discussion_points": [
            "current directional context set vs richer context topology",
            "single shared memory table vs per-stage memory table",
            "best insertion depths for grid3d risk prediction",
            "how much memory budget to trade for backbone compute budget",
        ],
    }


__all__ = [
    "Engram3DBlock",
    "LearnableVoxelTokenizer3D",
    "Model",
    "MultiHeadHashEmbedding",
    "StructureStem3D",
    "UNet3DEngramModelConfig",
    "WindEngram3DBlock",
    "get_engram3d_design_notes",
]
