from functools import lru_cache, partial

import numpy as np
import torch
import torch.nn as nn

try:
    from timm.models.vision_transformer import Mlp, PatchEmbed, trunc_normal_
except ImportError as exc:  # pragma: no cover - optional dependency guard
    Mlp = None
    PatchEmbed = None
    trunc_normal_ = None
    TIMM_IMPORT_ERROR = exc
else:
    TIMM_IMPORT_ERROR = None

from ._arrow_layers import (
    MemEffMLA,
    RMSNorm,
    XFORMERS_IMPORT_ERROR,
    apply_rotary_emb,
    compute_axial_cis,
    compute_mixed_cis_optimized as compute_mixed_cis,
    init_random_2d_freqs,
    init_t_xy,
    memory_efficient_attention,
    require_xformers,
)
from ._arrow_moe import SP_MOE as MoE
from ._arrow_pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_ring_pos_embed, get_2d_sincos_pos_embed


def ensure_arrow_dependencies_available() -> None:
    missing = []
    first_error = None
    if TIMM_IMPORT_ERROR is not None:
        missing.append("timm")
        first_error = first_error or TIMM_IMPORT_ERROR
    if XFORMERS_IMPORT_ERROR is not None:
        missing.append("xformers")
        first_error = first_error or XFORMERS_IMPORT_ERROR
    if missing:
        message = "ARROW requires optional dependencies {}. Install them before instantiating the ARROW model.".format(
            ", ".join("'{}'".format(name) for name in missing)
        )
        raise ImportError(message) from first_error


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class WeatherEmbedding(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
        ring_pos_embed=False,
    ):
        super().__init__()
        if PatchEmbed is None or trunc_normal_ is None:
            ensure_arrow_dependencies_available()
        self.img_size = tuple(img_size)
        self.patch_size = int(patch_size)
        self.variables = list(variables)
        self.ring_pos_embed = bool(ring_pos_embed)
        self.token_embeds = nn.ModuleList([PatchEmbed(None, patch_size, 1, embed_dim) for _ in range(len(variables))])
        self.num_patches = (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size)
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.initialize_weights()

    def initialize_weights(self):
        if self.ring_pos_embed:
            pos_embed = get_2d_ring_pos_embed(
                self.pos_embed.shape[-1],
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
                cls_token=False,
            )
        else:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
                cls_token=False,
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.variables)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))
        for token_embed in self.token_embeds:
            weight = token_embed.proj.weight.data
            trunc_normal_(weight.view([weight.shape[0], -1]), std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {var: idx for idx, var in enumerate(self.variables)}
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        batch_size, _, num_tokens, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)
        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)
        x = x.squeeze()
        return x.unflatten(dim=0, sizes=(batch_size, num_tokens))

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        for i in range(len(var_ids)):
            var_id = var_ids[i]
            embed_variable = self.token_embeds[var_id](x[:, i : i + 1])
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)
        return self.aggregate_variables(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Linear(1, hidden_size)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1))


class MemEffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, freqs_cis=None):
        require_xformers()
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k, v = self.apply_rope(q, k, v, freqs_cis)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale, p=self.attn_drop.p)
        x = x.reshape([batch_size, seq_len, channels])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def apply_rope(self, q, k, v, freqs_cis):
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)


class Block(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_mla=False,
        use_moe=False,
        routed_num_experts=5,
        shared_num_experts=1,
        selected_experts=2,
        **block_kwargs,
    ):
        super().__init__()
        if Mlp is None:
            ensure_arrow_dependencies_available()
        self.use_mla = bool(use_mla)
        self.use_moe = bool(use_moe)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        if use_mla:
            self.attn = MemEffMLA(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_moe:
            self.mlp = MoE(
                input_size=hidden_size,
                output_size=hidden_size,
                hidden_size=mlp_hidden_dim,
                routed_num_experts=routed_num_experts,
                shared_num_experts=shared_num_experts,
                noisy_gating=True,
                k=selected_experts,
            )
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, freqs_cis=None, time_interval=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis)
        if self.use_moe:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), time_interval)
            self.aux_loss_dict = self.mlp.aux_loss_dict
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class ArrowBackbone(nn.Module):
    def __init__(
        self,
        in_img_size,
        variables,
        static_channel_names=None,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        rope_type="nope",
        rope_theta=100.0,
        use_mla=False,
        use_moe=False,
        routed_num_experts=5,
        shared_num_experts=1,
        selected_experts=2,
        list_time_intervals=(6, 12, 24),
        ring_pos_embed=False,
    ):
        super().__init__()
        ensure_arrow_dependencies_available()
        if rope_type not in {"axial", "mixed", "nope"}:
            raise ValueError("unsupported rope_type '{}'".format(rope_type))
        in_img_size = tuple(int(item) for item in in_img_size)
        patch_size = int(patch_size)
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = list(variables)
        self.static_channel_names = list(static_channel_names or [])
        self.all_variables = self.variables + self.static_channel_names
        self.patch_size = patch_size
        self.rope_type = rope_type
        self.use_moe = bool(use_moe)
        self.time_interval_map = {int(t): i for i, t in enumerate(list_time_intervals)}
        self.embedding = WeatherEmbedding(
            variables=self.all_variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
            ring_pos_embed=ring_pos_embed,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        if self.rope_type == "mixed":
            self.compute_cis = partial(compute_mixed_cis, num_heads=num_heads)
            freqs = []
            for _ in range(depth):
                freqs.append(init_random_2d_freqs(dim=hidden_size // num_heads, num_heads=num_heads, theta=rope_theta))
            freqs = torch.stack(freqs, dim=1).view(2, depth, -1)
            self.freqs = nn.Parameter(torch.stack([freqs.clone() for _ in list_time_intervals], dim=0), requires_grad=True)
            t_x, t_y = init_t_xy(end_x=in_img_size[0] // patch_size, end_y=in_img_size[1] // patch_size)
            self.register_buffer("freqs_t_x", t_x)
            self.register_buffer("freqs_t_y", t_y)
        elif self.rope_type == "axial":
            self.compute_cis = partial(compute_axial_cis, dim=hidden_size // num_heads, theta=rope_theta)
            self.freqs_cis = self.compute_cis(end_x=in_img_size[0] // patch_size, end_y=in_img_size[1] // patch_size)
        else:
            self.freqs_cis = None
        self.blocks = nn.ModuleList(
            [
                Block(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_mla=use_mla,
                    use_moe=use_moe,
                    routed_num_experts=routed_num_experts,
                    shared_num_experts=shared_num_experts,
                    selected_experts=selected_experts,
                )
                for _ in range(depth)
            ]
        )
        self.head = FinalLayer(hidden_size, patch_size, len(self.variables))
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, height=None, width=None):
        patch = self.patch_size
        channels = len(self.variables)
        height = self.in_img_size[0] // patch if height is None else height // patch
        width = self.in_img_size[1] // patch if width is None else width // patch
        if height * width != x.shape[1]:
            raise ValueError("token grid {}x{} is incompatible with {} tokens".format(height, width, x.shape[1]))
        x = x.reshape(shape=(x.shape[0], height, width, patch, patch, channels))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        return x.reshape(shape=(x.shape[0], channels, height * patch, width * patch))

    def create_input(self, x, static_inputs=None):
        if not self.static_channel_names:
            return x
        if static_inputs is None:
            raise ValueError("ARROW expected static_inputs because static_channel_names were configured")
        if static_inputs.ndim != 4:
            raise ValueError("ARROW expects static_inputs as [B, C, H, W], but received {}".format(tuple(static_inputs.shape)))
        if static_inputs.shape[1] != len(self.static_channel_names):
            raise ValueError(
                "ARROW expected {} static channels, but received {}".format(
                    len(self.static_channel_names),
                    static_inputs.shape[1],
                )
            )
        static_inputs = static_inputs.to(device=x.device, dtype=x.dtype)
        return torch.cat([x, static_inputs], dim=1)

    def forward(self, x, variables=None, time_interval=None, static_inputs=None):
        if x.ndim != 4:
            raise ValueError("ARROW backbone expects x as [B, C, H, W], but received {}".format(tuple(x.shape)))
        if time_interval is None:
            raise ValueError("ARROW backbone requires time_interval")
        self.aux_loss = 0.0
        variables = list(self.variables if variables is None else variables)
        if tuple(variables) != tuple(self.variables):
            raise ValueError("ARROW backbone variables {} do not match configured {}".format(variables, self.variables))
        all_variables = variables + self.static_channel_names
        if len(self.static_channel_names) != 0:
            x = self.create_input(x, static_inputs=static_inputs)
        x = self.embedding(x, all_variables)
        x = self.embed_norm_layer(x)
        self.moe_noises = []
        time_interval_emb = self.t_embedder(time_interval)
        time_interval_index = torch.as_tensor(
            [self.time_interval_map[int(round(float(t) * 10.0))] for t in time_interval.detach().cpu().tolist()],
            device=x.device,
            dtype=torch.long,
        )
        if self.rope_type == "mixed":
            freqs = self.freqs[time_interval_index]
            freqs_cis = self.compute_cis(freqs, self.freqs_t_x, self.freqs_t_y)
            for i, block in enumerate(self.blocks):
                x = block(x, time_interval_emb, freqs_cis=freqs_cis[:, i], time_interval=time_interval_index)
                if self.use_moe:
                    self.moe_noises.append(block.aux_loss_dict["noises_dist"])
        elif self.rope_type == "axial":
            freqs_cis = self.freqs_cis.to(x.device)
            for block in self.blocks:
                x = block(x, time_interval_emb, freqs_cis=freqs_cis, time_interval=time_interval_index)
                if self.use_moe:
                    self.moe_noises.append(block.aux_loss_dict["noises_dist"])
        else:
            for block in self.blocks:
                x = block(x, time_interval_emb, time_interval=time_interval_index)
                if self.use_moe:
                    self.moe_noises.append(block.aux_loss_dict["noises_dist"])
        if self.use_moe:
            self.moe_noises = torch.stack(self.moe_noises, dim=0)
        x = self.head(x, time_interval_emb)
        return self.unpatchify(x)
