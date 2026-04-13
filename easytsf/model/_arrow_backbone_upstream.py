import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_, Mlp, PatchEmbed
from xformers.ops import memory_efficient_attention, unbind
from functools import lru_cache
from GlobWeather.models.hub.Arrow_layers import compute_axial_cis, apply_rotary_emb, init_random_2d_freqs, init_t_xy
from GlobWeather.models.hub.Arrow_layers import compute_mixed_cis_optimized as compute_mixed_cis
from GlobWeather.models.hub.Arrow_layers import MemEffMLA
from functools import partial
from GlobWeather.models.hub.MoE import SP_MOE as MoE

import numpy as np

from GlobWeather.utils.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_2d_ring_pos_embed
from GlobWeather.models.hub.Arrow_layers import RMSNorm
import os

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
        ring_pos_embed=False
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables
        self.ring_pos_embed = ring_pos_embed

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(variables))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        # try 1
        # self.channel_query = nn.Parameter(torch.zeros(self.num_patches, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
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

        # token embedding layer
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.variables:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)   # transpose or permute
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        # try 1
        # var_query = self.channel_query.unsqueeze(0).expand(b, -1, -1, -1).flatten(0, 1)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i : i + 1]) # B, L, D
            embeds.append(embed_variable)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.channel_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = x + self.pos_embed.unsqueeze(1)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, freqs_cis=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = unbind(qkv, 0)   # B x H x L x D
        q, k, v = self.apply_rope(q, k, v, freqs_cis)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale, p=self.attn_drop.p)
        x = x.reshape([B, L, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def apply_rope(self, q, k, v, freqs_cis):
        # q, k: B x H x L x D -> B x L x H x D
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        return q, k, v


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, 
                 use_mla=False, use_moe=False, routed_num_experts=5, 
                 shared_num_experts=1, selected_experts=2,
                 **block_kwargs):
        super().__init__()
        self.use_mla = use_mla
        self.use_moe = use_moe
        # self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        if use_mla:
            self.attn = MemEffMLA(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        
        # mlp
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if use_moe:
            self.mlp = MoE(input_size=hidden_size, output_size=hidden_size, 
                           hidden_size=mlp_hidden_dim, routed_num_experts=routed_num_experts, 
                           shared_num_experts=shared_num_experts, noisy_gating=True, k=selected_experts)
        else:
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, freqs_cis=None, time_interval=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs_cis=freqs_cis)
        if self.use_moe:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp), time_interval)
            # self.aux_loss = self.mlp.aux_loss
            self.aux_loss_dict = self.mlp.aux_loss_dict
        else:
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class arrow(nn.Module):
    def __init__(self, 
        in_img_size,
        variables,
        const_variables,
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
        list_time_intervals=[6, 12, 24],
        ring_pos_embed=False
    ):
        super().__init__()
        assert rope_type in ["axial", "mixed", "nope"]
        
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.variables = variables
        self.const_variables = const_variables
        if len(const_variables) != 0:
            self.read_constants("./constants")
        self.all_variables = self.variables + self.const_variables

        self.patch_size = patch_size
        self.rope_type = rope_type
        self.use_moe = use_moe
        self.time_interval_map = {t: i for i, t in enumerate(list_time_intervals)}
        
        # embedding
        self.embedding = WeatherEmbedding(
            variables=self.all_variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
            ring_pos_embed=ring_pos_embed
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size)
        
        # interval embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # rope in backbone
        if self.rope_type == "mixed":
            self.compute_cis = partial(compute_mixed_cis, num_heads=num_heads)
            freqs = []
            for _ in range(depth):
                freqs.append(
                    init_random_2d_freqs(dim=hidden_size // num_heads, num_heads=num_heads, theta=rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, depth, -1)
            self.freqs = nn.Parameter(torch.stack([freqs.clone() for _ in list_time_intervals], dim=0), requires_grad=True)

            t_x, t_y = init_t_xy(end_x=in_img_size[0] // patch_size, end_y=in_img_size[1] // patch_size)
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        elif self.rope_type == "axial":
            self.compute_cis = partial(compute_axial_cis, dim=hidden_size//num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x=in_img_size[0] // patch_size, end_y=in_img_size[1] // patch_size)
            self.freqs_cis = freqs_cis
        else:
            self.freqs_cis = None

        # backbone
        self.blocks = nn.ModuleList([
            Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                  use_mla=use_mla, use_moe=use_moe, routed_num_experts=routed_num_experts,
                  shared_num_experts=shared_num_experts, selected_experts=selected_experts) for _ in range(depth)
        ])
        
        # prediction layer
        self.head = FinalLayer(hidden_size, patch_size, len(variables))

        self.initialize_weights()

    def read_constants(self, root_dir):
        self.constants = []
        V = len(self.const_variables)
        for const in self.const_variables:
            constant = np.load(os.path.join(root_dir, "{}.npy".format(const)))
            self.constants.append(torch.from_numpy(constant))
        self.constants = torch.stack(self.constants)
        const_max = self.constants.view(V, -1).max(dim=1)[0].view(V, 1, 1)
        const_min = self.constants.view(V, -1).min(dim=1)[0].view(V, 1, 1)

        self.constants = (self.constants - const_min) / (const_max - const_min + 1e-8)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        trunc_normal_(self.t_embedder.mlp.weight, std=0.02)
        
        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = len(self.variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def create_input(self, x):
        # x: B x V x H x W
        B = x.shape[0]
        constants = self.constants.to(x.device)
        constants = constants.unsqueeze(0).repeat(B, 1, 1, 1)
        x = torch.concat([x, constants], dim=1)
        return x

    def forward(self, x, variables, time_interval):
        self.aux_loss = 0.0
        all_variables = variables + self.const_variables
        if len(self.const_variables) != 0:
            x = self.create_input(x)
        x = self.embedding(x, all_variables) # B, L(num_tokens), D(embed_dim)
        x = self.embed_norm_layer(x)
        self.moe_noises = []

        time_interval_emb = self.t_embedder(time_interval)
        time_interval = torch.tensor([self.time_interval_map[int(t*10)] for t in time_interval], device=x.device)
        if self.rope_type == "mixed":
            # freqs = torch.stack([getattr(self, f'freqs_{int(t*10)}') for t in time_interval])
            freqs = self.freqs[time_interval]
            freqs_cis = self.compute_cis(freqs, self.freqs_t_x, self.freqs_t_y)
            for i, block in enumerate(self.blocks):
                x = block(x, time_interval_emb, freqs_cis=freqs_cis[:, i], time_interval=time_interval)
                if self.use_moe:
                    self.moe_noises.append(block.aux_loss_dict["noises_dist"])
        elif self.rope_type == "axial":
            freqs_cis = self.freqs_cis.to(x.device)
            for i, block in enumerate(self.blocks):
                x = block(x, time_interval_emb, freqs_cis=freqs_cis, time_interval=time_interval)
                if self.use_moe:
                    self.moe_noises.append(block.aux_loss_dict["noises_dist"])
        else:
            for block in self.blocks:
                x = block(x, time_interval_emb, time_interval=time_interval) # B, L, D
                if self.use_moe:
                    self.moe_noises.append(block.aux_loss_dict["noises_dist"])
        
        if self.use_moe:
            self.moe_noises = torch.stack(self.moe_noises, dim=0)
        x = self.head(x, time_interval_emb) # B, L, D'
        x = self.unpatchify(x)  # B, V, H, W
        
        return x
