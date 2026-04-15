import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from xformers.ops import memory_efficient_attention


def rms_norm(x, normalized_shape, weight=None, eps=1e-6):
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    
    ndim = x.dim()
    dims = tuple(range(ndim - len(normalized_shape), ndim))
    
    variance = x.pow(2).mean(dims, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    
    if weight is not None:
        broadcast_shape = [1] * (ndim - len(normalized_shape)) + list(weight.shape)
        x_normalized = x_normalized * weight.view(broadcast_shape)
    
    return x_normalized


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return rms_norm(x, (self.dim,), self.weight, self.eps)

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)

    return freqs_cis

def compute_mixed_cis_optimized(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    if len(freqs.shape) == 3:
        depth = freqs.shape[1]
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = torch.einsum('n,dhc->dhnc', t_x, freqs[0].view(depth, num_heads, -1))
            freqs_y = torch.einsum('n,dhc->dhnc', t_y, freqs[1].view(depth, num_heads, -1))
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    elif len(freqs.shape) == 4:
        B = freqs.shape[0]
        depth = freqs.shape[2]
        with torch.cuda.amp.autocast(enabled=False):
            freqs_x = torch.einsum('n,bdhc->bdhnc', t_x, freqs[:, 0].view(B, depth, num_heads, -1))
            freqs_y = torch.einsum('n,bdhc->bdhnc', t_y, freqs[:, 1].view(B, depth, num_heads, -1))
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    else:
        raise ValueError(f"Unsupported freqs shape: {freqs.shape}")
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def init_t_xy(end_x: int, end_y: int, sphere: bool = True):
    y_coords = torch.arange(end_y, dtype=torch.float32)
    x_coords = torch.arange(end_x, dtype=torch.float32)

    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    t_y = y_grid.reshape(-1)
    t_x = x_grid.reshape(-1)
    
    return t_x, t_y

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    if x.ndim == freqs_cis.ndim:
        return freqs_cis
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq/xk: B x H x L x D
    # freqs_cis: H x L x D/2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class MemEffMLA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        q_lora_rank: int = 128,
        kv_lora_rank: int = 256,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.wq_a = nn.Linear(dim, q_lora_rank, bias=qkv_bias)
            self.q_norm = RMSNorm(q_lora_rank)
            self.wq_b = nn.Linear(q_lora_rank, dim, bias=qkv_bias)
        
        self.wkv_a = nn.Linear(dim, kv_lora_rank, bias=qkv_bias)
        self.kv_norm = RMSNorm(kv_lora_rank)
        self.wkv_b = nn.Linear(kv_lora_rank, dim * 2, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, attn_bias=None, freqs_cis=None):
        B, L, _ = x.shape
        
        if self.q_lora_rank == 0:
            q = self.wq(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x))).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        kv = self.wkv_a(x)
        kv = self.kv_norm(kv)
        kv = self.wkv_b(kv).reshape(B, L, self.num_heads, 2 * self.head_dim)
        k, v = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)

        k = k.permute(0, 2, 1, 3)  # B, H, L, D
        v = v.permute(0, 2, 1, 3)  # B, H, L, D
        
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias, scale=self.scale, p=self.attn_drop)
        x = x.permute(0, 2, 1, 3).reshape(B, L, -1)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
