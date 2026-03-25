import math

import torch
import torch.nn as nn


def normalized_graph_support(graph):
    identity = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
    support = graph + identity
    degree = support.sum(dim=-1).clamp_min(1e-6)
    inv_sqrt_degree = degree.pow(-0.5)
    return inv_sqrt_degree.unsqueeze(-1) * support * inv_sqrt_degree.unsqueeze(-2)


def build_graph_prior(graph, neighbor_order):
    support = normalized_graph_support(graph)
    prior = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
    power = support
    for _ in range(max(1, int(neighbor_order))):
        prior = prior + power
        power = torch.matmul(power, support)
    prior = prior / prior.max().clamp_min(1e-6)
    return prior.clamp(0.0, 1.0)


def build_grid_prior(grid_shape, neighbor_order, device=None, dtype=None):
    if len(grid_shape) not in {2, 3}:
        raise ValueError("grid prior supports only 2D or 3D token grids")

    axes = [torch.arange(size, device=device, dtype=torch.float32) for size in grid_shape]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(grid_shape))
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    chebyshev = diff.abs().amax(dim=-1)
    manhattan = diff.abs().sum(dim=-1)

    prior = torch.zeros_like(chebyshev)
    within_hop = chebyshev <= max(1, int(neighbor_order))
    prior[within_hop] = 1.0 / (1.0 + manhattan[within_hop])
    prior.fill_diagonal_(1.0)
    if dtype is not None:
        prior = prior.to(dtype=dtype)
    return prior


def _pearson_corr(ts):
    _, _, series_length = ts.shape
    mean = ts.mean(dim=2, keepdim=True)
    std = ts.std(dim=2, unbiased=False, keepdim=True)
    centered = ts - mean
    cov = torch.matmul(centered, centered.transpose(1, 2)) / series_length
    std_outer = torch.matmul(std, std.transpose(1, 2))
    std_outer = torch.where(std_outer == 0, torch.tensor(1e-8, device=ts.device, dtype=ts.dtype), std_outer)
    corr = cov / std_outer
    eye = torch.eye(corr.size(1), device=ts.device, dtype=ts.dtype).unsqueeze(0)
    return corr * (1 - eye) + eye


class TemporalPredictionHead(nn.Module):
    def __init__(self, hidden, forecast_len, head_dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden, forecast_len)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x.transpose(2, 1)


class TokenPredictionHead(nn.Module):
    def __init__(self, hidden, forecast_len, token_dim, head_dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(hidden, forecast_len * token_dim)
        self.dropout = nn.Dropout(head_dropout)
        self.forecast_len = forecast_len
        self.token_dim = token_dim

    def forward(self, x):
        batch_size, num_tokens = x.shape[:2]
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x.view(batch_size, num_tokens, self.forecast_len, self.token_dim).permute(0, 2, 1, 3).contiguous()


class GraphProjectBlock(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.graph_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, support):
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.dropout(torch.nn.functional.gelu(self.fc1(hidden)))
        hidden = self.fc2(hidden)
        hidden = torch.einsum("ij,bjpd->bipd", support, hidden)
        hidden = self.dropout(self.graph_proj(hidden))
        return hidden + residual


class GridProjectBlock(nn.Module):
    def __init__(self, d_model, spatial_ndim, dropout=0.2):
        super().__init__()
        if spatial_ndim == 2:
            conv_cls = nn.Conv2d
        elif spatial_ndim == 3:
            conv_cls = nn.Conv3d
        else:
            raise ValueError("grid project block supports only 2D or 3D grids")

        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.depthwise = conv_cls(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pointwise = conv_cls(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, grid_shape):
        residual = hidden
        batch_size, num_tokens, patch_num, model_dim = hidden.shape
        if num_tokens != math.prod(grid_shape):
            raise ValueError("token count does not match grid shape: {} vs {}".format(num_tokens, grid_shape))

        hidden = self.norm(hidden)
        hidden = self.dropout(torch.nn.functional.gelu(self.fc1(hidden)))
        hidden = self.fc2(hidden)
        hidden = hidden.permute(0, 2, 3, 1).reshape(batch_size * patch_num, model_dim, *grid_shape)
        hidden = self.depthwise(hidden)
        hidden = self.dropout(torch.nn.functional.gelu(self.pointwise(hidden)))
        hidden = hidden.reshape(batch_size, patch_num, model_dim, num_tokens).permute(0, 3, 1, 2).contiguous()
        return hidden + residual


class SpatialContrastive(nn.Module):
    def __init__(
        self,
        num_tokens,
        model_dim,
        m_dim=None,
        k_order=3,
        de=4,
        threshold=0.3,
        structure_prior_weight=0.5,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.m_dim = m_dim if m_dim is not None else max(1, num_tokens // 10 + 1)
        self.k_order = k_order
        self.threshold = threshold
        self.structure_prior_weight = structure_prior_weight
        self.q = nn.Parameter(torch.randn(num_tokens, self.m_dim))
        self.v1 = nn.Parameter(torch.randn(self.m_dim, de))
        self.v2 = nn.Parameter(torch.randn(self.m_dim, de))
        self.f = nn.Linear(model_dim, self.k_order)
        self.a = None
        self.prior = None

    def polynomial(self, embedding):
        embedding = embedding.permute(0, 2, 1, 3).reshape(-1, embedding.shape[1], embedding.shape[-1])
        q_power = self.q
        coeff = self.f(embedding).unsqueeze(-2).expand(-1, -1, self.m_dim, -1)
        q_mixture = coeff[..., 0] * q_power
        for index in range(1, self.k_order):
            q_power = q_power * self.q
            q_mixture = coeff[..., index] * q_power + q_mixture
        return q_mixture

    def composition(self, ts, q_mixture, prior):
        v_matrix = torch.mm(self.v1, self.v2.transpose(0, 1)).unsqueeze(0).expand(q_mixture.size(0), -1, -1)
        learned = torch.sigmoid(torch.bmm(torch.bmm(q_mixture, v_matrix), q_mixture.transpose(1, 2)))
        pearson = (_pearson_corr(ts) + 1.0) / 2.0
        dynamic = (pearson + learned) / 2.0
        prior = prior.to(dtype=dynamic.dtype, device=dynamic.device).unsqueeze(0).expand(dynamic.size(0), -1, -1)
        mixed = (1.0 - self.structure_prior_weight) * dynamic + self.structure_prior_weight * prior
        eye = torch.eye(mixed.size(-1), device=mixed.device, dtype=mixed.dtype).unsqueeze(0)
        mixed = mixed * (1 - eye) + eye
        return mixed, prior

    def cal_corr(self, ts, embedding, prior):
        q_mixture = self.polynomial(embedding)
        self.a, self.prior = self.composition(ts, q_mixture, prior)

    def forward(self, features, polarity):
        if self.a is None or self.prior is None:
            raise RuntimeError("correlation matrix must be computed before contrastive loss")

        distance = self.get_feature_dis(features)
        eye = torch.eye(self.a.size(-1), device=self.a.device, dtype=self.a.dtype).unsqueeze(0)
        pos_mask = (self.prior > 0).float() * (1 - eye)
        neg_mask = (self.prior <= 0).float() * (1 - eye)

        if polarity == "neg":
            adjacency = torch.clamp(1.0 - self.a, min=0.0) * neg_mask
            fallback = neg_mask
        else:
            adjacency = torch.clamp(self.a - self.threshold, min=0.0) * pos_mask
            fallback = pos_mask

        adjacency_sum = adjacency.sum(dim=-1, keepdim=True)
        adjacency = torch.where(adjacency_sum > 0, adjacency, fallback)
        return self.cal_loss(distance, adjacency), adjacency

    @staticmethod
    def cal_loss(distance, adjacency):
        distance = torch.exp(distance)
        distance_sum = torch.sum(distance, dim=-1)
        distance_sum_pos = torch.sum(distance * adjacency, dim=-1)
        return -torch.log(distance_sum_pos * distance_sum.pow(-1) + 1e-8).mean()

    @staticmethod
    def get_feature_dis(features):
        distance = torch.matmul(features, features.transpose(-2, -1))
        mask = torch.eye(distance.shape[-1], device=features.device, dtype=features.dtype).unsqueeze(0)
        norm = torch.sum(features ** 2, dim=2, keepdim=True).sqrt()
        norm = torch.matmul(norm, norm.transpose(-2, -1)) + 1e-8
        distance = distance / norm
        return (1 - mask) * distance
