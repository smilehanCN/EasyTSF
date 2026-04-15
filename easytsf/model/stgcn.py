import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.filter_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=(0, padding))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        return self.dropout(gated)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x, support):
        propagated = torch.einsum("ij,bcjt->bcit", support, x)
        return self.proj(propagated)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, dropout):
        super().__init__()
        self.temp_in = TemporalConv(in_channels, hidden_channels, kernel_size, dropout)
        self.graph_conv = GraphConv(hidden_channels, hidden_channels)
        self.temp_out = TemporalConv(hidden_channels, out_channels, kernel_size, dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else nn.Identity()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x, support):
        residual = self.residual(x)
        x = self.temp_in(x)
        x = F.relu(self.graph_conv(x, support))
        x = self.temp_out(x)
        return self.norm(x + residual)


class Model(nn.Module):
    def __init__(self, hist_len, pred_len, hidden_dim, block_num=2, kernel_size=3, dropout=0.1, graph=None):
        super().__init__()
        if graph is None:
            raise ValueError("STGCN requires graph during initialization")
        graph = torch.as_tensor(graph, dtype=torch.float32)
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("STGCN graph must be a square matrix, but received shape {}".format(tuple(graph.shape)))
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.input_proj = nn.Conv2d(1, hidden_dim, kernel_size=(1, 1))
        self.blocks = nn.ModuleList(
            STGCNBlock(hidden_dim, hidden_dim, hidden_dim, kernel_size, dropout)
            for _ in range(block_num)
        )
        self.output_proj = nn.Conv2d(hidden_dim, 1, kernel_size=(1, 1))
        self.time_proj = nn.Linear(hist_len, pred_len)
        self.register_buffer("graph", graph, persistent=False)
        self.register_buffer("support", self._get_normalized_support(graph), persistent=False)

    @staticmethod
    def _get_normalized_support(graph):
        identity = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        support = graph + identity
        degree = support.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt_degree = degree.pow(-0.5)
        return inv_sqrt_degree.unsqueeze(-1) * support * inv_sqrt_degree.unsqueeze(-2)

    def forward(self, var_x, marker_x, marker_y):
        del marker_x, marker_y
        x = var_x.transpose(1, 2).unsqueeze(1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, self.support)

        x = self.output_proj(x).squeeze(1)
        prediction = self.time_proj(x)
        return prediction.transpose(1, 2)
