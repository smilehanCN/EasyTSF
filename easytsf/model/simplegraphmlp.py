import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hist_len, pred_len, hidden_dim, dropout=0.0):
        super().__init__()
        self.pred_len = int(pred_len)
        self.time_mlp = nn.Sequential(
            nn.Linear(int(hist_len), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.pred_len),
        )
        self._cached_support = None
        self._cached_support_key = None

    def _get_normalized_support(self, graph):
        cache_key = (graph.device.type, graph.device.index, graph.dtype, tuple(graph.shape))
        if self._cached_support is not None and self._cached_support_key == cache_key:
            return self._cached_support

        identity = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
        support = graph + identity
        degree = support.sum(dim=-1).clamp_min(1e-6)
        inv_sqrt_degree = degree.pow(-0.5)
        support = inv_sqrt_degree.unsqueeze(-1) * support * inv_sqrt_degree.unsqueeze(-2)

        self._cached_support = support
        self._cached_support_key = cache_key
        return support

    def forward(self, var_x, marker_x, graph):
        del marker_x
        if graph.dim() != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("SimpleGraphMLP requires a square graph adjacency matrix")
        if graph.shape[0] != var_x.shape[-1]:
            raise ValueError(
                "SimpleGraphMLP expected graph with {} nodes but received {}".format(
                    var_x.shape[-1],
                    graph.shape[0],
                )
            )

        support = self._get_normalized_support(graph)
        node_inputs = var_x.transpose(1, 2)
        propagated = torch.einsum("ij,bjl->bil", support, node_inputs)
        mixed = 0.5 * (node_inputs + propagated)
        prediction = self.time_mlp(mixed)
        return prediction.transpose(1, 2).contiguous()
