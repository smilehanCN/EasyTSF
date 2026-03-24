import torch

from .mtsf import MTSFTask


class STFTask(MTSFTask):
    def __init__(self, graph, **kwargs):
        super().__init__(**kwargs)
        graph_tensor = torch.as_tensor(graph, dtype=torch.float32)
        self.register_buffer("graph", graph_tensor, persistent=False)

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, _ = self._prepare_batch(batch)
        label = var_y[:, -self.hparams.pred_len:, :]
        prediction = self.model(var_x, marker_x, self.graph)
        return prediction, label
