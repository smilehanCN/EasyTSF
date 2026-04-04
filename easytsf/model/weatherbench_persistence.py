import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = int(pred_len)
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, var_x, marker_x, marker_y, static_inputs=None, target_input_indices=None):
        del marker_x, marker_y, static_inputs
        if var_x.ndim != 5:
            raise ValueError(
                "WeatherBenchPersistence expects var_x as [B, hist_len, C, H, W], but received shape {}".format(
                    tuple(var_x.shape),
                )
            )

        prediction = var_x[:, -1:, :, :, :]
        if target_input_indices is not None:
            if not torch.is_tensor(target_input_indices):
                target_input_indices = torch.as_tensor(target_input_indices, device=var_x.device, dtype=torch.long)
            prediction = prediction.index_select(dim=2, index=target_input_indices)
        prediction = prediction.repeat(1, self.pred_len, 1, 1, 1)
        return prediction + self.bias.view(1, 1, 1, 1, 1)
