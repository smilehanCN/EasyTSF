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

    def forward(self, var_x, marker_x, grid_mask=None, coord=None):
        del marker_x, coord
        spatial_shape = tuple(var_x.shape[3:])
        mask = None
        if grid_mask is not None:
            if tuple(grid_mask.shape) != spatial_shape:
                raise ValueError(
                    "SimpleGridMLP expected grid_mask shape {}, but received {}".format(
                        spatial_shape,
                        tuple(grid_mask.shape),
                    )
                )
            mask = grid_mask.view(1, 1, 1, *spatial_shape).to(device=var_x.device, dtype=var_x.dtype)
            var_x = var_x * mask

        batch_size, _, channel_num = var_x.shape[:3]
        tokens = var_x.reshape(batch_size, var_x.shape[1], -1).transpose(1, 2)
        prediction = self.time_mlp(tokens).transpose(1, 2)
        prediction = prediction.reshape(batch_size, self.pred_len, channel_num, *spatial_shape)

        if mask is not None:
            prediction = prediction * mask
        return prediction.contiguous()
