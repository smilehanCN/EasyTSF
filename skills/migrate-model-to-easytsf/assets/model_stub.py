import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hist_len, pred_len, var_num):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.var_num = int(var_num)
        # Add explicit model-specific hyperparameters here.

    def forward(self, var_x, marker_x, marker_y):
        # var_x: [B, hist_len, N]
        # marker_x: [B, hist_len, T] or None
        # marker_y: [B, pred_len, T] or None
        del marker_x, marker_y
        raise NotImplementedError("Implement the migrated model and return [B, pred_len, N].")
