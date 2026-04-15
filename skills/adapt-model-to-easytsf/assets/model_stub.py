import torch.nn as nn


class Model(nn.Module):
    def __init__(self, hist_len, pred_len, var_num):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.var_num = int(var_num)
        # Add explicit task-aware hyperparameters here.

    def forward(self, var_x, marker_x, marker_y):
        # Current concrete runtime path: sequence_prediction through mtsf.
        # Update the signature when implementing a new task contract.
        del marker_x, marker_y
        raise NotImplementedError("Implement the adapted model for the target prediction task.")
