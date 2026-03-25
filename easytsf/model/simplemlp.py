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

    def forward(self, var_x, marker_x):
        del marker_x
        tokens = var_x.transpose(1, 2)
        prediction = self.time_mlp(tokens)
        return prediction.transpose(1, 2).contiguous()
