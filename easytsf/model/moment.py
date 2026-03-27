import torch.nn as nn

from ._moment_utils import MomentForecastBackbone


class Model(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        moment_model_name_or_path,
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
        head_dropout=0.1,
    ):
        super().__init__()
        self.model = MomentForecastBackbone(
            hist_len=hist_len,
            pred_len=pred_len,
            moment_model_name_or_path=moment_model_name_or_path,
            freeze_encoder=freeze_encoder,
            freeze_embedder=freeze_embedder,
            freeze_head=freeze_head,
            head_dropout=head_dropout,
            error_prefix="MOMENT support",
        )

    def forward(self, var_x, marker_x, marker_y):
        del marker_x, marker_y
        return self.model.forecast(var_x)
