import torch
import torch.nn as nn


try:
    from momentfm import MOMENTPipeline
except ImportError as exc:
    MOMENTPipeline = None
    _MOMENT_IMPORT_ERROR = exc
else:
    _MOMENT_IMPORT_ERROR = None


def _freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


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
        if MOMENTPipeline is None:
            raise ImportError(
                "MOMENT support requires the optional dependency 'momentfm'. "
                "Install it first, for example: pip install momentfm"
            ) from _MOMENT_IMPORT_ERROR

        model_kwargs = {
            "task_name": "forecasting",
            "seq_len": hist_len,
            "forecast_horizon": pred_len,
            "freeze_encoder": freeze_encoder,
            "freeze_embedder": freeze_embedder,
            "freeze_head": freeze_head,
            "head_dropout": head_dropout,
        }
        self.model = MOMENTPipeline.from_pretrained(
            moment_model_name_or_path,
            model_kwargs=model_kwargs,
        )
        self.model.init()
        if freeze_head:
            self.model.head = _freeze_module(self.model.head)

    def forward(self, var_x, marker_x):
        del marker_x
        x_enc = var_x.permute(0, 2, 1).contiguous()
        input_mask = torch.ones(
            (x_enc.shape[0], x_enc.shape[-1]),
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        outputs = self.model(x_enc=x_enc, input_mask=input_mask)
        return outputs.forecast.permute(0, 2, 1).contiguous()
