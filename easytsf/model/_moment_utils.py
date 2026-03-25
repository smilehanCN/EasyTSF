import torch
import torch.nn as nn


try:
    from momentfm import MOMENTPipeline
    from transformers import AutoConfig
except ImportError as exc:
    MOMENTPipeline = None
    AutoConfig = None
    _MOMENT_IMPORT_ERROR = exc
else:
    _MOMENT_IMPORT_ERROR = None


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def load_moment_pipeline(moment_model_name_or_path, model_kwargs):
    try:
        return MOMENTPipeline.from_pretrained(
            moment_model_name_or_path,
            model_kwargs=model_kwargs,
        )
    except TypeError as exc:
        if "required positional argument: 'config'" not in str(exc):
            raise
        config = AutoConfig.from_pretrained(moment_model_name_or_path)
        return MOMENTPipeline.from_pretrained(
            moment_model_name_or_path,
            config=config,
            model_kwargs=model_kwargs,
        )


class MomentForecastBackbone(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        moment_model_name_or_path,
        freeze_encoder=True,
        freeze_embedder=True,
        freeze_head=False,
        head_dropout=None,
        error_prefix="MOMENT support",
    ):
        super().__init__()
        if MOMENTPipeline is None:
            raise ImportError(
                "{} requires the optional dependency 'momentfm'. "
                "Install it first, for example: pip install momentfm".format(error_prefix)
            ) from _MOMENT_IMPORT_ERROR

        model_kwargs = {
            "task_name": "forecasting",
            "seq_len": hist_len,
            "forecast_horizon": pred_len,
            "freeze_encoder": freeze_encoder,
            "freeze_embedder": freeze_embedder,
            "freeze_head": freeze_head,
        }
        if head_dropout is not None:
            model_kwargs["head_dropout"] = head_dropout

        self.pipeline = load_moment_pipeline(
            moment_model_name_or_path,
            model_kwargs,
        )
        self.pipeline.init()
        self.model = getattr(self.pipeline, "model", self.pipeline)
        if freeze_head:
            self.model.head = freeze_module(self.model.head)

        self.patch_size = getattr(self.model.config, "patch_len")
        self.stride = getattr(self.model.config, "patch_stride_len", self.patch_size)
        self.model_dim = getattr(self.model.config, "d_model")
        self.patch_num = getattr(self.model, "head_nf", self.model_dim) // self.model_dim

    def get_settings(self):
        return self.model_dim, self.patch_size, self.stride, self.patch_num

    def _build_input_mask(self, inputs):
        return torch.ones(
            (inputs.shape[0], inputs.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )

    def forecast(self, inputs):
        x_enc = inputs.permute(0, 2, 1).contiguous()
        input_mask = self._build_input_mask(inputs)
        outputs = self.pipeline(x_enc=x_enc, input_mask=input_mask)
        return outputs.forecast.permute(0, 2, 1).contiguous()

    def forecast_for_plugin(self, inputs):
        x_enc = inputs.permute(0, 2, 1).contiguous()
        input_mask = self._build_input_mask(inputs)
        embeddings = self.model.embed(x_enc=x_enc, input_mask=input_mask, reduction="none").embeddings
        outputs = self.model.head(embeddings)
        outputs = self.model.normalizer(x=outputs, mode="denorm")
        return outputs.permute(0, 2, 1).contiguous(), embeddings

    def denorm(self, inputs):
        outputs = inputs.permute(0, 2, 1).contiguous()
        outputs = self.model.normalizer(x=outputs, mode="denorm")
        return outputs.permute(0, 2, 1).contiguous()
