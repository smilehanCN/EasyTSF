from __future__ import annotations

from .itransformer import Model as iTransformerModel
from .mixlinear import Model as MixLinearModel
from .sparsetsf import Model as SparseTSFModel
from .stid import Model as STIDModel
from .stgcn import Model as STGCNModel
from .timebase import Model as TimeBaseModel
from .tqnet import Model as TQNetModel


MODEL_REGISTRY = {
    "iTransformer": iTransformerModel,
    "MixLinear": MixLinearModel,
    "TQNet": TQNetModel,
    "STGCN": STGCNModel,
    "STID": STIDModel,
    "SparseTSF": SparseTSFModel,
    "TimeBase": TimeBaseModel,
}


def get_model_class(model_name: str):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        raise ValueError(
            "unknown model '{}'; registered models are {}".format(
                model_name,
                sorted(MODEL_REGISTRY),
            )
        ) from exc
