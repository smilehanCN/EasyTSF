from __future__ import annotations

from .afno3d import Model as AFNO3DModel
from .fno3d import Model as FNO3DModel
from .itransformer import Model as iTransformerModel
from .mixlinear import Model as MixLinearModel
from .patchstg_flat3d import Model as PatchSTGFlat3DModel
from .pcmlp import Model as PCMLPModel
from .simvpv2_3d import Model as SimVPv23DModel
from .sparsetsf import Model as SparseTSFModel
from .stid import Model as STIDModel
from .timebase import Model as TimeBaseModel
from .tqnet import Model as TQNetModel
from .unet3d import Model as UNet3DModel
from .unet3d_patchcat import Model as UNet3DPatchCatModel


MODEL_REGISTRY = {
    "afno3d": AFNO3DModel,
    "fno3d": FNO3DModel,
    "iTransformer": iTransformerModel,
    "MixLinear": MixLinearModel,
    "PCMLP": PCMLPModel,
    "simvpv2_3d": SimVPv23DModel,
    "TQNet": TQNetModel,
    "STID": STIDModel,
    "SparseTSF": SparseTSFModel,
    "TimeBase": TimeBaseModel,
    "unet3d": UNet3DModel,
    "unet3d_patchcat": UNet3DPatchCatModel,
    "patchstg_flat3d": PatchSTGFlat3DModel,
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
