from __future__ import annotations

from .arrow import Model as ARROWModel
from .fredn import Model as FreDNMultivariate3DModel
from .itransformer import Model as iTransformerModel
from .mixlinear import Model as MixLinearModel
from .patchstg_flat3d import Model as PatchSTGFlat3DModel
from .pcmlp import Model as PCMLPModel
from .sparsetsf import Model as SparseTSFModel
from .stid import Model as STIDModel
from .stgcn import Model as STGCNModel
from .timebase import Model as TimeBaseModel
from .tqnet import Model as TQNetModel
from .unet3d_patchcat import Model as UNet3DPatchCatModel
from .unet3d_wf4cast import Model as UNet3DModel
from .weatherbench_persistence import Model as WeatherBenchPersistenceModel


MODEL_REGISTRY = {
    "ARROW": ARROWModel,
    "iTransformer": iTransformerModel,
    "MixLinear": MixLinearModel,
    "PCMLP": PCMLPModel,
    "TQNet": TQNetModel,
    "STGCN": STGCNModel,
    "STID": STIDModel,
    "SparseTSF": SparseTSFModel,
    "TimeBase": TimeBaseModel,
    "WeatherBenchPersistence": WeatherBenchPersistenceModel,
    "unet3d": UNet3DModel,
    "unet3d_patchcat": UNet3DPatchCatModel,
    "patchstg_flat3d": PatchSTGFlat3DModel,
    "fredn_multivariate3d": FreDNMultivariate3DModel,
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
