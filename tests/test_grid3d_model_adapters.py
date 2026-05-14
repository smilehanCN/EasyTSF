import pytest
import torch

from easytsf.model import MODEL_REGISTRY, get_model_class
from easytsf.task import get_task_spec, validate_task_runtime_conf


GRID3D_MODELS = get_task_spec("grid3d_forecasting").supported_models


def test_grid3d_supported_models_are_registered():
    missing = [model_name for model_name in GRID3D_MODELS if model_name not in MODEL_REGISTRY]
    assert missing == []


@pytest.mark.parametrize("model_name", GRID3D_MODELS)
def test_grid3d_supported_models_pass_task_preflight(model_name):
    runtime_conf = {
        "task": "grid3d_forecasting",
        "model": model_name,
        "val_metric": "val/loss",
    }

    assert validate_task_runtime_conf(runtime_conf) == get_task_spec("grid3d_forecasting")


def test_representative_grid3d_model_forward_shape_contract():
    model_cls = get_model_class("fno3d")
    model = model_cls(
        history_len=2,
        pred_len=1,
        in_channels=3,
        out_channels=2,
        coord_channels=3,
        fno_width=4,
        fno_layers=1,
        fno_modes=(1, 1, 1),
        fno_padding=0,
        fno_projection_dim=8,
        use_coords=True,
    )
    inputs = torch.randn(1, 2, 3, 4, 4, 3)
    coords = torch.randn(1, 3, 4, 4, 3)

    output = model(inputs, coords=coords)

    assert output.shape == (1, 1, 2, 4, 4, 3)
