import pytest
import torch

from easytsf.model import get_model_class
from easytsf.task import get_task_spec, validate_task_runtime_conf


@pytest.mark.parametrize(
    "task_name,model_name",
    (
        ("grid3d_forecasting", "unet3d"),
        ("grid3d_forecasting", "unet3d_engram"),
        ("grid3d_forecasting", "unet3d_patchcat"),
        ("grid3d_forecasting", "patchstg_flat3d"),
        ("grid3d_forecasting", "fredn_multivariate3d"),
        ("grid3d_forecasting", "fno3d"),
        ("grid3d_forecasting", "afno3d"),
    ),
)
def test_validate_task_runtime_conf_accepts_grid3d_models(task_name, model_name):
    runtime_conf = {
        "task": task_name,
        "model": model_name,
        "val_metric": "val/loss",
    }
    assert validate_task_runtime_conf(runtime_conf) == get_task_spec(task_name)


def test_grid3d_models_forward_shape_contract():
    batch_size, history_len, pred_len = 1, 4, 2
    channels = 3
    spatial_shape = (8, 8, 4)
    inputs = torch.randn(batch_size, history_len, channels, *spatial_shape)
    coords = torch.randn(batch_size, 3, *spatial_shape)

    model_builders = {
        "unet3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            base_channels=4,
            patch_size=(1, 1, 1),
            downsample_scale=(1, 1, 1),
            kernel_size=(3, 3, 3),
            expansion=2,
        ),
        "unet3d_engram": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            base_channels=4,
            patch_size=(1, 1, 1),
            downsample_scale=(1, 1, 1),
            kernel_size=(3, 3, 3),
            expansion=2,
            use_engram=False,
        ),
        "unet3d_patchcat": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            base_channels=4,
            input_embed_dim=8,
            patch_size=(1, 1, 1),
            downsample_scale=(1, 1, 1),
            kernel_size=(3, 3, 3),
            expansion=2,
            use_coords=True,
        ),
        "patchstg_flat3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            patch_size_3d=(2, 2, 1),
            embed_dim=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            use_coords=True,
            spatial_downsample_factor_3d=(1, 1, 1),
        ),
        "fredn_multivariate3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            embed_size=8,
            hidden_size=16,
            hidden_layers=1,
            dropout=0.0,
            use_revin=True,
            revin_affine=True,
            revin_subtract_last=False,
            voxel_chunk_size=128,
            spatial_downsample_factor_3d=(1, 1, 1),
        ),
        "fno3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            fno_width=8,
            fno_layers=2,
            fno_modes=(2, 2, 2),
            fno_padding=0,
            fno_projection_dim=16,
            use_coords=True,
        ),
        "afno3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            patch_size=(2, 2, 1),
            afno_embed_dim=8,
            afno_depth=1,
            afno_num_blocks=2,
            afno_hidden_size_factor=1,
            afno_mlp_ratio=2.0,
            afno_dropout=0.0,
            afno_drop_path_rate=0.0,
            afno_sparsity_threshold=0.0,
            afno_hard_thresholding_fraction=1.0,
            afno_double_skip=True,
            use_coords=True,
            grid_shape=spatial_shape,
        ),
    }

    for model_name, builder in model_builders.items():
        model_cls = get_model_class(model_name)
        model = builder(model_cls)
        output = model(inputs, coords=coords)
        assert output.shape == (batch_size, pred_len, channels, *spatial_shape)


def test_grid3d_models_classification_output_shape_contract():
    batch_size, history_len, pred_len = 1, 4, 2
    channels = 3
    risk_num_classes = 3
    risk_num_heads = 4
    classification_channels = risk_num_classes * risk_num_heads
    spatial_shape = (8, 8, 4)
    inputs = torch.randn(batch_size, history_len, channels, *spatial_shape)
    coords = torch.randn(batch_size, 3, *spatial_shape)

    model_builders = {
        "unet3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            base_channels=4,
            patch_size=(1, 1, 1),
            downsample_scale=(1, 1, 1),
            kernel_size=(3, 3, 3),
            expansion=2,
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
        ),
        "unet3d_engram": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            base_channels=4,
            patch_size=(1, 1, 1),
            downsample_scale=(1, 1, 1),
            kernel_size=(3, 3, 3),
            expansion=2,
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
            use_engram=False,
        ),
        "unet3d_patchcat": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            base_channels=4,
            input_embed_dim=8,
            patch_size=(1, 1, 1),
            downsample_scale=(1, 1, 1),
            kernel_size=(3, 3, 3),
            expansion=2,
            use_coords=True,
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
        ),
        "patchstg_flat3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            patch_size_3d=(2, 2, 1),
            embed_dim=16,
            depth=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            use_coords=True,
            spatial_downsample_factor_3d=(1, 1, 1),
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
        ),
        "fredn_multivariate3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            embed_size=8,
            hidden_size=16,
            hidden_layers=1,
            dropout=0.0,
            use_revin=True,
            revin_affine=True,
            revin_subtract_last=False,
            voxel_chunk_size=128,
            spatial_downsample_factor_3d=(1, 1, 1),
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
        ),
        "fno3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            fno_width=8,
            fno_layers=2,
            fno_modes=(2, 2, 2),
            fno_padding=0,
            fno_projection_dim=16,
            use_coords=True,
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
        ),
        "afno3d": lambda cls: cls(
            history_len=history_len,
            pred_len=pred_len,
            in_channels=channels,
            coord_channels=3,
            patch_size=(2, 2, 1),
            afno_embed_dim=8,
            afno_depth=1,
            afno_num_blocks=2,
            afno_hidden_size_factor=1,
            afno_mlp_ratio=2.0,
            afno_dropout=0.0,
            afno_drop_path_rate=0.0,
            afno_sparsity_threshold=0.0,
            afno_hard_thresholding_fraction=1.0,
            afno_double_skip=True,
            use_coords=True,
            output_mode="classification",
            risk_num_classes=risk_num_classes,
            risk_num_heads=risk_num_heads,
            grid_shape=spatial_shape,
        ),
    }

    for model_name, builder in model_builders.items():
        model_cls = get_model_class(model_name)
        model = builder(model_cls)
        output = model(inputs, coords=coords)
        assert output.shape == (batch_size, pred_len, classification_channels, *spatial_shape)
