import pytest
import torch

from easytsf.model.arrow import Model


def test_arrow_model_surfaces_missing_optional_dependencies():
    with pytest.raises(ImportError, match="timm|xformers"):
        Model(
            hist_len=1,
            pred_len=1,
            height=2,
            width=2,
            input_channel_names=("t2m", "u10"),
            target_channel_names=("t2m",),
        )


class DummyBackbone(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls = []
        self.moe_noises = None

    def forward(self, x, variables=None, time_interval=None, static_inputs=None):
        self.calls.append(
            {
                "x": x.detach().clone(),
                "variables": tuple(variables or ()),
                "time_interval": time_interval.detach().clone(),
                "static_inputs": None if static_inputs is None else static_inputs.detach().clone(),
            }
        )
        return torch.ones_like(x)


class DummyScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, input_data, mask=None):
        output = (input_data - self.mean) / self.std
        if mask is None:
            return output
        return torch.where(mask, output, input_data)

    def inverse_transform(self, input_data, mask=None):
        output = input_data * self.std + self.mean
        if mask is None:
            return output
        return torch.where(mask, output, input_data)


def test_arrow_model_rollout_uses_interval_scalers_and_target_subset(monkeypatch):
    import easytsf.model.arrow as arrow_module

    monkeypatch.setattr(arrow_module, "ensure_arrow_dependencies_available", lambda: None)
    monkeypatch.setattr(arrow_module, "ArrowBackbone", DummyBackbone)

    model = Model(
        hist_len=1,
        pred_len=2,
        height=2,
        width=2,
        input_channel_names=("t2m", "u10"),
        target_channel_names=("u10",),
        static_channel_names=("lsm",),
        static_var_num=1,
        arrow_train_intervals=(6, 12),
    )

    input_scaler = DummyScaler(torch.zeros(1, 2, 1, 1), torch.ones(1, 2, 1, 1))
    interval_diff_scalers = {
        6: DummyScaler(torch.zeros(1, 2, 1, 1), torch.full((1, 2, 1, 1), 2.0)),
        12: DummyScaler(torch.zeros(1, 2, 1, 1), torch.full((1, 2, 1, 1), 3.0)),
    }

    var_x = torch.zeros(1, 1, 2, 2, 2)
    marker_x = torch.tensor([[0]], dtype=torch.int64)
    marker_y = torch.tensor([[6, 18]], dtype=torch.int64) * (3600 * 10**9)
    static_inputs = torch.ones(1, 1, 2, 2)
    prediction = model(
        var_x,
        marker_x,
        marker_y,
        static_inputs=static_inputs,
        target_input_indices=torch.tensor([1]),
        interval_diff_scalers=interval_diff_scalers,
        input_state_scaler=input_scaler,
    )

    assert prediction.shape == (1, 2, 1, 2, 2)
    assert torch.allclose(prediction[:, 0], torch.full((1, 1, 2, 2), 2.0))
    assert torch.allclose(prediction[:, 1], torch.full((1, 1, 2, 2), 5.0))
    assert len(model.backbone.calls) == 2
    assert torch.allclose(model.backbone.calls[0]["time_interval"], torch.tensor([0.6]))
    assert torch.allclose(model.backbone.calls[1]["time_interval"], torch.tensor([1.2]))
    assert model.backbone.calls[0]["static_inputs"].shape == (1, 1, 2, 2)


def test_arrow_model_rejects_intervals_outside_config(monkeypatch):
    import easytsf.model.arrow as arrow_module

    monkeypatch.setattr(arrow_module, "ensure_arrow_dependencies_available", lambda: None)
    monkeypatch.setattr(arrow_module, "ArrowBackbone", DummyBackbone)

    model = Model(
        hist_len=1,
        pred_len=1,
        height=2,
        width=2,
        input_channel_names=("t2m",),
        target_channel_names=("t2m",),
        arrow_train_intervals=(6, 12),
    )

    with pytest.raises(ValueError, match="configured"):
        model(
            torch.zeros(1, 1, 1, 2, 2),
            torch.tensor([[0]], dtype=torch.int64),
            torch.tensor([[18]], dtype=torch.int64) * (3600 * 10**9),
            interval_diff_scalers={6: DummyScaler(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1))},
            input_state_scaler=DummyScaler(torch.zeros(1, 1, 1, 1), torch.ones(1, 1, 1, 1)),
        )
