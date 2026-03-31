import torch

from easytsf.task.mtsf import MTSFTask


class IdentityScaler:
    def transform(self, input_data, mask=None):
        del mask
        return input_data

    def inverse_transform(self, input_data, mask=None):
        del mask
        return input_data

    def set_stats(self, mean, std):
        del mean, std
        return self


class NonContiguousPredictionModel(torch.nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, var_x, marker_x, marker_y):
        del marker_x, marker_y
        batch_size, _, var_num = var_x.shape
        prediction = torch.randn(batch_size, var_num, self.pred_len, dtype=var_x.dtype, device=var_x.device)
        prediction = prediction.permute(0, 2, 1)
        assert not prediction.is_contiguous()
        return prediction


def test_test_step_accepts_non_contiguous_predictions(monkeypatch):
    monkeypatch.setattr(MTSFTask, "_build_scaler", lambda self: IdentityScaler())
    monkeypatch.setattr(MTSFTask, "_build_model", lambda self: NonContiguousPredictionModel(pred_len=2))

    task = MTSFTask(
        task="mtsf",
        model="dummy",
        data_root="unused",
        dataset="unused",
        hist_len=4,
        pred_len=2,
        var_num=3,
        optimizer="Adam",
        lr=1e-3,
        lr_scheduler="StepLR",
        lr_step_size=1,
        lr_gamma=0.1,
    )
    task.log = lambda *args, **kwargs: None

    batch = {
        "inputs": torch.randn(2, 4, 3),
        "inputs_timestamps": torch.randn(2, 4, 1),
        "targets": torch.randn(2, 2, 3),
        "targets_timestamps": torch.randn(2, 2, 1),
    }

    task.test_step(batch, 0)

    assert torch.isfinite(task.test_mae.compute())
