import torch

from .mtsf import MTSFTask


class _BaseGridTSFTask(MTSFTask):
    spatial_ndim = None

    def __init__(self, grid_mask=None, coord=None, **kwargs):
        super().__init__(**kwargs)
        if grid_mask is not None:
            grid_mask = torch.as_tensor(grid_mask, dtype=torch.float32)
        if coord is not None:
            coord = torch.as_tensor(coord, dtype=torch.float32)
        self.register_buffer("grid_mask", grid_mask, persistent=False)
        self.register_buffer("coord", coord, persistent=False)

    def _validate_grid_dim(self, var_x):
        if self.spatial_ndim is None:
            return
        expected_var_dim = self.spatial_ndim + 3
        if var_x.dim() != expected_var_dim:
            raise ValueError(
                "{} expects grid batch tensors with {} dims, but received shape {}".format(
                    self.__class__.__name__,
                    expected_var_dim,
                    tuple(var_x.shape),
                )
            )

    def _validate_prediction_shape(self, prediction, label):
        if prediction.shape != label.shape:
            raise ValueError(
                "{} model output shape {} does not match label shape {}".format(
                    self.__class__.__name__,
                    tuple(prediction.shape),
                    tuple(label.shape),
                )
            )

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, _ = self._prepare_batch(batch)
        self._validate_grid_dim(var_x)
        label = var_y[:, -self.hparams.pred_len:, ...]
        prediction = self.model(var_x, marker_x, self.grid_mask, self.coord)
        self._validate_prediction_shape(prediction, label)
        return prediction, label


class Grid2DTSFTask(_BaseGridTSFTask):
    spatial_ndim = 2


class Grid3DTSFTask(_BaseGridTSFTask):
    spatial_ndim = 3


class GridSTFTask(_BaseGridTSFTask):
    pass
