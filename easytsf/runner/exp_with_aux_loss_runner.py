import os

import numpy as np
import torch

from easytsf.runner.exp_base_runner import LTSFRunner


class LTSFwithAuxLossRunner(LTSFRunner):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        stat = np.load(os.path.join(self.hparams.data_root, '{}.npz'.format(self.hparams.dataset_name)))
        self.register_buffer('mean', torch.tensor(stat['mean']).float())
        self.register_buffer('std', torch.tensor(stat['std']).float())

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = self._prepare_batch(batch)
        label = var_y[:, -self.hparams.pred_len:, :]
        prediction, aux_loss = self.model(var_x, marker_x)
        return prediction, label, aux_loss

    def training_step(self, batch, batch_idx):
        prediction, label, aux_loss = self.forward(batch, batch_idx)
        # pred_loss = 0.5 * self.mae_loss_func(prediction, label) + 0.5 * self.mse_loss_func(prediction, label)
        pred_loss = self.mse_loss_func(prediction, label)
        self.log('train/loss', pred_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return pred_loss + aux_loss

    def validation_step(self, batch, batch_idx):
        prediction, label, aux_loss = self.forward(batch, batch_idx)
        pred_loss = self.loss_function(prediction, label)
        self.log('val/loss', pred_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return pred_loss + aux_loss

    def test_step(self, batch, batch_idx):
        prediction, label, aux_loss = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)
