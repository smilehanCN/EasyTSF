import importlib
import inspect
import os

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs


class LTSFMyNetVXRunner(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

        stat = np.load(os.path.join(self.hparams.data_root, '{}.npz'.format(self.hparams.dataset_name)))
        self.register_buffer('mean', torch.tensor(stat['mean']).float())
        self.register_buffer('std', torch.tensor(stat['std']).float())

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = tuple(
            tensor if tensor.dtype == torch.float32 else tensor.float() for tensor in batch
        )
        pred_y, pred_x = self.model(var_x, marker_x)
        return pred_y, pred_x, var_y, var_x

    def training_step(self, batch, batch_idx):
        pred_y, pred_x, var_y, var_x = self.forward(batch, batch_idx)
        pred_mse_loss, pred_mae_loss, pred_patch_loss, pred_cycle_loss, pred_overall_loss = self.model.loss_function(pred_y, var_y)
        if self.model.rec_loss_weight == 0:
            rec_mse_loss, rec_mae_loss, rec_patch_loss, rec_cycle_loss, rec_overall_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            rec_mse_loss, rec_mae_loss, rec_patch_loss, rec_cycle_loss, rec_overall_loss = self.model.loss_function(pred_x, var_x[:, -pred_x.shape[1]:, ...])

        pred_loss = pred_mse_loss + pred_mae_loss + pred_patch_loss + pred_cycle_loss + pred_overall_loss
        rec_loss = rec_mse_loss + rec_mae_loss + rec_patch_loss + rec_cycle_loss + rec_overall_loss
        loss = pred_loss + self.model.rec_loss_weight * rec_loss
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/pred_loss', pred_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/rec_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_y, pred_x, var_y, var_x = self.forward(batch, batch_idx)
        pred_mse_loss, pred_mae_loss, pred_patch_loss, pred_cycle_loss, pred_overall_loss = self.model.loss_function(pred_y, var_y)
        pred_loss = pred_mse_loss + pred_mae_loss + pred_patch_loss + pred_cycle_loss + pred_overall_loss
        if self.model.rec_loss_weight == 0:
            rec_loss = 0.0
        else:
            rec_mse_loss, rec_mae_loss, rec_patch_loss, rec_cycle_loss, rec_overall_loss = self.model.loss_function(pred_x, var_x[:, -pred_x.shape[1]:, ...])
            rec_loss = rec_mse_loss + rec_mae_loss + rec_patch_loss + rec_cycle_loss + rec_overall_loss
        loss = pred_loss + self.model.rec_loss_weight * rec_loss
        self.log('val/mse', pred_mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/pred_loss', pred_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/rec_loss', rec_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred_y, pred_x, var_y, var_x = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(pred_y, var_y)
        mse = torch.nn.functional.mse_loss(pred_y, var_y)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mape', mse, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.95), weight_decay=1e-5)
        else:
            raise ValueError('Invalid optimizer type!')

        if self.hparams.lr_scheduler == 'StepLR':
            lr_scheduler = {
                "scheduler": lrs.StepLR(
                    optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
            }
        elif self.hparams.lr_scheduler == 'MultiStepLR':
            lr_scheduler = {
                "scheduler": lrs.MultiStepLR(
                    optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)
            }
        elif self.hparams.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = {
                "scheduler": lrs.ReduceLROnPlateau(
                    optimizer, mode='min', factor=self.hparams.lrs_factor, patience=self.hparams.lrs_patience),
                "monitor": self.hparams.val_metric
            }
        elif self.hparams.lr_scheduler == 'WSD':
            assert self.hparams.lr_warmup_end_epochs < self.hparams.lr_stable_end_epochs < self.hparams.max_epochs

            def wsd_lr_lambda(epoch):
                if epoch < self.hparams.lr_warmup_end_epochs:
                    return (epoch + 1) / self.hparams.lr_warmup_end_epochs
                if self.hparams.lr_warmup_end_epochs <= epoch < self.hparams.lr_stable_end_epochs:
                    return 1.0
                if self.hparams.lr_stable_end_epochs <= epoch <= self.hparams.max_epochs:
                    return 1.0 - ((epoch + 1 - self.hparams.lr_stable_end_epochs) / (
                            self.hparams.max_epochs - self.hparams.lr_stable_end_epochs + 1))

            lr_scheduler = {
                "scheduler": lrs.LambdaLR(optimizer, lr_lambda=wsd_lr_lambda),
            }
        elif self.hparams.lr_scheduler == 'CycleNetLRS':
            # https://github.com/ACAT-SCUT/CycleNet/blob/ffeaa16ca037951a9d8f981700269f9693a411d4/utils/tools.py#L19
            def lr_lambda(epoch):
                if epoch < 3:
                    return 1
                else:
                    return 0.8 ** ((epoch - 3) // 1)

            lr_scheduler = {
                "scheduler": lrs.LambdaLR(optimizer, lr_lambda=lr_lambda),
            }
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def load_model(self):
        model_name = self.hparams.model_name
        Model = getattr(importlib.import_module('.' + model_name, package='easytsf.model'), model_name)
        self.model = self.instancialize(Model)

    def instancialize(self, Model):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        model_class_args = inspect.getfullargspec(Model.__init__).args[1:]  # 获取模型参数
        interface_args = self.hparams.keys()
        model_args_instance = {}
        for arg in model_class_args:
            if arg in interface_args:
                model_args_instance[arg] = getattr(self.hparams, arg)
        return Model(**model_args_instance)

    def inverse_transform_var(self, data):
        return (data * self.std) + self.mean

    def inverse_transform_time_marker(self, time_marker):
        time_marker[..., 0] = time_marker[..., 0] * (int((24 * 60) / self.hparams.freq - 1))
        time_marker[..., 1] = time_marker[..., 1] * 6
        time_marker[..., 2] = time_marker[..., 2] * 30
        time_marker[..., 3] = time_marker[..., 3] * 365

        if "max_event_per_day" in self.hparams:
            time_marker[..., -1] = time_marker[..., -1] * self.hparams.max_event_per_day

        return time_marker
