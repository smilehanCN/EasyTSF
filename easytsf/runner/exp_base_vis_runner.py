import importlib
import inspect
import os

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs

from easytsf.runner.data_runner import load_dataset_stats


def mean_absolute_percentage_error(y_true: torch.Tensor, y_pred: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    计算MAPE（平均绝对百分比误差）

    参数:
        y_true (torch.Tensor): 真实值张量
        y_pred (torch.Tensor): 预测值张量
        epsilon (float): 用于避免除零的小值，默认1e-8

    返回:
        torch.Tensor: MAPE，百分比形式
    """
    # 确保y_true和y_pred形状相同
    if y_true.shape != y_pred.shape:
        raise ValueError("形状不匹配：y_true和y_pred必须有相同的形状")

    # 计算绝对值的实际值，并限制最小值以避免除零
    abs_y_true = torch.clamp(torch.abs(y_true), min=epsilon)

    # 计算每个元素的绝对百分比误差
    absolute_percentage_errors = torch.abs((y_pred - y_true) / abs_y_true)

    # 计算均值并转换为百分比
    mape = 100.0 * torch.mean(absolute_percentage_errors)

    return mape


def weighted_absolute_percentage_error(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        epsilon: float = 1e-8
) -> torch.Tensor:
    """
    计算WAPE（加权平均绝对百分比误差），公式为：总绝对误差 / 总真实值绝对值

    参数:
        y_true (torch.Tensor): 真实值张量
        y_pred (torch.Tensor): 预测值张量
        epsilon (float): 用于避免除零的小值，默认1e-8

    返回:
        torch.Tensor: WAPE，百分比形式
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("形状不匹配：y_true和y_pred必须有相同的形状")

    # 计算总绝对误差和总真实值绝对值
    total_absolute_error = torch.sum(torch.abs(y_pred - y_true))
    total_absolute_y_true = torch.sum(torch.abs(y_true))

    # 避免除零：如果真实值全为零且预测正确，返回0.0
    denominator = torch.clamp(total_absolute_y_true, min=epsilon)

    # 计算WAPE并转换为百分比
    wape = 100.0 * (total_absolute_error / denominator)

    return wape


class VisLTSFRunner(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()

        mean, std = load_dataset_stats(
            os.path.join(self.hparams.data_root, '{}.npz'.format(self.hparams.dataset_name)),
            use_mmap=getattr(self.hparams, "use_mmap", False),
            cache_npz_as_npy=getattr(self.hparams, "cache_npz_as_npy", None),
        )
        self.register_buffer('mean', torch.tensor(mean).float())
        self.register_buffer('std', torch.tensor(std).float())
        self.test_result = []

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        label = var_y[:, -self.hparams.pred_len:, :]

        prediction = self.model(var_x, marker_x)
        return prediction, label

    def training_step(self, batch, batch_idx):
        if self.hparams.use_mix_loss:
            prediction, label = self.forward(batch, batch_idx)
            loss = 0.5 * self.mae_loss_func(prediction, label) + 0.5 * self.mse_loss_func(prediction, label)
        else:
            loss = self.loss_function(*self.forward(batch, batch_idx))
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx))
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        var_x, _, _, _ = [_.float() for _ in batch]
        prediction, label = self.forward(batch, batch_idx)
        raw_history = self.inverse_transform_var(var_x)
        raw_prediction = self.inverse_transform_var(prediction)
        raw_label = self.inverse_transform_var(label)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)

        if self.hparams.save_samples and batch_idx < 1000:
            self.test_result.append({
                "raw_history": raw_history.cpu(),
                "raw_prediction": raw_prediction.cpu(),
                "raw_label": raw_label.cpu(),
                "scaled_prediction": prediction.cpu(),
                "scaled_label": label.cpu(),
            })

    def on_test_epoch_end(self):
        if self.hparams.save_samples:
            raw_history = torch.cat([batch['raw_history'] for batch in self.test_result])
            raw_prediction = torch.cat([batch['raw_prediction'] for batch in self.test_result])
            raw_target = torch.cat([batch['raw_label'] for batch in self.test_result])
            scaled_prediction = torch.cat([batch['scaled_prediction'] for batch in self.test_result])
            scaled_label = torch.cat([batch['scaled_label'] for batch in self.test_result])
            save_path = os.path.join(
                self.hparams.save_root,
                "{}_{}".format(self.hparams.model_name, self.hparams.dataset_name),
                self.hparams.conf_hash,'seed_{}'.format(self.hparams.seed), 'test_output.npz')
            print(save_path)
            np.savez(
                save_path,
                history=raw_history.numpy(),
                prediction=raw_prediction.numpy(),
                target=raw_target.numpy(),
                scaled_prediction=scaled_prediction.numpy(),
                scaled_label=scaled_label.numpy(),
            )

    def configure_loss(self):
        self.loss_function = nn.MSELoss()
        self.mae_loss_func = nn.L1Loss()
        self.mse_loss_func = nn.MSELoss()

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
                    return 1.0 - (epoch + 1 - self.hparams.lr_stable_end_epochs) / (
                            self.hparams.max_epochs - self.hparams.lr_stable_end_epochs)

            lr_scheduler = {
                "scheduler": lrs.LambdaLR(optimizer, lr_lambda=wsd_lr_lambda),
            }
        elif self.hparams.lr_scheduler == 'OneCycleLR':
            lr_scheduler = {
                "scheduler": lrs.OneCycleLR(
                    optimizer, max_lr=self.hparams.lr, pct_start=self.hparams.lrs_pct_start, epochs=self.hparams.max_epochs, steps_per_epoch=self.hparams.steps_per_epoch),
                # "monitor": self.hparams.val_metric
                'interval': 'step',  # 在每一步更新
                'frequency': 1,
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
