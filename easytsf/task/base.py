from __future__ import annotations

import inspect
from pathlib import Path

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from easytsf.data.scaler import StandardScaler, load_standard_scaler_stats, resolve_dataset_scaler_policy
from easytsf.model import get_model_class


class BaseForecastTask(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self._setup_task_state()
        self.model = self._instantiate_registered_model()
        self.loss_function = self._build_loss_function()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()

    def _setup_task_state(self):
        self.dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        self.scaler_policy = resolve_dataset_scaler_policy(
            self.dataset_dir,
            data_is_standardized=getattr(self.hparams, "data_is_standardized"),
        )
        self.scaler = self._build_scaler()
        return None

    def _build_scaler(self):
        if self.scaler_policy.requires_forward_transform:
            mean, std = self._fit_scaler_stats()
        else:
            mean, std = load_standard_scaler_stats(self.scaler_policy.stats_path)
        return StandardScaler(mean, std)

    def _fit_scaler_stats(self):
        mmap_mode = "r" if bool(getattr(self.hparams, "use_mmap", False)) else None
        # Default data layout is assumed to be [L, C, ...].
        train_variable = np.load(self.dataset_dir / "train_data.npy", mmap_mode=mmap_mode, allow_pickle=False)
        if train_variable.ndim < 2:
            raise ValueError(
                "train_data.npy must use layout T,C,..., got shape {}".format(
                    tuple(int(size) for size in train_variable.shape)
                )
            )
        reduce_dims = tuple(axis for axis in range(train_variable.ndim) if axis != 1)
        mean = np.asarray(train_variable.mean(axis=reduce_dims, dtype=np.float64), dtype=np.float32)
        std = np.asarray(train_variable.std(axis=reduce_dims, dtype=np.float64), dtype=np.float32)
        stats_shape = [1] * train_variable.ndim
        stats_shape[1] = train_variable.shape[1]
        mean = mean.reshape(stats_shape)
        std = std.reshape(stats_shape)
        std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)
        return mean, std

    def _build_loss_function(self):
        return nn.MSELoss()

    def _instantiate_registered_model(self):
        model_name = self.hparams.model
        model_cls = get_model_class(model_name)
        model_args = {}
        for name, parameter in inspect.signature(model_cls.__init__).parameters.items():
            if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if hasattr(self.hparams, name):
                model_args[name] = getattr(self.hparams, name)
        return model_cls(**model_args)

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        return (
            self.scaler.inverse_transform(prediction, mask=targets_mask),
            self.scaler.inverse_transform(label, mask=targets_mask),
        )

    def preprocess_batch(self, batch):
        processed_batch = dict(batch)
        if self.scaler_policy.requires_forward_transform:
            for key in ("inputs", "targets"):
                value = processed_batch.get(key)
                if value is None:
                    continue
                processed_batch[key] = self.scaler.transform(value)
        return processed_batch

    def _apply(self, fn):
        """Apply a function to all tensors in the module, including scaler stats.
        
        This method is called by PyTorch when moving the module to a different device
        or changing its dtype. We override it to ensure scaler mean/std tensors are
        also moved to the correct device.
        
        Args:
            fn: Function to apply to all tensors
            
        Returns:
            self
        """
        super()._apply(fn)
        scaler = getattr(self, "scaler", None)
        if scaler is not None:
            scaler.set_stats(fn(scaler.mean), fn(scaler.std))
        return self

    def _forward(self, batch):
        raise NotImplementedError

    def _iter_test_metric_pairs(self, batch, prediction, label):
        metric_space = getattr(self.hparams, "test_metric_space", "original")
        if metric_space == "original":
            prediction, label = self.postprocess_outputs(prediction, label)
        yield prediction, label

    def training_step(self, batch, batch_idx):
        prediction, label = self._forward(batch)
        loss = self.loss_function(prediction, label)
        aux_loss = self.model.get_aux_loss() if hasattr(self.model, "get_aux_loss") else None
        if aux_loss is not None:
            loss = loss + getattr(self.hparams, "aux_loss_weight", 1.0) * aux_loss
            self.log("train/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, label = self._forward(batch)
        loss = self.loss_function(prediction, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_test_epoch_start(self):
        self.test_mae.reset()
        self.test_mse.reset()

    def test_step(self, batch, batch_idx):
        prediction, label = self._forward(batch)
        for pred_item, label_item in self._iter_test_metric_pairs(batch, prediction, label):
            pred_item = pred_item.contiguous()
            label_item = label_item.contiguous()
            self.test_mae.update(pred_item, label_item)
            self.test_mse.update(pred_item, label_item)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_params = (
            self.model.get_param_groups(default_lr=self.hparams.lr)
            if hasattr(self.model, "get_param_groups")
            else self.parameters()
        )
        optimizer_name = self.hparams.optimizer
        optimizer_kwargs = {"lr": self.hparams.lr}
        if optimizer_name == "Adam":
            optimizer_cls = torch.optim.Adam
        elif optimizer_name == "AdamW":
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs.update(betas=(0.9, 0.95), weight_decay=1e-5)
        else:
            raise ValueError("supported optimizers: Adam, AdamW; got {}".format(optimizer_name))
        optimizer = optimizer_cls(optimizer_params, **optimizer_kwargs)

        scheduler_name = self.hparams.lr_scheduler
        if scheduler_name == "StepLR":
            scheduler = {"scheduler": lrs.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)}
        elif scheduler_name == "OneCycleLR":
            steps_per_epoch = getattr(self.hparams, "steps_per_epoch", None)
            if steps_per_epoch is None:
                raise ValueError("steps_per_epoch is required for OneCycleLR")
            scheduler = {
                "scheduler": lrs.OneCycleLR(
                    optimizer,
                    max_lr=self.hparams.lr,
                    pct_start=self.hparams.lrs_pct_start,
                    epochs=self.hparams.max_epochs,
                    steps_per_epoch=steps_per_epoch,
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            raise ValueError("supported lr_scheduler: StepLR, OneCycleLR; got {}".format(scheduler_name))

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
