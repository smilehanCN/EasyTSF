import inspect
from pathlib import Path

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from easytsf.model import get_model_class
from easytsf.data.scaler import StandardScaler


class MTSFTask(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.scaler = self._build_scaler()
        self.model = self._build_model()
        self.loss_function = nn.MSELoss()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_rmse = MeanSquaredError(squared=False)

    def _build_model(self):
        model_name = self.hparams.model
        model_cls = get_model_class(model_name)
        model_args = {}
        for name, parameter in inspect.signature(model_cls.__init__).parameters.items():
            if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if hasattr(self.hparams, name):
                model_args[name] = getattr(self.hparams, name)
                continue
            if parameter.default is inspect.Parameter.empty:
                raise ValueError("config must define required model argument '{}' for {}".format(name, model_name))
        return model_cls(**model_args)

    def _build_scaler(self):
        dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        mmap_mode = "r" if bool(getattr(self.hparams, "use_mmap", False)) else None
        train_variable = np.load(dataset_dir / "train_data.npy", mmap_mode=mmap_mode, allow_pickle=False)
        return StandardScaler.fit(train_variable)

    def _apply(self, fn):
        super()._apply(fn)
        self.scaler.set_stats(fn(self.scaler.mean), fn(self.scaler.std))
        return self

    def preprocess_batch(self, batch):
        var_x = batch["inputs"]
        marker_x = batch.get("inputs_timestamps")
        var_y = batch["targets"]
        marker_y = batch.get("targets_timestamps")
        tensors = (var_x, marker_x, var_y, marker_y)
        var_x, marker_x, var_y, marker_y = tuple(
            tensor if tensor.dtype == torch.float32 else tensor.float() for tensor in tensors
        )
        return (
            self.scaler.transform(var_x),
            marker_x,
            self.scaler.transform(var_y),
            marker_y,
        )

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        return (
            self.scaler.inverse_transform(prediction, mask=targets_mask),
            self.scaler.inverse_transform(label, mask=targets_mask),
        )

    def _forward(self, batch):
        var_x, marker_x, var_y, marker_y = self.preprocess_batch(batch)
        label = var_y[:, -self.hparams.pred_len :, ...]

        prediction = self.model(var_x, marker_x, marker_y)
        return prediction, label

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

    def test_step(self, batch, batch_idx):
        prediction, label = self._forward(batch)
        metric_space = getattr(self.hparams, "test_metric_space", "original")
        if metric_space == "original":
            prediction, label = self.postprocess_outputs(prediction, label)
        prediction = prediction.contiguous()
        label = label.contiguous()
        self.test_mae.update(prediction, label)
        self.test_mse.update(prediction, label)
        self.test_rmse.update(prediction, label)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if hasattr(self.model, "get_param_groups"):
            param_groups = self.model.get_param_groups(default_lr=self.hparams.lr)
        else:
            param_groups = None
        optimizer_params = param_groups if param_groups else self.parameters()
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(optimizer_params, lr=self.hparams.lr)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=self.hparams.lr,
                betas=(0.9, 0.95),
                weight_decay=1e-5,
            )
        else:
            raise ValueError("invalid optimizer type: {}".format(self.hparams.optimizer))

        if self.hparams.lr_scheduler == "StepLR":
            scheduler = {"scheduler": lrs.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)}
        elif self.hparams.lr_scheduler == "MultiStepLR":
            scheduler = {"scheduler": lrs.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.gamma)}
        elif self.hparams.lr_scheduler == "ReduceLROnPlateau":
            scheduler = {
                "scheduler": lrs.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.hparams.lrs_factor,
                    patience=self.hparams.lrs_patience,
                ),
                "monitor": self.hparams.val_metric,
            }
        elif self.hparams.lr_scheduler == "OneCycleLR":
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
            raise ValueError("invalid lr_scheduler type: {}".format(self.hparams.lr_scheduler))

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
