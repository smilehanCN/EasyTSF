import inspect
import json
from pathlib import Path

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from easytsf.data.scaler import StandardScaler
from easytsf.model import get_model_class


class WeatherBenchTask(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
        with (self.dataset_dir / "meta.json").open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        with (self.dataset_dir / "channels.json").open("r", encoding="utf-8") as handle:
            self.channels = json.load(handle)
        with (self.dataset_dir / "static_channels.json").open("r", encoding="utf-8") as handle:
            self.static_channels = json.load(handle)

        self.input_channel_names = list(getattr(self.hparams, "input_channel_names", None) or self.meta["input_channels"])
        self.target_channel_names = list(getattr(self.hparams, "target_channel_names", None) or self.meta["target_channels"])

        channel_name_to_index = {channel["name"]: int(channel["index"]) for channel in self.channels}
        self.input_channel_indices = [channel_name_to_index[name] for name in self.input_channel_names]
        self.target_channel_indices = [channel_name_to_index[name] for name in self.target_channel_names]
        input_channel_positions = {name: index for index, name in enumerate(self.input_channel_names)}
        if not set(self.target_channel_names).issubset(set(self.input_channel_names)):
            raise ValueError("target_channel_names must be a subset of input_channel_names")

        self.register_buffer(
            "target_input_index_tensor",
            torch.as_tensor(
                [input_channel_positions[name] for name in self.target_channel_names],
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.input_scaler, self.target_scaler = self._build_scalers()
        self.model = self._build_model()
        self.loss_function = nn.MSELoss()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()

    def _build_model(self):
        model_name = self.hparams.model
        model_cls = get_model_class(model_name)
        model_args = {}
        derived_args = {
            "hist_len": int(self.hparams.hist_len),
            "pred_len": int(self.hparams.pred_len),
            "var_num": len(self.input_channel_names),
            "input_var_num": len(self.input_channel_names),
            "target_var_num": len(self.target_channel_names),
            "static_var_num": len(self.static_channels),
            "grid_shape": tuple(self.meta["grid_shape"]),
            "height": int(self.meta["grid_shape"][0]),
            "width": int(self.meta["grid_shape"][1]),
        }
        for name, parameter in inspect.signature(model_cls.__init__).parameters.items():
            if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if hasattr(self.hparams, name):
                model_args[name] = getattr(self.hparams, name)
                continue
            if name in derived_args:
                model_args[name] = derived_args[name]
                continue
            if parameter.default is inspect.Parameter.empty:
                raise ValueError("config must define required model argument '{}' for {}".format(name, model_name))
        return model_cls(**model_args)

    def _build_scalers(self):
        with np.load(self.dataset_dir / "stats.npz") as stats:
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
        if mean.ndim == 1:
            mean = mean[None, :, None, None]
        if std.ndim == 1:
            std = std[None, :, None, None]

        input_mean = mean[:, self.input_channel_indices, :, :]
        input_std = std[:, self.input_channel_indices, :, :]
        target_mean = mean[:, self.target_channel_indices, :, :]
        target_std = std[:, self.target_channel_indices, :, :]
        return StandardScaler(input_mean, input_std), StandardScaler(target_mean, target_std)

    def _apply(self, fn):
        super()._apply(fn)
        self.input_scaler.set_stats(fn(self.input_scaler.mean), fn(self.input_scaler.std))
        self.target_scaler.set_stats(fn(self.target_scaler.mean), fn(self.target_scaler.std))
        return self

    def preprocess_batch(self, batch):
        var_x = batch["inputs"].float()
        marker_x = batch.get("inputs_timestamps")
        var_y = batch["targets"].float()
        marker_y = batch.get("targets_timestamps")
        static_inputs = batch.get("static_inputs")
        if static_inputs is not None:
            static_inputs = static_inputs.float()
        return (
            self.input_scaler.transform(var_x),
            marker_x,
            self.target_scaler.transform(var_y),
            marker_y,
            static_inputs,
        )

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        return (
            self.target_scaler.inverse_transform(prediction, mask=targets_mask),
            self.target_scaler.inverse_transform(label, mask=targets_mask),
        )

    def _forward(self, batch):
        var_x, marker_x, var_y, marker_y, static_inputs = self.preprocess_batch(batch)
        label = var_y[:, -self.hparams.pred_len :, ...]

        model_forward_parameters = inspect.signature(self.model.forward).parameters
        model_kwargs = {}
        if "static_inputs" in model_forward_parameters and static_inputs is not None:
            model_kwargs["static_inputs"] = static_inputs
        if "target_input_indices" in model_forward_parameters:
            model_kwargs["target_input_indices"] = self.target_input_index_tensor

        prediction = self.model(var_x, marker_x, marker_y, **model_kwargs)
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
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, sync_dist=True)

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
