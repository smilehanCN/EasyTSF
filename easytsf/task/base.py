from __future__ import annotations

import json
import inspect
from collections.abc import Iterable
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
        self.model = self._instantiate_registered_model(self._get_model_derived_args())
        self.loss_function = self._build_loss_function()
        self.test_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()

    def _setup_task_state(self):
        if self._enable_base_single_scaler():
            self.dataset_dir = Path(self.hparams.data_root).expanduser() / str(self.hparams.dataset)
            self.meta = self._load_task_meta()
            self.scaler_policy = resolve_dataset_scaler_policy(self.dataset_dir, meta=self.meta)
            self.scaler = self._build_scaler()
        self._setup_additional_task_state()
        return None

    def _setup_additional_task_state(self):
        return None

    def _enable_base_single_scaler(self):
        return False

    def _load_task_meta(self):
        meta_path = self.dataset_dir / "meta.json"
        if not meta_path.exists():
            return {}
        with meta_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _get_model_derived_args(self):
        return {}

    def _iter_standard_scalers(self) -> Iterable[StandardScaler]:
        scaler = getattr(self, "scaler", None)
        if scaler is None:
            return ()
        return (scaler,)

    def _get_scaler_fit_reduce_dims(self):
        raise NotImplementedError("single-scaler tasks must implement _get_scaler_fit_reduce_dims")

    def _format_single_scaler_stats(self, mean, std):
        raise NotImplementedError("single-scaler tasks must implement _format_single_scaler_stats")

    def _get_preprocess_float_keys(self):
        return ()

    def _get_preprocess_scale_keys(self):
        return ()

    def _build_scaler(self):
        if not self._enable_base_single_scaler():
            raise NotImplementedError("tasks without base single-scaler support must implement _build_scaler")

        if self.scaler_policy.requires_forward_transform:
            mean, std = self._fit_single_scaler_stats()
        else:
            mean, std = load_standard_scaler_stats(self.scaler_policy.stats_path)
        mean, std = self._format_single_scaler_stats(mean, std)
        return StandardScaler(mean, std)

    def _fit_single_scaler_stats(self):
        mmap_mode = "r" if bool(getattr(self.hparams, "use_mmap", False)) else None
        train_variable = np.load(self.dataset_dir / "train_data.npy", mmap_mode=mmap_mode, allow_pickle=False)
        reduce_dims = self._get_scaler_fit_reduce_dims()
        mean = np.asarray(train_variable.mean(axis=reduce_dims, dtype=np.float64), dtype=np.float32)
        std = np.asarray(train_variable.std(axis=reduce_dims, dtype=np.float64), dtype=np.float32)
        std = np.where(std == 0.0, 1.0, std).astype(np.float32, copy=False)
        return mean, std

    def _build_loss_function(self):
        return nn.MSELoss()

    def _instantiate_registered_model(self, derived_args=None):
        model_name = self.hparams.model
        model_cls = get_model_class(model_name)
        model_args = {}
        derived_args = dict(derived_args or {})
        for name, parameter in inspect.signature(model_cls.__init__).parameters.items():
            if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if hasattr(self.hparams, name):
                model_args[name] = getattr(self.hparams, name)
                continue
            if name in derived_args:
                model_args[name] = derived_args[name]
        return model_cls(**model_args)

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        if not self._enable_base_single_scaler():
            raise NotImplementedError("tasks without base single-scaler support must implement postprocess_outputs")
        return (
            self.scaler.inverse_transform(prediction, mask=targets_mask),
            self.scaler.inverse_transform(label, mask=targets_mask),
        )

    def preprocess_batch(self, batch):
        if not self._enable_base_single_scaler():
            raise NotImplementedError("tasks without base single-scaler support must implement preprocess_batch")

        processed_batch = dict(batch)
        for key in self._get_preprocess_float_keys():
            value = processed_batch.get(key)
            if value is None or value.dtype == torch.float32:
                continue
            processed_batch[key] = value.float()

        if self.scaler_policy.requires_forward_transform:
            for key in self._get_preprocess_scale_keys():
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
        for scaler in self._iter_standard_scalers():
            if scaler is None:
                continue
            scaler.set_stats(fn(scaler.mean), fn(scaler.std))
        return self

    def _forward(self, batch):
        raise NotImplementedError

    def _compute_validation_loss(self, batch, prediction, label):
        del batch
        return self.loss_function(prediction, label), {}

    def _iter_test_metric_pairs(self, batch, prediction, label):
        del batch
        metric_space = getattr(self.hparams, "test_metric_space", "original")
        if metric_space == "original":
            prediction, label = self.postprocess_outputs(prediction, label)
        yield prediction, label

    def training_step(self, batch, batch_idx):
        del batch_idx
        prediction, label = self._forward(batch)
        loss = self.loss_function(prediction, label)
        aux_loss = self.model.get_aux_loss() if hasattr(self.model, "get_aux_loss") else None
        if aux_loss is not None:
            loss = loss + getattr(self.hparams, "aux_loss_weight", 1.0) * aux_loss
            self.log("train/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        prediction, label = self._forward(batch)
        loss, log_kwargs = self._compute_validation_loss(batch, prediction, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, **dict(log_kwargs))
        return loss

    def on_test_epoch_start(self):
        self.test_mae.reset()
        self.test_mse.reset()

    def test_step(self, batch, batch_idx):
        del batch_idx
        prediction, label = self._forward(batch)
        for pred_item, label_item in self._iter_test_metric_pairs(batch, prediction, label):
            pred_item = pred_item.contiguous()
            label_item = label_item.contiguous()
            self.test_mae.update(pred_item, label_item)
            self.test_mse.update(pred_item, label_item)
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
