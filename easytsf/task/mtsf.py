import importlib
import inspect
import warnings

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs

from easytsf.model import get_model_contract


class MTSFTask(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.data_spec = kwargs.get("data_spec")
        self.save_hyperparameters(ignore=["graph", "grid_mask", "coord", "data_spec"])
        self.model = self._build_model()
        self.loss_function = nn.MSELoss()
        self.mae_loss_func = nn.L1Loss()
        self.mse_loss_func = nn.MSELoss()

    @staticmethod
    def _prepare_batch(batch):
        if isinstance(batch, dict):
            inputs = batch["inputs"]
            targets = batch["targets"]
            inputs_timestamps = batch.get("inputs_timestamps")
            targets_timestamps = batch.get("targets_timestamps")

            if inputs_timestamps is None:
                inputs_timestamps = inputs.new_empty((inputs.shape[0], inputs.shape[1], 0))
            if targets_timestamps is None:
                targets_timestamps = targets.new_empty((targets.shape[0], targets.shape[1], 0))
            tensors = (inputs, inputs_timestamps, targets, targets_timestamps)
        else:
            tensors = batch
        return tuple(tensor if tensor.dtype == torch.float32 else tensor.float() for tensor in tensors)

    def _build_model(self):
        model_name = self.hparams.model_name
        contract = get_model_contract(model_name)
        contract.validate(getattr(self.hparams, "task_name", "mtsf"), self.data_spec)
        if contract.is_legacy:
            warnings.warn(
                "model '{}' is marked as legacy and is outside the maintained preset/smoke matrix. {}".format(
                    model_name,
                    contract.note,
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        module_name = contract.module_name
        module = importlib.import_module(".{}".format(module_name), package="easytsf.model")
        if not hasattr(module, "Model"):
            raise ValueError("easytsf.model.{} must define a top-level Model class".format(module_name))
        model_cls = getattr(module, "Model")
        model_args = {}
        for arg in inspect.getfullargspec(model_cls.__init__).args[1:]:
            if hasattr(self.hparams, arg):
                model_args[arg] = getattr(self.hparams, arg)
        return model_cls(**model_args)

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, _ = self._prepare_batch(batch)
        label = var_y[:, -self.hparams.pred_len:, :]
        prediction = self.model(var_x, marker_x)
        return prediction, label

    def training_step(self, batch, batch_idx):
        prediction, label = self.forward(batch, batch_idx)
        if getattr(self.hparams, "use_mix_loss", False):
            loss = 0.5 * self.mae_loss_func(prediction, label) + 0.5 * self.mse_loss_func(prediction, label)
        else:
            loss = self.loss_function(prediction, label)
        aux_loss = self.model.get_aux_loss() if hasattr(self.model, "get_aux_loss") else None
        if aux_loss is not None:
            loss = loss + getattr(self.hparams, "aux_loss_weight", 1.0) * aux_loss
            self.log("train/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, label = self.forward(batch, batch_idx)
        loss = self.loss_function(prediction, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        prediction, label = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        self.log("test/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/mse", mse, on_step=False, on_epoch=True, sync_dist=True)

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
        elif self.hparams.lr_scheduler == "WSD":
            if not self.hparams.lr_warmup_end_epochs < self.hparams.lr_stable_end_epochs < self.hparams.max_epochs:
                raise ValueError("WSD scheduler requires lr_warmup_end_epochs < lr_stable_end_epochs < max_epochs")

            def wsd_lr_lambda(epoch):
                if epoch < self.hparams.lr_warmup_end_epochs:
                    return (epoch + 1) / self.hparams.lr_warmup_end_epochs
                if epoch < self.hparams.lr_stable_end_epochs:
                    return 1.0
                return 1.0 - (
                    (epoch + 1 - self.hparams.lr_stable_end_epochs)
                    / (self.hparams.max_epochs - self.hparams.lr_stable_end_epochs + 1)
                )

            scheduler = {"scheduler": lrs.LambdaLR(optimizer, lr_lambda=wsd_lr_lambda)}
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
        elif self.hparams.lr_scheduler == "CycleNetLRS":
            def lr_lambda(epoch):
                if epoch < 3:
                    return 1
                return 0.8 ** (epoch - 3)

            scheduler = {"scheduler": lrs.LambdaLR(optimizer, lr_lambda=lr_lambda)}
        else:
            raise ValueError("invalid lr_scheduler type: {}".format(self.hparams.lr_scheduler))

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
