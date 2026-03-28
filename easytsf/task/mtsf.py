import importlib
import inspect
import warnings

import lightning.pytorch as L
import torch
import torch.optim.lr_scheduler as lrs

from easytsf.model import get_model_contract
from easytsf.scaler import (
    build_valid_mask,
    fill_invalid_values,
    inverse_transform_by_stats,
    masked_mae,
    masked_mse,
    transform_by_stats,
)


class MTSFTask(L.LightningModule):
    def __init__(self, scaler_stats=None, graph=None, **kwargs):
        super().__init__()
        self.data_spec = kwargs.get("data_spec")
        self._model_side_inputs = {}
        if graph is not None:
            self._model_side_inputs["graph"] = torch.as_tensor(graph, dtype=torch.float32)
        self.save_hyperparameters(ignore=["graph", "grid_mask", "coord", "data_spec", "scaler_stats"])
        self.model = self._build_model()
        self.loss_function = masked_mse
        self.mae_loss_func = masked_mae
        self.mse_loss_func = masked_mse
        self.null_val = kwargs.get("null_val")
        self.rescale = bool(kwargs.get("rescale", False))
        self.null_to_num = 0.0

        mean = None
        std = None
        if scaler_stats is not None:
            if "mean" not in scaler_stats or "std" not in scaler_stats:
                raise ValueError("scaler_stats must define 'mean' and 'std'")
            mean = torch.as_tensor(scaler_stats["mean"], dtype=torch.float32)
            std = torch.as_tensor(scaler_stats["std"], dtype=torch.float32)
        self.register_buffer("scaler_mean", mean, persistent=False)
        self.register_buffer("scaler_std", std, persistent=False)

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
            elif arg in self._model_side_inputs:
                model_args[arg] = self._model_side_inputs[arg]
        return model_cls(**model_args)

    def _make_valid_mask(self, tensor):
        return build_valid_mask(tensor, self.null_val)

    def preprocess_batch(self, batch):
        var_x, marker_x, var_y, marker_y = self._prepare_batch(batch)
        inputs_mask = self._make_valid_mask(var_x)
        targets_mask = self._make_valid_mask(var_y)

        if self.scaler_mean is not None and self.scaler_std is not None:
            var_x = transform_by_stats(var_x, self.scaler_mean, self.scaler_std, mask=inputs_mask)
            var_y = transform_by_stats(var_y, self.scaler_mean, self.scaler_std, mask=targets_mask)

        var_x = fill_invalid_values(var_x, inputs_mask, fill_value=self.null_to_num)
        var_y = fill_invalid_values(var_y, targets_mask, fill_value=self.null_to_num)
        return {
            "inputs": var_x,
            "inputs_timestamps": marker_x,
            "targets": var_y,
            "targets_timestamps": marker_y,
            "inputs_mask": inputs_mask,
            "targets_mask": targets_mask,
        }

    def postprocess_outputs(self, prediction, label, targets_mask):
        if self.rescale and self.scaler_mean is not None and self.scaler_std is not None:
            prediction = inverse_transform_by_stats(prediction, self.scaler_mean, self.scaler_std)
            label = inverse_transform_by_stats(label, self.scaler_mean, self.scaler_std, mask=targets_mask)
        return prediction, label

    def _run_model(self, var_x, marker_x, marker_y):
        return self.model(var_x, marker_x, marker_y)

    def _validate_model_inputs(self, var_x):
        del var_x

    def _validate_prediction_shape(self, prediction, label):
        if prediction.shape != label.shape:
            raise ValueError(
                "{} model output shape {} does not match label shape {}".format(
                    self.__class__.__name__,
                    tuple(prediction.shape),
                    tuple(label.shape),
                )
            )

    def _forward_with_context(self, batch):
        prepared_batch = self.preprocess_batch(batch)
        var_x = prepared_batch["inputs"]
        marker_x = prepared_batch["inputs_timestamps"]
        label = prepared_batch["targets"][:, -self.hparams.pred_len:, ...]
        marker_y = prepared_batch["targets_timestamps"]
        targets_mask = prepared_batch["targets_mask"][:, -self.hparams.pred_len:, ...]

        self._validate_model_inputs(var_x)
        prediction = self._run_model(var_x, marker_x, marker_y)
        self._validate_prediction_shape(prediction, label)
        return {
            "prediction": prediction,
            "label": label,
            "targets_mask": targets_mask,
        }

    def forward(self, batch, batch_idx):
        del batch_idx
        outputs = self._forward_with_context(batch)
        return outputs["prediction"], outputs["label"]

    def training_step(self, batch, batch_idx):
        del batch_idx
        outputs = self._forward_with_context(batch)
        prediction = outputs["prediction"]
        label = outputs["label"]
        targets_mask = outputs["targets_mask"]
        if getattr(self.hparams, "use_mix_loss", False):
            loss = 0.5 * self.mae_loss_func(prediction, label, targets_mask) + 0.5 * self.mse_loss_func(prediction, label, targets_mask)
        else:
            loss = self.loss_function(prediction, label, targets_mask)
        aux_loss = self.model.get_aux_loss() if hasattr(self.model, "get_aux_loss") else None
        if aux_loss is not None:
            loss = loss + getattr(self.hparams, "aux_loss_weight", 1.0) * aux_loss
            self.log("train/aux_loss", aux_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        outputs = self._forward_with_context(batch)
        loss = self.loss_function(outputs["prediction"], outputs["label"], outputs["targets_mask"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        del batch_idx
        outputs = self._forward_with_context(batch)
        prediction, label = self.postprocess_outputs(
            outputs["prediction"],
            outputs["label"],
            outputs["targets_mask"],
        )
        mae = self.mae_loss_func(prediction, label, outputs["targets_mask"])
        mse = self.mse_loss_func(prediction, label, outputs["targets_mask"])
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
