import importlib
import inspect
import warnings

import lightning.pytorch as L
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs

from easytsf.model import get_model_contract


class MTSFTask(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model()
        self.loss_function = F.mse_loss
        self.mae_loss_func = F.l1_loss
        self.mse_loss_func = F.mse_loss

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
        contract.validate(getattr(self.hparams, "task_name", "mtsf"))
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
        for name, parameter in inspect.signature(model_cls.__init__).parameters.items():
            if name == "self" or parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
                continue
            if hasattr(self.hparams, name):
                model_args[name] = getattr(self.hparams, name)
                continue
            if parameter.default is inspect.Parameter.empty:
                raise ValueError("config must define required model argument '{}' for {}".format(name, model_name))
        return model_cls(**model_args)

    def preprocess_batch(self, batch):
        var_x, marker_x, var_y, marker_y = self._prepare_batch(batch)
        return {
            "inputs": var_x,
            "inputs_timestamps": marker_x,
            "targets": var_y,
            "targets_timestamps": marker_y,
        }

    def postprocess_outputs(self, prediction, label, targets_mask=None):
        del targets_mask
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

        self._validate_model_inputs(var_x)
        prediction = self._run_model(var_x, marker_x, marker_y)
        self._validate_prediction_shape(prediction, label)
        return {
            "prediction": prediction,
            "label": label,
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
        del batch_idx
        outputs = self._forward_with_context(batch)
        loss = self.loss_function(outputs["prediction"], outputs["label"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        del batch_idx
        outputs = self._forward_with_context(batch)
        prediction, label = self.postprocess_outputs(outputs["prediction"], outputs["label"])
        mae = self.mae_loss_func(prediction, label)
        mse = self.mse_loss_func(prediction, label)
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
