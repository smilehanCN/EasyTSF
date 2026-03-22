import importlib
import inspect

import lightning.pytorch as L
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs


class ForecastTask(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._build_model()
        self.loss_function = nn.MSELoss()
        self.mae_loss_func = nn.L1Loss()
        self.mse_loss_func = nn.MSELoss()

    @staticmethod
    def _prepare_batch(batch):
        return tuple(tensor if tensor.dtype == torch.float32 else tensor.float() for tensor in batch)

    def _build_model(self):
        model_name = self.hparams.model_name
        module = importlib.import_module(".{}".format(model_name), package="easytsf.model")
        if not hasattr(module, model_name):
            raise ValueError("model {} is not defined in easytsf.model.{}".format(model_name, model_name))
        model_cls = getattr(module, model_name)
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
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(),
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
