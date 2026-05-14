import argparse
import math
from pathlib import Path

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from easytsf.task import validate_task_runtime_conf

from .config import finalize_runtime_conf, load_experiment_config, parse_config_overrides


def _resolve_device_count(devices) -> int:
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    return 1


def _apply_train_batch_limit(train_batches: int, limit_train_batches) -> int:
    if limit_train_batches in {None, ""}:
        return train_batches
    if isinstance(limit_train_batches, str) and limit_train_batches.lower() in {"all", "none"}:
        return train_batches
    if isinstance(limit_train_batches, bool):
        return train_batches
    if isinstance(limit_train_batches, int):
        return max(1, min(train_batches, int(limit_train_batches)))
    if isinstance(limit_train_batches, float):
        if limit_train_batches <= 0:
            raise ValueError("limit_train_batches must be positive when set")
        if limit_train_batches <= 1.0:
            return max(1, int(math.ceil(train_batches * limit_train_batches)))
        return max(1, min(train_batches, int(limit_train_batches)))
    raise TypeError("unsupported limit_train_batches type: {}".format(type(limit_train_batches).__name__))


def prepare_runtime_conf_for_task(runtime_conf, task_spec=None):
    prepared_conf = dict(runtime_conf)
    if task_spec is None:
        task_spec = validate_task_runtime_conf(prepared_conf)

    datamodule = task_spec.datamodule_cls(**prepared_conf)
    export_task_hparams = getattr(datamodule, "export_task_hparams", None)
    if callable(export_task_hparams):
        prepared_conf.update(dict(export_task_hparams() or {}))
    train_batches = max(1, len(datamodule.train_dataloader()))
    device_count = _resolve_device_count(prepared_conf.get("devices", 1))
    accumulate_grad_batches = max(1, int(prepared_conf.get("accumulate_grad_batches", 1)))
    strategy = str(prepared_conf.get("strategy", "auto"))
    if device_count > 1 and strategy.startswith("ddp"):
        train_batches = int(math.ceil(train_batches / float(device_count)))
    train_batches = _apply_train_batch_limit(train_batches, prepared_conf.get("limit_train_batches"))
    prepared_conf["train_batches_per_epoch"] = train_batches
    prepared_conf["steps_per_epoch"] = max(1, int(math.ceil(train_batches / float(accumulate_grad_batches))))
    return task_spec, datamodule, prepared_conf


def run_experiment(runtime_conf, extra_callbacks=None):
    L.seed_everything(runtime_conf["seed"], verbose=bool(runtime_conf.get("seed_verbose", True)))

    task_spec, datamodule, runtime_conf = prepare_runtime_conf_for_task(runtime_conf)
    task = task_spec.task_cls(**runtime_conf)
    val_metric_mode = runtime_conf.get("val_metric_mode", "min")

    exp_dir = Path(runtime_conf["exp_dir"]).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(exp_dir / "checkpoints"),
            monitor=runtime_conf["val_metric"],
            mode=val_metric_mode,
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=runtime_conf["val_metric"],
            mode=val_metric_mode,
            patience=runtime_conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    callbacks.extend(extra_callbacks or [])

    trainer = L.Trainer(
        accelerator=runtime_conf["accelerator"],
        devices=runtime_conf["devices"],
        strategy=runtime_conf.get("strategy", "auto"),
        precision=runtime_conf.get("precision", "32-true"),
        accumulate_grad_batches=int(runtime_conf.get("accumulate_grad_batches", 1)),
        logger=CSVLogger(save_dir=str(exp_dir), name="", version=""),
        callbacks=callbacks,
        max_epochs=runtime_conf["max_epochs"],
        limit_train_batches=runtime_conf.get("limit_train_batches", 1.0),
        check_val_every_n_epoch=runtime_conf.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=runtime_conf.get("num_sanity_val_steps", 2),
        log_every_n_steps=runtime_conf.get("log_every_n_steps", 50),
        gradient_clip_algorithm=runtime_conf.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=runtime_conf["gradient_clip_val"],
        default_root_dir=str(exp_dir),
        enable_checkpointing=True,
        enable_progress_bar=runtime_conf.get("enable_progress_bar", True),
        enable_model_summary=runtime_conf.get("enable_model_summary", True),
    )
    trainer.fit(task, datamodule=datamodule)

    ckpt_path = None
    checkpoint_cb = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
    if checkpoint_cb is not None:
        best_path = checkpoint_cb.best_model_path
        last_path = checkpoint_cb.last_model_path
        if best_path and Path(best_path).exists():
            ckpt_path = best_path
        elif last_path and Path(last_path).exists():
            ckpt_path = last_path

    return trainer.test(task, datamodule=datamodule, ckpt_path=ckpt_path)


def build_cli_parser():
    parser = argparse.ArgumentParser(description="Run a single EasyTSF experiment.")
    parser.add_argument("experiment", help="Experiment preset ref or yaml path.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Runtime override. Repeat to pass multiple values.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli_parser().parse_args()
    overrides = parse_config_overrides(args.overrides)
    runtime_conf = finalize_runtime_conf(load_experiment_config(args.experiment), overrides)
    run_experiment(runtime_conf)
