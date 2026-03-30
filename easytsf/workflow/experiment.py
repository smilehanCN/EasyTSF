import argparse
from pathlib import Path

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from easytsf.task import get_task_registry_entry

from .config import finalize_runtime_conf, load_experiment_config, parse_config_overrides


def run_experiment(runtime_conf, extra_callbacks=None):
    L.seed_everything(runtime_conf["seed"])

    task_entry = get_task_registry_entry(runtime_conf.get("task_name", "mtsf"))
    datamodule = task_entry.datamodule_cls(**runtime_conf)
    runtime_conf["steps_per_epoch"] = max(1, len(datamodule.train_dataloader())) # for OneCycleScheduler
    task = task_entry.task_cls(**runtime_conf)

    exp_dir = Path(runtime_conf["exp_dir"]).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=str(exp_dir / "checkpoints"),
            monitor=runtime_conf["val_metric"],
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=runtime_conf["val_metric"],
            mode="min",
            patience=runtime_conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    callbacks.extend(extra_callbacks or [])

    trainer = L.Trainer(
        accelerator=runtime_conf["accelerator"],
        devices=runtime_conf["devices"],
        precision=runtime_conf.get("precision", "32-true"),
        logger=CSVLogger(save_dir=str(exp_dir), name="", version=""),
        callbacks=callbacks,
        max_epochs=runtime_conf["max_epochs"],
        gradient_clip_algorithm=runtime_conf.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=runtime_conf["gradient_clip_val"],
        default_root_dir=str(exp_dir),
        enable_checkpointing=True,
    )
    trainer.fit(task, datamodule=datamodule)
    return trainer.test(task, datamodule=datamodule, ckpt_path="best")


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
