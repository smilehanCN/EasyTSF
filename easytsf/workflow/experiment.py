from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from easytsf.task import get_task_registry_entry

from .config import DEFAULT_TASK_NAME, finalize_runtime_conf, load_config, load_experiment_config


@dataclass
class ExperimentBundle:
    conf: dict[str, Any]
    datamodule: Any
    task: Any
    trainer: L.Trainer | None = None


def _serialize_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value.item() if hasattr(value, "item") and callable(value.item) else value


def build_callbacks(conf, extra_callbacks=None): 
    callbacks = [
        ModelCheckpoint(dirpath=str(Path(conf["exp_dir"]) / "checkpoints"), monitor=conf["val_metric"], mode="min", save_top_k=1, save_last=True, every_n_epochs=1),
        EarlyStopping(monitor=conf["val_metric"], mode="min", patience=conf["es_patience"]),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    callbacks.extend(extra_callbacks or [])
    return callbacks

def build_logger(conf):
    exp_dir = Path(conf["exp_dir"])
    save_dir = exp_dir.parents[1] if len(exp_dir.parents) >= 2 else Path(conf["save_root"]) / f'{conf["model_name"]}_{conf["dataset_name"]}'
    return CSVLogger(save_dir=str(save_dir), name=conf["conf_hash"], version="seed_{}".format(conf["seed"]))


def build_trainer(conf, extra_callbacks=None):
    return L.Trainer(
        accelerator=conf["accelerator"],
        devices=conf["devices"],
        precision=conf.get("precision", "32-true"),
        logger=build_logger(conf),
        callbacks=build_callbacks(conf, extra_callbacks=extra_callbacks),
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["exp_dir"],
        enable_checkpointing=True,
    )


def _ensure_runtime_conf(conf):
    resolved_conf = dict(conf)
    if "conf_hash" not in resolved_conf or "exp_dir" not in resolved_conf:
        resolved_conf = finalize_runtime_conf(resolved_conf)
    return resolved_conf


def _prepare_training_conf(conf, datamodule):
    resolved_conf = dict(conf)
    if resolved_conf.get("lr_scheduler") == "OneCycleLR" and resolved_conf.get("steps_per_epoch") is None:
        resolved_conf["steps_per_epoch"] = max(1, len(datamodule.train_dataloader()))
    return resolved_conf


def build_experiment(conf, training=False, extra_callbacks=None):
    resolved_conf = _ensure_runtime_conf(conf)
    task_entry = get_task_registry_entry(resolved_conf.get("task_name", DEFAULT_TASK_NAME))
    datamodule = task_entry.datamodule_cls(**resolved_conf)
    runtime_conf = _prepare_training_conf(resolved_conf, datamodule) if training else dict(resolved_conf)
    task = task_entry.task_cls(**runtime_conf)
    trainer = build_trainer(runtime_conf, extra_callbacks=extra_callbacks) if training else None
    return ExperimentBundle(conf=runtime_conf, datamodule=datamodule, task=task, trainer=trainer)


def resolve_ckpt_path(conf, name="best"):
    explicit_path = Path(str(name)).expanduser()
    if explicit_path.suffix == ".ckpt":
        if explicit_path.exists():
            return str(explicit_path.resolve())
        raise FileNotFoundError("checkpoint not found: {}".format(explicit_path))

    ckpt_dir = Path(conf["exp_dir"]) / "checkpoints"
    if name == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
        if ckpt_path.exists():
            return str(ckpt_path.resolve())
        raise FileNotFoundError("checkpoint not found: {}".format(ckpt_path))

    candidates = sorted(path for path in ckpt_dir.glob("*.ckpt") if path.name != "last.ckpt")
    if not candidates:
        raise FileNotFoundError("best checkpoint not found under {}".format(ckpt_dir))
    return str(candidates[0].resolve())


def _metric_value(value):
    value = _serialize_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _collect_metrics(conf, trainer):
    callback_metrics = {name: _metric_value(value) for name, value in trainer.callback_metrics.items()}
    try:
        ckpt_path = resolve_ckpt_path(conf, "best")
    except FileNotFoundError:
        ckpt_path = None
    return {
        "task_name": conf.get("task_name", DEFAULT_TASK_NAME),
        "model_name": conf["model_name"],
        "dataset_name": conf["dataset_name"],
        "hist_len": int(conf["hist_len"]),
        "pred_len": int(conf["pred_len"]),
        "seed": int(conf["seed"]),
        "conf_hash": conf["conf_hash"],
        "exp_dir": str(Path(conf["exp_dir"]).resolve()),
        "ckpt_path": ckpt_path,
        "status": "success",
        "error": None,
        "val_metric_name": conf.get("val_metric"),
        "val_metric_value": callback_metrics.get(conf.get("val_metric")),
        "mae": callback_metrics.get("test/mae"),
        "mse": callback_metrics.get("test/mse"),
    }


def run_experiment(conf, extra_callbacks=None):
    resolved_conf = _ensure_runtime_conf(conf)
    L.seed_everything(resolved_conf["seed"])
    experiment = build_experiment(resolved_conf, training=True, extra_callbacks=extra_callbacks)
    Path(experiment.conf["exp_dir"]).mkdir(parents=True, exist_ok=True)
    experiment.trainer.fit(experiment.task, datamodule=experiment.datamodule)
    try:
        resolve_ckpt_path(experiment.conf, "best")
        ckpt_path = "best"
    except FileNotFoundError:
        ckpt_path = None
    experiment.trainer.test(experiment.task, datamodule=experiment.datamodule, ckpt_path=ckpt_path)
    return _collect_metrics(experiment.conf, experiment.trainer)


def run_training(conf):
    return run_experiment(conf)


def run_evaluation(conf, ckpt_path="best"):
    resolved_conf = _ensure_runtime_conf(conf)
    L.seed_everything(resolved_conf["seed"])
    experiment = build_experiment(resolved_conf, training=True)
    experiment.trainer.test(
        experiment.task,
        datamodule=experiment.datamodule,
        ckpt_path=resolve_ckpt_path(experiment.conf, ckpt_path),
    )
    return _collect_metrics(experiment.conf, experiment.trainer)


def experiment_main(conf, extra_callbacks=None):
    return run_experiment(conf, extra_callbacks=extra_callbacks)
