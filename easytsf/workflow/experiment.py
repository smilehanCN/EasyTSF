import hashlib
import importlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"
TASK_CONFIG_DIR = CONFIG_ROOT / "tasks"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
SEARCH_SPACE_DIR = CONFIG_ROOT / "search_spaces"
STUDY_CONFIG_DIR = CONFIG_ROOT / "studies"
CONFIG_SECTIONS = ("model", "data", "train", "runtime")
DEFAULT_TASK_NAME = "mtsf"


@dataclass
class ExperimentComponents:
    conf: dict
    trainer: L.Trainer
    datamodule: Any
    task: Any


def _normalize_hash_value(value):
    if isinstance(value, dict):
        return {key: _normalize_hash_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize_hash_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize_hash_value(item) for item in value)
    return value


def cal_conf_hash(config, useless_key=None, hash_len=10):
    if useless_key is None:
        useless_key = ["save_root", "data_root", "seed", "ckpt_path", "conf_hash", "exp_dir", "use_wandb", "use_ray"]

    filtered_config = {
        key: _normalize_hash_value(value)
        for key, value in config.items()
        if key not in useless_key
    }
    conf_str = json.dumps(filtered_config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    md5 = hashlib.md5()
    md5.update(conf_str.encode("utf-8"))
    return md5.hexdigest()[:hash_len]


def load_module_from_path(module_name, exp_conf_path):
    spec = importlib.util.spec_from_file_location(module_name, exp_conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_devices(devices):
    if devices is None:
        return "auto"
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return [int(d) for d in devices]
    if not isinstance(devices, str):
        return devices

    value = devices.strip()
    if value == "":
        return "auto"
    if value.lower() == "auto":
        return "auto"
    if value in {"-1", "all"}:
        return -1
    if "," in value:
        parts = [p.strip() for p in value.split(",") if p.strip() != ""]
        if len(parts) == 0:
            return "auto"
        return [int(p) for p in parts]
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def add_shared_runtime_args(parser):
    parser.add_argument("-c", "--config", required=True, type=str, help="experiment config id or YAML path")
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", type=str, help="save root")
    parser.add_argument("--accelerator", default="auto", type=str, help="accelerator to use")
    parser.add_argument("--devices", default="auto", type=str, help="device ids/count, e.g. auto, 1, 0,1")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    return parser


def add_config_override_args(parser):
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override config values, e.g. --set data.pred_len=336 --set train.lr=1e-4",
    )
    return parser


def add_tune_args(parser):
    parser.add_argument("-p", "--param_space", default=None, type=str, help="search-space id or Python path")
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--cpus_per_trial", default=2, type=int)
    parser.add_argument("--gpus_per_trial", default=0.5, type=float)
    return parser


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping: {}".format(path))
    return data


def _save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True, allow_unicode=True)


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _empty_section_map():
    return {section: {} for section in CONFIG_SECTIONS}


def _normalize_section_map(data, source_name):
    unknown_sections = set(data.keys()) - set(CONFIG_SECTIONS)
    if unknown_sections:
        raise ValueError("unsupported config sections in {}: {}".format(source_name, sorted(unknown_sections)))
    normalized = {}
    for section in CONFIG_SECTIONS:
        value = data.get(section, {})
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError("{} section '{}' must be a mapping".format(source_name, section))
        normalized[section] = dict(value)
    return normalized


def parse_config_overrides(override_items):
    overrides = _empty_section_map()
    for item in override_items or []:
        if "=" not in item:
            raise ValueError("config override must use SECTION.KEY=VALUE: {}".format(item))
        key_ref, raw_value = item.split("=", 1)
        if "." not in key_ref:
            raise ValueError("config override must use SECTION.KEY=VALUE: {}".format(item))
        section, key = key_ref.split(".", 1)
        if section not in CONFIG_SECTIONS:
            raise ValueError("unknown config section '{}' in override '{}'".format(section, item))
        if not key or "." in key:
            raise ValueError("only one level of keys is supported in override '{}'".format(item))
        overrides[section][key] = yaml.safe_load(raw_value)
    return overrides


def _resolve_experiment_path(config_ref):
    ref_path = Path(config_ref).expanduser()
    if ref_path.exists():
        return ref_path.resolve()

    relative_ref = Path(config_ref)
    if relative_ref.suffix not in {".yaml", ".yml"}:
        relative_ref = relative_ref.with_suffix(".yaml")
    resolved_path = (EXPERIMENT_CONFIG_DIR / relative_ref).resolve()
    if resolved_path.exists():
        return resolved_path
    raise FileNotFoundError("experiment config not found: {}".format(config_ref))


def _resolve_search_space_path(param_space_ref):
    ref_path = Path(param_space_ref).expanduser()
    if ref_path.exists():
        return ref_path.resolve()

    relative_ref = Path(param_space_ref)
    if relative_ref.suffix != ".py":
        relative_ref = relative_ref.with_suffix(".py")
    resolved_path = (SEARCH_SPACE_DIR / relative_ref).resolve()
    if resolved_path.exists():
        return resolved_path
    raise FileNotFoundError("search-space config not found: {}".format(param_space_ref))


def _load_task_defaults(task_name):
    task_config_path = TASK_CONFIG_DIR / "{}.yaml".format(task_name)
    if not task_config_path.exists():
        raise FileNotFoundError("task defaults not found: {}".format(task_config_path))
    return _normalize_section_map(_load_yaml(task_config_path), "{} task defaults".format(task_name))


def _merge_sectioned_config(*configs):
    merged = _empty_section_map()
    for config in configs:
        for section in CONFIG_SECTIONS:
            merged[section].update(config.get(section, {}))
    return merged


def _flatten_config(sectioned_config):
    flat_config = {}
    for section in CONFIG_SECTIONS:
        flat_config.update(sectioned_config[section])
    required_keys = ("model_name", "dataset_name", "hist_len", "pred_len")
    missing_keys = [key for key in required_keys if key not in flat_config]
    if missing_keys:
        raise ValueError("missing required config keys: {}".format(missing_keys))
    return flat_config


def load_config(config_ref, overrides=None):
    experiment_path = _resolve_experiment_path(config_ref)
    experiment_config = _normalize_section_map(_load_yaml(experiment_path), str(experiment_path))
    normalized_overrides = _normalize_section_map(overrides or {}, "config overrides")
    task_name = (
        normalized_overrides["runtime"].get("task_name")
        or experiment_config["runtime"].get("task_name")
        or DEFAULT_TASK_NAME
    )
    task_defaults = _load_task_defaults(task_name)

    dataset_name = normalized_overrides["data"].get("dataset_name") or experiment_config["data"].get("dataset_name")
    if not dataset_name:
        raise ValueError("experiment config must define data.dataset_name: {}".format(experiment_path))

    merged_config = _merge_sectioned_config(task_defaults, experiment_config, normalized_overrides)
    return _flatten_config(merged_config)


def load_param_space(param_space_ref):
    param_space_path = _resolve_search_space_path(param_space_ref)
    module_hash = hashlib.md5(str(param_space_path).encode("utf-8")).hexdigest()[:10]
    module = load_module_from_path("easytsf_param_space_{}".format(module_hash), str(param_space_path))
    if not hasattr(module, "param_space"):
        raise ValueError("search-space module must define param_space: {}".format(param_space_path))
    return module.param_space


def finalize_runtime_conf(base_conf, overrides=None, use_ray=False):
    conf = dict(base_conf)
    if overrides:
        conf.update({key: value for key, value in overrides.items() if value is not None})

    conf["devices"] = parse_devices(conf.get("devices", "auto"))
    conf["accelerator"] = conf.get("accelerator", "auto")
    conf["use_ray"] = bool(use_ray)
    conf["conf_hash"] = cal_conf_hash(conf, hash_len=10)
    save_dir = os.path.join(conf["save_root"], "{}_{}".format(conf["model_name"], conf["dataset_name"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["conf_hash"], "seed_{}".format(conf["seed"]))
    return conf


def build_runtime_overrides(args, include_ckpt_path=False):
    overrides = {
        "seed": int(args.seed),
        "data_root": args.data_root,
        "save_root": args.save_root,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "use_wandb": args.use_wandb,
    }
    if include_ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    return overrides


def get_resolved_config_path(conf):
    return Path(conf["exp_dir"]) / "resolved_config.yaml"


def get_metrics_path(conf):
    return Path(conf["exp_dir"]) / "metrics.json"


def ensure_experiment_dir(conf):
    Path(conf["exp_dir"]).mkdir(parents=True, exist_ok=True)


def save_resolved_config(conf):
    ensure_experiment_dir(conf)
    resolved_config = {key: _serialize_value(value) for key, value in sorted(conf.items())}
    _save_yaml(get_resolved_config_path(conf), resolved_config)
    return get_resolved_config_path(conf)


def load_saved_metrics(conf):
    metrics_path = get_metrics_path(conf)
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _serialize_value(value):
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _extract_scalar_metric(value):
    value = _serialize_value(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _build_metric_record(conf, status, ckpt_path=None, metrics=None, error=None, val_metric_value=None):
    metric_values = {}
    if metrics:
        metric_values["mae"] = _extract_scalar_metric(metrics.get("test/mae"))
        metric_values["mse"] = _extract_scalar_metric(metrics.get("test/mse"))
    return {
        "task_name": conf.get("task_name", DEFAULT_TASK_NAME),
        "model_name": conf["model_name"],
        "dataset_name": conf["dataset_name"],
        "hist_len": int(conf["hist_len"]),
        "pred_len": int(conf["pred_len"]),
        "seed": int(conf["seed"]),
        "conf_hash": conf["conf_hash"],
        "exp_dir": str(Path(conf["exp_dir"]).resolve()),
        "ckpt_path": None if ckpt_path is None else str(ckpt_path),
        "status": status,
        "error": error,
        "val_metric_name": conf.get("val_metric"),
        "val_metric_value": _extract_scalar_metric(val_metric_value),
        **metric_values,
    }


def save_metrics(conf, metrics):
    ensure_experiment_dir(conf)
    serialized_metrics = {key: _serialize_value(value) for key, value in metrics.items()}
    _save_json(get_metrics_path(conf), serialized_metrics)
    return get_metrics_path(conf)


def build_callbacks(conf, training=True):
    if not training:
        return []

    checkpoint_dir = Path(conf["exp_dir"]) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=conf["val_metric"],
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=conf["val_metric"],
            mode="min",
            patience=conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if conf.get("use_ray"):
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

        callbacks.append(
            TuneReportCheckpointCallback(
                {conf["val_metric"]: conf["val_metric"]},
                save_checkpoints=False,
                on="validation_end",
            )
        )

    return callbacks


def build_logger(conf):
    save_dir = os.path.join(conf["save_root"], "{}_{}".format(conf["model_name"], conf["dataset_name"]))
    version = "seed_{}".format(conf["seed"])
    if conf.get("use_wandb"):
        return WandbLogger(save_dir=save_dir, name=conf["conf_hash"], version=version)
    return CSVLogger(save_dir=save_dir, name=conf["conf_hash"], version=version)


def build_experiment(conf, training=True):
    from easytsf.task import get_task_registry_entry

    L.seed_everything(conf["seed"])
    ensure_experiment_dir(conf)
    finalized_conf = dict(conf)
    task_name = finalized_conf.get("task_name", DEFAULT_TASK_NAME)
    task_entry = get_task_registry_entry(task_name)
    datamodule = task_entry.datamodule_cls(**conf)
    task_entry.validate_data_spec(finalized_conf["dataset_name"], datamodule.data_spec)

    finalized_conf.update(datamodule.get_resolved_conf_updates())

    finalized_conf["data_spec"] = datamodule.data_spec
    if datamodule.data_spec.channel_num is not None:
        finalized_conf["channel_num"] = int(datamodule.data_spec.channel_num)
    if datamodule.data_spec.spatial_shape:
        finalized_conf["spatial_shape"] = list(datamodule.data_spec.spatial_shape)
    if datamodule.data_spec.spatial_ndim:
        finalized_conf["spatial_ndim"] = int(datamodule.data_spec.spatial_ndim)
    finalized_conf["steps_per_epoch"] = max(1, len(datamodule.train_dataloader()))
    task = task_entry.task_cls(**task_entry.build_task_kwargs(datamodule), **finalized_conf)
    trainer = L.Trainer(
        accelerator=finalized_conf["accelerator"],
        devices=finalized_conf["devices"],
        precision=finalized_conf.get("precision", "32-true"),
        logger=build_logger(finalized_conf),
        callbacks=build_callbacks(finalized_conf, training=training),
        max_epochs=finalized_conf["max_epochs"],
        gradient_clip_algorithm=finalized_conf.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=finalized_conf["gradient_clip_val"],
        default_root_dir=finalized_conf["exp_dir"],
        enable_checkpointing=training,
    )
    return ExperimentComponents(
        conf=finalized_conf,
        trainer=trainer,
        datamodule=datamodule,
        task=task,
    )


def resolve_ckpt_path(conf, ckpt_path):
    if ckpt_path not in {"best", "last"}:
        return ckpt_path

    checkpoint_dir = Path(conf["exp_dir"]) / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError("checkpoint directory does not exist: {}".format(checkpoint_dir))

    if ckpt_path == "last":
        last_ckpt = checkpoint_dir / "last.ckpt"
        if not last_ckpt.exists():
            raise FileNotFoundError("last checkpoint not found: {}".format(last_ckpt))
        return str(last_ckpt)

    candidates = sorted(
        path for path in checkpoint_dir.glob("*.ckpt")
        if path.name != "last.ckpt"
    )
    if not candidates:
        raise FileNotFoundError("best checkpoint not found under {}".format(checkpoint_dir))
    if len(candidates) > 1:
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _get_model_checkpoint_callback(trainer):
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback
    return None


def run_training(conf):
    experiment = build_experiment(conf, training=True)
    save_resolved_config(experiment.conf)
    experiment.trainer.fit(model=experiment.task, datamodule=experiment.datamodule)

    checkpoint_callback = _get_model_checkpoint_callback(experiment.trainer)
    best_ckpt_path = None
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_ckpt_path = checkpoint_callback.best_model_path

    fit_metrics = dict(experiment.trainer.callback_metrics)
    test_results = experiment.trainer.test(experiment.task, datamodule=experiment.datamodule, ckpt_path="best")
    test_metrics = test_results[0] if test_results else {}
    metrics = _build_metric_record(
        experiment.conf,
        status="success",
        ckpt_path=best_ckpt_path,
        metrics=test_metrics,
        val_metric_value=fit_metrics.get(experiment.conf["val_metric"]),
    )
    save_metrics(experiment.conf, metrics)
    return metrics


def run_evaluation(conf, ckpt_path):
    experiment = build_experiment(conf, training=False)
    save_resolved_config(experiment.conf)
    resolved_ckpt_path = resolve_ckpt_path(experiment.conf, ckpt_path)
    test_results = experiment.trainer.test(
        experiment.task,
        datamodule=experiment.datamodule,
        ckpt_path=resolved_ckpt_path,
    )
    test_metrics = test_results[0] if test_results else {}
    metrics = _build_metric_record(
        experiment.conf,
        status="success",
        ckpt_path=resolved_ckpt_path,
        metrics=test_metrics,
    )
    save_metrics(experiment.conf, metrics)
    return metrics
