import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from easytsf.task import get_task_registry_entry


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PROJECT_ROOT / "config"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
CONFIG_SECTIONS = ("model", "data", "train", "runtime")
DEFAULT_TASK_NAME = "mtsf"
REQUIRED_EXPERIMENT_KEYS = {
    "model": ("model_name",),
    "data": ("dataset_name", "hist_len", "pred_len", "var_num", "precompute_window_index"),
    "train": (
        "batch_size",
        "max_epochs",
        "lr",
        "lr_scheduler",
        "optimizer",
        "es_patience",
        "gradient_clip_val",
        "gradient_clip_algorithm",
        "val_metric",
        "use_mix_loss",
    ),
    "runtime": ("task_name", "num_workers", "pin_memory", "persistent_workers", "prefetch_factor", "use_mmap"),
}
SCHEDULER_REQUIRED_TRAIN_KEYS = {
    "StepLR": ("lr_step_size", "lr_gamma"),
    "MultiStepLR": ("milestones", "gamma"),
    "ReduceLROnPlateau": ("lrs_factor", "lrs_patience"),
    "WSD": ("lr_warmup_end_epochs", "lr_stable_end_epochs"),
    "OneCycleLR": ("lrs_pct_start",),
}


@dataclass
class ExperimentBundle:
    conf: dict[str, Any]
    datamodule: Any
    task: Any
    trainer: L.Trainer


def serialize_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module {} from {}".format(module_name, module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_hash_value(value):
    value = serialize_value(value)
    if isinstance(value, dict):
        return {key: _normalize_hash_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_hash_value(item) for item in value]
    return value


def calculate_config_hash(config, ignored_keys=None, hash_len=10):
    if ignored_keys is None:
        ignored_keys = ["save_root", "data_root", "seed", "ckpt_path", "conf_hash", "exp_dir"]
    filtered_config = {
        key: _normalize_hash_value(value)
        for key, value in config.items()
        if key not in ignored_keys
    }
    payload = json.dumps(filtered_config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.md5()
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()[:hash_len]


def parse_devices(devices):
    if devices is None:
        return "auto"
    if isinstance(devices, int):
        return devices
    if isinstance(devices, (list, tuple)):
        return [int(device_id) for device_id in devices]
    if not isinstance(devices, str):
        return devices
    value = devices.strip()
    if value == "" or value.lower() == "auto":
        return "auto"
    if value in {"-1", "all"}:
        return -1
    if "," in value:
        parts = [part.strip() for part in value.split(",") if part.strip() != ""]
        if not parts:
            return "auto"
        return [int(part) for part in parts]
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _load_yaml(path):
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping: {}".format(path))
    return data


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


def _merge_sectioned_config(*configs):
    merged = _empty_section_map()
    for config in configs:
        for section in CONFIG_SECTIONS:
            merged[section].update(config.get(section, {}))
    return merged


def _validate_supported_task_name(task_name, source_name):
    if task_name != DEFAULT_TASK_NAME:
        raise ValueError(
            "unsupported task_name '{}' in {}; supported task names are ['{}']".format(
                task_name,
                source_name,
                DEFAULT_TASK_NAME,
            )
        )


def _validate_complete_experiment_config(sectioned_config, source_name):
    missing_keys = []
    for section, keys in REQUIRED_EXPERIMENT_KEYS.items():
        missing_keys.extend(
            "{}.{}".format(section, key)
            for key in keys
            if key not in sectioned_config[section]
        )
    if missing_keys:
        raise ValueError("{} is missing required config keys: {}".format(source_name, missing_keys))

    task_name = sectioned_config["runtime"]["task_name"]
    _validate_supported_task_name(task_name, source_name)

    model_name = sectioned_config["model"]["model_name"]
    if model_name == "TQNet" and "time_feature_descriptions" not in sectioned_config["data"]:
        raise ValueError("{} is missing required config keys: ['data.time_feature_descriptions']".format(source_name))

    scheduler_name = sectioned_config["train"]["lr_scheduler"]
    required_scheduler_keys = SCHEDULER_REQUIRED_TRAIN_KEYS.get(scheduler_name, ())
    missing_scheduler_keys = [
        "train.{}".format(key)
        for key in required_scheduler_keys
        if key not in sectioned_config["train"]
    ]
    if missing_scheduler_keys:
        raise ValueError("{} is missing required config keys: {}".format(source_name, missing_scheduler_keys))


def _flatten_config(sectioned_config):
    flat_config = {}
    for section in CONFIG_SECTIONS:
        flat_config.update(sectioned_config[section])
    required_keys = ("model_name", "dataset_name", "hist_len", "pred_len")
    missing_keys = [key for key in required_keys if key not in flat_config]
    if missing_keys:
        raise ValueError("missing required config keys: {}".format(missing_keys))
    task_name = flat_config.get("task_name", DEFAULT_TASK_NAME)
    _validate_supported_task_name(task_name, "config")
    if task_name == DEFAULT_TASK_NAME and "var_num" not in flat_config:
        raise ValueError("mtsf config must define data.var_num")
    if flat_config.get("model_name") == "TQNet" and "time_feature_descriptions" not in flat_config:
        raise ValueError("TQNet config must define data.time_feature_descriptions")
    return flat_config


def resolve_experiment_path(config_ref):
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


def load_experiment_config(config_ref):
    experiment_path = resolve_experiment_path(config_ref)
    experiment_config = _normalize_section_map(_load_yaml(experiment_path), str(experiment_path))
    _validate_complete_experiment_config(experiment_config, str(experiment_path))
    return _flatten_config(experiment_config)


def finalize_runtime_conf(base_conf, runtime_overrides=None):
    conf = dict(base_conf)
    for key, value in dict(runtime_overrides or {}).items():
        if value is not None:
            conf[key] = value

    required_runtime_keys = ("data_root", "save_root", "seed")
    missing_runtime_keys = [key for key in required_runtime_keys if conf.get(key) in {None, ""}]
    if missing_runtime_keys:
        raise ValueError("missing runtime config keys: {}".format(missing_runtime_keys))

    conf["seed"] = int(conf["seed"])
    conf["devices"] = parse_devices(conf.get("devices"))
    conf["accelerator"] = conf.get("accelerator", "auto") or "auto"
    conf["conf_hash"] = calculate_config_hash(conf, hash_len=10)
    if not conf.get("exp_dir"):
        exp_root = Path(conf["save_root"]).expanduser() / "{}_{}".format(conf["model_name"], conf["dataset_name"])
        conf["exp_dir"] = str((exp_root / conf["conf_hash"] / "seed_{}".format(conf["seed"])).resolve())
    return conf


def build_callbacks(conf):
    checkpoint_dir = Path(conf["exp_dir"]) / "checkpoints"
    return [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor=conf["val_metric"],
            mode="min",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
        ),
        EarlyStopping(monitor=conf["val_metric"], mode="min", patience=conf["es_patience"]),
        LearningRateMonitor(logging_interval="epoch"),
    ]


def build_logger(conf):
    exp_root = Path(conf["save_root"]).expanduser() / "{}_{}".format(conf["model_name"], conf["dataset_name"])
    return CSVLogger(save_dir=str(exp_root), name=conf["conf_hash"], version="seed_{}".format(conf["seed"]))


def build_trainer(conf, enable_progress_bar=True):
    return L.Trainer(
        accelerator=conf["accelerator"],
        devices=conf["devices"],
        precision=conf.get("precision", "32-true"),
        logger=build_logger(conf),
        callbacks=build_callbacks(conf),
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["exp_dir"],
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_progress_bar,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )


def _prepare_training_conf(conf, datamodule):
    resolved_conf = dict(conf)
    if resolved_conf.get("lr_scheduler") == "OneCycleLR" and resolved_conf.get("steps_per_epoch") is None:
        resolved_conf["steps_per_epoch"] = max(1, len(datamodule.train_dataloader()))
    return resolved_conf


def build_experiment(conf, enable_progress_bar=True):
    resolved_conf = finalize_runtime_conf(conf)
    task_entry = get_task_registry_entry(resolved_conf.get("task_name", DEFAULT_TASK_NAME))
    datamodule = task_entry.datamodule_cls(**resolved_conf)
    runtime_conf = _prepare_training_conf(resolved_conf, datamodule)
    task = task_entry.task_cls(**runtime_conf)
    trainer = build_trainer(runtime_conf, enable_progress_bar=enable_progress_bar)
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
    value = serialize_value(value)
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def collect_metrics(conf, trainer):
    callback_metrics = {name: _metric_value(value) for name, value in trainer.callback_metrics.items()}
    try:
        ckpt_path = resolve_ckpt_path(conf, "best")
    except FileNotFoundError:
        ckpt_path = None

    metrics = {
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
        "val_metric_name": conf["val_metric"],
        "val_metric_value": callback_metrics.get(conf["val_metric"]),
        "mae": callback_metrics.get("test/mae"),
        "mse": callback_metrics.get("test/mse"),
        "rmse": callback_metrics.get("test/rmse"),
    }
    if metrics["val_metric_value"] is not None:
        metrics[conf["val_metric"]] = metrics["val_metric_value"]
    return metrics


def build_failure_metrics(conf, error):
    metrics = {
        "task_name": conf.get("task_name", DEFAULT_TASK_NAME),
        "model_name": conf["model_name"],
        "dataset_name": conf["dataset_name"],
        "hist_len": int(conf["hist_len"]),
        "pred_len": int(conf["pred_len"]),
        "seed": int(conf["seed"]),
        "conf_hash": conf["conf_hash"],
        "exp_dir": str(Path(conf["exp_dir"]).resolve()),
        "ckpt_path": None,
        "status": "failed",
        "error": str(error),
        "val_metric_name": conf["val_metric"],
        "val_metric_value": None,
        "mae": None,
        "mse": None,
        "rmse": None,
    }
    metrics[conf["val_metric"]] = None
    return metrics


def run_experiment(conf, enable_progress_bar=True):
    resolved_conf = finalize_runtime_conf(conf)
    L.seed_everything(resolved_conf["seed"], workers=True)
    experiment = build_experiment(resolved_conf, enable_progress_bar=enable_progress_bar)
    Path(experiment.conf["exp_dir"]).mkdir(parents=True, exist_ok=True)
    experiment.trainer.fit(experiment.task, datamodule=experiment.datamodule)
    try:
        resolve_ckpt_path(experiment.conf, "best")
        ckpt_path = "best"
    except FileNotFoundError:
        ckpt_path = None
    experiment.trainer.test(experiment.task, datamodule=experiment.datamodule, ckpt_path=ckpt_path)
    return collect_metrics(experiment.conf, experiment.trainer)


def run_experiment_from_path(config_ref, runtime_overrides, enable_progress_bar=True):
    base_conf = load_experiment_config(config_ref)
    conf = finalize_runtime_conf(base_conf, runtime_overrides=runtime_overrides)
    return run_experiment(conf, enable_progress_bar=enable_progress_bar)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a single EasyTSF experiment.")
    parser.add_argument("experiment_yaml", help="Experiment YAML path or config/experiments reference.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--save-root", required=True, help="Artifact root directory.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed.")
    parser.add_argument("--devices", default=None, help="Lightning devices value.")
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator value.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runtime_overrides = {
        "data_root": args.data_root,
        "save_root": args.save_root,
        "seed": args.seed,
        "devices": args.devices,
        "accelerator": args.accelerator,
    }
    try:
        result = run_experiment_from_path(
            args.experiment_yaml,
            runtime_overrides=runtime_overrides,
            enable_progress_bar=True,
        )
    except Exception as error:
        failure = {
            "experiment": args.experiment_yaml,
            "status": "failed",
            "error": str(error),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
