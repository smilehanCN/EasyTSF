import hashlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
BENCHMARK_CONFIG_DIR = CONFIG_ROOT / "benchmarks"
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
        useless_key = ["save_root", "data_root", "seed", "ckpt_path", "conf_hash", "exp_dir", "use_wandb"]

    filtered_config = {
        key: _normalize_hash_value(value)
        for key, value in config.items()
        if key not in useless_key
    }
    conf_str = json.dumps(filtered_config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    md5 = hashlib.md5()
    md5.update(conf_str.encode("utf-8"))
    return md5.hexdigest()[:hash_len]


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
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
        return [int(device_id) for device_id in devices]
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
        parts = [part.strip() for part in value.split(",") if part.strip() != ""]
        if len(parts) == 0:
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


def _save_yaml(path, data):
    import yaml

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
    import yaml

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


def load_experiment_config(config_ref, overrides=None):
    experiment_path = _resolve_experiment_path(config_ref)
    experiment_config = _normalize_section_map(_load_yaml(experiment_path), str(experiment_path))
    _validate_complete_experiment_config(experiment_config, str(experiment_path))
    normalized_overrides = _normalize_section_map(overrides or {}, "config overrides")
    merged_config = _merge_sectioned_config(experiment_config, normalized_overrides)
    _validate_complete_experiment_config(merged_config, "merged config")
    return _flatten_config(merged_config)


def finalize_runtime_conf(base_conf, overrides=None):
    conf = dict(base_conf)
    if overrides:
        conf.update({key: value for key, value in overrides.items() if value is not None})

    conf["devices"] = parse_devices(conf.get("devices", "auto"))
    conf["accelerator"] = conf.get("accelerator", "auto")
    conf["conf_hash"] = cal_conf_hash(conf, hash_len=10)
    if not conf.get("exp_dir"):
        exp_root = Path(conf["save_root"]) / "{}_{}".format(conf["model_name"], conf["dataset_name"])
        conf["exp_dir"] = str(exp_root / conf["conf_hash"] / "seed_{}".format(conf["seed"]))
    return conf


load_config = load_experiment_config
