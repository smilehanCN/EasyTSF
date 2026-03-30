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
        return yaml.safe_load(handle) or {}

def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)


def _empty_section_map():
    return {section: {} for section in CONFIG_SECTIONS}


def _normalize_section_map(data):
    normalized = {}
    for section in CONFIG_SECTIONS:
        value = data.get(section, {})
        if value is None:
            value = {}
        normalized[section] = dict(value)
    return normalized


def parse_config_overrides(override_items):
    import yaml

    overrides = _empty_section_map()
    for item in override_items or []:
        key_ref, raw_value = item.split("=", 1)
        section, key = key_ref.split(".", 1)
        overrides[section][key] = yaml.safe_load(raw_value)
    return overrides


def _resolve_experiment_path(config_ref):
    ref_path = Path(config_ref).expanduser()
    if ref_path.exists():
        return ref_path.resolve()

    relative_ref = Path(config_ref)
    if relative_ref.suffix not in {".yaml", ".yml"}:
        relative_ref = relative_ref.with_suffix(".yaml")
    return (EXPERIMENT_CONFIG_DIR / relative_ref).resolve()


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
    flat_config["task_name"] = flat_config.get("task_name", DEFAULT_TASK_NAME)
    return flat_config


def load_experiment_config(config_ref, overrides=None):
    experiment_path = _resolve_experiment_path(config_ref)
    experiment_config = _normalize_section_map(_load_yaml(experiment_path))
    normalized_overrides = _normalize_section_map(overrides or {})
    merged_config = _merge_sectioned_config(experiment_config, normalized_overrides)
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
