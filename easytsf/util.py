import hashlib
import importlib
import importlib.util
import json
import sys


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
