import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as L
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from easytsf.data import DataInterface
from easytsf.task import ForecastTask
from easytsf.util import cal_conf_hash, load_module_from_path, parse_devices


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = PROJECT_ROOT / "config"
TASK_CONFIG_PATH = CONFIG_ROOT / "tasks" / "forecast.yaml"
DATASET_CATALOG_PATH = CONFIG_ROOT / "datasets" / "catalog.yaml"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
SEARCH_SPACE_DIR = CONFIG_ROOT / "search_spaces"
CONFIG_SECTIONS = ("model", "data", "train", "runtime")


@dataclass
class ExperimentComponents:
    conf: dict
    trainer: L.Trainer
    datamodule: DataInterface
    task: ForecastTask


def add_shared_runtime_args(parser):
    parser.add_argument("-c", "--config", required=True, type=str, help="experiment config id or YAML path")
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", type=str, help="save root")
    parser.add_argument("--accelerator", default="auto", type=str, help="accelerator to use")
    parser.add_argument("--devices", default="auto", type=str, help="device ids/count, e.g. auto, 1, 0,1")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=0, help="seed")
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


def _load_dataset_defaults(dataset_name):
    catalog = _load_yaml(DATASET_CATALOG_PATH)
    if dataset_name not in catalog:
        raise KeyError("dataset '{}' is not defined in {}".format(dataset_name, DATASET_CATALOG_PATH))
    dataset_entry = catalog[dataset_name]
    if not isinstance(dataset_entry, dict):
        raise ValueError("dataset '{}' entry must be a mapping".format(dataset_name))
    return _normalize_section_map(dataset_entry, "dataset '{}'".format(dataset_name))


def _merge_sectioned_config(*configs):
    merged = {section: {} for section in CONFIG_SECTIONS}
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


def load_config(config_ref):
    task_defaults = _normalize_section_map(_load_yaml(TASK_CONFIG_PATH), "task defaults")
    experiment_path = _resolve_experiment_path(config_ref)
    experiment_config = _normalize_section_map(_load_yaml(experiment_path), str(experiment_path))
    dataset_name = experiment_config["data"].get("dataset_name")
    if not dataset_name:
        raise ValueError("experiment config must define data.dataset_name: {}".format(experiment_path))
    dataset_defaults = _load_dataset_defaults(dataset_name)
    return _flatten_config(_merge_sectioned_config(task_defaults, dataset_defaults, experiment_config))


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


def build_callbacks(conf, training=True):
    if not training:
        return []

    callbacks = [
        ModelCheckpoint(
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
    L.seed_everything(conf["seed"])
    datamodule = DataInterface(**conf)
    finalized_conf = dict(conf)
    finalized_conf["steps_per_epoch"] = max(1, len(datamodule.train_dataloader()))
    task = ForecastTask(**finalized_conf)
    trainer = L.Trainer(
        accelerator=finalized_conf["accelerator"],
        devices=finalized_conf["devices"],
        precision=finalized_conf.get("precision", "32-true"),
        logger=build_logger(finalized_conf),
        callbacks=build_callbacks(finalized_conf, training=training),
        max_epochs=finalized_conf["max_epochs"],
        gradient_clip_algorithm=finalized_conf.get("gradient_clip_algorithm", "norm"),
        gradient_clip_val=finalized_conf["gradient_clip_val"],
        default_root_dir=finalized_conf["save_root"],
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


def run_training(conf):
    experiment = build_experiment(conf, training=True)
    experiment.trainer.fit(model=experiment.task, datamodule=experiment.datamodule)
    experiment.trainer.test(experiment.task, datamodule=experiment.datamodule, ckpt_path="best")
    return experiment


def run_test(conf, ckpt_path):
    experiment = build_experiment(conf, training=False)
    experiment.trainer.test(
        experiment.task,
        datamodule=experiment.datamodule,
        ckpt_path=resolve_ckpt_path(experiment.conf, ckpt_path),
    )
    return experiment
