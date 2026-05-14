from __future__ import annotations

import argparse
import os
from pathlib import Path

from easytsf.task import validate_task_runtime_conf

from .config import finalize_runtime_conf, load_experiment_config, load_module_from_path
from .experiment import run_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_CONFIG = {
    "backend": "ray",
    "num_samples": 1,
    "cpus_per_trial": 1,
    "gpus_per_trial": 0.0,
    "num_gpus": 0,
}


def _resolve_existing_path(path_ref, benchmark_path: Path) -> Path:
    path = Path(path_ref).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        Path.cwd() / path,
        REPO_ROOT / path,
        benchmark_path.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_output_path(path_ref, benchmark_path: Path) -> Path:
    path = Path(path_ref).expanduser()
    if path.is_absolute():
        return path.resolve()
    if (Path.cwd() / path).exists():
        return (Path.cwd() / path).resolve()
    if (REPO_ROOT / path).exists() or not (benchmark_path.parent / path).exists():
        return (REPO_ROOT / path).resolve()
    return (benchmark_path.parent / path).resolve()


def _coerce_search_config(raw_conf: dict) -> dict:
    search_config = dict(DEFAULT_SEARCH_CONFIG)
    search_config.update(dict(raw_conf.get("search_config") or {}))
    search_config["backend"] = str(raw_conf.get("backend", search_config.get("backend", "ray"))).lower()
    search_config["num_samples"] = int(search_config.get("num_samples", 1))
    search_config["cpus_per_trial"] = int(search_config.get("cpus_per_trial", 1))
    search_config["gpus_per_trial"] = float(search_config.get("gpus_per_trial", 0.0))
    search_config["num_gpus"] = int(search_config.get("num_gpus", 0))
    if search_config.get("max_concurrent_trials") not in {None, ""}:
        search_config["max_concurrent_trials"] = int(search_config["max_concurrent_trials"])
    return search_config


def _apply_base_overrides(base_conf: dict, raw_conf: dict) -> dict:
    merged = dict(base_conf)
    for key in ("base_overrides", "runtime_overrides"):
        overrides = raw_conf.get(key)
        if overrides:
            merged.update(dict(overrides))
    return merged


def load_benchmark(benchmark_ref):
    benchmark_path = Path(benchmark_ref).expanduser().resolve()
    raw_conf = load_module_from_path(benchmark_path.stem, str(benchmark_path)).benchmark_config
    if not isinstance(raw_conf, dict):
        raise TypeError("benchmark_config in '{}' must be a dict".format(benchmark_path))

    name = str(raw_conf.get("name") or benchmark_path.stem)
    experiment_path = _resolve_existing_path(raw_conf["experiment"], benchmark_path)
    search_save_dir = _resolve_output_path(
        raw_conf.get("search_save_dir", "save/benchmarks/{}".format(name)),
        benchmark_path,
    )
    search_config = _coerce_search_config(raw_conf)

    base_conf = _apply_base_overrides(load_experiment_config(experiment_path), raw_conf)
    metric = str(raw_conf.get("metric") or search_config.get("metric") or base_conf["val_metric"])
    mode = str(raw_conf.get("mode") or search_config.get("mode") or base_conf.get("val_metric_mode", "min"))
    base_conf["val_metric"] = metric
    base_conf["val_metric_mode"] = mode

    task_spec = validate_task_runtime_conf(base_conf)
    return {
        "name": name,
        "benchmark_path": str(benchmark_path),
        "experiment_path": str(experiment_path),
        "base_conf": base_conf,
        "search_config": search_config,
        "search_save_dir": str(search_save_dir),
        "param_space": dict(raw_conf.get("param_space") or {}),
        "metric": metric,
        "mode": mode,
        "task_name": task_spec.name,
        "task_spec": task_spec,
    }


def _build_tune_reporter(param_space, metric, mode):
    from ray.tune import CLIReporter

    return CLIReporter(
        parameter_columns=list(param_space.keys()),
        metric_columns=[metric],
        metric=metric,
        mode=mode,
        sort_by_metric=True,
    )


def _build_tune_scheduler(search_config):
    scheduler_name = str(search_config.get("scheduler", "fifo")).lower()
    if scheduler_name in {"fifo", "none", ""}:
        from ray.tune.schedulers import FIFOScheduler

        return FIFOScheduler()
    raise ValueError("unsupported benchmark scheduler: {}".format(scheduler_name))


def _tune_train_func(hyper_conf, base_conf, verbose=False):
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

    conf = finalize_runtime_conf(base_conf, overrides=hyper_conf)
    conf["exp_dir"] = str(Path(tune.get_context().get_trial_dir()).resolve())
    if not verbose:
        conf["seed_verbose"] = False
        conf["enable_progress_bar"] = False
        conf["enable_model_summary"] = False
    run_experiment(
        conf,
        extra_callbacks=[
            TuneReportCheckpointCallback(
                metrics={conf["val_metric"]: conf["val_metric"]},
                save_checkpoints=False,
                on="validation_end",
            )
        ],
    )


def _run_ray_benchmark(benchmark_conf, resume=True, verbose=False):
    base_conf = dict(benchmark_conf["base_conf"])
    search_save_dir = Path(benchmark_conf["search_save_dir"]).expanduser().resolve()
    search_storage_path = search_save_dir.parent
    search_experiment_name = search_save_dir.name
    search_config = benchmark_conf["search_config"]
    param_space = benchmark_conf["param_space"]
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    import ray
    from ray import tune

    if not ray.is_initialized():
        ray_init_kwargs = dict(search_config.get("ray_init") or {})
        if search_config["num_gpus"] > 0 and "num_gpus" not in ray_init_kwargs:
            ray_init_kwargs["num_gpus"] = search_config["num_gpus"]
        ray.init(**ray_init_kwargs)

    metric = benchmark_conf["metric"]
    metric_mode = benchmark_conf["mode"]
    reporter = _build_tune_reporter(param_space, metric, metric_mode) if verbose else None
    trainable = tune.with_parameters(_tune_train_func, base_conf=base_conf, verbose=verbose)
    resources = search_config.get("resources")
    if resources is None:
        resources = {"cpu": search_config["cpus_per_trial"], "gpu": search_config["gpus_per_trial"]}
    trainable = tune.with_resources(trainable, resources=resources)

    should_restore = resume and tune.Tuner.can_restore(str(search_save_dir))
    if should_restore:
        tuner = tune.Tuner.restore(
            str(search_save_dir),
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=True,
            param_space=param_space,
        )
    else:
        tune_config_kwargs = {
            "metric": metric,
            "mode": metric_mode,
            "scheduler": _build_tune_scheduler(search_config),
            "num_samples": search_config["num_samples"],
        }
        if search_config.get("max_concurrent_trials") not in {None, ""}:
            tune_config_kwargs["max_concurrent_trials"] = search_config["max_concurrent_trials"]
        tuner = tune.Tuner(
            trainable=trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
            run_config=tune.RunConfig(
                name=search_experiment_name,
                storage_path=str(search_storage_path),
                verbose=1 if verbose else 0,
                progress_reporter=reporter,
            ),
        )

    return tuner.fit()


def run_benchmark(benchmark_ref, resume=True, verbose=False):
    benchmark_conf = load_benchmark(benchmark_ref)
    backend = str(benchmark_conf["search_config"].get("backend", "ray")).lower()
    if backend != "ray":
        raise ValueError("unsupported benchmark backend: {}".format(backend))
    return _run_ray_benchmark(benchmark_conf, resume=resume, verbose=verbose)


def build_cli_parser():
    parser = argparse.ArgumentParser(description="Run an EasyTSF benchmark.")
    parser.add_argument("benchmark", help="Benchmark python file path.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable reuse of existing benchmark artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable Ray Tune progress tables and Lightning progress output.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli_parser().parse_args()
    run_benchmark(args.benchmark, resume=args.resume, verbose=args.verbose)
