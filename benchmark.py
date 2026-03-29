import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

from experiment import (
    CONFIG_ROOT,
    build_failure_metrics,
    finalize_runtime_conf,
    load_experiment_config,
    load_module_from_path,
    run_experiment,
    serialize_value,
)


BENCHMARK_CONFIG_DIR = CONFIG_ROOT / "benchmarks"
BENCHMARK_ALLOWED_KEYS = {"name", "experiment", "seeds", "search_config", "param_space"}
BENCHMARK_SEARCH_CONFIG_DEFAULTS = {
    "num_samples": 1,
    "cpus_per_trial": 2,
    "gpus_per_trial": 0.0,
    "num_gpus": 0,
}
BENCHMARK_SEARCH_ALLOWED_KEYS = set(BENCHMARK_SEARCH_CONFIG_DEFAULTS)
RUN_COLUMNS_BASE = [
    "benchmark_name",
    "benchmark_run_hash",
    "experiment",
    "status",
    "error",
    "task_name",
    "model_name",
    "dataset_name",
    "hist_len",
    "pred_len",
    "seed",
    "val_metric_name",
    "val_metric_value",
    "mae",
    "mse",
    "rmse",
    "conf_hash",
    "ckpt_path",
    "exp_dir",
]
SUMMARY_COLUMNS_BASE = [
    "benchmark_name",
    "benchmark_run_hash",
    "experiment",
    "val_metric_name",
    "val_metric_mean",
    "val_metric_std",
    "mae_mean",
    "mae_std",
    "mse_mean",
    "mse_std",
    "rmse_mean",
    "rmse_std",
    "num_successful_seeds",
    "expected_num_seeds",
    "all_seeds_succeeded",
]


def _json_ready(value):
    value = serialize_value(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(data), handle, indent=2, sort_keys=True, ensure_ascii=False)


def _resolve_benchmark_path(benchmark_ref):
    ref_path = Path(benchmark_ref).expanduser()
    if ref_path.exists():
        return ref_path.resolve()

    relative_ref = Path(benchmark_ref)
    if relative_ref.suffix != ".py":
        relative_ref = relative_ref.with_suffix(".py")
    resolved_path = (BENCHMARK_CONFIG_DIR / relative_ref).resolve()
    if resolved_path.exists():
        return resolved_path
    raise FileNotFoundError("benchmark config not found: {}".format(benchmark_ref))


def _load_python_benchmark(path):
    module_hash = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    module = load_module_from_path("easytsf_benchmark_{}".format(module_hash), str(path))
    if not hasattr(module, "benchmark"):
        raise ValueError("benchmark module must define benchmark: {}".format(path))
    return module.benchmark


def _normalize_search_config(raw_search_config, benchmark_path):
    if raw_search_config is None:
        raw_search_config = {}
    if not isinstance(raw_search_config, dict):
        raise ValueError("benchmark search_config must be a mapping: {}".format(benchmark_path))
    unknown_keys = set(raw_search_config) - BENCHMARK_SEARCH_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError("unsupported search_config keys in {}: {}".format(benchmark_path, sorted(unknown_keys)))

    search_config = {**BENCHMARK_SEARCH_CONFIG_DEFAULTS, **raw_search_config}
    search_config["num_samples"] = int(search_config["num_samples"])
    search_config["cpus_per_trial"] = int(search_config["cpus_per_trial"])
    search_config["gpus_per_trial"] = float(search_config["gpus_per_trial"])
    search_config["num_gpus"] = int(search_config["num_gpus"])
    return search_config


def load_benchmark_config(benchmark_ref):
    benchmark_path = _resolve_benchmark_path(benchmark_ref)
    raw_conf = _load_python_benchmark(benchmark_path)
    if not isinstance(raw_conf, dict):
        raise ValueError("benchmark config must be a mapping: {}".format(benchmark_path))

    unknown_keys = set(raw_conf) - BENCHMARK_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError("unsupported benchmark keys in {}: {}".format(benchmark_path, sorted(unknown_keys)))

    benchmark_name = raw_conf.get("name")
    if not isinstance(benchmark_name, str) or benchmark_name.strip() == "":
        raise ValueError("benchmark name must be a non-empty string: {}".format(benchmark_path))

    experiment_ref = raw_conf.get("experiment")
    if not isinstance(experiment_ref, str) or experiment_ref.strip() == "":
        raise ValueError("benchmark experiment must be a non-empty string: {}".format(benchmark_path))

    if "param_space" not in raw_conf:
        raise ValueError("benchmark param_space must be explicitly provided: {}".format(benchmark_path))
    param_space = raw_conf["param_space"]
    if not isinstance(param_space, dict):
        raise ValueError("benchmark param_space must be a mapping: {}".format(benchmark_path))
    if "seed" in param_space:
        raise ValueError("benchmark param_space must not define seed: {}".format(benchmark_path))

    seeds = raw_conf.get("seeds")
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError("benchmark seeds must be a non-empty list: {}".format(benchmark_path))

    base_conf = load_experiment_config(experiment_ref)
    return {
        "path": benchmark_path,
        "name": benchmark_name.strip(),
        "experiment": experiment_ref,
        "base_conf": dict(base_conf),
        "seeds": [int(seed) for seed in seeds],
        "search_config": _normalize_search_config(raw_conf.get("search_config"), benchmark_path),
        "param_space": dict(param_space),
    }


def _serialize_for_hash(value):
    value = serialize_value(value)
    if isinstance(value, dict):
        return {key: _serialize_for_hash(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_serialize_for_hash(item) for item in value]
    return value


def build_benchmark_run_hash(benchmark_conf, runtime_overrides, hash_len=10):
    payload = {
        "benchmark_name": benchmark_conf["name"],
        "benchmark_path": str(benchmark_conf["path"]),
        "experiment": benchmark_conf["experiment"],
        "seeds": list(benchmark_conf["seeds"]),
        "search_config": _serialize_for_hash(benchmark_conf["search_config"]),
        "param_space": _serialize_for_hash(benchmark_conf["param_space"]),
        "runtime_overrides": _serialize_for_hash(
            {
                "data_root": runtime_overrides.get("data_root"),
                "devices": runtime_overrides.get("devices"),
                "accelerator": runtime_overrides.get("accelerator"),
            }
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.md5()
    digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()[:hash_len]


def build_benchmark_dir(save_root, benchmark_name, benchmark_run_hash):
    return (Path(save_root).expanduser() / "benchmarks" / benchmark_name / benchmark_run_hash).resolve()


def _resource_spec(search_config):
    resources = {"cpu": search_config["cpus_per_trial"]}
    if search_config["gpus_per_trial"] > 0:
        resources["gpu"] = search_config["gpus_per_trial"]
    return resources


def _trial_trainable(trial_config, base_conf, runtime_overrides):
    conf = finalize_runtime_conf(base_conf, runtime_overrides={**runtime_overrides, **trial_config})
    try:
        return run_experiment(conf, enable_progress_bar=False)
    except Exception as error:
        return build_failure_metrics(conf, error)


def build_runs_rows(result_grid, benchmark_conf, benchmark_run_hash, param_keys):
    rows = []
    for result in result_grid:
        metrics = dict(result.metrics or {})
        row = {
            "benchmark_name": benchmark_conf["name"],
            "benchmark_run_hash": benchmark_run_hash,
            "experiment": benchmark_conf["experiment"],
        }
        for key in param_keys:
            row[key] = serialize_value(result.config.get(key))
        row["seed"] = serialize_value(result.config.get("seed"))
        for key in RUN_COLUMNS_BASE[3:]:
            row[key] = serialize_value(metrics.get(key))
        rows.append(row)
    rows.sort(key=lambda row: tuple(serialize_value(row.get(key)) for key in ["seed", *param_keys]))
    return rows


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_runs_report(benchmark_dir, rows, param_keys):
    columns = RUN_COLUMNS_BASE[:3] + param_keys + RUN_COLUMNS_BASE[3:]
    normalized_rows = [{column: row.get(column) for column in columns} for row in rows]
    return _write_csv(Path(benchmark_dir) / "runs.csv", columns, normalized_rows)


def _metric_mean(values):
    return statistics.mean(values) if values else None


def _metric_std(values):
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return statistics.pstdev(values)


def build_summary_rows(rows, param_keys, seeds):
    grouped = {}
    for row in rows:
        group_key = tuple(serialize_value(row.get(key)) for key in param_keys)
        grouped.setdefault(group_key, []).append(row)

    summary_rows = []
    for group_key in sorted(grouped):
        group_rows = grouped[group_key]
        successful_rows = [row for row in group_rows if row.get("status") == "success"]
        val_metric_values = [float(row["val_metric_value"]) for row in successful_rows if row.get("val_metric_value") is not None]
        mae_values = [float(row["mae"]) for row in successful_rows if row.get("mae") is not None]
        mse_values = [float(row["mse"]) for row in successful_rows if row.get("mse") is not None]
        rmse_values = [float(row["rmse"]) for row in successful_rows if row.get("rmse") is not None]
        summary_row = {
            "benchmark_name": group_rows[0]["benchmark_name"],
            "benchmark_run_hash": group_rows[0]["benchmark_run_hash"],
            "experiment": group_rows[0]["experiment"],
            "val_metric_name": group_rows[0].get("val_metric_name"),
            "val_metric_mean": _metric_mean(val_metric_values),
            "val_metric_std": _metric_std(val_metric_values),
            "mae_mean": _metric_mean(mae_values),
            "mae_std": _metric_std(mae_values),
            "mse_mean": _metric_mean(mse_values),
            "mse_std": _metric_std(mse_values),
            "rmse_mean": _metric_mean(rmse_values),
            "rmse_std": _metric_std(rmse_values),
            "num_successful_seeds": len(successful_rows),
            "expected_num_seeds": len(seeds),
            "all_seeds_succeeded": len(successful_rows) == len(seeds),
        }
        for key, value in zip(param_keys, group_key, strict=True):
            summary_row[key] = value
        summary_rows.append(summary_row)
    return summary_rows


def write_summary_report(benchmark_dir, rows, param_keys, seeds):
    columns = SUMMARY_COLUMNS_BASE[:3] + param_keys + SUMMARY_COLUMNS_BASE[3:]
    summary_rows = build_summary_rows(rows, param_keys, seeds)
    normalized_rows = [{column: row.get(column) for column in columns} for row in summary_rows]
    return _write_csv(Path(benchmark_dir) / "summary.csv", columns, normalized_rows), summary_rows


def select_best_config(summary_rows, param_keys):
    eligible_rows = [
        row for row in summary_rows
        if row.get("all_seeds_succeeded") and row.get("val_metric_mean") is not None
    ]
    if not eligible_rows:
        return {}
    best_row = min(eligible_rows, key=lambda row: float(row["val_metric_mean"]))
    return {key: serialize_value(best_row.get(key)) for key in param_keys}


def run_benchmark(benchmark_ref, runtime_overrides):
    benchmark_conf = load_benchmark_config(benchmark_ref)
    benchmark_run_hash = build_benchmark_run_hash(benchmark_conf, runtime_overrides)
    benchmark_dir = build_benchmark_dir(runtime_overrides["save_root"], benchmark_conf["name"], benchmark_run_hash)
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    param_keys = sorted(benchmark_conf["param_space"].keys())
    ray_results_dir = benchmark_dir / "ray_results"
    benchmark_manifest = {
        "benchmark_name": benchmark_conf["name"],
        "benchmark_run_hash": benchmark_run_hash,
        "benchmark_ref": str(benchmark_conf["path"]),
        "experiment": benchmark_conf["experiment"],
        "seeds": list(benchmark_conf["seeds"]),
        "param_space": benchmark_conf["param_space"],
        "search_config": benchmark_conf["search_config"],
        "runtime_overrides": {
            "data_root": runtime_overrides.get("data_root"),
            "save_root": runtime_overrides.get("save_root"),
            "devices": runtime_overrides.get("devices"),
            "accelerator": runtime_overrides.get("accelerator"),
        },
    }
    _save_json(benchmark_dir / "benchmark.json", benchmark_manifest)

    import ray
    from ray import tune

    owns_ray = not ray.is_initialized()
    try:
        if owns_ray:
            if benchmark_conf["search_config"]["num_gpus"] > 0:
                ray.init(num_gpus=benchmark_conf["search_config"]["num_gpus"])
            else:
                ray.init()

        trial_space = dict(benchmark_conf["param_space"])
        trial_space["seed"] = tune.grid_search(list(benchmark_conf["seeds"]))
        trainable = tune.with_parameters(
            _trial_trainable,
            base_conf=benchmark_conf["base_conf"],
            runtime_overrides=runtime_overrides,
        )
        trainable = tune.with_resources(trainable, _resource_spec(benchmark_conf["search_config"]))

        tuner = tune.Tuner(
            trainable=trainable,
            param_space=trial_space,
            tune_config=tune.TuneConfig(
                metric=benchmark_conf["base_conf"]["val_metric"],
                mode="min",
                num_samples=benchmark_conf["search_config"]["num_samples"],
            ),
            run_config=tune.RunConfig(
                name="ray",
                storage_path=str(ray_results_dir.resolve()),
            ),
        )
        result_grid = tuner.fit()
        rows = build_runs_rows(result_grid, benchmark_conf, benchmark_run_hash, param_keys)
    finally:
        if owns_ray and ray.is_initialized():
            ray.shutdown()

    runs_path = write_runs_report(benchmark_dir, rows, param_keys)
    summary_path, summary_rows = write_summary_report(benchmark_dir, rows, param_keys, benchmark_conf["seeds"])
    best_config = select_best_config(summary_rows, param_keys)
    best_config_path = benchmark_dir / "best_config.json"
    _save_json(best_config_path, best_config)

    return {
        "benchmark_name": benchmark_conf["name"],
        "benchmark_run_hash": benchmark_run_hash,
        "benchmark_dir": str(benchmark_dir),
        "runs_path": str(runs_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "best_config_path": str(best_config_path.resolve()),
        "best_config": best_config,
        "ray_results_dir": str(ray_results_dir.resolve()),
        "trial_count": len(rows),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a Ray Tune benchmark for EasyTSF experiments.")
    parser.add_argument("benchmark_py", help="Benchmark Python path or config/benchmarks reference.")
    parser.add_argument("--data-root", required=True, help="Dataset root directory.")
    parser.add_argument("--save-root", required=True, help="Artifact root directory.")
    parser.add_argument("--devices", default=None, help="Lightning devices value passed to each trial.")
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator value passed to each trial.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    runtime_overrides = {
        "data_root": args.data_root,
        "save_root": args.save_root,
        "devices": args.devices,
        "accelerator": args.accelerator,
    }
    try:
        result = run_benchmark(args.benchmark_py, runtime_overrides=runtime_overrides)
    except Exception as error:
        failure = {
            "benchmark": args.benchmark_py,
            "status": "failed",
            "error": str(error),
        }
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
