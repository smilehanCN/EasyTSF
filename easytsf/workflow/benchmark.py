import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from .config import finalize_runtime_conf
from .config import BENCHMARK_CONFIG_DIR, _save_json, load_experiment_config, load_module_from_path
from .experiment import _serialize_value, resolve_ckpt_path, run_experiment


BENCHMARK_ALLOWED_KEYS = {"name", "seeds", "search_config", "experiment", "param_space"}
BENCHMARK_SEARCH_CONFIG_DEFAULTS = {
    "num_samples": 1,
    "cpus_per_trial": 2,
    "gpus_per_trial": 0.5,
    "num_gpus": 0,
}
BENCHMARK_SEARCH_ALLOWED_KEYS = set(BENCHMARK_SEARCH_CONFIG_DEFAULTS)


def _slug(value):
    slug = "".join(char if char.isalnum() else "_" for char in str(value or "")).strip("_").lower()
    if slug == "":
        raise ValueError("cannot derive slug from empty value")
    return slug


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


def _benchmark_name_from_conf(conf):
    return "{}_{}".format(_slug(conf["model_name"]), _slug(conf["dataset_name"]))


def _build_benchmark_dir(save_root, benchmark_name):
    return Path(save_root) / "benchmarks" / benchmark_name


def _build_benchmark_exp_dir(benchmark_dir, conf_hash, seed):
    return Path(benchmark_dir) / conf_hash / "seed_{}".format(seed)


def _build_benchmark_artifact_paths(benchmark_dir):
    benchmark_dir = Path(benchmark_dir)
    search_dir = benchmark_dir / "search"
    return {
        "benchmark_dir": benchmark_dir,
        "resolved_benchmark_path": benchmark_dir / "benchmark.json",
        "search_dir": search_dir,
        "trial_report_path": search_dir / "trial_report.csv",
        "best_trial_report_path": search_dir / "best_trial_report.csv",
        "best_params_path": search_dir / "best_params.json",
        "tune_meta_path": search_dir / "tune_meta.json",
        "runs_path": benchmark_dir / "runs.csv",
        "summary_path": benchmark_dir / "summary.csv",
    }


def _has_resumeable_search_artifacts(paths):
    required_paths = [
        paths["best_params_path"],
        paths["trial_report_path"],
        paths["best_trial_report_path"],
    ]
    return all(path.exists() for path in required_paths)


def _extract_scalar_metric(value):
    value = _serialize_value(value)
    if value in {None, ""}:
        return None
    try:
        value = float(value)
        return None if math.isnan(value) else value
    except (TypeError, ValueError):
        return value


def load_saved_metrics(conf):
    path = Path(conf["exp_dir"]) / "metrics.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    def last_logged_value(name):
        if not name:
            return None
        for row in reversed(rows):
            value = row.get(name)
            if value not in {None, ""}:
                return _extract_scalar_metric(value)
        return None

    mae = last_logged_value("test/mae")
    mse = last_logged_value("test/mse")
    rmse = last_logged_value("test/rmse")
    if mae is None and mse is None and rmse is None:
        return None
    try:
        ckpt_path = resolve_ckpt_path(conf, "best")
    except FileNotFoundError:
        ckpt_path = None
    return {
        "task_name": conf.get("task_name", "mtsf"),
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
        "val_metric_value": last_logged_value(conf.get("val_metric")),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
    }


def load_benchmark(benchmark_ref):
    benchmark_path = _resolve_benchmark_path(benchmark_ref)
    raw_conf = _load_python_benchmark(benchmark_path)
    if not isinstance(raw_conf, dict):
        raise ValueError("benchmark config must be a mapping: {}".format(benchmark_path))

    unknown_keys = set(raw_conf) - BENCHMARK_ALLOWED_KEYS
    if unknown_keys:
        raise ValueError("unsupported benchmark keys in {}: {}".format(benchmark_path, sorted(unknown_keys)))

    experiment_ref = raw_conf.get("experiment")
    if not isinstance(experiment_ref, str) or experiment_ref.strip() == "":
        raise ValueError("benchmark experiment must be a non-empty string: {}".format(benchmark_path))

    if "param_space" not in raw_conf:
        raise ValueError("benchmark param_space must be explicitly provided: {}".format(benchmark_path))
    param_space = raw_conf.get("param_space")
    if not isinstance(param_space, dict):
        raise ValueError("benchmark param_space must be a mapping: {}".format(benchmark_path))

    seeds = raw_conf.get("seeds", [0])
    if not isinstance(seeds, list) or len(seeds) == 0:
        raise ValueError("benchmark seeds must be a non-empty list: {}".format(benchmark_path))

    base_conf = load_experiment_config(experiment_ref)
    derived_name = _benchmark_name_from_conf(base_conf)
    configured_name = raw_conf.get("name")
    if configured_name is not None and configured_name != derived_name:
        raise ValueError(
            "benchmark name must match the derived model_dataset '{}': {}".format(derived_name, benchmark_path)
        )

    return {
        "path": benchmark_path,
        "name": derived_name,
        "experiment": experiment_ref,
        "base_conf": dict(base_conf),
        "seeds": [int(seed) for seed in seeds],
        "search_config": _normalize_search_config(raw_conf.get("search_config"), benchmark_path),
        "param_space": dict(param_space),
    }


def _build_failure_record(conf, error):
    return {
        "benchmark_name": conf["benchmark_name"],
        "experiment": conf["experiment_ref"],
        "resume_hit": False,
        "task_name": conf.get("task_name", "mtsf"),
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
        "val_metric_name": conf.get("val_metric"),
        "val_metric_value": None,
        "mae": None,
        "mse": None,
    }


def _build_tune_failure_record(benchmark_name, experiment_ref, search_dir, conf, error):
    metrics = _build_failure_record(
        {
            **conf,
            "benchmark_name": benchmark_name,
            "experiment_ref": experiment_ref,
        },
        "tune failed: {}".format(error),
    )
    metrics["exp_dir"] = str(Path(search_dir).resolve())
    return metrics


def _build_benchmark_row(benchmark_name, experiment_ref, metrics, resume_hit):
    row = {
        "benchmark_name": benchmark_name,
        "experiment": experiment_ref,
        "resume_hit": bool(resume_hit),
    }
    row.update(metrics)
    return row


def _write_runs_report(benchmark_dir, rows):
    benchmark_dir = Path(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    runs_path = benchmark_dir / "runs.csv"
    columns = [
        "benchmark_name",
        "experiment",
        "resume_hit",
        "task_name",
        "model_name",
        "dataset_name",
        "hist_len",
        "pred_len",
        "seed",
        "status",
        "mae",
        "mse",
        "rmse",
        "val_metric_name",
        "val_metric_value",
        "conf_hash",
        "ckpt_path",
        "exp_dir",
        "error",
    ]
    normalized_rows = []
    for row in rows:
        normalized_rows.append({column: row.get(column) for column in columns})
    normalized_rows.sort(key=lambda row: (row["conf_hash"], row["seed"]))

    with runs_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(normalized_rows)
    return runs_path, normalized_rows


def _write_summary_report(benchmark_dir, runs_rows):
    benchmark_dir = Path(benchmark_dir)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary_path = benchmark_dir / "summary.csv"
    summary_columns = [
        "task_name",
        "model_name",
        "dataset_name",
        "hist_len",
        "pred_len",
        "mae_mean",
        "mae_std",
        "mse_mean",
        "mse_std",
        "rmse_mean",
        "rmse_std",
        "num_seeds",
    ]
    grouped = {}
    for row in runs_rows:
        if row.get("status") != "success":
            continue
        group_key = (row["task_name"], row["model_name"], row["dataset_name"], row["hist_len"], row["pred_len"])
        grouped.setdefault(group_key, {"mae": [], "mse": [], "rmse": []})
        if row.get("mae") is not None:
            grouped[group_key]["mae"].append(float(row["mae"]))
        if row.get("mse") is not None:
            grouped[group_key]["mse"].append(float(row["mse"]))
        if row.get("rmse") is not None:
            grouped[group_key]["rmse"].append(float(row["rmse"]))

    summary_rows = []
    for group_key in sorted(grouped):
        metric_group = grouped[group_key]
        mae_values = metric_group["mae"]
        mse_values = metric_group["mse"]
        rmse_values = metric_group["rmse"]
        summary_rows.append(
            {
                "task_name": group_key[0],
                "model_name": group_key[1],
                "dataset_name": group_key[2],
                "hist_len": group_key[3],
                "pred_len": group_key[4],
                "mae_mean": statistics.mean(mae_values) if mae_values else None,
                "mae_std": statistics.pstdev(mae_values) if len(mae_values) > 1 else 0.0 if mae_values else None,
                "mse_mean": statistics.mean(mse_values) if mse_values else None,
                "mse_std": statistics.pstdev(mse_values) if len(mse_values) > 1 else 0.0 if mse_values else None,
                "rmse_mean": statistics.mean(rmse_values) if rmse_values else None,
                "rmse_std": statistics.pstdev(rmse_values) if len(rmse_values) > 1 else 0.0 if rmse_values else None,
                "num_seeds": len(mae_values),
            }
        )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def _write_placeholder_search_reports(paths, best_params):
    rows = [{"trial_id": 0, **best_params}] if best_params else [{"trial_id": 0}]
    fieldnames = sorted(rows[0].keys())
    for target_path in (paths["trial_report_path"], paths["best_trial_report_path"]):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _build_tune_reporter(param_space, metric, mode):
    from ray.tune import CLIReporter

    return CLIReporter(
        parameter_columns=list(param_space.keys()),
        metric_columns=[metric],
        metric=metric,
        mode=mode,
        sort_by_metric=True,
    )


def _save_tune_reports(result_grid, metric, mode, report_dir):
    target_dir = Path(report_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    trial_report_path = target_dir / "trial_report.csv"
    best_report_path = target_dir / "best_trial_report.csv"

    result_grid.get_dataframe().to_csv(trial_report_path, index=False)
    result_grid.get_dataframe(filter_metric=metric, filter_mode=mode).to_csv(best_report_path, index=False)
    return trial_report_path, best_report_path


def _build_tune_callbacks(conf):
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

    return [
        TuneReportCheckpointCallback(
            {conf["val_metric"]: conf["val_metric"]},
            save_checkpoints=False,
            on="validation_end",
        )
    ]


def _tune_train_func(hyper_conf, base_conf, trial_root):
    conf = finalize_runtime_conf(base_conf, overrides=hyper_conf)
    conf["exp_dir"] = str(_build_benchmark_exp_dir(trial_root, conf["conf_hash"], conf["seed"]))
    run_experiment(conf, extra_callbacks=_build_tune_callbacks(conf))


def _run_tune_search(param_space, init_conf, search_dir, search_config):
    search_dir = Path(search_dir)
    if len(param_space) == 0:
        return {
            "metric": init_conf["val_metric"],
            "mode": "min",
            "experiment_path": str(search_dir.resolve()),
            "trial_report_path": str((search_dir / "trial_report.csv").resolve()),
            "best_trial_report_path": str((search_dir / "best_trial_report.csv").resolve()),
            "best_config": {},
            "best_metrics": {},
            "num_errors": 0,
            "errors": [],
            "status": "fixed",
        }

    import ray
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler

    if not ray.is_initialized():
        if search_config["num_gpus"] > 0:
            ray.init(num_gpus=search_config["num_gpus"])
        else:
            ray.init()

    metric = init_conf["val_metric"]
    scheduler = FIFOScheduler()
    reporter = _build_tune_reporter(param_space, metric, "min")
    trainable = tune.with_parameters(_tune_train_func, base_conf=init_conf, trial_root=search_dir / "trials")
    trainable = tune.with_resources(
        trainable,
        resources={"cpu": search_config["cpus_per_trial"], "gpu": search_config["gpus_per_trial"]},
    )

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode="min",
            scheduler=scheduler,
            num_samples=search_config["num_samples"],
        ),
        run_config=tune.RunConfig(
            name="ray",
            storage_path=str((search_dir / "ray_results").expanduser().resolve()),
            progress_reporter=reporter,
        ),
    )

    result_grid = tuner.fit()
    trial_report_path, best_report_path = _save_tune_reports(result_grid, metric, "min", report_dir=search_dir)
    best_result = result_grid.get_best_result(metric=metric, mode="min", scope="all")
    return {
        "metric": metric,
        "mode": "min",
        "experiment_path": str(Path(result_grid.experiment_path).resolve()),
        "trial_report_path": str(trial_report_path.resolve()),
        "best_trial_report_path": str(best_report_path.resolve()),
        "best_config": {key: _serialize_value(value) for key, value in dict(best_result.config).items()},
        "best_metrics": {key: _serialize_value(value) for key, value in dict(best_result.metrics).items()},
        "num_errors": int(result_grid.num_errors),
        "errors": [str(error) for error in result_grid.errors],
        "status": "searched",
    }


def run_benchmark(
    benchmark_ref,
    runtime_overrides=None,
    dry_run=False,
    resume=True,
    fail_fast=False,
):
    benchmark_conf = load_benchmark(benchmark_ref)
    runtime_overrides = dict(runtime_overrides or {})
    benchmark_dir = _build_benchmark_dir(runtime_overrides.get("save_root", "save"), benchmark_conf["name"])
    artifact_paths = _build_benchmark_artifact_paths(benchmark_dir)
    base_conf = dict(benchmark_conf["base_conf"])
    seeds = list(benchmark_conf["seeds"])

    if dry_run:
        print(
            "[dry-run] benchmark={} experiment={} dataset={} hist_len={} pred_len={} tune_seed={} eval_seeds={} param_keys={} benchmark_dir={}".format(
                benchmark_conf["name"],
                benchmark_conf["experiment"],
                base_conf["dataset_name"],
                base_conf["hist_len"],
                base_conf["pred_len"],
                seeds[0],
                seeds,
                sorted(benchmark_conf["param_space"].keys()),
                benchmark_dir,
            )
        )
        return {
            "benchmark_name": benchmark_conf["name"],
            "benchmark_dir": str(benchmark_dir.resolve()),
            "run_count": len(seeds),
            "rows": [],
            "runs_path": str(artifact_paths["runs_path"].resolve()),
            "summary_path": str(artifact_paths["summary_path"].resolve()),
            "search_dir": str(artifact_paths["search_dir"].resolve()),
            "trial_report_path": str(artifact_paths["trial_report_path"].resolve()),
            "best_trial_report_path": str(artifact_paths["best_trial_report_path"].resolve()),
            "best_params_path": str(artifact_paths["best_params_path"].resolve()),
            "tune_meta_path": str(artifact_paths["tune_meta_path"].resolve()),
        }

    artifact_paths["benchmark_dir"].mkdir(parents=True, exist_ok=True)
    _save_json(
        artifact_paths["resolved_benchmark_path"],
        {
            "benchmark_name": benchmark_conf["name"],
            "benchmark_ref": str(benchmark_conf["path"]),
            "experiment": benchmark_conf["experiment"],
            "seeds": list(seeds),
            "search_config": dict(benchmark_conf["search_config"]),
            "param_space_keys": sorted(benchmark_conf["param_space"].keys()),
            "base_conf": {key: _serialize_value(value) for key, value in sorted(base_conf.items())},
        },
    )

    if resume and _has_resumeable_search_artifacts(artifact_paths):
        print("[resume-search] {} -> {}".format(benchmark_conf["name"], artifact_paths["search_dir"]))
        best_params = dict(json.loads(artifact_paths["best_params_path"].read_text(encoding="utf-8")))
        tune_meta = json.loads(artifact_paths["tune_meta_path"].read_text(encoding="utf-8")) if artifact_paths["tune_meta_path"].exists() else {}
    else:
        print("[search] {} -> {}".format(benchmark_conf["name"], artifact_paths["search_dir"]))
        search_init_conf = {**base_conf, **runtime_overrides, "seed": seeds[0]}
        try:
            search_result = _run_tune_search(
                param_space=benchmark_conf["param_space"],
                init_conf=search_init_conf,
                search_dir=artifact_paths["search_dir"],
                search_config=benchmark_conf["search_config"],
            )
            best_params = dict(search_result["best_config"])
            if search_result["status"] == "fixed":
                _write_placeholder_search_reports(artifact_paths, best_params)
            tune_meta = {
                "benchmark_name": benchmark_conf["name"],
                "experiment": benchmark_conf["experiment"],
                "search_dir": str(artifact_paths["search_dir"].resolve()),
                "search_config": dict(benchmark_conf["search_config"]),
                "param_keys": sorted(benchmark_conf["param_space"].keys()),
                "status": "success",
                "mode": search_result["mode"],
                "metric": search_result["metric"],
                "best_config": dict(best_params),
                "best_metrics": dict(search_result["best_metrics"]),
                "trial_report_path": search_result["trial_report_path"],
                "best_trial_report_path": search_result["best_trial_report_path"],
                "experiment_path": search_result["experiment_path"],
                "num_errors": int(search_result["num_errors"]),
                "errors": list(search_result["errors"]),
                "search_status": search_result["status"],
            }
            _save_json(artifact_paths["best_params_path"], best_params)
            _save_json(artifact_paths["tune_meta_path"], tune_meta)
        except Exception as error:
            _save_json(
                artifact_paths["tune_meta_path"],
                {
                    "benchmark_name": benchmark_conf["name"],
                    "experiment": benchmark_conf["experiment"],
                    "search_dir": str(artifact_paths["search_dir"].resolve()),
                    "search_config": dict(benchmark_conf["search_config"]),
                    "param_keys": sorted(benchmark_conf["param_space"].keys()),
                    "status": "failed",
                    "error": str(error),
                },
            )
            rows = []
            for seed in seeds:
                failed_conf = finalize_runtime_conf(
                    base_conf,
                    overrides={
                        **runtime_overrides,
                        "seed": seed,
                        "exp_dir": str(_build_benchmark_exp_dir(benchmark_dir, "search_failed", seed)),
                    },
                )
                rows.append(
                    _build_benchmark_row(
                        benchmark_conf["name"],
                        benchmark_conf["experiment"],
                        _build_tune_failure_record(
                            benchmark_conf["name"],
                            benchmark_conf["experiment"],
                            artifact_paths["search_dir"],
                            failed_conf,
                            error,
                        ),
                        resume_hit=False,
                    )
                )
            runs_path, runs_rows = _write_runs_report(benchmark_dir, rows)
            summary_path = _write_summary_report(benchmark_dir, runs_rows)
            if fail_fast:
                raise RuntimeError("benchmark stopped because fail_fast=1") from error
            return {
                "benchmark_name": benchmark_conf["name"],
                "benchmark_dir": str(benchmark_dir.resolve()),
                "run_count": len(rows),
                "rows": rows,
                "runs_path": str(runs_path.resolve()),
                "summary_path": str(summary_path.resolve()),
                "search_dir": str(artifact_paths["search_dir"].resolve()),
                "trial_report_path": str(artifact_paths["trial_report_path"].resolve()),
                "best_trial_report_path": str(artifact_paths["best_trial_report_path"].resolve()),
                "best_params_path": str(artifact_paths["best_params_path"].resolve()),
                "tune_meta_path": str(artifact_paths["tune_meta_path"].resolve()),
            }

    rows = []
    stop_error = None
    for seed in seeds:
        conf = finalize_runtime_conf(base_conf, overrides={**runtime_overrides, **best_params, "seed": seed})
        conf["benchmark_name"] = benchmark_conf["name"]
        conf["experiment_ref"] = benchmark_conf["experiment"]
        conf["exp_dir"] = str(_build_benchmark_exp_dir(benchmark_dir, conf["conf_hash"], conf["seed"]))

        existing_metrics = load_saved_metrics(conf) if resume else None
        if existing_metrics and existing_metrics.get("status") == "success":
            print("[resume] {} seed={} -> {}".format(benchmark_conf["name"], conf["seed"], conf["exp_dir"]))
            rows.append(
                _build_benchmark_row(
                    benchmark_conf["name"],
                    benchmark_conf["experiment"],
                    existing_metrics,
                    resume_hit=True,
                )
            )
            continue

        print("[run] {} seed={} -> {}".format(benchmark_conf["name"], conf["seed"], conf["exp_dir"]))
        try:
            result = run_experiment(conf)
            metrics = result if isinstance(result, dict) else load_saved_metrics(conf)
            if metrics is None:
                try:
                    ckpt_path = resolve_ckpt_path(conf, "best")
                except FileNotFoundError:
                    ckpt_path = None
                metrics = {
                    "task_name": conf.get("task_name", "mtsf"),
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
                    "val_metric_value": None,
                    "mae": None,
                    "mse": None,
                }
        except Exception as error:
            metrics = _build_failure_record(conf, error)
            print("[failed] {} seed={} error={}".format(benchmark_conf["name"], conf["seed"], error))
            rows.append(
                _build_benchmark_row(
                    benchmark_conf["name"],
                    benchmark_conf["experiment"],
                    metrics,
                    resume_hit=False,
                )
            )
            if fail_fast:
                stop_error = error
                break
            continue

        rows.append(
            _build_benchmark_row(
                benchmark_conf["name"],
                benchmark_conf["experiment"],
                metrics,
                resume_hit=False,
            )
        )

    runs_path, runs_rows = _write_runs_report(benchmark_dir, rows)
    summary_path = _write_summary_report(benchmark_dir, runs_rows)
    result = {
        "benchmark_name": benchmark_conf["name"],
        "benchmark_dir": str(benchmark_dir.resolve()),
        "run_count": len(rows),
        "rows": rows,
        "runs_path": str(runs_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "search_dir": str(artifact_paths["search_dir"].resolve()),
        "trial_report_path": str(artifact_paths["trial_report_path"].resolve()),
        "best_trial_report_path": str(artifact_paths["best_trial_report_path"].resolve()),
        "best_params_path": str(artifact_paths["best_params_path"].resolve()),
        "tune_meta_path": str(artifact_paths["tune_meta_path"].resolve()),
    }
    if stop_error is not None:
        raise RuntimeError("benchmark stopped because fail_fast=1") from stop_error
    return result
