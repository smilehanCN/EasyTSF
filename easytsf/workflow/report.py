import csv
import hashlib
import math
from pathlib import Path

import pandas as pd
import yaml

from .config import load_module_from_path


FIXED_COLUMNS = ["model", "dataset", "hist_len", "pred_len"]
FIXED_COLUMN_MAP = {
    "model": "model",
    "dataset": "dataset",
    "hist_len": "hist_len",
    "pred_len": "pred_len",
}


def _load_benchmark_config(benchmark_ref):
    benchmark_path = Path(benchmark_ref).expanduser().resolve()
    module_name = "easytsf_benchmark_report_{}".format(hashlib.md5(str(benchmark_path).encode("utf-8")).hexdigest())
    raw_conf = load_module_from_path(module_name, str(benchmark_path)).benchmark_config
    return {
        "benchmark_path": benchmark_path,
        "param_space": dict(raw_conf["param_space"]),
        "search_save_dir": Path(raw_conf["search_save_dir"]).expanduser().resolve(),
    }


def _parameter_columns(param_space):
    return [key for key in param_space if key not in FIXED_COLUMNS]


def _extract_numeric_metric(value, run_dir, column):
    if value in {None, ""}:
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Failed to parse metric column '{}' in '{}'".format(column, run_dir)
        ) from exc
    return None if math.isnan(value) else value


def _find_run_dirs(results_dir):
    return sorted(
        metrics_path.parent
        for metrics_path in results_dir.rglob("metrics.csv")
        if (metrics_path.parent / "hparams.yaml").is_file()
    )


def _load_hparams(run_dir):
    hparams_path = run_dir / "hparams.yaml"
    try:
        with hparams_path.open("r", encoding="utf-8") as handle:
            hparams = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - exact parser exception type is implementation-specific
        raise ValueError("Failed to parse hparams in '{}'".format(run_dir)) from exc
    if not isinstance(hparams, dict):
        raise ValueError("Expected mapping hparams in '{}'".format(run_dir))
    return hparams


def _read_metrics(run_dir):
    metrics_path = run_dir / "metrics.csv"
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    except Exception as exc:  # pragma: no cover - csv parser exception types depend on runtime
        raise ValueError("Failed to parse metrics in '{}'".format(run_dir)) from exc

    test_columns = sorted(name for name in fieldnames if name.startswith("test"))
    val_columns = sorted(name for name in fieldnames if name.startswith("val"))
    metrics = {}

    for column in test_columns:
        metrics[column] = None
        for row in reversed(rows):
            value = _extract_numeric_metric(row.get(column), run_dir, column)
            if value is not None:
                metrics[column] = value
                break

    for column in val_columns:
        values = []
        for row in rows:
            value = _extract_numeric_metric(row.get(column), run_dir, column)
            if value is not None:
                values.append(value)
        metrics[column] = min(values) if values else None

    return metrics, test_columns, val_columns


def _build_run_record(run_dir, parameter_columns):
    hparams = _load_hparams(run_dir)
    metrics, test_columns, val_columns = _read_metrics(run_dir)

    record = {}
    for target_key, source_key in FIXED_COLUMN_MAP.items():
        if source_key not in hparams:
            raise ValueError("Missing hparam '{}' in '{}'".format(source_key, run_dir))
        record[target_key] = hparams[source_key]

    if "seed" not in hparams:
        raise ValueError("Missing hparam 'seed' in '{}'".format(run_dir))
    record["seed"] = hparams["seed"]

    for column in parameter_columns:
        record[column] = hparams.get(column)

    record.update(metrics)
    return record, test_columns, val_columns


def _build_run_dataframe(run_dirs, parameter_columns):
    records = []
    all_test_columns = set()
    all_val_columns = set()

    for run_dir in run_dirs:
        record, test_columns, val_columns = _build_run_record(run_dir, parameter_columns)
        records.append(record)
        all_test_columns.update(test_columns)
        all_val_columns.update(val_columns)

    if not records:
        raise ValueError("No valid run directories found")

    test_columns = sorted(all_test_columns)
    val_columns = sorted(all_val_columns)
    ordered_columns = FIXED_COLUMNS + parameter_columns + ["seed"] + test_columns + val_columns

    run_df = pd.DataFrame.from_records(records)
    for column in ordered_columns:
        if column not in run_df.columns:
            run_df[column] = pd.NA
    run_df = run_df.loc[:, ordered_columns]
    return run_df, test_columns, val_columns


def _raise_on_duplicate_seed(run_df, group_columns):
    duplicate_mask = run_df.duplicated(subset=group_columns + ["seed"], keep=False)
    if not duplicate_mask.any():
        return

    duplicated = run_df.loc[duplicate_mask, group_columns + ["seed"]].drop_duplicates().to_dict("records")
    raise ValueError("Duplicate seed found for benchmark report groups: {}".format(duplicated))


def _aggregate_metrics(run_df, parameter_columns, test_columns, val_columns):
    group_columns = FIXED_COLUMNS + parameter_columns
    _raise_on_duplicate_seed(run_df, group_columns)

    metric_columns = test_columns + val_columns
    grouped = run_df.groupby(group_columns, dropna=False, sort=True)
    mean_df = grouped[metric_columns].mean().add_suffix("_mean")
    std_df = grouped[metric_columns].std(ddof=1).add_suffix("_std")
    report_df = pd.concat([mean_df, std_df], axis=1).reset_index()

    metric_output_columns = []
    for column in test_columns:
        metric_output_columns.extend(["{}_mean".format(column), "{}_std".format(column)])
    for column in val_columns:
        metric_output_columns.extend(["{}_mean".format(column), "{}_std".format(column)])

    report_df = report_df.loc[:, group_columns + metric_output_columns]
    report_df = report_df.sort_values(group_columns, kind="stable", na_position="last").reset_index(drop=True)
    return report_df


def build_benchmark_report(benchmark_ref, results_dir=None, out_path=None):
    benchmark_conf = _load_benchmark_config(benchmark_ref)
    parameter_columns = _parameter_columns(benchmark_conf["param_space"])

    resolved_results_dir = Path(results_dir).expanduser().resolve() if results_dir else benchmark_conf["search_save_dir"]
    if not resolved_results_dir.exists():
        raise FileNotFoundError("Benchmark results directory does not exist: {}".format(resolved_results_dir))

    run_dirs = _find_run_dirs(resolved_results_dir)
    if not run_dirs:
        raise ValueError("No valid run directories found under '{}'".format(resolved_results_dir))

    run_df, test_columns, val_columns = _build_run_dataframe(run_dirs, parameter_columns)
    report_df = _aggregate_metrics(run_df, parameter_columns, test_columns, val_columns)

    if out_path is not None:
        output_path = Path(out_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)

    return report_df


__all__ = ["build_benchmark_report"]
