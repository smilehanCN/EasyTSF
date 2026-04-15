import argparse
import csv
import math
from pathlib import Path

import pandas as pd
import yaml

from .benchmark import load_benchmark


FIXED_COLUMNS = ["model", "dataset", "hist_len", "pred_len", "conf_hash"]


def _extract_numeric_metric(value):
    if value in {None, ""}:
        return None
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    value = float(value)
    return None if math.isnan(value) else value


def build_benchmark_report(benchmark_ref, results_dir=None, out_path=None):
    benchmark_conf = load_benchmark(benchmark_ref)
    parameter_columns = [key for key in benchmark_conf["param_space"] if key not in FIXED_COLUMNS]
    resolved_results_dir = (
        Path(results_dir).expanduser().resolve()
        if results_dir
        else Path(benchmark_conf["search_save_dir"]).expanduser().resolve()
    )

    records = []
    all_test_columns = set()
    all_val_columns = set()

    for metrics_path in sorted(resolved_results_dir.rglob("metrics.csv")):
        run_dir = metrics_path.parent
        hparams_path = run_dir / "hparams.yaml"
        if not hparams_path.is_file():
            continue

        with hparams_path.open("r", encoding="utf-8") as handle:
            hparams = yaml.safe_load(handle) or {}

        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        test_columns = sorted(name for name in fieldnames if name.startswith("test"))
        val_columns = sorted(name for name in fieldnames if name.startswith("val"))
        metrics = {}

        for column in test_columns:
            metrics[column] = None
            for row in reversed(rows):
                value = _extract_numeric_metric(row.get(column))
                if value is not None:
                    metrics[column] = value
                    break

        for column in val_columns:
            values = [
                value
                for value in (_extract_numeric_metric(row.get(column)) for row in rows)
                if value is not None
            ]
            metrics[column] = min(values) if values else None

        record = {column: hparams[column] for column in FIXED_COLUMNS}
        record["seed"] = hparams["seed"]
        for column in parameter_columns:
            record[column] = hparams.get(column)
        record.update(metrics)

        records.append(record)
        all_test_columns.update(test_columns)
        all_val_columns.update(val_columns)

    test_columns = sorted(all_test_columns)
    val_columns = sorted(all_val_columns)
    ordered_columns = FIXED_COLUMNS + parameter_columns + ["seed"] + test_columns + val_columns

    run_df = pd.DataFrame.from_records(records)
    for column in ordered_columns:
        if column not in run_df.columns:
            run_df[column] = pd.NA
    run_df = run_df.loc[:, ordered_columns]

    group_columns = FIXED_COLUMNS + parameter_columns
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

    if out_path is not None:
        output_path = Path(out_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)

    return report_df


def build_cli_parser():
    parser = argparse.ArgumentParser(description="Build an EasyTSF benchmark report.")
    parser.add_argument("benchmark", help="Benchmark python file path.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override benchmark search_save_dir.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output csv file path. Print csv to stdout when omitted.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli_parser().parse_args()
    report_df = build_benchmark_report(args.benchmark, results_dir=args.results_dir, out_path=args.out_path)
    if args.out_path is None:
        print(report_df.to_csv(index=False), end="")


__all__ = ["build_benchmark_report"]
