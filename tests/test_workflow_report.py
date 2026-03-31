import csv

import pandas as pd
import pytest
import yaml

from easytsf.workflow.report import build_benchmark_report


def _write_benchmark_config(path, search_save_dir, param_space):
    path.write_text(
        "benchmark_config = {\n"
        "    'name': 'demo_benchmark',\n"
        "    'search_save_dir': " + repr(str(search_save_dir)) + ",\n"
        "    'search_config': {\n"
        "        'num_samples': 1,\n"
        "        'cpus_per_trial': 1,\n"
        "        'gpus_per_trial': 0,\n"
        "        'num_gpus': 0,\n"
        "    },\n"
        "    'experiment': 'unused.yaml',\n"
        "    'param_space': " + repr(param_space) + ",\n"
        "}\n",
        encoding="utf-8",
    )


def _write_hparams(path, **kwargs):
    path.write_text(yaml.safe_dump(kwargs, sort_keys=False), encoding="utf-8")


def _write_metrics(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _create_run(run_dir, *, model, dataset, hist_len, pred_len, seed, lr, dropout, metric_rows):
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_hparams(
        run_dir / "hparams.yaml",
        model=model,
        dataset=dataset,
        hist_len=hist_len,
        pred_len=pred_len,
        seed=seed,
        lr=lr,
        dropout=dropout,
    )
    _write_metrics(run_dir / "metrics.csv", metric_rows)


def test_build_benchmark_report_aggregates_runs_and_writes_csv(tmp_path):
    results_dir = tmp_path / "search"
    benchmark_path = tmp_path / "core.py"
    out_path = tmp_path / "reports" / "summary.csv"
    _write_benchmark_config(
        benchmark_path,
        results_dir,
        {
            "model": "ignored_model",
            "dataset": "ignored_dataset",
            "hist_len": 96,
            "pred_len": 12,
            "lr": 0.001,
            "dropout": 0.2,
        },
    )

    _create_run(
        results_dir / "trial_a",
        model="DemoNet",
        dataset="ETTh1",
        hist_len=96,
        pred_len=12,
        seed=1,
        lr=0.001,
        dropout=0.2,
        metric_rows=[
            {"epoch": "0", "val/loss": "0.8", "val/mae": "0.7", "test/mae": "", "test/mse": ""},
            {"epoch": "1", "val/loss": "0.4", "val/mae": "0.5", "test/mae": "", "test/mse": ""},
            {"epoch": "2", "val/loss": "", "val/mae": "", "test/mae": "0.2", "test/mse": "0.04"},
        ],
    )
    _create_run(
        results_dir / "nested" / "trial_b",
        model="DemoNet",
        dataset="ETTh1",
        hist_len=96,
        pred_len=12,
        seed=2,
        lr=0.001,
        dropout=0.2,
        metric_rows=[
            {"epoch": "0", "val/loss": "0.6", "val/mae": "0.4", "test/mae": "", "test/mse": ""},
            {"epoch": "1", "val/loss": "0.9", "val/mae": "0.45", "test/mae": "", "test/mse": ""},
            {"epoch": "2", "val/loss": "", "val/mae": "", "test/mae": "0.3", "test/mse": "0.09"},
        ],
    )
    _create_run(
        results_dir / "trial_c",
        model="DemoNet",
        dataset="ETTh1",
        hist_len=96,
        pred_len=12,
        seed=3,
        lr=0.001,
        dropout=0.1,
        metric_rows=[
            {"epoch": "0", "val/loss": "0.2", "val/mae": "0.25", "test/mae": "", "test/mse": ""},
            {"epoch": "1", "val/loss": "0.3", "val/mae": "0.35", "test/mae": "", "test/mse": ""},
            {"epoch": "2", "val/loss": "", "val/mae": "", "test/mae": "0.1", "test/mse": "0.01"},
        ],
    )

    ignored_dir = results_dir / "ignored"
    ignored_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(ignored_dir / "metrics.csv", [{"epoch": "0", "val/loss": "1.0"}])

    report_df = build_benchmark_report(str(benchmark_path), out_path=str(out_path))

    assert list(report_df.columns) == [
        "model",
        "dataset",
        "hist_len",
        "pred_len",
        "lr",
        "dropout",
        "test/mae_mean",
        "test/mae_std",
        "test/mse_mean",
        "test/mse_std",
        "val/loss_mean",
        "val/loss_std",
        "val/mae_mean",
        "val/mae_std",
    ]
    assert report_df.to_dict("records")[0]["dropout"] == 0.1
    assert report_df.to_dict("records")[1]["dropout"] == 0.2

    aggregated_row = report_df.to_dict("records")[1]
    assert aggregated_row["model"] == "DemoNet"
    assert aggregated_row["dataset"] == "ETTh1"
    assert aggregated_row["hist_len"] == 96
    assert aggregated_row["pred_len"] == 12
    assert aggregated_row["lr"] == pytest.approx(0.001)
    assert aggregated_row["test/mae_mean"] == pytest.approx(0.25)
    assert aggregated_row["test/mae_std"] == pytest.approx(0.0707106781)
    assert aggregated_row["test/mse_mean"] == pytest.approx(0.065)
    assert aggregated_row["test/mse_std"] == pytest.approx(0.0353553391)
    assert aggregated_row["val/loss_mean"] == pytest.approx(0.5)
    assert aggregated_row["val/loss_std"] == pytest.approx(0.1414213562)
    assert aggregated_row["val/mae_mean"] == pytest.approx(0.45)
    assert aggregated_row["val/mae_std"] == pytest.approx(0.0707106781)

    written_df = pd.read_csv(out_path)
    assert list(written_df.columns) == list(report_df.columns)
    assert written_df.shape == (2, 14)


def test_build_benchmark_report_uses_results_dir_override(tmp_path):
    configured_results_dir = tmp_path / "configured_results"
    override_results_dir = tmp_path / "override_results"
    benchmark_path = tmp_path / "core.py"
    _write_benchmark_config(benchmark_path, configured_results_dir, {"lr": 0.001})

    _create_run(
        override_results_dir / "trial_a",
        model="DemoNet",
        dataset="Electricity",
        hist_len=24,
        pred_len=12,
        seed=7,
        lr=0.001,
        dropout=0.0,
        metric_rows=[
            {"epoch": "0", "val/loss": "0.2", "test/mae": ""},
            {"epoch": "1", "val/loss": "", "test/mae": "0.1"},
        ],
    )

    report_df = build_benchmark_report(str(benchmark_path), results_dir=str(override_results_dir))

    assert report_df.shape == (1, 9)
    assert report_df.to_dict("records")[0]["dataset"] == "Electricity"
    assert report_df.to_dict("records")[0]["test/mae_std"] != report_df.to_dict("records")[0]["test/mae_std"]


def test_build_benchmark_report_raises_when_no_valid_runs_exist(tmp_path):
    results_dir = tmp_path / "search"
    results_dir.mkdir()
    benchmark_path = tmp_path / "core.py"
    _write_benchmark_config(benchmark_path, results_dir, {"lr": 0.001})

    with pytest.raises(ValueError, match="No valid run directories found under"):
        build_benchmark_report(str(benchmark_path))


def test_build_benchmark_report_raises_on_duplicate_seed_per_group(tmp_path):
    results_dir = tmp_path / "search"
    benchmark_path = tmp_path / "core.py"
    _write_benchmark_config(benchmark_path, results_dir, {"lr": 0.001, "dropout": 0.2})

    metric_rows = [
        {"epoch": "0", "val/loss": "0.3", "test/mae": ""},
        {"epoch": "1", "val/loss": "", "test/mae": "0.2"},
    ]
    _create_run(
        results_dir / "trial_a",
        model="DemoNet",
        dataset="ETTm1",
        hist_len=24,
        pred_len=12,
        seed=1,
        lr=0.001,
        dropout=0.2,
        metric_rows=metric_rows,
    )
    _create_run(
        results_dir / "trial_b",
        model="DemoNet",
        dataset="ETTm1",
        hist_len=24,
        pred_len=12,
        seed=1,
        lr=0.001,
        dropout=0.2,
        metric_rows=metric_rows,
    )

    with pytest.raises(ValueError, match="Duplicate seed found"):
        build_benchmark_report(str(benchmark_path))


def test_build_benchmark_report_raises_on_invalid_metric_values(tmp_path):
    results_dir = tmp_path / "search"
    benchmark_path = tmp_path / "core.py"
    _write_benchmark_config(benchmark_path, results_dir, {"lr": 0.001})

    _create_run(
        results_dir / "trial_a",
        model="DemoNet",
        dataset="ETTh2",
        hist_len=48,
        pred_len=24,
        seed=3,
        lr=0.001,
        dropout=0.0,
        metric_rows=[
            {"epoch": "0", "val/loss": "0.2", "test/mae": ""},
            {"epoch": "1", "val/loss": "", "test/mae": "oops"},
        ],
    )

    with pytest.raises(ValueError, match="Failed to parse metric column 'test/mae'"):
        build_benchmark_report(str(benchmark_path))


def test_build_benchmark_report_rejects_legacy_hparam_names(tmp_path):
    results_dir = tmp_path / "search"
    benchmark_path = tmp_path / "core.py"
    _write_benchmark_config(benchmark_path, results_dir, {"lr": 0.001})

    run_dir = results_dir / "trial_a"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_hparams(
        run_dir / "hparams.yaml",
        model_name="DemoNet",
        dataset_name="ETTh1",
        hist_len=96,
        pred_len=12,
        seed=1,
        lr=0.001,
        dropout=0.2,
    )
    _write_metrics(
        run_dir / "metrics.csv",
        [
            {"epoch": "0", "val/loss": "0.4", "test/mae": ""},
            {"epoch": "1", "val/loss": "", "test/mae": "0.2"},
        ],
    )

    with pytest.raises(ValueError, match="Missing hparam 'model'"):
        build_benchmark_report(str(benchmark_path))
