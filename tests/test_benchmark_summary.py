import csv

from benchmark import build_summary_rows, select_best_config, write_runs_report, write_summary_report


def _base_row(**overrides):
    row = {
        "benchmark_name": "smoke_benchmark",
        "benchmark_run_hash": "abc123",
        "experiment": "tests/fixtures/experiments/itransformer_smoke.yaml",
        "d_model": 8,
        "dropout": 0.0,
        "seed": 0,
        "status": "success",
        "error": None,
        "task_name": "mtsf",
        "model_name": "iTransformer",
        "dataset_name": "SmokeSet",
        "hist_len": 8,
        "pred_len": 4,
        "val_metric_name": "val/loss",
        "val_metric_value": 1.0,
        "mae": 1.0,
        "mse": 2.0,
        "rmse": 1.414,
        "conf_hash": "hash",
        "ckpt_path": "/tmp/model.ckpt",
        "exp_dir": "/tmp/run",
    }
    row.update(overrides)
    return row


def test_benchmark_summary_groups_by_hyperparameters_and_selects_best_complete_config(tmp_path):
    rows = [
        _base_row(seed=0, d_model=8, dropout=0.0, val_metric_value=2.0, mae=2.0, mse=4.0, rmse=2.0),
        _base_row(seed=1, d_model=8, dropout=0.0, val_metric_value=4.0, mae=4.0, mse=16.0, rmse=4.0, conf_hash="hash2"),
        _base_row(seed=0, d_model=16, dropout=0.1, val_metric_value=1.0, mae=1.0, mse=1.0, rmse=1.0, conf_hash="hash3"),
        _base_row(
            seed=1,
            d_model=16,
            dropout=0.1,
            status="failed",
            error="boom",
            val_metric_value=None,
            mae=None,
            mse=None,
            rmse=None,
            conf_hash="hash4",
            ckpt_path=None,
        ),
    ]

    runs_path = write_runs_report(tmp_path, rows, ["d_model", "dropout"])
    summary_path, summary_rows = write_summary_report(tmp_path, rows, ["d_model", "dropout"], [0, 1])
    best_config = select_best_config(summary_rows, ["d_model", "dropout"])

    with runs_path.open("r", encoding="utf-8", newline="") as handle:
        run_report_rows = list(csv.DictReader(handle))
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_report_rows = list(csv.DictReader(handle))

    assert len(run_report_rows) == 4
    assert len(summary_report_rows) == 2
    assert best_config == {"d_model": 8, "dropout": 0.0}

    grouped_summary = build_summary_rows(rows, ["d_model", "dropout"], [0, 1])
    summary_by_config = {(row["d_model"], row["dropout"]): row for row in grouped_summary}
    assert summary_by_config[(8, 0.0)]["val_metric_mean"] == 3.0
    assert summary_by_config[(8, 0.0)]["all_seeds_succeeded"] is True
    assert summary_by_config[(16, 0.1)]["num_successful_seeds"] == 1
    assert summary_by_config[(16, 0.1)]["all_seeds_succeeded"] is False
