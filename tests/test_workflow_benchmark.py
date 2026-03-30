import csv
from pathlib import Path

from easytsf.workflow import benchmark


def _write_logged_metrics(exp_dir, val_metric_value=0.25, mae=1.5, mse=2.5, rmse=1.58):
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val/loss", "test/mae", "test/mse", "test/rmse"])
        writer.writeheader()
        writer.writerow({"epoch": 0, "val/loss": val_metric_value})
        writer.writerow({"epoch": 1, "test/mae": mae, "test/mse": mse, "test/rmse": rmse})
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "epoch=0.ckpt").write_text("checkpoint", encoding="utf-8")


def test_load_saved_metrics_reads_metrics_csv_and_checkpoint(tmp_path):
    exp_dir = tmp_path / "exp"
    _write_logged_metrics(exp_dir, val_metric_value=0.5, mae=1.2, mse=2.3, rmse=1.52)

    metrics = benchmark.load_saved_metrics(
        {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "seed": 0,
            "conf_hash": "abc123",
            "exp_dir": str(exp_dir),
            "val_metric": "val/loss",
        }
    )

    assert metrics["val_metric_value"] == 0.5
    assert metrics["mae"] == 1.2
    assert metrics["mse"] == 2.3
    assert metrics["rmse"] == 1.52
    assert metrics["ckpt_path"] == str((exp_dir / "checkpoints" / "epoch=0.ckpt").resolve())


def test_run_benchmark_uses_metrics_csv_instead_of_run_experiment_return(tmp_path, monkeypatch):
    benchmark_conf = {
        "path": tmp_path / "benchmark.py",
        "name": "demo_model_demo_dataset",
        "experiment": "demo",
        "base_conf": {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "val_metric": "val/loss",
            "save_root": str(tmp_path),
        },
        "seeds": [0],
        "search_config": {},
        "param_space": {},
    }
    run_calls = []

    monkeypatch.setattr(benchmark, "load_benchmark", lambda _: dict(benchmark_conf))

    def fake_run_experiment(conf):
        run_calls.append(str(conf["exp_dir"]))
        _write_logged_metrics(conf["exp_dir"], val_metric_value=0.4, mae=8.0, mse=9.0, rmse=3.0)
        return [{"test/mae": 999.0, "test/mse": 999.0}]

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)

    result = benchmark.run_benchmark("demo", runtime_overrides={"save_root": str(tmp_path)}, resume=False)

    assert len(run_calls) == 1
    assert result["rows"][0]["resume_hit"] is False
    assert result["rows"][0]["mae"] == 8.0
    assert result["rows"][0]["mse"] == 9.0
    assert result["rows"][0]["rmse"] == 3.0
    assert result["rows"][0]["val_metric_value"] == 0.4
    assert result["rows"][0]["ckpt_path"].endswith("epoch=0.ckpt")


def test_run_benchmark_resume_reuses_existing_metrics_csv(tmp_path, monkeypatch):
    benchmark_conf = {
        "path": tmp_path / "benchmark.py",
        "name": "demo_model_demo_dataset",
        "experiment": "demo",
        "base_conf": {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "val_metric": "val/loss",
            "save_root": str(tmp_path),
        },
        "seeds": [0],
        "search_config": {},
        "param_space": {},
    }
    run_calls = []

    monkeypatch.setattr(benchmark, "load_benchmark", lambda _: dict(benchmark_conf))

    def fake_run_experiment(conf):
        run_calls.append(str(conf["exp_dir"]))
        _write_logged_metrics(conf["exp_dir"], val_metric_value=0.3, mae=7.0, mse=8.0, rmse=2.8)
        return [{"ignored": True}]

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)

    first_result = benchmark.run_benchmark("demo", runtime_overrides={"save_root": str(tmp_path)}, resume=True)
    assert first_result["rows"][0]["resume_hit"] is False
    assert len(run_calls) == 1

    def fail_if_called(conf):
        raise AssertionError("run_experiment should not be called on resume")

    monkeypatch.setattr(benchmark, "run_experiment", fail_if_called)
    second_result = benchmark.run_benchmark("demo", runtime_overrides={"save_root": str(tmp_path)}, resume=True)

    assert len(run_calls) == 1
    assert second_result["rows"][0]["resume_hit"] is True
    assert second_result["rows"][0]["mae"] == 7.0
