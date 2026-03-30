import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from easytsf.workflow import benchmark


class _FakeResultGrid:
    def __init__(self, config, metrics, experiment_path):
        self._config = dict(config)
        self._metrics = dict(metrics)
        self.experiment_path = str(experiment_path)
        self.num_errors = 0
        self.errors = []

    def get_best_result(self, metric=None, mode=None, scope=None):
        return SimpleNamespace(config=self._config, metrics=self._metrics)


def _install_fake_ray(monkeypatch):
    tune_module = ModuleType("ray.tune")
    integration_module = ModuleType("ray.tune.integration")
    lightning_integration_module = ModuleType("ray.tune.integration.pytorch_lightning")
    schedulers_module = ModuleType("ray.tune.schedulers")
    tune_state = {"current_trial_dir": None, "reported_metrics": None}

    class FakeCLIReporter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTuneConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunConfig:
        def __init__(self, **kwargs):
            self.name = kwargs["name"]
            self.storage_path = kwargs["storage_path"]
            self.progress_reporter = kwargs.get("progress_reporter")

    class FakeTuner:
        init_calls = []
        restore_calls = []
        can_restore_calls = []
        can_restore_response = False

        @classmethod
        def can_restore(cls, path):
            cls.can_restore_calls.append(path)
            return cls.can_restore_response

        @classmethod
        def restore(cls, path, trainable, param_space=None, **kwargs):
            cls.restore_calls.append((path, dict(param_space or {}), dict(kwargs)))
            return cls(
                trainable=trainable,
                param_space=param_space or {},
                tune_config=None,
                run_config=SimpleNamespace(name=Path(path).name, storage_path=str(Path(path).parent)),
            )

        def __init__(self, trainable, param_space, tune_config, run_config):
            self.trainable = trainable
            self.param_space = dict(param_space or {})
            self.tune_config = tune_config
            self.run_config = run_config
            type(self).init_calls.append(self)

        def fit(self):
            trial_dir = Path(self.run_config.storage_path) / self.run_config.name / "trial_00000"
            trial_dir.mkdir(parents=True, exist_ok=True)
            tune_state["current_trial_dir"] = str(trial_dir.resolve())
            tune_state["reported_metrics"] = None
            metrics = self.trainable(dict(self.param_space))
            if tune_state["reported_metrics"] is not None:
                metrics = dict(tune_state["reported_metrics"])
            experiment_path = Path(self.run_config.storage_path) / self.run_config.name
            return _FakeResultGrid(self.param_space, metrics, experiment_path.resolve())

    def with_parameters(fn, **kwargs):
        def trainable(config):
            return fn(config, **kwargs)

        return trainable

    def with_resources(trainable, resources):
        trainable.resources = resources
        return trainable

    tune_module.CLIReporter = FakeCLIReporter
    tune_module.TuneConfig = FakeTuneConfig
    tune_module.RunConfig = FakeRunConfig
    tune_module.Tuner = FakeTuner
    tune_module.get_context = lambda: SimpleNamespace(get_trial_dir=lambda: tune_state["current_trial_dir"])
    tune_module.report = lambda metrics, checkpoint=None: tune_state.__setitem__("reported_metrics", dict(metrics))
    tune_module.with_parameters = with_parameters
    tune_module.with_resources = with_resources

    class FakeTuneReportCheckpointCallback:
        def __init__(self, metrics=None, filename="checkpoint", save_checkpoints=True, on="validation_end"):
            self.metrics = metrics
            self.filename = filename
            self.save_checkpoints = save_checkpoints
            self.on = on

        def on_validation_end(self, trainer, pl_module=None):
            reported_metrics = {}
            for key, metric_name in self.metrics.items():
                reported_metrics[key] = trainer.callback_metrics[metric_name]
            tune_module.report(reported_metrics)

    lightning_integration_module.TuneReportCheckpointCallback = FakeTuneReportCheckpointCallback
    integration_module.pytorch_lightning = lightning_integration_module

    class FakeFIFOScheduler:
        pass

    schedulers_module.FIFOScheduler = FakeFIFOScheduler

    ray_module = ModuleType("ray")
    ray_state = {"initialized": False}

    def is_initialized():
        return ray_state["initialized"]

    def init(num_gpus=None):
        ray_state["initialized"] = True

    ray_module.is_initialized = is_initialized
    ray_module.init = init
    ray_module.tune = tune_module

    monkeypatch.setitem(sys.modules, "ray", ray_module)
    monkeypatch.setitem(sys.modules, "ray.tune", tune_module)
    monkeypatch.setitem(sys.modules, "ray.tune.integration", integration_module)
    monkeypatch.setitem(sys.modules, "ray.tune.integration.pytorch_lightning", lightning_integration_module)
    monkeypatch.setitem(sys.modules, "ray.tune.schedulers", schedulers_module)
    return tune_module, lightning_integration_module


def test_load_benchmark_uses_explicit_path_and_config_name(tmp_path):
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text("model_name: demo_model\ndataset_name: demo_dataset\n", encoding="utf-8")
    search_path = tmp_path / "search"
    benchmark_path = tmp_path / "core.py"
    benchmark_path.write_text(
        "benchmark_config = {\n"
        "    'name': 'custom_benchmark',\n"
        "    'search_name': " + repr(str(search_path)) + ",\n"
        "    'experiment': " + repr(str(experiment_path)) + ",\n"
        "    'search_config': {\n"
        "        'num_samples': 1,\n"
        "        'cpus_per_trial': 2,\n"
        "        'gpus_per_trial': 0.5,\n"
        "        'num_gpus': 0,\n"
        "    },\n"
        "    'param_space': {},\n"
        "}\n",
        encoding="utf-8",
    )

    loaded = benchmark.load_benchmark(str(benchmark_path))

    assert loaded["name"] == "custom_benchmark"
    assert loaded["search_dir"] == str(search_path.resolve())
    assert loaded["base_conf"]["model_name"] == "demo_model"
    assert loaded["param_space"] == {}


def test_run_benchmark_reports_val_metric_via_lightning_callback(tmp_path, monkeypatch, capsys):
    fake_tune, fake_lightning_integration = _install_fake_ray(monkeypatch)
    run_calls = []
    benchmark_conf = {
        "name": "demo_model_demo_dataset",
        "base_conf": {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "val_metric": "val/loss",
            "save_root": str(tmp_path),
            "data_root": str(tmp_path),
            "seed": 0,
        },
        "search_config": {
            "num_samples": 1,
            "cpus_per_trial": 2,
            "gpus_per_trial": 0.5,
            "num_gpus": 0,
        },
        "search_dir": str(tmp_path / "search"),
        "param_space": {"lr": 0.01},
    }

    monkeypatch.setattr(benchmark, "load_benchmark", lambda _: dict(benchmark_conf))

    def fake_run_experiment(conf, extra_callbacks=None):
        run_calls.append((str(conf["exp_dir"]), list(extra_callbacks or [])))
        trainer = SimpleNamespace(callback_metrics={"val/loss": 0.42}, sanity_checking=False)
        for callback in extra_callbacks or []:
            callback.on_validation_end(trainer, pl_module=None)

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)

    result = benchmark.run_benchmark(str(tmp_path / "benchmark.py"), resume=False)

    assert os.environ["RAY_CHDIR_TO_TRIAL_DIR"] == "0"
    assert Path(run_calls[0][0]).name == "trial_00000"
    assert len(run_calls[0][1]) == 1
    assert isinstance(run_calls[0][1][0], fake_lightning_integration.TuneReportCheckpointCallback)
    assert run_calls[0][1][0].metrics == {"val/loss": "val/loss"}
    assert run_calls[0][1][0].save_checkpoints is False
    assert run_calls[0][1][0].on == "validation_end"
    assert result == 0.42
    assert "[best] val/loss=0.42" in capsys.readouterr().out
    assert fake_tune.Tuner.restore_calls == []
    assert not (tmp_path / "search" / "trial_report.csv").exists()
    assert not (tmp_path / "search" / "best_trial_report.csv").exists()


def test_run_benchmark_uses_tuner_restore_for_resume(tmp_path, monkeypatch, capsys):
    fake_tune, _ = _install_fake_ray(monkeypatch)
    fake_tune.Tuner.can_restore_response = True
    benchmark_conf = {
        "name": "demo_model_demo_dataset",
        "base_conf": {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "val_metric": "val/loss",
            "save_root": str(tmp_path),
            "data_root": str(tmp_path),
            "seed": 0,
        },
        "search_config": {
            "num_samples": 1,
            "cpus_per_trial": 2,
            "gpus_per_trial": 0.5,
            "num_gpus": 0,
        },
        "search_dir": str(tmp_path / "search"),
        "param_space": {"lr": 0.02},
    }

    monkeypatch.setattr(benchmark, "load_benchmark", lambda _: dict(benchmark_conf))

    def fake_run_experiment(conf, extra_callbacks=None):
        trainer = SimpleNamespace(callback_metrics={"val/loss": 0.31}, sanity_checking=False)
        for callback in extra_callbacks or []:
            callback.on_validation_end(trainer, pl_module=None)

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)

    result = benchmark.run_benchmark(str(tmp_path / "benchmark.py"), resume=True)

    assert fake_tune.Tuner.restore_calls
    restore_path, restore_param_space, restore_kwargs = fake_tune.Tuner.restore_calls[0]
    assert restore_path.endswith("ray_results/ray")
    assert restore_param_space == {"lr": 0.02}
    assert restore_kwargs["resume_unfinished"] is True
    assert restore_kwargs["resume_errored"] is True
    assert result == 0.31
    assert "[best] val/loss=0.31" in capsys.readouterr().out


def test_run_benchmark_returns_min_val_metric(tmp_path, monkeypatch):
    _install_fake_ray(monkeypatch)
    benchmark_conf = {
        "name": "demo_model_demo_dataset",
        "base_conf": {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "val_metric": "val/loss",
            "save_root": str(tmp_path),
            "data_root": str(tmp_path),
            "seed": 0,
        },
        "search_config": {
            "num_samples": 1,
            "cpus_per_trial": 2,
            "gpus_per_trial": 0.5,
            "num_gpus": 0,
        },
        "search_dir": str(tmp_path / "custom_search"),
        "param_space": {},
    }
    run_calls = []

    monkeypatch.setattr(benchmark, "load_benchmark", lambda _: dict(benchmark_conf))

    def fake_run_experiment(conf, extra_callbacks=None):
        run_calls.append(str(conf["exp_dir"]))
        trainer = SimpleNamespace(callback_metrics={"val/loss": 0.4}, sanity_checking=False)
        for callback in extra_callbacks or []:
            callback.on_validation_end(trainer, pl_module=None)

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)

    result = benchmark.run_benchmark(str(tmp_path / "benchmark.py"), resume=False)

    assert len(run_calls) == 1
    assert result == 0.4


def test_run_benchmark_resume_uses_tuner_restore(tmp_path, monkeypatch):
    fake_tune, _ = _install_fake_ray(monkeypatch)
    fake_tune.Tuner.can_restore_response = True
    benchmark_conf = {
        "name": "demo_model_demo_dataset",
        "base_conf": {
            "task_name": "mtsf",
            "model_name": "demo_model",
            "dataset_name": "demo_dataset",
            "hist_len": 24,
            "pred_len": 12,
            "val_metric": "val/loss",
            "save_root": str(tmp_path),
            "data_root": str(tmp_path),
            "seed": 0,
        },
        "search_config": {
            "num_samples": 1,
            "cpus_per_trial": 2,
            "gpus_per_trial": 0.5,
            "num_gpus": 0,
        },
        "search_dir": str(tmp_path / "custom_search"),
        "param_space": {},
    }

    monkeypatch.setattr(benchmark, "load_benchmark", lambda _: dict(benchmark_conf))

    def fake_run_experiment(conf, extra_callbacks=None):
        trainer = SimpleNamespace(callback_metrics={"val/loss": 0.3}, sanity_checking=False)
        for callback in extra_callbacks or []:
            callback.on_validation_end(trainer, pl_module=None)

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)

    result = benchmark.run_benchmark(str(tmp_path / "benchmark.py"), resume=True)

    assert fake_tune.Tuner.restore_calls
    assert result == 0.3
