from types import SimpleNamespace

import pytest

from easytsf.workflow import experiment


class FakeDataModule:
    init_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        FakeDataModule.init_kwargs = dict(kwargs)

    def train_dataloader(self):
        return [object()] * 4


class FakeTask:
    init_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        FakeTask.init_kwargs = dict(kwargs)


class FakeTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.events = []
        self.callback_metrics = {"val/loss": 0.25}
        FakeTrainer.instances.append(self)

    def fit(self, task, datamodule=None):
        self.events.append(("fit", task, datamodule))

    def test(self, task, datamodule=None, ckpt_path=None):
        self.events.append(("test", task, datamodule, ckpt_path))
        return {"val/loss": self.callback_metrics["val/loss"]}


def test_run_experiment_returns_val_metric_after_fit(tmp_path, monkeypatch):
    seed_calls = []
    FakeTrainer.instances.clear()

    monkeypatch.setattr(experiment.L, "Trainer", FakeTrainer)
    monkeypatch.setattr(experiment.L, "seed_everything", lambda seed, verbose=True: seed_calls.append((seed, verbose)))
    monkeypatch.setattr(
        experiment,
        "get_task_registry_entry",
        lambda _: SimpleNamespace(datamodule_cls=FakeDataModule, task_cls=FakeTask),
    )

    exp_dir = tmp_path / "run"
    result = experiment.run_experiment(
        {
            "task_name": "mtsf",
            "model": "demo_model",
            "dataset": "demo_dataset",
            "save_root": str(tmp_path),
            "seed": 7,
            "hist_len": 12,
            "pred_len": 3,
            "accelerator": "cpu",
            "devices": 1,
            "val_metric": "val/loss",
            "es_patience": 2,
            "max_epochs": 5,
            "gradient_clip_val": 0.0,
            "lr_scheduler": "OneCycleLR",
            "exp_dir": str(exp_dir),
        },
        extra_callbacks=["extra-callback"],
    )

    trainer = FakeTrainer.instances[-1]
    assert result == {"val/loss": 0.25}
    assert seed_calls == [(7, True)]
    assert FakeDataModule.init_kwargs["exp_dir"] == str(exp_dir)
    assert FakeTask.init_kwargs["steps_per_epoch"] == 4
    assert trainer.kwargs["logger"].log_dir.rstrip("/") == str(exp_dir.resolve())
    assert trainer.kwargs["default_root_dir"] == str(exp_dir.resolve())
    assert "extra-callback" in trainer.kwargs["callbacks"]
    assert [event[0] for event in trainer.events] == ["fit", "test"]


def test_cli_parser_rejects_removed_eval_flags():
    parser = experiment.build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--eval"])
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--ckpt-path", "best"])
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--print-conf"])


def test_run_experiment_requires_explicit_task_name(monkeypatch):
    monkeypatch.setattr(experiment.L, "seed_everything", lambda *args, **kwargs: None)

    with pytest.raises(KeyError, match="task_name"):
        experiment.run_experiment(
            {
                "model": "demo_model",
                "dataset": "demo_dataset",
                "save_root": "unused",
                "seed": 7,
                "hist_len": 12,
                "pred_len": 3,
                "accelerator": "cpu",
                "devices": 1,
                "val_metric": "val/loss",
                "es_patience": 2,
                "max_epochs": 5,
                "gradient_clip_val": 0.0,
                "lr_scheduler": "OneCycleLR",
                "exp_dir": "unused",
            }
        )
