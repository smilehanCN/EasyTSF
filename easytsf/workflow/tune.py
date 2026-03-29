import hashlib
from pathlib import Path

from .config import (
    CONFIG_ROOT,
    finalize_runtime_conf,
    load_module_from_path,
)
from .experiment import _serialize_value, run_experiment


SEARCH_SPACE_DIR = CONFIG_ROOT / "search_spaces"


def add_tune_args(parser):
    parser.add_argument("-p", "--param_space", default=None, type=str, help="search-space id or Python path")
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--cpus_per_trial", default=2, type=int)
    parser.add_argument("--gpus_per_trial", default=0.5, type=float)
    return parser


def _resolve_search_space_path(param_space_ref):
    ref_path = Path(param_space_ref).expanduser()
    if ref_path.exists():
        return ref_path.resolve()

    relative_ref = Path(param_space_ref)
    if relative_ref.suffix != ".py":
        relative_ref = relative_ref.with_suffix(".py")
    return (SEARCH_SPACE_DIR / relative_ref).resolve()


def load_param_space(param_space_ref):
    param_space_path = _resolve_search_space_path(param_space_ref)
    module_hash = hashlib.md5(str(param_space_path).encode("utf-8")).hexdigest()[:10]
    return load_module_from_path("easytsf_param_space_{}".format(module_hash), str(param_space_path)).param_space


def _build_tune_reporter(param_space, metric, mode):
    from ray.tune import CLIReporter

    return CLIReporter(
        parameter_columns=list(param_space.keys()),
        metric_columns=[metric],
        metric=metric,
        mode=mode,
        sort_by_metric=True,
    )


def _save_tune_reports(result_grid, metric, mode, report_dir=None):
    target_dir = Path(result_grid.experiment_path) if report_dir is None else Path(report_dir).expanduser().resolve()
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


def _tune_train_func(hyper_conf, base_conf):
    conf = finalize_runtime_conf(base_conf, overrides=hyper_conf)
    run_experiment(conf, extra_callbacks=_build_tune_callbacks(conf))


def run_tune_search(
    param_space,
    init_conf,
    num_samples=1,
    cpus_per_trial=2,
    gpus_per_trial=1,
    num_gpus=0,
    mode="min",
    experiment_name=None,
    storage_path=None,
    report_dir=None,
):
    import ray
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler

    if not ray.is_initialized():
        if num_gpus > 0:
            ray.init(num_gpus=num_gpus)
        else:
            ray.init()

    metric = init_conf["val_metric"]
    if experiment_name is None:
        experiment_name = "RAY_{}_{}".format(init_conf["model_name"], init_conf["dataset_name"])
    resolved_storage_path = Path(init_conf["save_root"]).expanduser().resolve() if storage_path is None else Path(storage_path).expanduser().resolve()

    scheduler = FIFOScheduler()
    reporter = _build_tune_reporter(param_space, metric, mode)
    trainable = tune.with_parameters(_tune_train_func, base_conf=init_conf)
    trainable = tune.with_resources(
        trainable,
        resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
    )

    tuner = tune.Tuner(
        trainable=trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=str(resolved_storage_path),
            progress_reporter=reporter,
        ),
    )

    result_grid = tuner.fit()
    trial_report_path, best_report_path = _save_tune_reports(result_grid, metric, mode, report_dir=report_dir)
    best_result = result_grid.get_best_result(metric=metric, mode=mode, scope="all")

    return {
        "metric": metric,
        "mode": mode,
        "experiment_path": str(Path(result_grid.experiment_path).resolve()),
        "trial_report_path": str(trial_report_path.resolve()),
        "best_trial_report_path": str(best_report_path.resolve()),
        "best_config": {key: _serialize_value(value) for key, value in dict(best_result.config).items()},
        "best_metrics": {key: _serialize_value(value) for key, value in dict(best_result.metrics).items()},
        "num_errors": int(result_grid.num_errors),
        "errors": [str(error) for error in result_grid.errors],
    }
