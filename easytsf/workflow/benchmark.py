import argparse
import os
from pathlib import Path

from .config import finalize_runtime_conf, load_experiment_config, load_module_from_path
from .experiment import run_experiment


def load_benchmark(benchmark_ref):
    benchmark_path = Path(benchmark_ref).expanduser().resolve()
    raw_conf = load_module_from_path(benchmark_path.stem, str(benchmark_path)).benchmark_config
    experiment_path = Path(raw_conf["experiment"]).expanduser().resolve()
    search_dir = Path(raw_conf["search_name"]).expanduser().resolve()
    search_config = dict(raw_conf["search_config"])
    search_config["num_samples"] = int(search_config["num_samples"])
    search_config["cpus_per_trial"] = int(search_config["cpus_per_trial"])
    search_config["gpus_per_trial"] = float(search_config["gpus_per_trial"])
    search_config["num_gpus"] = int(search_config["num_gpus"])
    return {
        "name": str(raw_conf.get("name")),
        "base_conf": load_experiment_config(experiment_path),
        "search_config": search_config,
        "search_dir": str(search_dir),
        "param_space": dict(raw_conf["param_space"]),
    }


def _build_tune_reporter(param_space, metric, mode):
    from ray.tune import CLIReporter

    return CLIReporter(
        parameter_columns=list(param_space.keys()),
        metric_columns=[metric],
        metric=metric,
        mode=mode,
        sort_by_metric=True,
    )


def _tune_train_func(hyper_conf, base_conf):
    from ray import tune
    from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

    conf = finalize_runtime_conf(base_conf, overrides=hyper_conf)
    conf["exp_dir"] = str(Path(tune.get_context().get_trial_dir()).resolve())
    run_experiment(
        conf,
        extra_callbacks=[
            TuneReportCheckpointCallback(
                metrics={conf["val_metric"]: conf["val_metric"]},
                save_checkpoints=False,
                on="validation_end",
            )
        ],
    )


def run_benchmark(
    benchmark_ref,
    resume=True,
):
    benchmark_conf = load_benchmark(benchmark_ref)
    base_conf = dict(benchmark_conf["base_conf"])
    search_dir = Path(benchmark_conf["search_dir"]).expanduser().resolve()
    search_config = benchmark_conf["search_config"]
    param_space = benchmark_conf["param_space"]
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"

    import ray
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler

    if not ray.is_initialized():
        if search_config["num_gpus"] > 0:
            ray.init(num_gpus=search_config["num_gpus"])
        else:
            ray.init()

    metric = base_conf["val_metric"]
    reporter = _build_tune_reporter(param_space, metric, "min")
    trainable = tune.with_parameters(_tune_train_func, base_conf=base_conf)
    trainable = tune.with_resources(
        trainable,
        resources={"cpu": search_config["cpus_per_trial"], "gpu": search_config["gpus_per_trial"]},
    )

    ray_storage_path = search_dir / "ray_results"
    ray_experiment_path = ray_storage_path / "ray"
    should_restore = resume and tune.Tuner.can_restore(str(ray_experiment_path))
    if should_restore:
        tuner = tune.Tuner.restore(
            str(ray_experiment_path),
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=True,
            param_space=param_space,
        )
    else:
        tuner = tune.Tuner(
            trainable=trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=metric,
                mode="min",
                scheduler=FIFOScheduler(),
                num_samples=search_config["num_samples"],
            ),
            run_config=tune.RunConfig(
                name="ray",
                storage_path=str(ray_storage_path),
                progress_reporter=reporter,
            ),
        )

    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric=metric, mode="min", scope="all")
    best_metric = best_result.metrics[metric]
    print("[best] {}={}".format(metric, best_metric))
    return best_metric


def build_cli_parser():
    parser = argparse.ArgumentParser(description="Run an EasyTSF benchmark.")
    parser.add_argument("benchmark", help="Benchmark python file path.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable reuse of existing benchmark artifacts.",
    )
    return parser


if __name__ == "__main__":
    args = build_cli_parser().parse_args()
    run_benchmark(args.benchmark, resume=args.resume)
