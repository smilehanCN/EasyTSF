from pathlib import Path

import argparse

from easytsf.experiment import (
    add_config_override_args,
    add_shared_runtime_args,
    add_tune_args,
    build_runtime_overrides,
    finalize_runtime_conf,
    load_config,
    load_param_space,
    parse_config_overrides,
    run_training,
)


def build_reporter(param_space, metric, mode):
    from ray.tune import CLIReporter

    return CLIReporter(
        parameter_columns=list(param_space.keys()),
        metric_columns=[metric],
        metric=metric,
        mode=mode,
        sort_by_metric=True,
    )


def save_tune_reports(result_grid, metric, mode):
    experiment_dir = Path(result_grid.experiment_path)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    trial_report_path = experiment_dir / "report.csv"
    best_report_path = experiment_dir / "report_best.csv"

    result_grid.get_dataframe().to_csv(trial_report_path, index=False)
    result_grid.get_dataframe(filter_metric=metric, filter_mode=mode).to_csv(best_report_path, index=False)

    return trial_report_path, best_report_path


def train_func(hyper_conf, base_conf):
    run_training(finalize_runtime_conf(base_conf, overrides=hyper_conf, use_ray=hyper_conf is not None))


def ray_tune_train(param_space, init_conf, num_samples=1, cpus_per_trial=2, gpus_per_trial=1, mode="min"):
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler

    metric = init_conf["val_metric"]
    experiment_name = "RAY_{}_{}".format(init_conf["model_name"], init_conf["dataset_name"])
    storage_path = str(Path(init_conf["save_root"]).expanduser().resolve())
    scheduler = FIFOScheduler()
    reporter = build_reporter(param_space, metric, mode)

    trainable = tune.with_parameters(train_func, base_conf=init_conf)
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
            storage_path=storage_path,
            progress_reporter=reporter,
        ),
    )

    result_grid = tuner.fit()
    trial_report_path, best_report_path = save_tune_reports(result_grid, metric, mode)
    best_result = result_grid.get_best_result(metric=metric, mode=mode, scope="all")

    print("Experiment path:", result_grid.experiment_path)
    print("Saved trial summary:", trial_report_path)
    print("Saved best-per-trial summary:", best_report_path)
    print("Best hyper-parameters found were:", best_result.config)
    print("Best result metrics:", best_result.metrics)

    if result_grid.num_errors:
        print("Errored trials:", result_grid.num_errors)
        for error in result_grid.errors:
            print(error)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    add_shared_runtime_args(parser)
    add_config_override_args(parser)
    add_tune_args(parser)
    return parser


def run_standard_train(args, init_exp_conf):
    base_conf = dict(init_exp_conf)
    base_conf.update(build_runtime_overrides(args))
    run_training(finalize_runtime_conf(base_conf))


def run_tune_search(args, init_exp_conf):
    import ray

    if not ray.is_initialized():
        if args.num_gpus > 0:
            ray.init(num_gpus=args.num_gpus)
        else:
            ray.init()

    tune_conf = dict(init_exp_conf)
    tune_conf.update(build_runtime_overrides(args))
    param_space = load_param_space(args.param_space)

    ray_tune_train(
        param_space=param_space,
        init_conf=tune_conf,
        num_samples=args.num_samples,
        cpus_per_trial=args.cpus_per_trial,
        gpus_per_trial=args.gpus_per_trial,
    )


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    init_exp_conf = load_config(args.config, overrides=parse_config_overrides(args.config_overrides))

    if args.param_space:
        run_tune_search(args, init_exp_conf)
    else:
        run_standard_train(args, init_exp_conf)


if __name__ == "__main__":
    main()
