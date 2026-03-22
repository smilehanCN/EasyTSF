import argparse
import importlib
import os
from pathlib import Path

import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from easytsf.util import cal_conf_hash
from easytsf.util import load_module_from_path
from easytsf.util import parse_devices


def load_config(exp_conf_path):
    exp_conf = load_module_from_path("exp_conf", exp_conf_path).exp_conf

    task_conf_module = importlib.import_module("config.base_conf.task")
    task_conf = task_conf_module.task_conf

    data_conf_module = importlib.import_module("config.base_conf.datasets")
    data_conf = eval("data_conf_module.{}_conf".format(exp_conf["dataset_name"]))

    fused_conf = {**task_conf, **data_conf}
    fused_conf.update(exp_conf)
    return fused_conf


def build_callbacks(conf):
    callbacks = [
        ModelCheckpoint(
            monitor=conf["val_metric"],
            mode="min",
            save_top_k=1,
            save_last=False,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=conf["val_metric"],
            mode="min",
            patience=conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if conf.get("use_ray"):
        from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

        callbacks.append(
            TuneReportCheckpointCallback(
                {conf["val_metric"]: conf["val_metric"]},
                save_checkpoints=False,
                on="validation_end",
            )
        )

    return callbacks


def build_runner(conf):
    from easytsf.runner.exp_base_runner import LTSFRunner
    from easytsf.runner.exp_mynetv6_runner import LTSFMyNetV6Runner
    from easytsf.runner.exp_mynetvx_runner import LTSFMyNetVXRunner
    from easytsf.runner.exp_reconstruction_runner import ReconstructionRunner
    from easytsf.runner.exp_univariate_runner import UnivariateRunner
    from easytsf.runner.exp_with_aux_loss_runner import LTSFwithAuxLossRunner

    if conf["exp_runner"] == "exp_with_aux_loss":
        return LTSFwithAuxLossRunner(**conf)
    if conf["exp_runner"] == "exp_base":
        return LTSFRunner(**conf)
    if conf["exp_runner"] == "exp_univariate":
        return UnivariateRunner(**conf)
    if conf["exp_runner"] == "exp_mynetv6_runner":
        return LTSFMyNetV6Runner(**conf)
    if conf["exp_runner"] == "exp_mynetvx_runner":
        return LTSFMyNetVXRunner(**conf)
    if conf["exp_runner"] == "exp_reconstruction_runner":
        return ReconstructionRunner(**conf)
    raise NotImplementedError


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


def train_func(hyper_conf, conf):
    from easytsf.runner.data_runner import DataInterface

    conf = dict(conf)
    if hyper_conf is not None:
        conf.update(hyper_conf)

    conf["devices"] = parse_devices(conf.get("devices", "auto"))
    conf["accelerator"] = conf.get("accelerator", "auto")
    conf["conf_hash"] = cal_conf_hash(conf, hash_len=10)

    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], "{}_{}".format(conf["model_name"], conf["dataset_name"]))
    if conf.get("use_wandb"):
        run_logger = WandbLogger(save_dir=save_dir, name=conf["conf_hash"], version="seed_{}".format(conf["seed"]))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=conf["conf_hash"], version="seed_{}".format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["conf_hash"], "seed_{}".format(conf["seed"]))

    trainer = L.Trainer(
        accelerator=conf["accelerator"],
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=run_logger,
        callbacks=build_callbacks(conf),
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf["gradient_clip_algorithm"] if "gradient_clip_algorithm" in conf else "norm",
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["save_root"],
    )

    data_module = DataInterface(**conf)
    train_loader = data_module.train_dataloader()
    conf["steps_per_epoch"] = max(1, len(train_loader))
    model = build_runner(conf)

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")


def ray_tune_train(param_space, init_conf, num_samples=1, cpus_per_trial=2, gpus_per_trial=1, mode="min"):
    from ray import tune
    from ray.tune.schedulers import FIFOScheduler

    metric = init_conf["val_metric"]
    experiment_name = "RAY_{}_{}".format(init_conf["model_name"], init_conf["dataset_name"])
    storage_path = str(Path(init_conf["save_root"]).expanduser().resolve())
    scheduler = FIFOScheduler()
    reporter = build_reporter(param_space, metric, mode)

    trainable = tune.with_parameters(train_func, conf=init_conf)
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


def build_arg_parser(require_param_space=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-p", "--param_space", required=require_param_space, type=str, default=None)
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", help="save root")
    parser.add_argument("--accelerator", default="auto", type=str, help="accelerator to use")
    parser.add_argument("--devices", default="auto", type=str, help="device ids/count, e.g. auto, 1, 0,1")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--cpus_per_trial", default=2, type=int)
    parser.add_argument("--gpus_per_trial", default=0.5, type=float)
    return parser


def build_training_conf(args, use_ray=False):
    return {
        "seed": int(args.seed),
        "data_root": args.data_root,
        "save_root": args.save_root,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "use_wandb": args.use_wandb,
        "use_ray": use_ray,
        "param_space_path": args.param_space,
    }


def run_standard_train(args, init_exp_conf):
    train_func(build_training_conf(args, use_ray=False), init_exp_conf)


def run_tune_search(args, init_exp_conf):
    import ray

    if not ray.is_initialized():
        if args.num_gpus > 0:
            ray.init(num_gpus=args.num_gpus)
        else:
            ray.init()

    tune_conf = dict(init_exp_conf)
    tune_conf.update(build_training_conf(args, use_ray=True))
    param_space = load_module_from_path("param_space", args.param_space).param_space

    ray_tune_train(
        param_space=param_space,
        init_conf=tune_conf,
        num_samples=args.num_samples,
        cpus_per_trial=args.cpus_per_trial,
        gpus_per_trial=args.gpus_per_trial,
    )


def main(argv=None, require_param_space=False):
    parser = build_arg_parser(require_param_space=require_param_space)
    args = parser.parse_args(argv)
    init_exp_conf = load_config(args.config)

    if args.param_space:
        run_tune_search(args, init_exp_conf)
    else:
        run_standard_train(args, init_exp_conf)


if __name__ == "__main__":
    main()
