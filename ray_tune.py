import argparse
import os

import ray
from ray import air, tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler

from easytsf.util import load_module_from_path
from train import load_config
from train import train_func


def ray_tune_train(param_space, init_conf, num_samples=1, cpus_per_trial=2, gpus_per_trial=1):
    scheduler = FIFOScheduler()
    reporter = CLIReporter(parameter_columns=list(param_space.keys()), metric_columns=[init_conf['val_metric']])
    storage_filename = 'RAY_{}_{}'.format(init_conf['model_name'], init_conf['dataset_name'])

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func, conf=init_conf), resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric=init_conf['val_metric'],
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name=storage_filename,
            storage_path=init_conf['save_root'],
            progress_reporter=reporter,
            verbose=0,
        ),
    )

    results = tuner.fit()
    res_df = results.get_dataframe()
    save_path = os.path.join(os.path.join(init_conf['save_root'], storage_filename), 'report.csv')
    res_df.to_csv(save_path)
    print(res_df)
    print("Best hyper-parameters found were: ", results.get_best_result().config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-p", "--param_space", default='config/private_conf/seeds.json', type=str)
    parser.add_argument("-d", "--data_root", default="/data3/smilehan/projects/github_eval/dataset", type=str)
    parser.add_argument("-s", "--save_root", default="/data3/smilehan/projects/github_eval/save", type=str)
    parser.add_argument("--devices", default=1, type=int, help="The devices to use, detail rules is show in README")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--num_gpus", default=2, type=int)
    parser.add_argument("--cpus_per_trial", default=2, type=int)
    parser.add_argument("--gpus_per_trial", default=0.5, type=float)
    args = parser.parse_args()

    os.environ["RAY_DEDUP_LOGS"] = "0"

    ray.init(num_gpus=args.num_gpus)

    training_conf = {
        "seed": int(args.seed),
        "param_space_path": args.param_space,
        "data_root": args.data_root,
        "save_root": args.save_root,
        "devices": args.devices,
        "use_ray": True,
    }
    init_exp_conf = load_config(args.config)
    for k, v in training_conf.items():
        init_exp_conf[k] = v

    param_space = load_module_from_path("param_space", args.param_space).param_space

    ray_tune_train(param_space, init_exp_conf, args.num_samples, args.cpus_per_trial, args.gpus_per_trial)
