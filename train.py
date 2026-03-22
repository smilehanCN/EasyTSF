import argparse
import importlib
import importlib.util
import os

import lightning.pytorch as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from easytsf.runner.data_runner import DataInterface
from easytsf.runner.exp_base_runner import LTSFRunner
from easytsf.runner.exp_with_aux_loss_runner import LTSFwithAuxLossRunner
from easytsf.runner.exp_univariate_runner import UnivariateRunner
from easytsf.runner.exp_mynetv6_runner import LTSFMyNetV6Runner
from easytsf.runner.exp_mynetvx_runner import LTSFMyNetVXRunner
from easytsf.runner.exp_reconstruction_runner import ReconstructionRunner
from easytsf.util import cal_conf_hash
from easytsf.util import load_module_from_path
from easytsf.util import parse_devices


def load_config(exp_conf_path):
    # 加载 exp_conf
    exp_conf = load_module_from_path("exp_conf", exp_conf_path).exp_conf

    # 加载 task_conf
    task_conf_module = importlib.import_module('config.base_conf.task')
    task_conf = task_conf_module.task_conf

    # 加载 data_conf
    data_conf_module = importlib.import_module('config.base_conf.datasets')
    data_conf = eval('data_conf_module.{}_conf'.format(exp_conf['dataset_name']))

    # conf 融合，参数优先级: exp_conf > task_conf = data_conf
    fused_conf = {**task_conf, **data_conf}
    fused_conf.update(exp_conf)

    return fused_conf


def train_func(hyper_conf, conf):
    if hyper_conf is not None:
        for k, v in hyper_conf.items():
            conf[k] = v
    conf["devices"] = parse_devices(conf.get("devices", "auto"))
    conf["accelerator"] = conf.get("accelerator", "auto")
    conf['conf_hash'] = cal_conf_hash(conf, hash_len=10)

    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], '{}_{}'.format(conf["model_name"], conf["dataset_name"]))
    if "use_wandb" in conf and conf["use_wandb"]:
        run_logger = WandbLogger(save_dir=save_dir, name=conf["conf_hash"], version='seed_{}'.format(conf["seed"]))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=conf["conf_hash"], version='seed_{}'.format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["conf_hash"], 'seed_{}'.format(conf["seed"]))

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
            mode='min',
            patience=conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if "use_ray" in conf and conf["use_ray"]:
        callbacks.append(TuneReportCheckpointCallback(
            {conf["val_metric"]: conf["val_metric"]}, save_checkpoints=False, on="validation_end"))

    trainer = L.Trainer(
        accelerator=conf["accelerator"],
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=run_logger,
        callbacks=callbacks,
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf["gradient_clip_algorithm"] if "gradient_clip_algorithm" in conf else "norm",
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["save_root"],
    )

    data_module = DataInterface(**conf)
    train_loader = data_module.train_dataloader()
    conf["steps_per_epoch"] = max(1, len(train_loader))
    if conf["exp_runner"] == "exp_with_aux_loss":
        model = LTSFwithAuxLossRunner(**conf)
    elif conf["exp_runner"] == "exp_base":
        model = LTSFRunner(**conf)
    elif conf["exp_runner"] == "exp_univariate":
        model = UnivariateRunner(**conf)
    elif conf["exp_runner"] == "exp_mynetv6_runner":
        model = LTSFMyNetV6Runner(**conf)
    elif conf["exp_runner"] == "exp_mynetvx_runner":
        model = LTSFMyNetVXRunner(**conf)
    elif conf["exp_runner"] == "exp_reconstruction_runner":
        model = ReconstructionRunner(**conf)
    else:
        raise NotImplementedError

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path='best')
    # trainer.test(model, datamodule=data_module, ckpt_path='/data2/smilehan/projects/EasyTSF/save/DoNet_ETTh1/3a154c2661/seed_0/checkpoints/epoch=9-step=660.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str)
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", help="save root")
    parser.add_argument("--accelerator", default="auto", type=str, help="accelerator to use")
    parser.add_argument("--devices", default="auto", type=str, help="device ids/count, e.g. auto, 1, 0,1")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    args = parser.parse_args()

    training_conf = {
        "seed": int(args.seed),
        "data_root": args.data_root,
        "save_root": args.save_root,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "use_wandb": args.use_wandb,
    }
    init_exp_conf = load_config(args.config)
    train_func(training_conf, init_exp_conf)
