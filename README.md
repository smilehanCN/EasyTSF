# EasyTSFNext

EasyTSFNext 是一个基于 PyTorch Lightning 的时间序列预测实验框架：通过“Python 配置文件 + 统一训练入口”，快速复现实验、切换模型与数据集。

相关介绍（中文）：https://mp.weixin.qq.com/s/bSwAbKBxON7FPebAiqltWg

## 功能概览

- 统一训练入口：`train.py` 读取配置并完成 train/val/test
- 模型库：`easytsf/model/`（如 PatchTST、iTransformer、STID、TimeLLM 等）
- 配置系统：`config/` 下的 Python 配置文件（`exp_conf = dict(...)`）
- 数据读取：默认读取预处理好的 `npz` 文件（`dataset/<dataset_name>.npz`）
- 可选：Ray Tune 超参搜索（`ray_tune.py`）

## 快速开始

### 1) 安装环境

建议 Python 3.10+。

```shell
conda create -n easytsfnext python=3.10
conda activate easytsfnext
pip install -r requirements.txt
```

如果你希望按 CUDA/CPU 环境自行安装 PyTorch，可先安装对应的 `torch`，再执行：

```shell
pip install lightning "ray[tune]" numpy pandas
```

可选依赖（按需安装）：

```shell
pip install wandb
pip install transformers
```

### 2) 准备数据

将数据放到 `dataset/` 目录下，文件名需与配置里的 `dataset_name` 对应：

- `dataset/ETTh1.npz`
- `dataset/Weather.npz`
- `dataset/Traffic.npz`

数据格式要求（npz 内的 key）：

- `scaled_variable`: 形状为 `[T, N]` 或 `[T, N, ...]` 的数值序列（建议已做标准化/归一化）
- `timestamp`: 长度为 `T` 的时间戳数组（可被 `pandas.DatetimeIndex` 解析）
- `mean` / `std`: 用于反归一化的统计量（与 `scaled_variable` 对应）

### 3) 训练与评估

从 `config/` 中选择一个实验配置（它是一个 Python 文件，内部包含 `exp_conf = dict(...)`），例如：

```shell
python train.py -c config/iTransformer/ETTh1_96for96.py -d dataset -s save --accelerator auto --devices auto --seed 0
```

说明：

- `-c/--config`: 实验配置文件路径
- `-d/--data_root`: 数据目录（默认 `dataset`）
- `-s/--save_root`: 保存目录（默认 `save`）
- `--accelerator`: 设备类型（默认 `auto`，可选 `cpu/gpu/auto`）
- `--devices`: 设备数量或索引（如 `1`、`0,1`、`auto`）
- `--use_wandb 1`: 启用 Weights & Biases（可选）

训练结束后会默认用 best checkpoint 进行一次 `test`（见 `train.py`）。

## 配置说明（config）

配置由三部分融合而来，优先级为：`exp_conf > (task_conf + dataset_conf)`：

- `config/base_conf/task.py`: 通用训练参数（batch_size、max_epochs、优化器等）
- `config/base_conf/datasets.py`: 数据集默认参数（data_split、freq、var_num 等）
- `config/<ModelName>/*.py`: 实验参数（模型名、hist_len/pred_len、runner 类型等）

你可以直接复制 `config/iTransformer/ETTh1_96for96.py` 新建自己的配置文件，然后修改其中的字段。

## 输出目录（save）

默认保存结构如下（由 `train.py` 生成）：

- `save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`
  - Lightning logs（CSV 或 WandB）
  - `checkpoints/`（best ckpt 等）

## 可选：仅测试/指定 ckpt

`test.py` 支持通过参数指定 checkpoint：

```shell
python test.py -c config/iTransformer/ETTh1_96for96.py -d dataset -s save --accelerator auto --devices auto --ckpt_path best
```

也可以指定具体 ckpt 路径：

```shell
python test.py -c config/iTransformer/ETTh1_96for96.py -d dataset -s save --accelerator gpu --devices 1 --ckpt_path save/xxx/checkpoints/epoch=9-step=660.ckpt
```

## 可选：Ray Tune 超参搜索

```shell
python ray_tune.py -c config/iTransformer/ETTh1_96for96.py -p path/to/param_space.py -d dataset -s save --accelerator auto --devices auto --num_samples 10 --num_gpus 1 --gpus_per_trial 0.5
```

其中 `-p` 指向的 Python 文件需要包含 `param_space`（字典）。

示例：

```python
from ray import tune

param_space = {
    "lr": tune.grid_search([1e-3, 5e-4]),
    "dropout": tune.choice([0.1, 0.2, 0.3]),
}
```

## 目录结构

- `train.py`: 训练入口（train + test）
- `test.py`: 测试入口（支持 `--ckpt_path`）
- `ray_tune.py`: Ray Tune 超参搜索入口
- `config/`: 配置文件
- `easytsf/model/`: 模型实现
- `easytsf/runner/`: LightningModule 与数据读取逻辑

## Cite

If you find this repo useful, please cite our paper:

```
@inproceedings{han2023are,
  title={KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting?},
  author={Xiao Han, Xinfeng Zhang, Yiling Wu, Zhenduo Zhang and Zhe Wu},
  booktitle={arXiv},
  year={2024},
}
```
