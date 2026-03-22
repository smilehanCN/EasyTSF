# EasyTSFNext

EasyTSFNext 是一个面向个人研究的轻量时序预测仓库，当前主目标是让多元时序预测的模型原型、单实验调试和批量 benchmark 复用同一条 forecast pipeline。

## 当前结构

- `train.py`：单实验训练入口；传入 `--param_space` 时执行 Ray Tune 搜索
- `test.py`：单实验测试入口；支持 `best`、`last` 或显式 checkpoint 路径
- `study.py`：批量科研入口；编排多个 `experiment` 并汇总多 seed 结果
- `config/tasks/forecast.yaml`：forecast 主链路默认配置
- `config/datasets/catalog.yaml`：数据集元信息和数据加载默认项
- `config/experiments/<ModelName>/*.yaml`：单实验 preset，推荐按“模型 x 数据集”组织
- `config/studies/<ModelName>/*.yaml`：批量评测声明，只枚举要跑的 case 和 seeds
- `config/search_spaces/<ModelName>/*.py`：Ray Tune 搜索空间
- `easytsf/data/`：`DataInterface` 和数据缓存/滑窗逻辑
- `easytsf/task/`：唯一公开任务层 `ForecastTask`
- `easytsf/model/`：模型实现

当前仓库只保留 forecast 主链路。`study` 不抽象数据逻辑，不引入新的 task 层，只做编排与聚合。

## 环境

推荐直接使用现成 conda 环境：

```shell
conda activate easytsf
pip install -r requirements.txt
```

## 快速开始

单实验训练：

```shell
python train.py -c iTransformer/ETTh1 -d dataset -s save --seed 0
```

通过 `--set` 直接覆盖 horizon 或训练超参：

```shell
python train.py -c iTransformer/ETTh1 --set data.hist_len=96 --set data.pred_len=336 --set train.lr=0.0001 --seed 0
python test.py -c iTransformer/ETTh1 --set data.hist_len=96 --set data.pred_len=336 --ckpt_path best --seed 0
```

也可以指定 `last` 或明确的 checkpoint 路径：

```shell
python test.py -c iTransformer/ETTh1 --ckpt_path last
python test.py -c iTransformer/ETTh1 --ckpt_path save/xxx/checkpoints/epoch=9-step=660.ckpt
```

批量 benchmark：

```shell
python study.py -s iTransformer/core
python study.py -s iTransformer/core --dry_run 1
python study.py -s iTransformer/core --resume 1
```

Ray Tune 搜索仍然可用，但只用于超参搜索，不再把重复 seed 当作标准 benchmark 工作流：

```shell
python train.py -c iTransformer/ETTh1 -p iTransformer/search -d dataset -s save --num_samples 10 --num_gpus 1 --gpus_per_trial 0.5
```

## 配置组织

配置仍按三层合并，优先级固定为：

`experiment > dataset > task`

对应位置：

- `config/tasks/forecast.yaml`
- `config/datasets/catalog.yaml`
- `config/experiments/<ModelName>/*.yaml`

YAML 只使用四个顶层 section：

- `model`
- `data`
- `train`
- `runtime`

这些 section 会在运行前展开成扁平配置，继续复用现有模型构造和 `DataInterface` 的参数契约。

推荐把 `experiment` 组织成“模型 x 数据集”的 preset，例如：

```yaml
model:
  model_name: iTransformer
  d_model: 256
data:
  dataset_name: ETTh1
  hist_len: 96
  pred_len: 96
train:
  lr: 0.0001
```

如果只是临时改 horizon、batch size 或学习率，优先用 `--set`，不要复制出新的 YAML。

`study` 配置只负责枚举要跑的 case：

```yaml
name: itransformer_core
seeds: [0, 1, 2]
cases:
  - experiment: iTransformer/ETTh1
    overrides:
      data:
        hist_len: 96
        pred_len: 96
  - experiment: iTransformer/Traffic
    overrides:
      data:
        hist_len: 96
        pred_len: 96
```

如果某个数据集的超参数长期和其他数据集不同，就直接写进各自的 `experiment preset`，不要堆到 `study` 里。

## 数据约定

数据文件位置默认是 `dataset/<dataset_name>.npz`，当前主链路实际依赖的 key 只有：

- `scaled_variable`
- `timestamp`

额外字段会被忽略。`DataInterface` 会负责：

- 读取 `.npz`
- 生成 `tod`、`dow`、`dom`、`doy` 等时间特征
- 按 `hist_len` / `pred_len` 切滑窗
- 按 `data_split` 生成 train / val / test loader

## 结果目录

单实验结果默认写到：

`save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`

每次运行会额外保存：

- `resolved_config.yaml`
- `metrics.json`
- `checkpoints/`

`metrics.json` 至少包含 `model_name`、`dataset_name`、`hist_len`、`pred_len`、`seed`、`mae`、`mse`、`status`、`exp_dir`、`ckpt_path`。

`study` 结果默认写到：

`save/studies/<study_name>/`

其中包含：

- `runs.csv`：一行一个 seed 运行
- `summary.csv`：按 `model_name,dataset_name,hist_len,pred_len` 聚合后的均值/方差

## 模型接入

新增标准 forecast 模型时，保持下面三步即可：

1. 在 `easytsf/model/<ModelName>.py` 中定义与 `model_name` 同名的类。
2. 让模型实现 `forward(var_x, marker_x)`，返回形状兼容的预测结果。
3. 在 `config/experiments/<ModelName>/` 下为常用数据集新增 preset。

旧的 `*_96for*.yaml` 仍可继续读取，但新工作流默认使用“数据集 preset + `--set` + `study` 枚举”的方式。
