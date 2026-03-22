# EasyTSFNext

EasyTSFNext 是一个面向个人研究的轻量时序预测仓库，核心目标是让不同模型共享同一条 forecast pipeline，减少实验脚手架差异。

## 当前结构

- `train.py`：统一训练入口；传入 `--param_space` 时执行 Ray Tune 搜索
- `test.py`：统一测试入口；支持 `best`、`last` 或显式 checkpoint 路径
- `config/tasks/forecast.yaml`：forecast 主链路默认配置
- `config/datasets/catalog.yaml`：数据集元信息和数据加载默认项
- `config/experiments/<ModelName>/*.yaml`：实验配置
- `config/search_spaces/<ModelName>/*.py`：Ray Tune 搜索空间
- `easytsf/data/`：`DataInterface` 和数据缓存/滑窗逻辑
- `easytsf/task/`：唯一公开任务层 `ForecastTask`
- `easytsf/model/`：模型实现
- `easytsf/layer/transformer.py`：当前仍在复用的 Transformer 组件

当前仓库保留的主链路只覆盖 forecast 任务。旧的 runner 变体、历史实验 config 和 `ray_tune.py` 包装层已经移除。

## 环境

推荐直接使用现成 conda 环境：

```shell
conda activate easytsf
pip install -r requirements.txt
```

## 快速开始

训练：

```shell
python train.py -c iTransformer/ETTh1_96for96 -d dataset -s save --accelerator auto --devices auto --seed 0
```

测试：

```shell
python test.py -c iTransformer/ETTh1_96for96 -d dataset -s save --accelerator auto --devices auto --ckpt_path best
```

也可以指定 `last` 或明确的 checkpoint 路径：

```shell
python test.py -c iTransformer/ETTh1_96for96 --ckpt_path last
python test.py -c iTransformer/ETTh1_96for96 --ckpt_path save/xxx/checkpoints/epoch=9-step=660.ckpt
```

Ray Tune 搜索：

```shell
python train.py -c iTransformer/ETTh1_96for96 -p iTransformer/search -d dataset -s save --num_samples 10 --num_gpus 1 --gpus_per_trial 0.5
```

## 配置组织

配置按三层合并，优先级固定为：

`experiment > dataset > task`

对应位置：

- `config/tasks/forecast.yaml`
- `config/datasets/catalog.yaml`
- `config/experiments/<ModelName>/*.yaml`

YAML 目前只使用四个顶层 section：

- `model`
- `data`
- `train`
- `runtime`

这些 section 会在运行前展开成扁平配置，继续复用现有模型构造和 `DataInterface` 的参数契约。

示例：

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
runtime:
  use_mmap: false
```

Ray Tune 搜索空间仍保留 Python 文件，因为需要直接定义 `tune.grid_search(...)` 等对象。

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

训练结果默认写到：

`save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`

测试脚本中的 `best` / `last` 会从该实验目录下的 `checkpoints/` 自动解析。

## 模型接入

新增标准 forecast 模型时，保持下面三步即可：

1. 在 `easytsf/model/<ModelName>.py` 中定义与 `model_name` 同名的类。
2. 让模型实现 `forward(var_x, marker_x)`，返回形状兼容的预测结果。
3. 在 `config/experiments/<ModelName>/` 下新增 YAML 实验配置。

如果只是模型私有逻辑，不要急于再抽新的公共层或重新引入 task 变体。
