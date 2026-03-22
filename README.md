# EasyTSFNext

EasyTSFNext 是一个面向个人学术研究的轻量时间序列预测代码库，基于 `lightning.pytorch` 和 `ray.tune` 构建。它的目标不是提供一套重工程化的平台，而是用尽可能少的样板代码，搭建统一、可复现、便于公平比较的实验 pipeline，让模型开发者把主要精力放在模型本身。

## 项目定位

- 用统一的训练、验证、测试入口组织实验，减少不同模型之间的流程偏差。
- 用配置驱动管理实验参数，保证对比实验的可复现性与可追溯性。
- 将数据读取、训练逻辑、日志与 checkpoint 管理收敛到公共模块，降低新模型接入成本。
- 使用 `ray.tune` 做高效调参，但不追求复杂的平台化封装。

## 设计原则

- 公平对比优先：不同模型尽量共享同一套数据切分、训练流程与评估方式。
- 配置驱动优先：实验差异放在 `exp_conf`，而不是散落在脚本中。
- 模型实现解耦：模型代码尽量只关注 `forward` 和必要的网络结构。
- 保持简洁：默认不做复杂数据格式校验，也不维护完整的工程测试体系。

## 核心能力

- 统一训练入口：`train.py` 负责配置加载、Trainer 构建、训练与自动测试。
- 独立测试入口：`test.py` 支持 `best`、`last` 或显式 `ckpt_path`。
- 配置系统：`task_conf + dataset_conf + exp_conf` 组合出最终实验配置。
- 公共数据管线：`DataInterface` 负责 `.npz` 数据读取、时间特征构造与滑窗切分。
- 公共训练骨架：不同 `Runner` 负责标准预测、辅助损失、单变量预测、重建等任务变体。
- 调参支持：`train.py` 在提供 `--param_space` 时切换到 Ray Tune 搜索模式。

## 代码结构

- `train.py`：统一入口；默认执行训练，提供 `--param_space` 时执行 Ray Tune 搜索
- `test.py`：统一测试入口
- `ray_tune.py`：兼容旧命令的包装脚本，内部委托给 `train.py`
- `config/base_conf/`：公共任务参数和数据集参数
- `config/<ModelName>/`：单个实验配置，定义 `exp_conf = dict(...)`
- `easytsf/runner/`：Lightning DataModule 和各类 Runner
- `easytsf/model/`：当前仓库中实际存在的模型实现
- `easytsf/layer/`：模型复用的公共层与 Transformer 组件

当前 `easytsf/model/` 中包含 `iTransformer`、`STID`、`SparseTSF`、`RLinear`、`MLP`、`TimeLLM` 等实现。`config/` 中保留了部分历史实验配置，使用前请以 `easytsf/model/` 下是否存在对应模型实现为准。

## 运行流程

统一实验链路如下：

`exp_conf` + `task_conf` + `dataset_conf`
-> `load_config()`
-> `DataInterface`
-> `Runner`
-> `Model`
-> `lightning.pytorch.Trainer`
-> `fit()`
-> `test(ckpt_path="best")`

其中：

- `train.py` 根据 `exp_runner` 选择具体的 Runner。
- Runner 根据 `model_name` 动态加载 `easytsf/model/<ModelName>.py` 中的同名类。
- 训练输出默认保存到 `save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`。

## 快速开始

### 1. 安装环境

当前仓库的最小依赖见 `requirements.txt`：

```shell
pip install -r requirements.txt
```

说明：普通训练和测试不再硬依赖 `ray[tune]`；只有使用 `--param_space` 进行 Ray Tune 搜索时才需要安装 `ray[tune]`。

如需创建独立环境，当前建议使用 Python 3.10+：

```shell
conda create -n easytsfnext python=3.10
conda activate easytsfnext
pip install -r requirements.txt
```

按需安装可选依赖：

```shell
pip install wandb
pip install transformers
```

### 2. 准备数据

将数据放到 `dataset/` 目录，文件名与配置中的 `dataset_name` 对应，例如：

```text
dataset/ETTh1.npz
dataset/Weather.npz
dataset/Traffic.npz
```

### 3. 训练

```shell
python train.py -c config/iTransformer/ETTh1_96for96.py -d dataset -s save --accelerator auto --devices auto --seed 0
```

### 4. 测试

```shell
python test.py -c config/iTransformer/ETTh1_96for96.py -d dataset -s save --accelerator auto --devices auto --ckpt_path best
```

也可以指定明确的 checkpoint 路径：

```shell
python test.py -c config/iTransformer/ETTh1_96for96.py -d dataset -s save --accelerator auto --devices auto --ckpt_path save/xxx/checkpoints/epoch=9-step=660.ckpt
```

### 5. Ray Tune 调参

```shell
python train.py -c config/iTransformer/ETTh1_96for96.py -p path/to/param_space.py -d dataset -s save --accelerator auto --devices auto --num_samples 10 --num_gpus 1 --gpus_per_trial 0.5
```

`param_space.py` 需要提供 `param_space` 字典，例如：

```python
from ray import tune

param_space = {
    "lr": tune.grid_search([1e-3, 5e-4]),
    "dropout": tune.choice([0.1, 0.2, 0.3]),
}
```

## 数据约定

当前代码真正依赖的数据格式是 `dataset/<dataset_name>.npz`，至少包含以下 key：

- `scaled_variable`：数值序列，通常形状为 `[T, N]`
- `timestamp`：长度为 `T` 的时间戳数组，可被 `pandas.DatetimeIndex` 解析
- `mean` / `std`：用于反归一化的统计量

`DataInterface` 会：

- 读取 `scaled_variable` 和 `timestamp`
- 按 `time_feature_cls` 生成 `tod`、`dow`、`dom`、`doy` 等时间特征
- 按 `hist_len` 和 `pred_len` 构造滑动窗口
- 按 `data_split = [train_len, val_len, test_len]` 切分 train/val/test

这里默认相信输入数据已经完成必要预处理，不额外增加复杂格式校验。

## 配置说明

配置由三层组成，优先级为：

`exp_conf > task_conf + dataset_conf`

对应位置如下：

- `config/base_conf/task.py`：公共训练参数，如 `batch_size`、`max_epochs`、`optimizer`、`exp_runner`
- `config/base_conf/datasets.py`：数据集元信息，如 `dataset_name`、`freq`、`data_split`、`var_num`
- `config/<ModelName>/*.py`：具体实验参数，如 `model_name`、`hist_len`、`pred_len`、网络超参数

如果要新增一个标准预测模型，通常只需要：

1. 在 `easytsf/model/<ModelName>.py` 中实现与 `model_name` 同名的类。
2. 让模型的 `forward(var_x, marker_x)` 返回形状兼容的预测结果。
3. 在 `config/<ModelName>/` 下添加对应的 `exp_conf` 文件。

如果模型需要特殊 loss 或返回额外中间量，再新增或复用对应的 Runner，并在配置中设置 `exp_runner`。

## 当前边界

- 这是个人研究代码库，优先保证实验迭代效率，而不是建设完整产品级基础设施。
- 默认不提供复杂数据校验、数据注册系统或多层抽象插件体系。
- 默认不维护完整单元测试矩阵，验证方式以脚本级和 smoke 级检查为主。
- 文档以当前仓库真实状态为准，不承诺仓库外或历史分支中的能力。

## Roadmap

后续计划围绕“轻量提质”进行重构，重点包括：

- 统一运行环境到 Python 3.11
- 升级依赖到 Lightning 2.6.1 和 PyTorch 2.9.1
- 清理和合并重复 Runner 逻辑，减少实验特例分散
- 提升类型标注、日志信息和错误提示质量
- 清理失效配置与过期文档，减少“配置存在但实现缺失”的歧义
- 补充轻量 smoke 验证，而不是引入完整重工程测试体系

## 协作说明

如果你准备让代码代理参与后续重构或模型接入，请先阅读仓库根目录下的 `AGENT.md`。其中会说明本项目的边界、结构职责和修改准则。

如需兼容旧命令，`python ray_tune.py ...` 仍可使用，但内部会复用 `train.py` 的同一套实现。
