# EasyTSF

EasyTSF 是一个面向多变量时序预测研究的轻量实验框架，目标是把仓库收口到一条清晰主链路：`mtsf`。

当前仓库只正式维护：

- 多变量时序预测任务 `mtsf`
- 单实验 preset
- 批量评测 `benchmark`
- 直接可运行的 workflow API

不再兼容 `grid` 数据路径、`npz` 数据布局、graph side input 或通用 TSF 抽象。

## 当前结构

- `config/experiments/<model_id>/*.yaml`：单实验 preset
- `config/benchmarks/<model_id>/*.py`：批量 benchmark 声明
- `easytsf/data/`：`MTSDataModule` 和 sequence-only 数据加载逻辑
- `easytsf/task/`：`MTSFTask` 与 task registry
- `easytsf/model/`：模型实现
- `easytsf/workflow/`：experiment / benchmark 的配置加载和执行

## 环境

推荐直接使用现成 conda 环境：

```shell
conda activate easytsf
pip install -r requirements.txt
```

如果需要标准包安装入口，额外执行：

```shell
pip install -e .
```

Foundation model 相关依赖保持可选安装。当前首个接入模型是 `MOMENT`，需要额外安装：

```shell
pip install momentfm
```

## 快速开始

单实验训练：

```python
from easytsf.workflow import finalize_runtime_conf, load_experiment_config, run_experiment

base_conf = load_experiment_config("tqnet/etth1")
conf = finalize_runtime_conf(
    base_conf,
    overrides={
        "data_root": "dataset",
        "save_root": "save",
        "seed": 0,
    },
)
run_experiment(conf)
```

评估 best checkpoint：

```python
from easytsf.workflow import run_evaluation

run_evaluation(conf, ckpt_path="best")
```

批量评测：

```python
from easytsf.workflow import run_benchmark

result = run_benchmark(
    "tqnet/core",
    runtime_overrides={
        "data_root": "dataset",
        "save_root": "save",
        "seed": 0,
    },
    dry_run=False,
    resume=True,
)
print(result["benchmark_dir"])
```

## 数据格式

`mtsf` 数据集使用目录布局：

```text
dataset/<dataset_name>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
```

约束如下：

- `*_data.npy` 必须是 `[L, N]`
- 单变量任务也必须写成 `[L, 1]`，不接受裸 `[L]`
- 三个 split 的列数必须一致
- 三个 split 都必须提供时间戳文件
- `meta.json` 需要提供频率和 `timestamps_description`

## 核心概念

### `task`

当前只维护一个公开 task：`easytsf/task/mtsf.py` 中的 `MTSFTask`。

它负责：

- batch 到预测/标签的转换
- loss 和 metric
- optimizer / scheduler 装配
- train / val / test step

模型接口统一为：

```python
forward(var_x, marker_x, marker_y)
```

默认标签兼容形态是 `[B, pred_len, N]`。

### `experiment`

`experiment` 是一次可直接训练/评估的单实验配方。它应至少描述：

- 使用哪个模型
- 跑哪个数据集
- `hist_len/pred_len`
- 训练超参
- 通用 loader/runtime 参数

配置来源固定为：

`experiment preset < runtime overrides`

experiment preset 是唯一的静态研究配方；`data_root`、`save_root`、`seed`、`devices`、`accelerator` 这类机器或运行环境参数继续通过运行时 override 注入。

### `benchmark`

`benchmark` 是一组 `experiment` 的批量执行与结果汇总声明。它只负责编排：

- 用哪个 experiment
- 跑哪些 seed
- 用什么搜索空间
- 如何 resume
- 如何汇总结果

## 测试

推荐最小验证：

```shell
python -m compileall easytsf
```

如果本地有数据，再补一个最小 `mtsf` smoke experiment 即可。
