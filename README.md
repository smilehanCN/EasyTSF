# EasyTSF

EasyTSF 是一个面向时序预测研究与 idea 快速验证的实验框架，重点解决“模型接入快、单实验调试快、批量 benchmark 组织清晰”这三件事。

它适合在统一流程下处理多种常见时序建模场景：

- 多变量时序
- 静态图时空序列
- 2D 规则网格时序
- 3D 规则网格时序

它主要帮助你更低成本地完成这些工作：

- 快速实现或复现模型
- 快速跑通单实验并调试训练流程
- 快速开展多数据集、多 horizon、多 seed 的批量 benchmark

仓库围绕几类核心抽象组织：

- `model`：模型实现
- `task`：问题级训练封装
- `experiment`：单实验配方
- `study`：批量 benchmark 声明
- `workflow`：负责把 `experiment` 和 `study` 跑起来的流程层

## 当前结构
- `train.py`：单实验训练入口；传入 `--param_space` 时执行 Ray Tune 搜索
- `evaluate.py`：单实验评估入口；支持 `best`、`last` 或显式 checkpoint 路径
- `study.py`：批量科研入口；编排多个 `experiment` 并汇总多 seed 结果
- `config/tasks/mtsf.yaml`：多元时序预测默认配置
- `config/tasks/stf.yaml`：静态图时空预测默认配置
- `config/tasks/grid2dtsf.yaml`：2D 规则网格时空预测默认配置
- `config/tasks/grid3dtsf.yaml`：3D 规则网格时空预测默认配置
- `config/experiments/<model_id>/*.yaml`：单实验 preset，推荐按“模型 x 数据集”组织，路径使用全小写 id
- `config/studies/<model_id>/*.yaml`：批量评测声明，只枚举要跑的 case 和 seeds，路径使用全小写 id
- `config/search_spaces/<model_id>/*.py`：Ray Tune 搜索空间，路径使用全小写 id
- `easytsf/data/`：`DataInterface`、`GridDataInterface`、BasicTS 风格序列数据加载与迁移工具
- `easytsf/task/`：当前包含 `MTSFTask`、`STFTask`、`Grid2DTSFTask`、`Grid3DTSFTask` 和一个实验性 `GridSTFTask` alias
- `easytsf/model/`：模型实现
- `easytsf/workflow/`：`experiment` 和 `study` 的流程实现

当前仓库正式维护四条主链路，覆盖 5 类常用 forecasting case：

- `mtsf`：标准时序预测，输入核心形态是 `(L, N)`；单变量预测按 `(L, 1)` 处理，不接受裸 `(L,)`
- `stf`：静态图时空预测，输入是 `(L, N)` 加一个静态 graph
- `grid2dtsf`：2D 规则网格时空预测，输入是 `(L, C, H, W)`
- `grid3dtsf`：3D 规则网格时空预测，输入是 `(L, C, X, Y, Z)`

`study` 不抽象数据逻辑，不维护 `dataset -> 超参` 规则，只做编排、resume 和聚合。

当前正式维护的模型集合由 `easytsf/model/contracts.py` 和 experiment preset 共同约束；当前 smoke 训练主要覆盖 Simple 系列 baseline，`iTransformer` 额外有前向校验：

- `mtsf`：`SimpleMLP`、`iTransformer`、`MOMENT`、`CoRA`
- `stf`：`SimpleGraphMLP`、`STGCN`、`CoRAGraph`
- `grid2dtsf/grid3dtsf`：`SimpleGridMLP`、`CoRAGrid`

其余历史模型仍保留在仓库里，但按 legacy 对待，不纳入默认 preset / maintained matrix 保障范围。

## 环境

推荐直接使用现成 conda 环境：

```shell
conda activate easytsf
pip install -r requirements.txt
```

如果需要标准包安装和测试入口，额外执行：

```shell
pip install -e .[dev]
```

Foundation model 相关依赖保持可选安装。当前首个接入模型是 `MOMENT`，需要额外安装：

```shell
pip install momentfm
```

如果上游包在当前 Python 版本下存在兼容性问题，优先回退到 Python 3.11 环境。

## 快速开始

单实验训练：

```shell
python train.py -c itransformer/etth1 -d dataset -s save --seed 0
```

静态图 STF 训练：

```shell
python train.py -c stgcn/pems03 -d dataset -s save --seed 0
python evaluate.py -c stgcn/pems03 --ckpt_path best --seed 0
```

Simple MLP smoke baseline：

```shell
python train.py -c simplemlp/pseudo -d dataset -s save --seed 0
python train.py -c simplemlp/etth1 -d dataset -s save --seed 0
python train.py -c simplegraphmlp/pems03 -d dataset -s save --seed 0
python train.py -c simplegridmlp/grid2d_demo -d dataset -s save --seed 0
python train.py -c simplegridmlp/grid3d_demo -d dataset -s save --seed 0
python train.py -c simplegridmlp/windfield3d_demo -d dataset -s save --seed 0
```

其中单变量实验 `simplemlp/pseudo` 依赖的数据格式是 `scaled_variable.shape == [L, 1]`；如果数据文件里存成裸 `[L]`，当前仓库会直接报错并要求改成 `[L, 1]`。

通过 `--set` 直接覆盖 horizon 或训练超参：

```shell
python train.py -c itransformer/etth1 --set data.hist_len=96 --set data.pred_len=336 --set train.lr=0.0001 --seed 0
python evaluate.py -c itransformer/etth1 --set data.hist_len=96 --set data.pred_len=336 --ckpt_path best --seed 0
```

也可以指定 `last` 或明确的 checkpoint 路径：

```shell
python evaluate.py -c itransformer/etth1 --ckpt_path last
python evaluate.py -c itransformer/etth1 --ckpt_path save/xxx/checkpoints/epoch=9-step=660.ckpt
```

测试与配置审计：

```shell
python -m unittest tests.test_config_contracts tests.test_smoke_mlp_support tests.test_grid_support
pytest
```

批量 benchmark：

```shell
python study.py -s itransformer/core
python study.py -s stgcn/core
python study.py -s simplegridmlp/core
python study.py -s itransformer/core --dry_run 1
python study.py -s itransformer/core --resume 1
```

Ray Tune 仍然只用于真正的超参搜索：

```shell
python train.py -c itransformer/etth1 -p itransformer/search -d dataset -s save --num_samples 10 --num_gpus 1 --gpus_per_trial 0.5
```

MOMENT 单实验训练：

```shell
python train.py -c moment/etth1 -d dataset -s save --seed 0
python evaluate.py -c moment/etth1 --ckpt_path best --seed 0
```

MOMENT 批量 benchmark：

```shell
python study.py -s moment/core
```

CoRA 单实验训练：

```shell
python train.py -c cora/ettm2_384for96 -d dataset -s save --seed 0
python evaluate.py -c cora/ettm2_384for96 --ckpt_path best --seed 0
```

CoRA 批量 benchmark：

```shell
python study.py -s cora/core
```

CoRAGraph / CoRAGrid 单实验训练：

```shell
python train.py -c coragraph/pems03 -d dataset -s save --seed 0
python train.py -c coragrid/grid2d_demo -d dataset -s save --seed 0
python train.py -c coragrid/grid3d_demo -d dataset -s save --seed 0
```

## 核心概念

先区分一句话版本：

- `task` 是“怎么训练和评估一个问题”的代码语义。
- `experiment` 是“一次具体实验跑什么配置”的配置语义。
- `study` 是“一批实验怎么组织和汇总”的配置语义。
- `workflow` 是“把 task / experiment / study 真正跑起来”的流程实现。

它们的关系可以理解为：

`task` 定义训练行为，`experiment` 选择一次运行的具体参数，`workflow` 负责执行一次 `experiment` 或一组 `study` case，`study` 只负责编排多个 `experiment`。

### `task`

`task` 是问题级训练封装层。它回答的是：

- 一个 batch 如何变成预测和标签
- loss 和 metric 如何定义
- optimizer 和 scheduler 如何配置
- train / val / test step 如何执行

当前仓库维护四个公开 task：

- `easytsf/task/mtsf.py` 中的 `MTSFTask`
- `easytsf/task/stf.py` 中的 `STFTask`
- `easytsf/task/gridstf.py` 中的 `Grid2DTSFTask`
- `easytsf/task/gridstf.py` 中的 `Grid3DTSFTask`

它们都不是“实验配置”，也不是“批量 benchmark 声明”，而是问题设定本身的训练语义承载点：

- `MTSFTask` 对应标准多元时序预测，模型接口是 `forward(var_x, marker_x, marker_y)`。
- `STFTask` 对应静态图时空预测，模型接口是 `forward(var_x, marker_x, graph)`。
- `Grid2DTSFTask` 对应 2D 规则网格时空预测，模型接口是 `forward(var_x, marker_x, grid_mask=None, coord=None)`，标签形态为 `[B, pred_len, C, H, W]`。
- `Grid3DTSFTask` 对应 3D 规则网格时空预测，模型接口是 `forward(var_x, marker_x, grid_mask=None, coord=None)`，标签形态为 `[B, pred_len, C, X, Y, Z]`。

也因此，新增模型时通常只需要改模型和 experiment preset；只有当问题设定发生变化时，才需要新增或调整 task。

### `experiment`

`experiment` 是一次可独立训练/评估的单实验配方。它回答的是：

- 这次实验属于哪个 task
- 这次实验用哪个模型
- 跑哪个数据集
- 默认 `hist_len/pred_len` 是什么
- 训练超参取什么值

因此，一个 `experiment` 通常已经包含：

- 模型参数
- 数据集选择
- 数据集专属训练超参
- 一组可直接运行的默认 `hist_len/pred_len`

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

`experiment` 是配置层概念，不是代码模块。它不负责真正执行训练，只负责把“一次实验应该怎么跑”描述清楚。

task 由 `runtime.task_name` 决定：

- 不写时默认是 `mtsf`
- 显式写 `runtime.task_name: stf` 时，workflow 会加载 `stf` 默认配置并构建 `STFTask`
- 显式写 `runtime.task_name: grid2dtsf` 时，workflow 会加载 `grid2dtsf` 默认配置并构建 `Grid2DTSFTask`
- 显式写 `runtime.task_name: grid3dtsf` 时，workflow 会加载 `grid3dtsf` 默认配置并构建 `Grid3DTSFTask`

### `workflow`

`workflow` 是流程实现层，位于 `easytsf/workflow/`。它回答的是：

- 如何加载并合并 task / dataset / experiment 配置
- 如何根据 `runtime.task_name` 构建 `DataInterface`、对应 task 和 trainer
- 如何执行单实验训练或评估
- 如何保存 `resolved_config.yaml`、`metrics.json`
- 如何对 study 做 resume、失败记录和结果聚合

当前有两个主要入口：

- `easytsf/workflow/experiment.py`：单实验流程
- `easytsf/workflow/study.py`：批量 benchmark 流程

`workflow` 是代码层概念，不是 YAML 配置对象。不要把它理解成又一层 experiment/study 配置。

### `study`

`study` 是一组 `experiment` 的批量执行与结果汇总声明。它回答的是：

- 跑哪些 case
- 每个 case 跑哪些 seed
- 如何 resume
- 如何汇总 MAE/MSE

典型配置如下：

```yaml
name: itransformer_core
seeds: [0, 1, 2]
cases:
  - experiment: itransformer/etth1
    overrides:
      data:
        hist_len: 96
        pred_len: 96
  - experiment: itransformer/traffic
    overrides:
      data:
        hist_len: 96
        pred_len: 96
```

如果某个数据集或某个 horizon 的特殊配方长期存在，就直接升级成新的 `experiment preset`，不要把复杂逻辑长期堆在 `study` 里。

`study` 不是第二套 experiment 配置系统，也不是 `dataset -> 超参` 的映射层。它的职责非常薄：编排、重复、汇总，不拥有数据集专属训练配方。

## 配置组织

配置主体按三层合并，优先级固定为：

`config overrides > experiment > task`

其中 `config overrides` 包括：

- `--set SECTION.KEY=VALUE`
- `study` 中 `cases[].overrides`

像 `--seed`、`--data_root`、`--save_root`、`--accelerator` 这类运行时参数不走四个 YAML section，而是在配置展开后单独覆盖。

对应位置：

- `config/tasks/{mtsf,stf,grid2dtsf,grid3dtsf}.yaml`
- `config/experiments/<model_id>/*.yaml`

YAML 只使用四个顶层 section：

- `model`
- `data`
- `train`
- `runtime`

这些 section 会在运行前展开成扁平配置，继续复用当前模型构造和 `DataInterface` 的参数契约。

其中 `runtime.task_name` 用来选择 task 默认配置与 task 实现；默认值是 `mtsf`。

## 数据约定

数据目录位置默认是 `dataset/<dataset_name>/`。当前仓库把 `meta.json` 视为唯一权威元信息入口，不再维护中心化 dataset catalog。

`mtsf` / `stf` 直接兼容 BasicTS 当前的序列目录布局：

- `meta.json`
- `train_data.npy`
- `val_data.npy`
- `test_data.npy`
- 可选 `train_timestamps.npy` / `val_timestamps.npy` / `test_timestamps.npy`
- `stf` 可选 `adj_mx.pkl`

其中：

- `meta.json` 至少负责 `name`、`frequency (minutes)`、`split_lengths`、`has_graph`
- 时间戳文件使用 BasicTS 风格的预处理特征，`DataInterface` 会按 `timestamps_description` 还原成当前模型接口使用的 marker 语义
- graph side file 固定约定为 `adj_mx.pkl`，兼容 BasicTS 的 raw adjacency pickle 和 `(sensor_ids, sensor_id_to_ind, adj_mx)` tuple pickle

对于原生 grid 数据集，`scaled_variable` 的形状约定为：

- 2D：`[L, C, H, W]`
- 3D：`[L, C, X, Y, Z]`

grid 数据集的可选 side input 固定放在目录 side file 中：

- `grid_mask.npy`：纯空间 mask，2D 为 `[H, W]`，3D 为 `[X, Y, Z]`
- `coord.npy`：坐标通道在前，2D 为 `[2, H, W]`，3D 为 `[3, X, Y, Z]`

旧版 flat `dataset/<dataset_name>.npz` 已不再兼容。额外文件会被忽略。`DataInterface` / `GridDataInterface` 负责：

- 对 sequence 任务读取 `train/val/test_*.npy`
- 对 grid 任务读取 `data.npz`
- 从目录内 `meta.json` 解析频率、split 长度和 side input 事实
- 按需读取 `adj_mx.pkl`
- 按需读取 `grid_mask` 和 `coord`
- 按 `hist_len` / `pred_len` 切滑窗
- 生成 train / val / test loader

如果你手里还是旧的 sequence 目录布局，可以用迁移脚本：

```shell
python scripts/migrate_sequence_dataset_to_basicts.py --data_root dataset --dataset_name ETTh1 --split_lengths 8640,2880,2880 --freq 60 --timestamp_features tod,dow,dom,doy
```

## 结果目录

单实验结果默认写到：

`save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`

每次训练或评估都会额外保存：

- `resolved_config.yaml`
- `metrics.json`

训练流程还会额外保存：

- `checkpoints/`

`metrics.json` 至少包含：

- `task_name`
- `model_name`
- `dataset_name`
- `hist_len`
- `pred_len`
- `seed`
- `mae`
- `mse`
- `status`
- `exp_dir`
- `ckpt_path`

`study` 结果默认写到：

`save/studies/<study_name>/`

其中包含：

- `runs.csv`：一行一个 seed 运行
- `summary.csv`：按 `task_name,model_name,dataset_name,hist_len,pred_len` 聚合后的均值/标准差（当前实现使用总体标准差）

## 模型接入

新增标准 task 模型时，保持下面几步即可：

1. 在 `easytsf/model/<model_id>.py` 中定义主类 `Model`。
2. 按所属 task 实现前向接口：
   - `mtsf` 模型：`forward(var_x, marker_x, marker_y)`
   - `stf` 模型：`forward(var_x, marker_x, graph)`
   - `grid2dtsf` 模型：`forward(var_x, marker_x, grid_mask=None, coord=None)`
   - `grid3dtsf` 模型：`forward(var_x, marker_x, grid_mask=None, coord=None)`
3. 返回与标签兼容的预测张量：
   - `mtsf` / `stf`：通常是 `[B, pred_len, N]`
   - `grid2dtsf`：通常是 `[B, pred_len, C, H, W]`
   - `grid3dtsf`：通常是 `[B, pred_len, C, X, Y, Z]`
4. 在 `config/experiments/<model_id>/` 下为常用数据集新增 preset。

模型文件名统一全小写，一个模型只占一个文件。如果只是模型私有逻辑，就放在该模型文件内部，不要急于抽新的公共层。
