# EasyTSFNext

EasyTSFNext 是一个面向科研 idea 快速验证的轻量时序预测仓库。当前核心目标很明确：

- 快速实现一个新模型
- 快速跑单实验调试
- 快速做多数据集、多 horizon、多 seed 的批量 benchmark

项目结构围绕几类边界清晰的术语组织：

- `model`：模型实现
- `task`：问题级训练封装
- `experiment`：单实验配方
- `study`：批量 benchmark 声明
- `workflow`：运行这些配置和任务的流程层

包内分层采用“领域主干 + workflow 流程层”的方式，不追求重工程化，不引入多任务注册器、复杂校验体系或完整测试矩阵。

## 当前结构

- `train.py`：单实验训练入口；传入 `--param_space` 时执行 Ray Tune 搜索
- `evaluate.py`：单实验评估入口；支持 `best`、`last` 或显式 checkpoint 路径
- `study.py`：批量科研入口；编排多个 `experiment` 并汇总多 seed 结果
- `config/tasks/mtsf.yaml`：多元时序预测默认配置
- `config/tasks/stf.yaml`：静态图时空预测默认配置
- `config/tasks/grid2dtsf.yaml`：2D 规则网格时空预测默认配置
- `config/tasks/grid3dtsf.yaml`：3D 规则网格时空预测默认配置
- `config/datasets/catalog.yaml`：数据集元信息和数据加载默认项
- `config/experiments/<model_id>/*.yaml`：单实验 preset，推荐按“模型 x 数据集”组织，路径使用全小写 id
- `config/studies/<model_id>/*.yaml`：批量评测声明，只枚举要跑的 case 和 seeds，路径使用全小写 id
- `config/search_spaces/<model_id>/*.py`：Ray Tune 搜索空间，路径使用全小写 id
- `easytsf/data/`：`DataInterface`、`GridDataInterface` 和数据缓存/滑窗逻辑
- `easytsf/task/`：当前包含 `MTSFTask`、`STFTask`、`Grid2DTSFTask`、`Grid3DTSFTask` 和一个实验性 `GridSTFTask` alias
- `easytsf/model/`：模型实现
- `easytsf/workflow/`：`experiment` 和 `study` 的流程实现

当前仓库正式维护四条主链路，覆盖 5 类常用 forecasting case：

- `mtsf`：标准时序预测，输入核心形态是 `(L, N)`；单变量预测按 `(L, 1)` 处理，不接受裸 `(L,)`
- `stf`：静态图时空预测，输入是 `(L, N)` 加一个静态 graph
- `grid2dtsf`：2D 规则网格时空预测，输入是 `(L, C, H, W)`
- `grid3dtsf`：3D 规则网格时空预测，输入是 `(L, C, X, Y, Z)`

`study` 不抽象数据逻辑，不维护 `dataset -> 超参` 规则，只做编排、resume 和聚合。

当前正式维护的模型矩阵由 preset 和 smoke 测试共同定义：

- `mtsf`：`SimpleMLP`、`iTransformer`、`MOMENT`、`CoRA`
- `stf`：`SimpleGraphMLP`、`STGCN`、`CoRAGraph`
- `grid2dtsf/grid3dtsf`：`SimpleGridMLP`、`CoRAGrid`

其余历史模型仍保留在仓库里，但按 legacy 对待，不纳入默认 preset / smoke 保障范围。

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

## Foundation Model

当前仓库把时序基础模型作为普通 `mtsf` 模型接入，继续复用现有 `train.py` / `evaluate.py` / `study.py` 主链路，不新增 zero-shot pipeline 或新的 task 抽象。

`v1` 首个接入模型是 `MOMENT`：

- experiment 入口：`config/experiments/moment/etth1.yaml`
- study 入口：`config/studies/moment/core.yaml`
- 模型配置键：
  - `moment_model_name_or_path`
  - `freeze_encoder`
  - `freeze_embedder`
  - `freeze_head`
  - `head_dropout`

当前实现只支持 forecasting fine-tune 和点预测评估：

- 不支持 zero-shot pipeline
- 不支持 probabilistic outputs
- 不支持 future covariates
- `marker_x` 会保留在统一接口中，但 `MOMENT` 当前不会消费它

`moment_model_name_or_path` 既可以填 Hugging Face 模型 id，也可以填本地模型目录。

基于 ICLR 2026 `CoRA` 的兼容实现也已加入当前框架。这里的接入方式是：

- 论文思路与官方仓库结构对齐，保留 `adapter -> projections_before -> contrastive -> projections_after -> gated fusion`
- 当前仓库里先只支持 `CoRA + MOMENT`
- experiment 入口：`config/experiments/cora/ettm2_384for96.yaml`
- study 入口：`config/studies/cora/core.yaml`
- `CoRA` 通过模型内部的 auxiliary contrastive loss 参与训练，不新增新的 task 或 CLI

在此基础上，仓库现在额外提供两个原生时空迁移版本：

- `CoRAGraph`：复用现有 `stf` 契约，把节点视作 graph token，引入图 diffusion prior、graph-aware projection block 和节点级 gated fusion
- `CoRAGrid`：复用 `grid2dtsf` / `grid3dtsf` 契约，把 2D/3D 规则网格切成 spatial patch token，引入局部邻域 prior、`Conv2d/Conv3d` spatial mixer 和 patch-level gated fusion

对应入口：

- `config/experiments/coragraph/pems03.yaml`
- `config/studies/coragraph/core.yaml`
- `config/experiments/coragrid/grid2d_demo.yaml`
- `config/experiments/coragrid/grid3d_demo.yaml`
- `config/studies/coragrid/core.yaml`

`CoRAGraph` / `CoRAGrid` 额外使用这些配置键：

- `structure_prior_weight`
- `neighbor_order`
- `spatial_patch_size`
- `spatial_mixer_type`

当前原生网格路径只支持 forecasting fine-tune，不支持：

- zero-shot foundation-model pipeline
- probabilistic outputs
- future covariates
- sphere / mesh / irregular grid

`gridstf` 目前只保留为实验性 task alias，方便复用同一套 grid datamodule / task 封装做快速验证，不进入默认 benchmark 主链路。

`CoRA` 相关模型配置键：

- `foundation_model`
- `plugin_dim`
- `num_before`
- `num_after`
- `beta`
- `dropout`
- `head_dropout`
- `plugin_lr`
- `backbone_lr`
- `gama`
- `K`
- `de`
- `thresold`

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

- `MTSFTask` 对应标准多元时序预测，模型接口是 `forward(var_x, marker_x)`。
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

配置按三层合并，优先级固定为：

`experiment > dataset > task`

对应位置：

- `config/tasks/{mtsf,stf,grid2dtsf,grid3dtsf}.yaml`
- `config/datasets/catalog.yaml`
- `config/experiments/<model_id>/*.yaml`

YAML 只使用四个顶层 section：

- `model`
- `data`
- `train`
- `runtime`

这些 section 会在运行前展开成扁平配置，继续复用当前模型构造和 `DataInterface` 的参数契约。

其中 `runtime.task_name` 用来选择 task 默认配置与 task 实现；默认值是 `mtsf`。

## 数据约定

数据文件位置默认是 `dataset/<dataset_name>.npz`，当前主链路实际依赖的核心 key 是：

- `scaled_variable`
- `timestamp`

对于 `stf` 数据集，静态 graph 不放在主 `.npz` 中，而是通过 dataset catalog 里的可选 `data.graph_path` 指向一个独立文件。该路径相对 `data_root` 解析，graph 默认是 dense float32 矩阵，形状为 `[N, N]`。

对于原生 grid 数据集，`scaled_variable` 的形状约定为：

- 2D：`[L, C, H, W]`
- 3D：`[L, C, X, Y, Z]`

grid 主 `.npz` 还可以包含两个可选 side input：

- `grid_mask`：纯空间 mask，2D 为 `[H, W]`，3D 为 `[X, Y, Z]`
- `coord`：坐标通道在前，2D 为 `[2, H, W]`，3D 为 `[3, X, Y, Z]`

额外字段会被忽略。`DataInterface` / `GridDataInterface` 负责：

- 读取 `.npz`
- 按需读取静态 graph
- 按需读取 `grid_mask` 和 `coord`
- 生成 `tod`、`dow`、`dom`、`doy` 等时间特征
- 按 `hist_len` / `pred_len` 切滑窗
- 按 `data_split` 生成 train / val / test loader

## 结果目录

单实验结果默认写到：

`save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`

每次训练或评估会额外保存：

- `resolved_config.yaml`
- `metrics.json`
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
- `summary.csv`：按 `task_name,model_name,dataset_name,hist_len,pred_len` 聚合后的均值/方差

## 模型接入

新增标准 task 模型时，保持下面几步即可：

1. 在 `easytsf/model/<model_id>.py` 中定义主类 `Model`。
2. 按所属 task 实现前向接口：
   - `mtsf` 模型：`forward(var_x, marker_x)`
   - `stf` 模型：`forward(var_x, marker_x, graph)`
   - `grid2dtsf` 模型：`forward(var_x, marker_x, grid_mask=None, coord=None)`
   - `grid3dtsf` 模型：`forward(var_x, marker_x, grid_mask=None, coord=None)`
3. 返回与标签兼容的预测张量：
   - `mtsf` / `stf`：通常是 `[B, pred_len, N]`
   - `grid2dtsf`：通常是 `[B, pred_len, C, H, W]`
   - `grid3dtsf`：通常是 `[B, pred_len, C, X, Y, Z]`
4. 在 `config/experiments/<model_id>/` 下为常用数据集新增 preset。

模型文件名统一全小写，一个模型只占一个文件。如果只是模型私有逻辑，就放在该模型文件内部，不要急于抽新的公共层。
