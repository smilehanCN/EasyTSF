# AGENT.md

本文件面向参与本仓库工作的 AI 代码代理。目标是让代理在尽量少打扰研究流程的前提下，理解当前结构边界，并在 forecast 主链路内做最小必要改动。

## 仓库使命

- 用统一 forecast pipeline 支撑时间序列预测实验。
- 在共享流程下保证不同模型的对比尽可能公平、可复现、可追踪。
- 将改动重点放在模型本身，而不是反复搭训练脚手架。

## 当前边界

- 只维护 forecasting 主链路。
- 不再维护历史 runner 变体、重建/可视化/辅助 loss 等旧实验分支。
- 不重新引入 `ray_tune.py`、Python `exp_conf` 配置或 `exp_runner` 分发链。
- 不为了“更工程化”再加插件系统、注册器体系或大规模测试矩阵。

## 架构地图

### CLI

- `train.py`：统一训练入口；传入 `--param_space` 时执行 Ray Tune
- `test.py`：统一测试入口；支持 `best`、`last` 或显式 checkpoint 路径

### 配置

- `config/tasks/forecast.yaml`：forecast 默认项
- `config/datasets/catalog.yaml`：数据集元信息
- `config/experiments/<ModelName>/*.yaml`：实验配置
- `config/search_spaces/<ModelName>/*.py`：Ray Tune 搜索空间

配置融合优先级固定为：

`experiment > dataset > task`

不要破坏这条规则，也不要把实验差异散回脚本常量。

### 数据

- `easytsf/data/data_module.py` 中的 `DataInterface` 是统一数据入口。
- 数据格式默认为 `dataset/<dataset_name>.npz`。
- 当前主链路只依赖 `scaled_variable` 和 `timestamp`。
- train/val/test 的滑窗切分逻辑集中在 `DataInterface`，不要为单个模型复制一份。

### 任务层

- `easytsf/task/forecast.py` 中的 `ForecastTask` 是唯一公开任务层。
- `pipeline` 指的是端到端训练链路，不是 LightningModule 名称。

### 模型层

- `easytsf/model/<ModelName>.py` 中应定义与 `model_name` 同名的类。
- 默认模型接口是 `forward(var_x, marker_x)`。
- 默认返回张量应与标签形状兼容，通常是 `[B, pred_len, N]`。

### 公共层

- 当前仍在使用的公共层主要是 `easytsf/layer/transformer.py`。
- 如果逻辑只属于某个模型，不要急于抽到公共层。

## 修改准则

- 新增标准模型时，优先只改 `easytsf/model/` 和对应 YAML config。
- 训练逻辑变化应尽量收敛在 `ForecastTask`，不要把 loss、调度细节散进模型。
- 不重新引入按 task 类型分发的多任务层结构，除非仓库目标发生变化。
- 文档必须以当前仓库真实状态为准，不要声明不存在的入口或能力。

## 验证准则

默认只要求轻量验证，不要求完整测试矩阵。优先做：

1. `conda activate easytsf`
2. `python train.py -h`
3. `python test.py -h`
4. YAML 实验配置可加载
5. 至少一个搜索空间模块可加载
6. 若本地有数据，再考虑最小 smoke run

对研究型改动，验证目标是“主链路未被破坏”，不是建立完整 CI。
