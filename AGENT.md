# AGENT.md

本文件面向参与本仓库工作的 AI 代码代理。目标是让代理在尽量少打扰研究流程的前提下，理解仓库边界、保持实验复现性，并用最小必要改动完成任务。

## 仓库使命

- 用统一的训练、验证、测试 pipeline 支撑时间序列预测实验。
- 在共享流程下保证不同模型的对比尽可能公平、可复现、可追踪。
- 将模型开发成本压缩到网络结构本身，避免每次重复搭训练脚手架。
- 保持研究仓库的简洁性，不把它扩展成通用平台或重工程框架。

## 工作边界

- 优先保持代码简洁、可读、可快速迭代。
- 不主动引入复杂数据校验、数据注册系统、插件系统或大规模测试矩阵。
- 不为了“看起来更工程化”而增加大量基础设施。
- 文档、日志、类型、错误提示可以提升，但前提是不破坏当前实验主链路。

## 架构地图

### CLI 入口

- `train.py`：统一入口；默认训练，提供 `--param_space` 时切换为 Ray Tune 搜索
- `test.py`：加载配置后执行测试，支持 `best` 或显式 `ckpt_path`
- `ray_tune.py`：兼容旧命令的轻量包装层，内部转发到 `train.py`

### 配置层

- `config/base_conf/task.py`：公共训练参数
- `config/base_conf/datasets.py`：数据集元信息
- `config/<ModelName>/*.py`：单个实验的 `exp_conf`

配置融合规则固定为：

`exp_conf > task_conf + dataset_conf`

不要破坏这条规则，也不要把实验差异散落回脚本常量中。

### 数据层

- `easytsf/runner/data_runner.py` 中的 `DataInterface` 是统一数据入口。
- 数据格式默认为 `dataset/<dataset_name>.npz`。
- 当前公共数据 key 是 `scaled_variable`、`timestamp`、`mean`、`std`。
- train/val/test 的滑窗切分逻辑集中在 `DataInterface`，不要为单个模型偷偷复制一份。

### 训练层

- `easytsf/runner/exp_base_runner.py` 是默认 Runner。
- `easytsf/runner/` 下的其他 Runner 负责特化场景，如辅助损失、单变量、重建或特定实验逻辑。
- `train.py` 通过 `exp_runner` 分发到具体 Runner。

### 模型层

- `easytsf/model/<ModelName>.py` 中应定义与 `model_name` 同名的类。
- 默认 Runner 约定模型接口为 `forward(var_x, marker_x)`。
- 默认返回值应与标签形状兼容，通常是 `[B, pred_len, N]`。
- 如果模型需要额外返回辅助项，必须与专用 Runner 和配置一起修改，不能只改模型。

### 公共层

- `easytsf/layer/` 放可复用层和 Transformer 组件。
- 若只是某个模型私有逻辑，不要急于抽到公共层。

## 修改准则

- 新增标准模型时，优先只改 `easytsf/model/` 和对应 `config/`。
- 训练逻辑变化应尽量收敛在 Runner 层，不要把 loss、日志、调度细节散进模型。
- 实验特例不要直接写进公共数据流程；如果必须支持，先判断是否值得抽成新的 Runner。
- 配置命名、日志路径、checkpoint 路径组织尽量保持现有约定，避免影响历史结果管理。
- 修改 README 或 AGENT 时，以当前仓库真实状态为准，不要替仓库声明不存在的能力。

## 复现准则

- 保留 `seed`、`conf_hash`、日志目录和 checkpoint 目录的基本组织方式。
- 保留 `train.py -> fit() -> test(best)` 的默认训练闭环，除非任务明确要求改变。
- 保留配置驱动的模型加载方式，不要把模型选择写死在代码里。
- 对比实验共享公共 pipeline 时，不要给单个模型附加隐藏特权逻辑。

## 验证准则

- 默认只要求轻量验证，不要求补完整单元测试。
优先做以下检查：
1. `python train.py -h`
2. `python test.py -h`
3. `python ray_tune.py -h`
4. 关键配置可加载，路径和命令与文档一致
5. 若本地有可用数据，再考虑最小化 smoke run

对研究型改动，验证目标是“主链路未被破坏”，不是建立完整测试矩阵。

## 重构优先级

后续重构按以下顺序优先推进：

1. 统一环境到 Python 3.11、Lightning 2.6.1、PyTorch 2.9.1
2. 去重并整理 Runner 逻辑，压缩重复实现
3. 提升类型标注、注释质量、日志和错误提示
4. 清理失效配置、历史遗留文档和实现缺口
5. 增加轻量 smoke 验证和必要的静态检查

## 非目标

- 不把仓库改造成通用 AutoML 平台
- 不引入复杂插件系统或多层注册器体系
- 不为文档阶段承诺完整 CI、发布流程或产品级质量保证
- 不为“形式完整”补大量与研究主线无关的基础设施
