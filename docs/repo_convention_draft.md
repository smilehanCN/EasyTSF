# EasyTSFNext 规范草案

本草案服务 EasyTSF 的 prediction task 设计。目标是让仓库保持清晰、可复现、低心智负担，并把每类任务需要的数据、task、model、workflow 契约显式写出来。

## 1. 设计原则

### 1.1 以任务契约为中心

仓库对外要明确四层公开契约：

- `data contract`
- `task contract`
- `model interface`
- `workflow surface`

当前文档使用的任务 taxonomy 是：

- `sequence_prediction`
- `graph_prediction`
- `grid_prediction`

### 1.2 显式优先于魔法

- 配置来源、合并优先级和结果目录必须清晰可追踪
- 模型构造依赖显式参数名，不依赖数据层回填
- 如果一段兼容逻辑只是在维持旧设计，应优先删除而不是保留

### 1.3 研究流程优先

规范服务下列主链路：

1. 接入模型
2. 跑通单实验
3. 固化为 experiment preset
4. 进入 benchmark 做超参数搜索与 trial 汇总

## 2. 目录与职责

- `easytsf/model/`：模型本体与模型私有 helper
- `easytsf/task/`：任务层与训练逻辑
- `easytsf/data/`：数据读取、时间特征恢复和 dataloader
- `easytsf/workflow/`：experiment / benchmark 编排
- `config/experiments/`：单实验预设
- `config/benchmarks/`：批量评测声明

不要把 workflow 逻辑回灌到 model / task / data，也不要让 data 层承担“从数据反推配置真相”的职责。

## 3. 当前实现状态

当前可运行代码仍集中在一个 sequence-oriented 的 `mtsf` 路径上：

- `easytsf/data/mts_data_module.py`
- `easytsf/task/mtsf.py`
- `easytsf/workflow/experiment.py`
- `easytsf/workflow/benchmark.py`
- `easytsf/workflow/report.py`

这条路径是当前 `sequence_prediction` 的一个具体实现，不应被表述成所有 prediction task 的唯一真理。

## 4. 模型与任务接入规范

新增一个 maintained prediction path 时，至少同时提交：

1. 任务的数据契约
2. 任务的 task contract
3. 模型接口约束
4. 至少一个 example experiment surface
5. 如需搜索，再补 benchmark surface
6. 文档条目，说明实现来源与使用方式

模型类契约必须与任务契约一致，并保持显式：

- 模型模块统一暴露顶层 `Model`
- 构造参数来自扁平配置键
- `forward(...)` 的参数必须显式表达该任务真正需要的输入
- 返回张量必须与该任务的标签契约兼容

如果 helper 只被当前模型使用，不要提取到共享层。

## 5. 配置规范

experiment preset 使用 flat YAML，并通过注释块区分：

- `# model`
- `# data`
- `# train`
- `# runtime`

配置合并顺序固定为：

`experiment preset < runtime overrides`

experiment preset 应自包含完整 recipe，并显式声明任务归属。当前 runnable sequence 路径仍使用 `task: mtsf`；未来任务不应被强行塞进这个键值所代表的旧叙事。

## 6. 数据规范

当前 sequence 路径的数据集布局为：

- `train_data.npy`
- `val_data.npy`
- `test_data.npy`
- `train_timestamps.npy`
- `val_timestamps.npy`
- `test_timestamps.npy`
- `meta.json`
- optional: `stats.npz`

其中：

- `*_data.npy` 必须是 `[L, N]`
- 单变量预测也写成 `[L, 1]`
- 三个 split 都必须提供时间戳文件
- `meta.json` 需要提供频率和 `timestamps_description`
- `meta.json` 可选提供 `data_is_standardized`
  - 为 `true` 时，`*_data.npy` 表示已经标准化后的存储，此时必须提供 `stats.npz`
  - 缺失或为 `false` 时，运行时始终把 `*_data.npy` 当作 raw data，并从 `train_data.npy` 拟合 scaler
- `stats.npz` 不会单独改变缩放策略；只有显式 `data_is_standardized: true` 才会启用

对 graph 和 grid prediction，不要默认沿用 sequence 数据契约。它们需要额外的 topology、mask、坐标或 side input 时，必须作为任务契约的一部分显式写出，而不是放进隐含兼容逻辑。
