# EasyTSFNext 规范草案

本草案只服务 EasyTSF 当前的维护范围：多变量时序预测 `mtsf`。目标是让仓库保持清晰、可复现、低心智负担，而不是继续扩展成通用时空建模框架。

## 1. 设计原则

### 1.1 只维护少数公开抽象

仓库只维护下列公开概念：

- `MTSFTask`
- `experiment`
- `benchmark`
- task-aware `model.forward(...)`

不新增 grid task、图结构 task、插件系统、AutoModel 或更大的注册分发树。

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
- `easytsf/data/`：sequence 数据读取、时间戳恢复、滑窗和 dataloader
- `easytsf/workflow/`：experiment / benchmark 编排
- `config/experiments/`：单实验预设
- `config/benchmarks/`：批量评测声明

不要把 workflow 逻辑回灌到 model / task / data，也不要让 data 层承担“从数据反推配置真相”的职责。

## 3. 模型接入规范

新增一个标准 `mtsf` 模型时，至少同时提交：

1. `easytsf/model/<model_id>.py`
2. 至少一个 `config/experiments/<model_id>/<dataset_id>.yaml`
3. 如需搜索，提供 `config/search_spaces/<model_id>/<name>.py`
4. 文档条目，说明实现来源与使用方式

模型类契约：

- 模型模块统一暴露顶层 `Model`
- 构造参数来自扁平配置键
- `forward(var_x, marker_x, marker_y)`
- 返回张量与标签形状兼容，默认 `[B, pred_len, N]`

如果 helper 只被当前模型使用，不要提取到共享层。

## 4. 配置规范

experiment preset 使用 flat YAML，并通过注释块区分：

- `# model`
- `# data`
- `# train`
- `# runtime`

配置合并顺序固定为：

`experiment preset < runtime overrides`

experiment preset 必须自包含完整 recipe，并显式声明 `task_name: mtsf`。如果要支持新的任务，必须先明确仓库目标变化，再讨论新增公开 task。

## 5. 数据规范

`mtsf` 数据集布局固定为：

- `train_data.npy`
- `val_data.npy`
- `test_data.npy`
- `train_timestamps.npy`
- `val_timestamps.npy`
- `test_timestamps.npy`
- `meta.json`

其中：

- `*_data.npy` 必须是 `[L, N]`
- 单变量预测也写成 `[L, 1]`
- 三个 split 都必须提供时间戳文件
- `meta.json` 需要提供频率和 `timestamps_description`

不再支持 `data.npz`、grid mask、坐标 side input 或通用 TSF 数据抽象。
