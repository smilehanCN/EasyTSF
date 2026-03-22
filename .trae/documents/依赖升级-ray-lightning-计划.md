# 依赖升级计划：Ray Tune & Lightning

## 目标

- 将项目依赖升级到与官方稳定文档一致的 API 版本：
  - Ray Tune：以 https://docs.ray.io/en/latest/tune/api/api.html 对应的最新版为准
  - Lightning：以 https://lightning.ai/docs/pytorch/stable/ 对应的稳定版为准（`lightning.pytorch`）
- 处理升级导致的 API 变更，保证以下能力可用：
  - `train.py` 正常训练并在训练后执行 `test(ckpt_path="best")`
  - `ray_tune.py` 可跑 Tune（至少能启动 Tuner 并汇报指标）
  - `test.py` 可通过指定 ckpt 跑 `trainer.test`
- 补齐可安装的依赖清单（当前仓库缺少 `requirements.txt/pyproject.toml`），并把 README 的环境/安装说明改为与实际一致。

## 现状摘要（来自当前仓库代码）

- Lightning 已使用 `lightning.pytorch` 导入路径（属于新包名体系）。
- Ray Tune 通过 `ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback` 集成 Lightning。
- `Trainer(devices=conf["devices"])` 当前从 CLI 读到的是字符串（如 `"0,"`）。Lightning 稳定版仍支持 `int | list[int] | str`（含 `"auto"` 与 GPU 字符串），但建议统一解析以避免歧义和跨平台差异。
- 仓库根目录没有依赖文件（`requirements.txt/pyproject.toml`），README 仍引用了不存在的 `requirements.txt` 和旧的 config 路径，需要同步修正。

## 升级策略

- 优先最小改动，保持现有训练逻辑和配置系统不变。
- 以“先让 CPU 跑通一个最小训练/测试 smoke test”为验收标准，再考虑 GPU、W&B、Ray 多卡等增强路径。
- 对高风险 API 做显式适配层（参数解析、配置对象选择、旧参数 deprecate 迁移），减少未来再升级成本。

## 最新 API 变更检查结论（本轮）

- Lightning（stable）：
  - `Trainer` 推荐显式传入 `accelerator` 与 `devices`，默认行为更偏向 `"auto"`。
  - `devices` 在文档中仍支持整数、列表、字符串（如 `"0,1"`、`"-1"`、`"auto"`）。
- Ray Tune（latest）：
  - `ray.tune.integration.pytorch_lightning.TuneReportCheckpointCallback` 在最新文档中仍可用。
  - `Tuner` 的 `run_config` 推荐使用 `ray.tune.RunConfig`。
  - `ray.train.RunConfig` 中 `progress_reporter/verbose/stop/sync_config` 已标注 deprecated；因此 Tune 场景避免继续绑定到 `air.RunConfig` 的旧字段习惯。

## 具体改动清单（实现阶段要做的事）

### 1) 增加依赖文件

- 新增 `requirements.txt`（或 `pyproject.toml`，二选一；默认选 `requirements.txt` 更贴合当前脚本工程形态）
  - `lightning`（稳定版）
  - `ray[tune]`（稳定版）
  - `torch`（不在此仓库强 pin CUDA 版本，但给出建议安装方式）
  - `numpy`、`pandas`
  - 可选：`wandb`、`transformers`
- 在 README 中明确“torch 的安装随 CUDA/CPU 环境而不同”，避免错误 pin。

### 2) 处理 Lightning API 兼容点

- `Trainer(devices=...)` 适配：
  - 保留旧输入兼容（如 `"0,"`），但在内部统一标准化为 Lightning 认可的 `int | list[int] | str`。
  - 新增 `--accelerator`（默认 `auto`），使 CPU/GPU 选择行为可控且与官方稳定文档一致。
  - 约定推荐输入：`--accelerator cpu --devices 1`（CPU smoke test）、`--accelerator gpu --devices 1`（单卡）。
- 检查 `precision`、callbacks、logger 的导入路径是否仍有效（保持 `lightning.pytorch.*`）。

### 3) 处理 Ray Tune API 兼容点

- 确认 Lightning 集成回调在新 Ray 版本的导入路径：
  - 优先使用当前路径 `ray.tune.integration.pytorch_lightning`
  - 保留最小 try-import 兜底，仅在确有迁移时启用
- 将 `run_config` 从 `air.RunConfig` 切到 `tune.RunConfig`（Tune API 一致性）：
  - 保留 `name`、`storage_path`
  - `progress_reporter` 使用 Tune 体系对象（`CLIReporter`）并通过 `tune.RunConfig` 传入
  - 明确避免依赖 `ray.train.RunConfig` 中已 deprecated 的字段语义
- `ray.init` 策略：
  - 默认改为 `ray.init()`（或仅在未初始化时调用）
  - GPU 资源通过 `tune.with_resources(..., {"gpu": gpus_per_trial})` 控制，避免双重资源来源冲突

### 4) 统一 train/test/ray_tune 的配置加载与 CLI

- 目前 `train.py` 和 `test.py` 都有 `load_config` 和 `train_func`，并且 `test.py` 内部硬编码 `ckpt_path`：
  - 将 `test.py` 改为支持 `--ckpt_path` 参数（默认 `"best"` 或空表示 best），避免用户手改源码
  - 复用 `train.py` 的 `load_config`，减少重复与分叉
- 在 `train.py/test.py/ray_tune.py` 对齐 `--accelerator` 与 `--devices` 的语义与默认值。

### 5) README 同步升级后的用法

- 更新安装方式：
  - `pip install lightning`（Lightning 官方稳定文档建议）
  - `pip install "ray[tune]"`（Ray Tune）
- 更新命令示例：
  - 训练示例使用仓库实际存在的 `config/iTransformer/ETTh1_96for96.py`
  - 测试示例展示 `--ckpt_path best` 或指定 ckpt 文件路径
  - Ray Tune 示例展示 `-p` 的 `param_space.py` 结构

### 6) 验证（必须可重复）

- 增加一个最小可跑的 smoke test 路径（不依赖外部数据集）：
  - 生成一个小的 `dataset/Dummy.npz`（包含 `scaled_variable/timestamp/mean/std`）
  - 增加对应的 `Dummy_conf`（在 `config/base_conf/datasets.py`）
  - 增加一个最小实验配置 `config/<Model>/Dummy_*.py`
  - CPU 下运行：
    - `python train.py -c ... --accelerator cpu --devices 1`
    - `python test.py -c ... --ckpt_path best`
  - 若启用 Ray：`python ray_tune.py ... --num_samples 1 --num_gpus 0 --gpus_per_trial 0`
  - 验证输出：
    - 训练目录存在 checkpoint，且 `trainer.test(..., ckpt_path="best")` 成功结束
    - Tune 产出 ResultGrid 与 `report.csv`，关键指标列不为空

## 交付物

- 新增依赖清单文件（`requirements.txt` 或 `pyproject.toml`）
- 更新 `train.py/test.py/ray_tune.py` 的兼容性改动
- 更新 README（安装/运行/参数说明与示例）
- 最小数据与配置（可选但强烈建议），确保 CI/本地可一键验证

## 风险与回退

- 风险：Lightning/Ray 新版对参数类型更严格（尤其是 `devices`）、Ray 与 Lightning 集成回调路径可能迁移。
- 回退：依赖文件可用版本区间约束（而非单点 pin），并保留兼容导入与参数解析，确保旧用法在合理范围内仍能工作。

## 实施顺序（执行时严格按序）

1. 落地依赖文件并更新 README 安装指引。
2. 改造 `train.py/test.py` 的 `accelerator/devices` 参数与解析函数。
3. 改造 `test.py` 的 `--ckpt_path`，移除硬编码 checkpoint。
4. 改造 `ray_tune.py`：`tune.RunConfig`、`ray.init` 资源策略、回调导入兼容。
5. 跑 CPU smoke test（train + test），再跑 1 次最小 Tune。
6. 回填 README 的最终命令示例与兼容说明。
