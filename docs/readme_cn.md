# EasyTSF 中文导读

这个文档不是英文 README 的逐字翻译，而是面向组内同学和中文读者的快速说明。根目录 [README.md](../README.md) 负责对外开源入口，这里用中文介绍仓库的 workflow 设计、skill 设计，以及如何迁移模型。

## 概览

EasyTSF（**E**xperiment-friendly **A**ssistant for **Y**our **T**ime-**S**eries **F**orecasting）强调两个方向：对人类研究者，提供可复现的 workflow；对 AI agent，提供可复用的 skills。

EasyTSF 是一个基于 Lightning 的轻量级时序预测算法库，强调可复现的 workflow 和对 agent 友好的 skills：

- Workflow design：面向研究人员，提供从模型代码到可复现实验和 benchmark 的低心智负担路径。
- Skill design：面向 AI agent，提供清晰的仓库约定，便于理解、迁移和使用项目。

中文读者也建议先配合阅读英文版 [README.md](../README.md)。

## Workflow Design

Workflow design 是 EasyTSF 面向人的协作契约，围绕时序预测研究和应用中的三个关键 workflow 展开：

### Experiment, Benchmark and Report

`experiment` 是基础单元。下面的 Quick Start 会先运行一次 `experiment`；`benchmark` 在这个基础上批量运行实验用于调参，`report` 再把这些结果整理成可读汇总。

- `experiment`：运行单次实验，用于调试模型代码和训练行为，也是 `benchmark` 的基础。
- `benchmark`：根据 benchmark 配置批量运行实验，是超参数搜索和优化的主要路径。
- `report`：对批量实验结果做汇总，便于比较和分析。

这些 workflow 构建在 Lightning 之上，同时把静态 preset 和运行时 override 分开，让研究人员可以更顺畅地从单次调试走到批量评估。

### Quick Start

需要 Python `>=3.11`。

安装到当前 Python 环境：

```bash
python -m pip install -e .
```

运行一个单次 experiment：

```bash
python -m easytsf.workflow.experiment config/experiments/tqnet/etth1.yaml
```

运行 benchmark：

```bash
python -m easytsf.workflow.benchmark config/benchmarks/tqnet/core.py
```

生成 benchmark 结果汇总：

```bash
python -m easytsf.workflow.report config/benchmarks/tqnet/core.py
```

## Skill Design

Skill design 是 EasyTSF 面向 AI 的协作契约。Skill 把仓库约定整理成可复用接口，帮助 AI agent 更稳定地理解工作流并使用项目。当前仓库内置了 `migrate-model-to-easytsf`，后续还可以继续扩展更多 Skill。

### Install the Skill

要在本地使用仓库里的 Skill，可以手动放到 Codex 的 skills 目录。Codex 会优先从 `${CODEX_HOME}/skills` 读取；如果没有设置 `CODEX_HOME`，通常就是 `~/.codex/skills`。

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/migrate-model-to-easytsf "${CODEX_HOME:-$HOME/.codex}/skills/"
```

如果你希望 Skill 与仓库内改动保持同步，可以改成软链：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$(pwd)/skills/migrate-model-to-easytsf" "${CODEX_HOME:-$HOME/.codex}/skills/migrate-model-to-easytsf"
```

### Migrate a Model into EasyTSF

把其他项目的模型迁进 EasyTSF 时，尽量保持改动面小而明确：

1. 模型文件放在 `easytsf/model/<model_id>.py`，并暴露顶层 `Model` 类。
2. `Model.__init__` 必须写显式参数名，因为 `MTSFTask` 会根据构造签名从 flat config 自动传参。
3. `forward` 适配成 `forward(var_x, marker_x, marker_y)`。
4. 输出要和标签兼容，通常是 `[B, pred_len, N]`。
5. 在 `easytsf/model/registry.py` 中注册模型。
6. 至少补一个 `config/experiments/<model_id>/` 下的可运行 preset。
7. 只有当模型已经适合做搜索时，再补 benchmark 配置。

如果外部模型依赖 graph side input、decoder cache，或者当前 EasyTSF workflow 没有提供的自定义数据契约，就应该先停下来重新设计，而不是悄悄扩张仓库边界。

当你已经有外部模型代码，并希望把它映射到 EasyTSF 时，可以直接使用这个 Skill：

```text
Use $migrate-model-to-easytsf to inspect this external forecasting model implementation and tell me whether it fits the current EasyTSF contract. If it fits, generate an EasyTSF-ready model plan plus one experiment preset draft. If it does not fit, stop with an incompatibility report.
```

如果你只是想读论文或做模型比较，直接使用普通 prompt 即可：

```text
Summarize this forecasting paper, compare it with the models already registered in EasyTSF, and tell me whether it looks compatible with the current EasyTSF contract before we touch any code.
```

更严格的仓库级迁移规则见 [AGENT.md](../AGENT.md)。

Skill 的事实来源在 [`skills/migrate-model-to-easytsf/`](../skills/migrate-model-to-easytsf/)。
