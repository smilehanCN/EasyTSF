# EasyTSF 中文导读

这个文档不是英文 README 的逐字翻译，而是面向组内同学和中文读者的快速说明。根目录 [README.md](../README.md) 负责对外开源入口，这里用中文介绍仓库的 workflow 设计、prediction task 设计，以及 AI skill 设计。

## 概览

EasyTSF（**E**xperiment-friendly **A**ssistant for **Y**our **T**ime-**S**eries **F**orecasting）强调两个方向：对人类研究者，提供可复现的 workflow；对 AI agent，提供可复用的 skills。

EasyTSF 是一个基于 Lightning 的轻量级 prediction task 算法库，强调可复现的 workflow 和对 agent 友好的 skills：

- Workflow design：面向研究人员，提供从模型代码到可复现实验和 benchmark 的低心智负担路径。
- Skill design：面向 AI agent，提供清晰的数据契约、任务契约、模型接口和 workflow surface。

中文读者也建议先配合阅读英文版 [README.md](../README.md)。

## Workflow Design

Workflow design 是 EasyTSF 面向人的协作契约，围绕三个公开 workflow surface 展开：

### Experiment, Benchmark and Report

`experiment` 是基础单元。下面的 Quick Start 会先运行一次 `experiment`；`benchmark` 在这个基础上批量运行实验用于调参，`report` 再把这些结果整理成可读汇总。

- `experiment`：运行单次实验，用于调试模型代码和训练行为，也是 `benchmark` 的基础。
- `benchmark`：根据 benchmark 配置批量运行实验，是超参数搜索和优化的主要路径。
- `report`：对批量实验结果做汇总，便于比较和分析。

这些 workflow 构建在 Lightning 之上，同时把静态 preset 和运行时 override 分开，让研究人员可以更顺畅地从单次调试走到批量评估。

## Prediction Tasks

EasyTSF 在文档和 Skill 中，把 prediction work 拆成四层公开契约：

1. `data contract`
2. `task contract`
3. `model interface`
4. `workflow surface`

当前采用的任务 taxonomy 是：

- `sequence_prediction`
- `graph_prediction`
- `grid_prediction`

当前可运行代码仍然主要落在一个 sequence-oriented 的 `mtsf` 路径上，同时也包含一个维护中的 `grid3d_forecasting` 路径，用于将 WindField4Cast 风格原始单步 `.nc` 数据导入为逐时间步 `.npy` cache 后进行 3D grid forecasting。对 graph prediction，请继续把它视为显式扩展目标，而不是隐含兼容的边角情况。

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

运行已接入的 MixLinear ETTh1 preset：

```bash
python -m easytsf.workflow.experiment config/experiments/mixlinear/etth1.yaml
```

运行已接入的 TimeBase ETTh1 preset：

```bash
python -m easytsf.workflow.experiment config/experiments/timebase/etth1.yaml
```

将 WindField4Cast 风格原始目录导入为 Grid3D 逐时间步 cache：

```bash
python scripts/grid3d_import.py --input-dir /path/to/raw_nc_dir --out-dir dataset/windfield4cast_demo
```

运行维护中的 UNet3D Grid3D preset：

```bash
python -m easytsf.workflow.experiment config/experiments/unet3d/windfield4cast_demo.yaml
```

运行 benchmark：

```bash
python -m easytsf.workflow.benchmark config/benchmarks/mixlinear/etth1.py
```

运行 TimeBase 的 benchmark：

```bash
python -m easytsf.workflow.benchmark config/benchmarks/timebase/etth1.py
```

生成 benchmark 结果汇总：

```bash
python -m easytsf.workflow.report config/benchmarks/mixlinear/etth1.py
```

## Skill Design

Skill design 是 EasyTSF 面向 AI 的协作契约。Skill 把仓库约定整理成可复用接口，帮助 AI agent 先识别 prediction task，再决定当前仓库能否直接承接，或是否需要明确扩展。当前仓库维护三个核心 Skill：

- `prepare-data-for-easytsf`：检查本地数据工件，识别 prediction task，并映射到当前仓库或扩展数据契约
- `adapt-model-to-easytsf`：分析外部模型代码，识别目标任务，并输出模型/任务/仓库的适配方案
- `run-workflow-with-easytsf`：围绕 experiment、benchmark、report 给出 task-aware workflow surface

### Install the Skill

要在本地使用仓库里的 Skill，可以手动放到 Codex 的 skills 目录。Codex 会优先从 `${CODEX_HOME}/skills` 读取；如果没有设置 `CODEX_HOME`，通常就是 `~/.codex/skills`。

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/* "${CODEX_HOME:-$HOME/.codex}/skills/"
```

如果你希望 Skill 与仓库内改动保持同步，可以改成逐个软链：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
for skill_dir in skills/*; do
  ln -s "$(pwd)/${skill_dir}" "${CODEX_HOME:-$HOME/.codex}/$(basename "${skill_dir}")"
done
```

### Prepare Data for EasyTSF

当用户已经有本地数据工件，希望先确认它属于哪类 prediction task，再决定怎么写配置或是否需要扩仓库时，可以直接使用这个 Skill：

```text
Use $prepare-data-for-easytsf to inspect this dataset directory, classify it as sequence, graph, or grid prediction data, and tell me whether the current EasyTSF repo can use it directly or needs a contract extension.
```

### Adapt a Model to EasyTSF

当用户已经有外部模型代码，想先判断它应该适配到哪类 prediction task，以及当前仓库是否能直接承接时，可以使用这个 Skill：

```text
Use $adapt-model-to-easytsf to inspect this model implementation, classify it as sequence, graph, or grid prediction, and tell me whether EasyTSF can adapt it directly or needs new task/data/workflow layers.
```

### Run an EasyTSF Workflow

当用户想获得 experiment、benchmark、report 的精确 workflow surface，或者希望 agent 说明当前仓库还缺哪些 workflow 层时，可以使用这个 Skill：

```text
Use $run-workflow-with-easytsf to classify this prediction task, tell me whether the current EasyTSF repo can run it directly, and give me the exact experiment or benchmark workflow surface.
```

## Current Implementation Note

当前可运行代码仍然集中在 sequence-oriented 的 `mtsf` 路径上，同时新增了一个可运行的 `grid3d_forecasting` 路径用于 3D 风场预测。`mtsf` 路径包括：

- `easytsf/data/mts_data_module.py`
- `easytsf/task/mtsf.py`
- `easytsf/workflow/experiment.py`
- `easytsf/workflow/benchmark.py`
- `easytsf/workflow/report.py`

Skill 和文档会直接使用 task-aware prediction 语言来描述未来扩展面；如果请求的是 graph 或 grid prediction，预期输出应是显式扩展方案，而不是强行把任务压回 sequence 假设。

当前维护的 sequence model surface 也包含可运行的 `MixLinear` 和 `TimeBase` preset：`config/experiments/mixlinear/etth1.yaml`、`config/experiments/timebase/etth1.yaml`，以及对应的 benchmark 示例：`config/benchmarks/mixlinear/etth1.py`、`config/benchmarks/timebase/etth1.py`。

更严格的仓库级协作规则见 [AGENT.md](../AGENT.md)。

Skill 的事实来源都在 [`skills/`](../skills/) 目录下。
