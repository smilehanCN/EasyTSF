# EasyTSF 中文导读

这个文档不是英文 README 的逐字翻译，而是面向组内同学和中文读者的快速上手说明。根目录 [README.md](../README.md) 负责对外开源入口，这里重点讲仓库怎么跑、怎么迁移模型、怎么配合 Codex 使用。

## 仓库定位

EasyTSF 当前维护的是一条很窄但稳定的主链路：

- 任务只维护 `mtsf`
- 数据只走 sequence-only 路径
- 实验入口是 `experiment preset < runtime overrides`
- benchmark 用 Ray Tune 编排
- 模型迁移优先服务“把别的项目模型接到当前仓库”，不是扩成通用时序平台

如果外部模型依赖 graph side input、额外 decoder state、特殊 datamodule 或其他当前 `mtsf` 不提供的输入，默认先评估并停下，不偷偷扩仓库边界。

## 最短运行链路

安装：

```bash
python -m pip install -e .
```

轻量校验：

```bash
python -m compileall easytsf
```

跑一个现成 experiment：

```bash
python -m easytsf.workflow.experiment config/experiments/tqnet/etth1.yaml \
  --set data_root=dataset \
  --set save_root=checkpoint \
  --set accelerator=auto \
  --set devices=auto
```

跑 benchmark：

```bash
python -m easytsf.workflow.benchmark config/benchmarks/tqnet/core.py
```

导出 benchmark 汇总：

```bash
python -m easytsf.workflow.report config/benchmarks/tqnet/core.py \
  --out save/benchmarks/tqnet_electricity/report.csv
```

## 模型迁移最小改动面

把其他项目的模型迁进 EasyTSF 时，先盯住这几个硬约束：

1. 模型文件要落在 `easytsf/model/<model_id>.py`，并暴露顶层 `Model` 类。
2. `Model.__init__` 必须写显式参数名，因为 `MTSFTask` 会根据构造签名从 flat config 自动传参。
3. `forward` 必须适配成 `forward(var_x, marker_x, marker_y)`。
4. 输出默认要和标签兼容，通常是 `[B, pred_len, N]`。
5. 新模型至少要补一个 `config/experiments/<model_id>/xxx.yaml`。

额外注意：

- `var_x` 是缩放后的输入变量
- `marker_x` / `marker_y` 是时间特征，如果模型不用，显式 `del` 掉
- `meta.json` 当前要求有 `frequency (minutes)` 和 `timestamps_description`
- 外部项目里的嵌套配置，迁进来后要摊平成 EasyTSF 的 flat YAML 键

## 用 Codex 迁移外部模型

先把仓库里的 skill 手动放到本地 Codex skills 目录。Codex 默认会从 `${CODEX_HOME}/skills` 读取；如果没设 `CODEX_HOME`，通常就是 `~/.codex/skills`。

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/migrate-model-to-easytsf "${CODEX_HOME:-$HOME/.codex}/skills/"
```

如果你希望 skill 跟仓库内改动保持同步，可以改成软链：

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$(pwd)/skills/migrate-model-to-easytsf" "${CODEX_HOME:-$HOME/.codex}/skills/migrate-model-to-easytsf"
```

这个 skill 的职责很窄：用户给出外部项目的模型代码、类定义、`forward` 逻辑或配置片段后，帮助判断能否迁入当前 EasyTSF `mtsf` 路径；如果能，就生成 EasyTSF 里的 model 和一个 experiment preset 草案所需的映射。

如果你只是想复习论文，不需要这个 skill，直接正常提问即可。

## 可复制 Prompt

仓库导读：

```text
先阅读这个仓库，告诉我 EasyTSF 当前真正维护的主链路是什么，哪些模型只是注册了代码但还没有完整 preset 或 benchmark，并给我一个最短上手路径。
```

基于外部项目代码做迁移评估：

```text
Use $migrate-model-to-easytsf to inspect this external forecasting model implementation. Check whether it fits the current EasyTSF mtsf path. If it does not fit, stop with an incompatibility report that names the unsupported inputs or abstractions.
```

基于外部项目代码生成 EasyTSF 版 model + experiment 草案：

```text
Use $migrate-model-to-easytsf to read this external model code and produce the EasyTSF migration result: the target Model.__init__ parameter list, the forward(var_x, marker_x, marker_y) adaptation, the required flat config keys, and one example experiment preset draft.
```

## 论文阅读和复习

如果只是读论文、做复习或做模型对比，直接用普通 prompt：

```text
总结这篇时序预测论文的核心结构，并和 EasyTSF 里已经注册的模型做对比，告诉我它在当前 mtsf 契约下是否容易迁移。
```

更严格的仓库级协作约束见 [AGENT.md](../AGENT.md)。
