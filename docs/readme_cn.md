# EasyTSF 中文说明

EasyTSF 是一个面向时序预测科研落地的轻量级工具库。项目目标不是做大而全的框架，而是把数据、任务、模型接口和实验工作流说清楚，让新的预测 idea 可以尽快变成可复现实验。

## 当前支持面

项目用四层契约描述一个预测任务：

1. 数据契约
2. 任务契约
3. 模型接口
4. 工作流入口

当前核心包只支持两个可运行任务：

| 任务 | 类型 | 数据模块 | 模型接口 | 状态 |
| --- | --- | --- | --- | --- |
| `mtsf` | `sequence_prediction` | `MTSDataModule` | `forward(var_x, marker_x, marker_y)` | 维护中 |
| `grid3d_forecasting` | `grid_prediction` | `Grid3DDataModule` | `forward(x, coords=None)` | 维护中 |

Graph prediction 和 Grid3D shear input/output ablation 都是扩展目标，不是当前核心包暴露的可运行任务。

## 安装

需要 Python `>=3.11`。

```bash
python -m pip install -e .
```

依赖声明在 `pyproject.toml` 中。本地数据、checkpoint、日志和 benchmark 输出不属于源码包。

## 常用工作流

运行单次实验：

```bash
python -m easytsf.workflow.experiment config/experiments/tqnet/etth1.yaml
```

运行 Grid3D demo：

```bash
python -m easytsf.workflow.experiment config/experiments/unet3d/windfield4cast_demo.yaml
```

运行 benchmark：

```bash
python -m easytsf.workflow.benchmark config/benchmarks/mixlinear/etth1.py
```

汇总 benchmark 报告：

```bash
python -m easytsf.workflow.report config/benchmarks/mixlinear/etth1.py --out reports/mixlinear_etth1.csv
```

配置优先级固定为：

```text
experiment preset < runtime overrides < benchmark param_space
```

## 数据契约

序列预测数据目录：

```text
<data_root>/<dataset>/
  train_data.npy
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  meta.json
  stats.npz       # 仅当 meta.data_is_standardized=true 时必需
```

Grid3D 数据目录：

```text
<data_root>/<dataset>/
  train_data.npy          # T,C,Y,X,Z
  val_data.npy
  test_data.npy
  train_timestamps.npy
  val_timestamps.npy
  test_timestamps.npy
  coord.npy               # 可选，3,Y,X,Z
  axes.npz                # 可选，物理坐标轴
  stats.npz               # 标准化存储时必需
  meta.json
```

将 WindField4Cast 风格原始数据导入为 Grid3D cache：

```bash
python scripts/grid3d_import.py --input-dir /path/to/raw_nc_dir --out-dir dataset/WindFieldDemo
```

## 目录定位

- `easytsf/data/`：数据读取、窗口构造、标准化工具
- `easytsf/task/`：当前只保留 `base.py`、`mtsf.py`、`grid3d_forecasting.py` 和注册入口
- `easytsf/model/`：模型适配器
- `easytsf/workflow/`：experiment、benchmark、report 入口
- `config/experiments/`：可运行实验 preset
- `config/benchmarks/`：Ray Tune benchmark 配置
- `scripts/`：稳定工具脚本，例如数据导入
- `recipes/`：历史科研 sweep、ablation、多 seed 启动等非核心入口
- `skills/`：给 agent 使用的项目契约

## 科研资产说明

WindShear 0416/0417 的 sweep、ablation、多 seed 和监控脚本属于历史科研 recipe，不作为 EasyTSF 核心 API。部分 recipe 可能引用已经移除或实验性的 task 名称，使用前需要先恢复对应 task 支持。

部分历史配置中保留了 `WindStear_V1_0417` 拼写，因为它对应已有本地实验数据目录。新文档和新命名应优先使用 `WindShear`。
