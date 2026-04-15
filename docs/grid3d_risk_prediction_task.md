# Grid3D Risk Prediction Task

本文档固定 `grid3d_risk_prediction` 的唯一正式协议。当前任务面向 `WindFieldV1_0414` 三维风场数据，目标是为无人机飞行安全与规划提供逐体素风险分类结果。

## 任务目标

给定历史三维风场，模型预测未来一步每个网格点的风险类别。

- 输入：历史 `u/v/w` 三个风速分量
- 输出：逐网格、多头分类 logits
- 头数：4 个，分别为 `shear_x`、`shear_y`、`shear_z`、`speed_cls`
- 类别数：3 类，分别为 `low`、`medium`、`high`
- 历史长度：`hist_len = 5`
- 预测长度：`pred_len = 1`
- `high_risk_class_index = 2`

模型输出通道数为 `risk_num_heads * risk_num_classes = 12`。logits 形状为 `[B, T, 12, Y, X, Z]`，标签形状为 `[B, T, 4, Y, X, Z]`。

## 数据契约

`grid3d_risk_prediction` 只支持重新导入后的 `WindFieldV1_0414` 新数据集，不兼容旧缓存。

新的 importer 会把物理坐标与 spacing 固化进数据集：

- `meta.json` 必须包含 `grid_spacing_m`
- `meta.json` 必须包含 `axis_layout`、`coord_min`、`coord_max`
- `axes.npz` 保存原始 `x/y/z`
- `coord.npy` 仍然保存归一化坐标，供模型输入使用

风险标签中的 shear 依赖物理 spacing，因此训练与统计都以数据集 `meta.json` 为准，不再以 YAML 中手填 `grid_spacing_m` 为准。

## 风速标签

设某时刻某网格点的风矢量为：

```text
W(y, x, z) = [u(y, x, z), v(y, x, z), w(y, x, z)]
```

风速标量定义为：

```text
speed(y, x, z) = ||W(y, x, z)||_2
```

风速三分类阈值通过实验 YAML 的 `risk_bins.speed` 配置。当前正式配置为：

```yaml
speed: [4.0, 6.0]
```

对应类别：

| Class | 区间 | 含义 |
| --- | --- | --- |
| 0 | `speed < 4.0 m/s` | low |
| 1 | `4.0 <= speed < 6.0 m/s` | medium |
| 2 | `speed >= 6.0 m/s` | high |

## 风切变标签

本任务沿用 wind shear 的风矢量差定义，而不是标量风速差。

```text
shear_x(y, x, z) = ||W(y, x + 1, z) - W(y, x, z)||_2 / dx
shear_y(y, x, z) = ||W(y + 1, x, z) - W(y, x, z)||_2 / dy
shear_z(y, x, z) = ||W(y, x, z + 1) - W(y, x, z)||_2 / dz
```

其中 `grid_spacing_m` 按 `[dy, dx, dz]` 存在于数据集 `meta.json` 中。边界处复制相邻差分结果，保证标签形状与目标风场一致。

三轴 shear 三分类阈值通过实验 YAML 的 `risk_bins.shear` 统一配置，并在实现中自动展开到 `shear_x/shear_y/shear_z`。当前正式配置为：

```yaml
shear: [0.05, 0.08]
```

对应类别：

| Class | 区间 | 含义 |
| --- | --- | --- |
| 0 | `shear < 0.05 m/s per 1m` | low |
| 1 | `0.05 <= shear < 0.08 m/s per 1m` | medium |
| 2 | `shear >= 0.08 m/s per 1m` | high |

当前任务不再接受 `shear_x/shear_y/shear_z` 分轴阈值配置，统一使用 `risk_bins.shear`。

## 训练与评估协议

当前正式训练协议：

- `head_loss_weights: [1.0, 1.0, 1.0, 1.0]`
- 主验证指标：`val/macro_f1`
- 主验证模式：`val_metric_mode: max`
- 关键安全指标：`val/high_risk_recall`、`test/high_risk_recall`

评估输出至少包含：

- `val/loss`
- `val/macro_f1`
- `val/high_risk_recall`
- `val/{head}_macro_f1`
- `val/{head}_high_risk_recall`
- `test/macro_f1`
- `test/high_risk_recall`
- `test/{head}_macro_f1`
- `test/{head}_high_risk_recall`
- `test/class_{k}_precision`
- `test/class_{k}_recall`

## 类别统计与 `risk_class_weights`

`risk_class_weights` 仍然只作用于训练期的 weighted cross entropy，不改变标签定义，也不改变推理或评估指标。

新的 3 类协议需要在真实 `WindFieldV1_0414` 数据上重新统计。当前仓库不内置正式统计数字，因为数据集位于远程服务器。

在远程服务器上重跑统计：

```bash
python scripts/analyze_grid3d_risk_bins.py \
  --config config/experiments/unet3d/grid3d_risk_prediction_windfield_v1_0414.yaml \
  --split train \
  --y-chunk 64 \
  --output-json logs/grid3d_risk_analysis/grid3d_risk_prediction_windfield_v1_0414_train.json
```

该脚本会输出：

- 每个 head 的 3 类计数和比例
- `recommended_risk_class_weights`

当前权重生成规则为共享 3 类权重：

```text
aggregate_count[k] = sum_head count[head, k]
weight[k] = max(aggregate_count) / aggregate_count[k]
```

脚本会保留两位小数；如果任一聚合类别计数为 0，脚本直接失败，表示当前阈值定义不适合正式训练协议。

## 正式配置与启动脚本

正式配置文件：

- `config/experiments/unet3d/grid3d_risk_prediction_windfield_v1_0414.yaml`
- `config/experiments/unet3d_patchcat/grid3d_risk_prediction_windfield_v1_0414.yaml`
- `config/experiments/patchstg_flat3d/grid3d_risk_prediction_windfield_v1_0414.yaml`
- `config/experiments/fredn_multivariate3d/grid3d_risk_prediction_windfield_v1_0414.yaml`

统一启动脚本：

```bash
GPUS=0,1,2,3 FREE_THRESHOLD_MIB=500 bash scripts/run_grid3d_risk_prediction_windfield_v1_0414.sh
```

旧的 4 类和 quantile 风险协议已移出正式入口，归档到 `archive/grid3d_risk_legacy/`，不再作为当前任务的可用协议。

## 实现位置

核心实现位于：

```text
scripts/grid3d_import.py
scripts/analyze_grid3d_risk_bins.py
easytsf/data/grid3d_data_module.py
easytsf/task/grid3d_risk_prediction.py
```

实现要点：

- importer 从原始 `x/y/z` 推导并校验物理 spacing
- 风险任务在物理量空间构造 speed 和 vector shear 标签
- validation 和 test 都累计 confusion 并输出风险指标
- workflow 支持 `val_metric_mode=max`，可按 `val/macro_f1` 选 checkpoint
