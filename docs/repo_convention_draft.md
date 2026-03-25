# EasyTSFNext 规范草案

本草案参考 Hugging Face `transformers` 官方仓库的设计和贡献规范，但只吸收适合 EasyTSFNext 当前定位的部分：统一接口、清晰分层、模型实现自包含、文档和验证与新增模型同行，不引入大而全的框架化抽象。

EasyTSFNext 的目标不是成为通用深度学习平台，而是成为一个适合多元时序预测研究的清晰、可复现、低心智负担模型库。

## 1. 设计原则

### 1.1 统一少数公开抽象

仓库只维护下列公开概念：

- `MTSFTask` / `STFTask` / `Grid2DTSFTask` / `Grid3DTSFTask`：当前公开任务层
- `experiment`：单实验可运行配方
- `study`：批量 benchmark 编排
- task-aware `model.forward(...)`：按任务设定选择模型调用契约

除非仓库目标发生变化，否则不新增 `AutoModel`、注册器、插件系统、多任务分发树等新抽象。

### 1.2 显式优先于魔法

- 配置来源、合并优先级、结果落盘路径必须清晰可追踪。
- 模型构造依赖显式参数名，不依赖隐式全局状态。
- 研究代码允许局部重复，但不接受“为了复用而复用”的过度抽象。

### 1.3 模型实现尽量自包含

参考 `transformers` 的做法，新增模型时优先把模型私有组件放在模型文件内部，只有在至少两个模型真实复用且抽出后更易理解时，才提升到共享层。

### 1.4 研究流程优先

规范要服务下列主链路：

1. 复现论文或外部实现
2. 接入模型并跑通单实验
3. 形成可复用的 experiment preset
4. 进入 study 做多 seed / 多数据集汇总
5. 沉淀结果与文档

### 1.5 可复现优先于“灵活”

- 同一配置应映射到稳定的结果目录和 `conf_hash`。
- `experiment > dataset > task` 的合并规则固定，不允许破坏。
- 任何长期存在的数据集专属 recipe 都应升级为 experiment preset，而不是藏在脚本分支或 study overrides 里。

### 1.6 任务边界优先于“统一一切”

当前仓库正式维护四条主链路：

- 多元时序：原生支持，标准输入形态为 `(L, N)`；单变量预测按 `(L, 1)` 作为 `mtsf` 的特例接入，不额外新增 `utsf` task，也不接受裸 `(L,)`。
- 静态图时空序列：原生支持，标准输入形态为 `(L, N)` 加一个静态 graph。
- 2D 规则网格时序：原生支持，标准输入形态为 `(L, C, H, W)`。
- 3D 规则网格时序：原生支持，标准输入形态为 `(L, C, X, Y, Z)`。
- 网格展平 baseline：如果预处理后能展平为 `(L, N)`，仍允许作为 `mtsf` baseline 使用。

但要明确区分两件事：

- “可以展平后复用 `mtsf` 方法”是 baseline 兼容策略。
- “原生支持时空/网格结构”是另一件事，不应被上述兼容策略偷换。

因此，本仓库当前把静态 graph 作为 `stf` 的一等输入，把规则 2D 网格作为 `grid2dtsf` 的一等输入，把规则 3D 网格作为 `grid3dtsf` 的一等输入，但不会把动态图、球面网格、非规则 mesh 或其他结构继续提前泛化成更多公开 task 抽象。只有当模型、数据接口或训练语义需要显式消费新的结构先验时，才再讨论是否新增独立 task。

现阶段的默认原则是：

- 接受展平后的 `mtsf-compatible` 输入。
- 不把 graph 语义混入 `MTSFTask` 公共契约。
- 静态 graph 通过 `STFTask` 和 graph-aware 模型显式消费。
- 2D 规则网格通过 `Grid2DTSFTask` 和 grid-aware 模型显式消费。
- 3D 规则网格通过 `Grid3DTSFTask` 和 grid-aware 模型显式消费。
- 不把“展平兼容”表述成“已经原生支持动态图、球面网格或 mesh”。
- 新任务若确实新增，优先共享 workflow，不强行统一 task 语义。

## 2. 目录与职责规范

### 2.1 顶层目录职责

- `easytsf/model/`：模型定义，只放模型本体与模型私有辅助类
- `easytsf/task/`：任务层与训练/评估逻辑
- `easytsf/data/`：数据读取、缓存、滑窗、loader
- `easytsf/workflow/`：experiment / study 编排
- `config/tasks/`：任务默认配置
- `config/datasets/`：数据集元信息与数据加载默认项
- `config/experiments/`：单实验预设
- `config/studies/`：批量评测声明
- `config/search_spaces/`：超参搜索空间

任何改动都应尽量落在唯一责任层，不把 workflow 逻辑回灌到 model / data / task。

### 2.2 文件命名

- Python 模型文件名统一小写：`easytsf/model/<model_id>.py`
- 配置目录与配置 id 统一小写：`config/experiments/<model_id>/<dataset_id>.yaml`
- 每个模型文件统一导出一个顶层入口类：`Model`
- `model_id`、study id、experiment id 使用稳定的小写标识，不使用大小写混排别名

### 2.3 每层允许承载的信息

`model`

- 结构定义
- 前向逻辑
- 模型私有 helper

`task`

- loss
- metric
- optimizer / scheduler 装配
- 与训练框架相关的 step 逻辑

`data`

- 数据格式约定
- 时间特征
- 滑窗切分
- dataloader 性能参数

`workflow`

- 配置加载与合并
- 结果目录与落盘
- trainer 组装
- study 聚合

禁止把以下逻辑放错层：

- 数据集特例 if/else 放进模型
- 模型超参默认值散落在 CLI 脚本
- study 期间动态改写模型实现
- 在 `study` 中长期维护数据集到超参的映射规则

## 3. 模型接入规范

### 3.1 新增模型的最低交付物

新增一个标准 `mtsf`、`stf`、`grid2dtsf` 或 `grid3dtsf` 模型时，必须同时提交：

1. `easytsf/model/<model_id>.py`
2. 至少一个 `config/experiments/<model_id>/<dataset_id>.yaml`
3. 如需搜索，提供 `config/search_spaces/<model_id>/<name>.py`
4. 文档条目，说明论文来源、实现假设与使用方式

如果缺少 experiment preset，则视为“代码未真正接入仓库主链路”。

### 3.2 模型类契约

模型类必须满足：

- 每个模型模块统一暴露顶层 `Model` 类
- 构造参数来自扁平配置键
- `mtsf` 模型公开接口为 `forward(var_x, marker_x)`，输入批次默认是 `[B, L, N]`，其中单变量 case 也统一写成 `[B, L, 1]`
- `stf` 模型公开接口为 `forward(var_x, marker_x, graph)`
- `grid2dtsf` 模型公开接口为 `forward(var_x, marker_x, grid_mask=None, coord=None)`
- `grid3dtsf` 模型公开接口为 `forward(var_x, marker_x, grid_mask=None, coord=None)`
- 返回张量与标签形状兼容：
  - `mtsf` / `stf` 默认为 `[B, pred_len, N]`
  - `grid2dtsf` 默认为 `[B, pred_len, C, H, W]`
  - `grid3dtsf` 默认为 `[B, pred_len, C, X, Y, Z]`

模型身份由文件名和 config id 承担，不再要求类名重复表达模型名。允许在文件内部保留论文名别名用于阅读，例如 `iTransformer = Model`，但主契约仍是顶层 `Model`。

允许模型在内部提供 `forecast()`、`encode()`、`decode()` 等辅助方法，但不应破坏统一 `forward` 契约。

### 3.3 模型文件组织

单模型单文件，文件内部建议按以下顺序组织：

1. 模块级说明
2. 私有 helper class / function
3. 核心模块定义
4. 主模型类

当 helper 只被该模型使用时，不应提取到共享 `layer/`。只有在跨模型稳定复用后，才允许升级为共享层。

### 3.4 模型配置键

模型参数命名应满足：

- 优先与论文或参考实现一致
- 避免无语义缩写
- 同类含义跨模型尽量统一，例如 `hist_len`、`pred_len`、`dropout`

新增模型不得引入与现有公共键冲突但语义不同的名字。

### 3.5 外部实现对齐要求

参考 `transformers` 新增模型流程，新增模型时应记录以下信息：

- 论文链接
- 官方或主流参考实现链接
- 当前实现与原实现的差异
- 是否做了仅为适配 EasyTSFNext 主链路的改写

如果没有这些上下文，后续复现和结果比较会很困难。

## 4. 配置规范

### 4.1 四段式 section 固定

YAML 只允许使用四个顶层 section：

- `model`
- `data`
- `train`
- `runtime`

不新增第五层公共 section。确有需要时，先判断是否能归入现有四段之一。

### 4.2 合并优先级固定

配置合并顺序固定为：

`task defaults < dataset defaults < experiment preset < runtime overrides`

其中研究层面最常用的表达仍保持：

`experiment > dataset > task`

CLI 仅负责运行时覆盖，不负责定义新的研究配方。

### 4.3 experiment 的职责

`experiment` 必须表达一个可直接运行、可复现的单实验方案，至少包含：

- 模型选择
- 数据集选择
- 默认 `hist_len / pred_len`
- 该数据集下能工作的训练超参

推荐按“模型 x 数据集”组织。若某个 horizon 组合长期存在，可使用稳定后缀，例如：

- `etth1.yaml`
- `etth1_96for336.yaml`

### 4.4 study 的职责边界

`study` 只负责：

- 跑哪些 case
- 每个 case 跑哪些 seed
- 需要哪些覆盖项
- 如何 resume
- 如何汇总结果

`study` 不负责：

- 存放长期训练 recipe
- 为不同数据集内置隐式超参规则
- 引入第二套 experiment 抽象

### 4.5 dataset catalog 的职责

`config/datasets/catalog.yaml` 只维护：

- 数据集元信息
- 数据加载默认项
- 可选静态 graph 路径
- 时间频率、split、time feature 等共享约定

不得把模型专属超参塞进 dataset catalog。

## 5. 文档规范

### 5.1 文档要跟着模型走

参考 `transformers` 的模型文档做法，新增模型时必须同步提供最小文档，至少说明：

- 这是什么模型，解决什么问题
- 论文与参考实现
- 在 EasyTSFNext 中的输入输出契约
- 推荐使用的 experiment preset
- 与原论文默认设置不一致的地方

### 5.2 文档写“仓库真实状态”

文档只能描述仓库里真实存在的内容，不写尚未落地的能力，不保留过时入口，不把历史脚本当作现状。

### 5.3 推荐的模型文档模板

每个模型文档建议包含：

1. Overview
2. Paper / Repo
3. EasyTSFNext interface
4. Available experiment presets
5. Repro notes
6. Known limitations

如果仓库后续补 `docs/models/`，建议以 `docs/models/<model_id>.md` 组织。

## 6. 结果与复现规范

### 6.1 结果目录必须稳定

单实验结果目录继续采用：

`save/<model_name>_<dataset_name>/<conf_hash>/seed_<seed>/`

目录下至少保留：

- `resolved_config.yaml`
- `metrics.json`
- `checkpoints/`

### 6.2 metrics 字段保持可聚合

`metrics.json` 至少保持以下稳定键：

- `model_name`
- `dataset_name`
- `hist_len`
- `pred_len`
- `seed`
- `status`
- `mae`
- `mse`
- `exp_dir`
- `ckpt_path`

新增字段可以追加，但不应破坏 `study` 现有聚合逻辑。

### 6.3 失败也要落可诊断信息

当训练失败时，应尽量保留：

- 失败状态
- 错误信息
- 对应配置
- 已分配的实验目录

研究仓库最怕“失败无痕”，因为这会让批量 benchmark 难以 resume 和排查。

## 7. 验证与测试规范

### 7.1 研究型仓库的最低验证标准

不要求复制 `transformers` 那种完整测试矩阵，但新增模型至少应完成：

1. `python train.py -h`
2. `python evaluate.py -h`
3. `python study.py -h`
4. 新增 experiment YAML 可成功加载
5. 如有 search space，对应模块可成功导入
6. 如果本地有数据，至少跑一个最小 smoke experiment

### 7.2 推荐新增的轻量自动检查

后续若补测试，优先增加以下轻量检查，而不是一开始就上大 CI：

- 配置文件可加载检查
- 模型可实例化检查
- 单 batch `forward` shape 检查
- study dry-run 检查

### 7.3 慢测试与 benchmark 分离

完整 benchmark、长时训练、多 seed 汇总不应作为常规提交门槛；它们属于研究评测，不属于每次代码改动都必须跑的快速测试。

## 8. 代码风格规范

### 8.1 风格总则

- 代码优先可读、可比对、可复现
- 局部重复优于错误抽象
- 宁可显式参数传递，不用隐式共享状态

### 8.2 推荐实践

- 使用清晰、稳定的参数名
- 保持模块边界简单
- 让默认路径服务主链路，而不是覆盖所有可能场景
- 新增公共层前先证明存在真实复用

### 8.3 不推荐实践

- 为单个模型设计公共基类
- 为未来可能支持的任务预留复杂扩展点
- 在模型层读取数据集路径或结果目录
- 在 workflow 层引入模型专属分支

## 9. 非目标

以下内容不属于当前规范追求：

- 通用模型注册中心
- `AutoConfig` / `AutoModel` 风格自动装配
- 多任务统一大框架
- 大规模兼容性测试矩阵
- 面向外部生态的通用插件 API

如果未来仓库目标变成“通用时序模型平台”，再讨论是否向更重的框架演进。

## 10. 建议的落地顺序

### Phase 1: 文档约定

- 以本草案作为仓库规范初稿
- 后续新增模型按本草案执行

### Phase 2: 最小模板化

- 补模型文档模板
- 补新增模型 checklist
- 补配置加载 smoke check

### Phase 3: 轻量自动校验

- 增加配置可加载检查
- 增加模型实例化与 shape 检查
- 增加 study dry-run 检查

## 参考来源

- `transformers` 仓库: <https://github.com/huggingface/transformers>
- `CONTRIBUTING.md`: <https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md>
- How to add a model to Transformers: <https://huggingface.co/docs/transformers/main/add_new_model>
- Modular Transformers: <https://huggingface.co/docs/transformers/main/modular_transformers>
- Document your model: <https://huggingface.co/docs/transformers/main/document_a_model>
- Testing: <https://huggingface.co/docs/transformers/main/testing>
- Checks on a Pull Request: <https://huggingface.co/docs/transformers/main/pr_checks>

## 一句话总结

对 EasyTSFNext 而言，最值得向 `transformers` 学的不是“大框架”，而是三件事：统一接口、模型实现自包含、文档和验证与新增模型一起交付。
