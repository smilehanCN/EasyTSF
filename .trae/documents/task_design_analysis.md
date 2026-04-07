# EasyTSF Task 模块设计分析与改进计划

## 一、当前设计概览

### 文件结构
```
easytsf/task/
├── __init__.py          # 模块导出
├── base.py              # 基础任务类 BaseForecastTask
├── registry.py          # 任务注册和规范定义
├── mtsf.py              # 多变量时间序列预测任务
├── grid3d_forecasting.py # 3D网格预测任务
└── weatherbench.py      # 天气预测基准任务
```

### 核心类关系
- `BaseForecastTask` (继承 `L.LightningModule`) - 提供训练/验证/测试的标准流程
- `MTSFTask`, `Grid3DForecastingTask`, `WeatherBenchTask` - 具体任务实现
- `TaskSpec`, `MetricSchema` - 任务规范数据类

---

## 二、发现的设计问题

### 问题 1: 代码重复 - _apply 方法

**现象:**
- `_apply` 方法在三个任务类中完全重复实现
- 这是 PyTorch Lightning 用于处理设备转移的方法（如 .to(device), .cuda()）
- 每个 Task 都需要手动处理 scaler 的 mean 和 std 的设备转移

**影响:**
- 维护成本高，修改一处需要同步多处
- 容易引入不一致性
- 违反 DRY (Don't Repeat Yourself) 原则

**代码示例 (重复的 _apply 方法):**
```python
# mtsf.py, grid3d_forecasting.py, weatherbench.py 都有相同代码
def _apply(self, fn):
    super()._apply(fn)
    self.scaler.set_stats(fn(self.scaler.mean), fn(self.scaler.std))
    return self
```

**根本原因:**
- StandardScaler 的 mean 和 std 是普通 tensor，不会自动跟随模型移动设备
- 每个 Task 需要手动处理 scaler 的设备转移

### 问题 2: 接口不一致

**现象:**
- `postprocess_outputs` 方法签名不统一
  - `base.py`: `postprocess_outputs(prediction, label, targets_mask=None)`
  - `grid3d_forecasting.py`: `postprocess_outputs(prediction, label)` - 缺少 targets_mask
  - `mtsf.py` 和 `weatherbench.py`: 包含 targets_mask 参数

**影响:**
- 增加使用复杂度
- 容易导致运行时错误
- 不符合里氏替换原则

### 问题 3: 职责混乱

**现象:**
`BaseForecastTask` 承担了过多职责:
1. 训练流程管理 (training_step, validation_step, test_step)
2. 模型实例化 (_instantiate_registered_model)
3. 优化器配置 (configure_optimizers)
4. 数据处理接口定义 (preprocess_batch, postprocess_outputs)
5. 任务状态管理 (_setup_task_state)

**影响:**
- 违反单一职责原则 (SRP)
- 类过于庞大，难以理解和维护
- 测试困难

### 问题 4: 扩展性差

**现象:**
添加新任务需要:
1. 创建新的任务类文件
2. 在 `registry.py` 的 `TASK_SPECS` 中注册
3. 在 `__init__.py` 中导出
4. 手动维护 `supported_models` 列表

**影响:**
- 扩展流程繁琐
- 容易遗漏步骤
- 不支持动态注册

### 问题 5: 缺少抽象层

**现象:**
- Scaler 相关逻辑分散在各个任务类中
- 没有统一的预处理/后处理抽象
- 数据转换逻辑与任务逻辑耦合

**影响:**
- 难以复用数据处理逻辑
- 不利于添加新的数据转换方式

### 问题 6: 配置管理问题

**现象:**
- 使用 `self.hparams` 字典传递配置，缺少类型安全
- `_get_model_derived_args` 逻辑复杂，难以理解
- 配置验证分散在多处

**影响:**
- 运行时才发现配置错误
- IDE 无法提供代码补全
- 文档化困难

### 问题 7: Registry 设计冗余

**现象:**
```python
# registry.py 中同时维护两个字典
TASK_SPECS = {...}  # 包含完整信息
TASK_REGISTRY = {name: (spec.datamodule_cls, spec.task_cls) for name, spec in TASK_SPECS.items()}
```

**影响:**
- `TASK_REGISTRY` 只是 `TASK_SPECS` 的子集视图
- 增加了维护负担
- 容易导致不一致

---

## 三、改进建议

### 建议 1: 在 BaseForecastTask 中统一处理 scaler 设备转移

**目标:** 在基类中统一处理 scaler 的设备转移，消除子类重复代码

**背景:**
- StandardScaler 已经是独立的封装类，功能完善
- 问题在于 scaler 的 mean/std 需要跟随模型移动设备
- 当前每个子类都重复实现 `_apply` 方法

**实现方案 1: 在基类中自动处理所有 scaler 属性**
```python
# base.py
class BaseForecastTask(L.LightningModule):
    def _apply(self, fn):
        super()._apply(fn)
        # 自动处理所有名为 scaler 或包含 scaler 的属性
        for attr_name in dir(self):
            if 'scaler' in attr_name.lower():
                scaler = getattr(self, attr_name, None)
                if scaler is not None and hasattr(scaler, 'set_stats'):
                    scaler.set_stats(fn(scaler.mean), fn(scaler.std))
        return self
```

**实现方案 2: 使用 register_buffer 注册 scaler 统计量**
```python
# 修改 StandardScaler 类，支持 register_buffer
class StandardScaler:
    def register_to_module(self, module, prefix='scaler'):
        """将 scaler 统计量注册为 module 的 buffer"""
        module.register_buffer(f'{prefix}_mean', self.mean)
        module.register_buffer(f'{prefix}_std', self.std)
        self._module = module
        self._prefix = prefix
    
    @property
    def mean(self):
        if hasattr(self, '_module'):
            return getattr(self._module, f'{self._prefix}_mean')
        return self._mean
    
    @property  
    def std(self):
        if hasattr(self, '_module'):
            return getattr(self._module, f'{self._prefix}_std')
        return self._std

# Task 类中使用
class MTSFTask(BaseForecastTask):
    def _setup_task_state(self):
        self.scaler = self._build_scaler()
        self.scaler.register_to_module(self)  # 自动处理设备转移
```

**影响范围:**
- `base.py`: 添加统一的 `_apply` 方法
- `mtsf.py`: 删除 `_apply` 方法
- `grid3d_forecasting.py`: 删除 `_apply` 方法
- `weatherbench.py`: 删除 `_apply` 方法（需要处理双 scaler）

### 建议 2: 统一接口签名

**目标:** 确保所有子类方法签名一致

**实现方案:**
1. 在 `BaseForecastTask` 中明确定义抽象方法
2. 所有子类必须遵循统一签名
3. 使用 `**kwargs` 处理可选参数

```python
# base.py
from abc import abstractmethod

class BaseForecastTask(L.LightningModule):
    @abstractmethod
    def postprocess_outputs(self, prediction, label, targets_mask=None):
        """所有子类必须实现此方法，包含 targets_mask 参数"""
        pass
```

### 建议 3: 分离职责 - 引入配置类

**目标:** 将配置管理从 BaseForecastTask 中分离

**实现方案:**
```python
# easytsf/task/config.py
from dataclasses import dataclass

@dataclass
class TaskConfig:
    model: str
    optimizer: str = "Adam"
    lr: float = 1e-3
    lr_scheduler: str = "StepLR"
    # ... 其他配置
    
    def validate(self):
        """配置验证逻辑"""
        pass

@dataclass  
class OptimizerConfig:
    optimizer_type: str
    lr: float
    weight_decay: float = 0.0
    # ... 优化器相关配置
```

### 建议 4: 改进注册机制

**目标:** 简化任务注册流程，支持装饰器注册

**实现方案:**
```python
# registry.py
TASK_SPECS = {}

def register_task(spec: TaskSpec):
    """装饰器方式注册任务"""
    def decorator(cls):
        TASK_SPECS[spec.name] = spec
        return cls
    return decorator

# 使用示例
@register_task(TaskSpec(
    name="mtsf",
    family="sequence_prediction",
    # ...
))
class MTSFTask(BaseForecastTask):
    pass
```

### 建议 5: 引入预处理管道抽象

**目标:** 将数据预处理逻辑抽象为可组合的管道

**实现方案:**
```python
# easytsf/data/transforms.py
from abc import ABC, abstractmethod

class Transform(ABC):
    @abstractmethod
    def __call__(self, batch):
        pass

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, batch):
        for t in self.transforms:
            batch = t(batch)
        return batch

class Normalize(Transform):
    def __init__(self, scaler):
        self.scaler = scaler
    
    def __call__(self, batch):
        batch['inputs'] = self.scaler.transform(batch['inputs'])
        return batch
```

### 建议 6: 简化 Registry 结构

**目标:** 移除冗余的 TASK_REGISTRY

**实现方案:**
```python
# registry.py - 简化后
TASK_SPECS = {...}  # 唯一的数据源

def get_task_components(task: str) -> tuple[type, type]:
    spec = get_task_spec(task)
    return spec.datamodule_cls, spec.task_cls

# 删除 TASK_REGISTRY 字典
```

### 建议 7: 增强类型安全

**目标:** 使用类型注解和数据类提高类型安全

**实现方案:**
```python
from typing import TypedDict, Literal

class TaskHyperParams(TypedDict, total=False):
    model: str
    optimizer: Literal["Adam", "AdamW"]
    lr: float
    lr_scheduler: Literal["StepLR", "MultiStepLR", "ReduceLROnPlateau", "OneCycleLR"]
    # ...

class BaseForecastTask(L.LightningModule):
    def __init__(self, **kwargs: Unpack[TaskHyperParams]):
        super().__init__()
        self.save_hyperparameters()
        # ...
```

---

## 四、重构优先级建议

### 高优先级 (立即修复)
1. **统一接口签名** - 修复 `postprocess_outputs` 不一致问题
2. **提取 ScalerMixin** - 消除代码重复
3. **简化 Registry** - 移除冗余的 TASK_REGISTRY

### 中优先级 (近期改进)
4. **分离配置类** - 提高配置管理的清晰度
5. **改进注册机制** - 支持装饰器注册
6. **增强类型安全** - 添加类型注解

### 低优先级 (长期优化)
7. **引入预处理管道** - 提高数据处理的灵活性

---

## 五、实施步骤

### 阶段 1: 修复接口不一致 (高优先级)
1. 在 `base.py` 中将 `postprocess_outputs` 改为抽象方法
2. 修改 `grid3d_forecasting.py` 的 `postprocess_outputs` 添加 `targets_mask` 参数
3. 运行测试确保兼容性

### 阶段 2: 消除 _apply 方法重复 (高优先级)
1. 选择实现方案（推荐方案 1，在基类中统一处理）
2. 在 `base.py` 中添加统一的 `_apply` 方法
3. 从三个任务类中删除重复的 `_apply` 方法
4. 运行测试确保设备转移功能正常

### 阶段 3: 简化 Registry (高优先级)
1. 删除 `TASK_REGISTRY` 字典
2. 更新所有引用 `TASK_REGISTRY` 的代码
3. 运行测试确保功能正常

### 阶段 4: 配置管理改进 (中优先级)
1. 创建 `easytsf/task/config.py` 文件
2. 定义配置数据类
3. 重构 `BaseForecastTask` 使用新的配置类
4. 更新文档和示例

### 阶段 5: 注册机制改进 (中优先级)
1. 实现 `register_task` 装饰器
2. 重构现有任务类使用装饰器注册
3. 更新测试用例

---

## 六、风险评估

### 低风险
- 统一接口签名
- 简化 Registry 结构

### 中风险
- 提取 ScalerMixin (需要仔细处理 weatherbench 的双 scaler 情况)
- 配置管理改进 (需要大量测试)

### 高风险
- 引入预处理管道 (可能影响现有数据流)

---

## 七、总结

当前 `easytsf/task` 模块的设计总体上是合理的，采用了清晰的继承层次和注册机制。但存在以下主要问题需要改进：

1. **代码重复** - 需要提取公共逻辑
2. **接口不一致** - 需要统一方法签名
3. **职责混乱** - 需要分离关注点
4. **扩展性差** - 需要改进注册机制

建议按照优先级分阶段实施改进，优先解决高优先级问题，确保向后兼容性。
