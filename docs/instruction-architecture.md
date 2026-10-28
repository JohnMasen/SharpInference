# 指令分层、发现与目录

## 项目职责

本轮只重构指令契约、实现和注册。逻辑图、执行图、VM 与 backend 的整体架构不变；
局部调整包括 Linear 构图展开、Affine 后处理及 VM 的指令集校验。
不提供旧项目兼容层，不增加指令版本机制，也不实现主动动态融合。

| 层 | 项目（省略 `SharpInference.`） | 职责 |
|---|---|---|
| 基础框架 | `Instructions` | 指令/签名/参数、provider、目录、DI 发现、图绑定和优化能力接口；不定义具体算子 |
| T0 契约 | `Instructions.Tier0` | 单个基础操作、端口/精度契约、集合清单与操作 ID |
| T1 契约 | `Instructions.Tier1` | 预定义多步融合、Affine、归一化等通用契约；引用 T0 描述组成步骤 |
| CPU 实现 | `Instructions.Tier0.CPU`、`Instructions.Tier1.CPU` | 对应层的完整 CPU 指令集、源码生成与数值实现 |
| DirectX 实现 | `Instructions.Tier0.DirectX`、`Instructions.Tier1.DirectX` | 对应层的完整 D3D12 指令集及 HLSL 生成 |
| 模型契约 | `Models.Phi4` | 模型逻辑图、专用操作声明及模型执行指令契约 |
| 模型实现 | `Models.Phi4.D3D12` | Phi4 专用 D3D12 执行指令；不混入通用 T1 |

以后可增加 `Tier0.Vulkan`、`Tier1.Vulkan`、`Tier0.CUDA` 等实现项目，
不需要让框架引用这些项目。项目名 `DirectX` 对应的实际架构仍是 D3D12。

T0 的“单步”指语义上的一个基础操作，不等于一个标量机器指令；
矩阵乘法、归约、索引、复制等仍属于 T0。T1 是显式定义的多步融合。
模型专用性由 `Domain` 与 `Category` 表达，不把模型名当作数值 Tier。

## DI + Scrutor 发现

```csharp
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Instructions;

var services = new ServiceCollection();
services.AddApplicationInstructions();
using var container = services.BuildServiceProvider();
var catalog = container.GetRequiredService<IInstructionCatalog>();
var rows = catalog.Rows;
var parameters = catalog.Parameters;
var signatures = catalog.Signatures;
```

应用必须在运行依赖中引用所需实现包；只引用 `Instructions` 获得消费能力，
不会凭空安装实现。发现过程如下：

1. 从 `DependencyContext.Default` 读取应用运行依赖清单，而非只看已加载程序集。
2. 获取清单内的托管程序集名称、去重排序，并由默认加载上下文加载。
3. Scrutor 扫描公开、非抽象的 `IInstructionProvider` 实现并注册为 singleton。
   构造函数依赖由 DI 注入，不要求无参构造函数或程序集特性。
4. 解析 `IInstructionCatalog` 时实例化 provider、完整枚举指令并建立目录，
   同时校验图绑定及优化能力引用。此阶段不创建 GPU 设备、不编译内核。
5. 优化器或 VM 编译器消费聚合后的 `IInstructionCollectionProvider`；
   该 DI 服务与 `IInstructionCatalog`、`InstructionRegistry` 指向同一目录实例。

同一实现类型重复扫描不会重复注册。不同 provider 争用同一集合则报错，
不按扫描顺序选胜者。加载、依赖注入、契约校验错误直接暴露，不静默忽略。
应用清单不可用时须显式选择程序集：

```csharp
services.AddInstructions(typeof(MyInstructionProvider).Assembly);
```

也可在调用 `AddInstructions` 前手工注册 `IInstructionProvider`。
扫描入口只有该 marker 接口；目录和目标过滤装饰器不会被扫描成 provider。
自动发现针对正常保留运行依赖的部署；本轮不承诺裁剪或 NativeAOT 自动扫描。

### 所有权与完整性

- `(Architecture, InstructionSet)` 唯一归一个 provider 完整拥有。
- 一个 provider 可以拥有多套集合，但任何一套都不能拆到多个 provider 拼装。
- 具体 provider 必须声明 `RequiredInstructions`，实际枚举须与清单完全一致。
- 重复指令、越权指令、缺项、额外项、空签名和重复端口等均在目录构建时报错。
- 不支持底层覆盖。另一实现必须使用不同集合身份，由优化器显式选择。
- 合法 backend 有责任实现所有必需 T0；框架不补齐、不回退，也不为不完整 backend
  提供可运行保证。清单校验检查已声明集合，并非框架为任意第三方 backend 自动补全契约。

## 目录与精度

目录是机器可读的数据，不要求消费者解析实现名称。

| 视图 | 字段 | 含义 |
|---|---|---|
| `Rows` | `Architecture`、`InstructionSet`、`Instruction` | 实现选择键；架构、集合身份和指令名缺一不可 |
| `Rows` | `Category`、`Domain`、`Comment` | T0/T1/Specialized、core/phi4 等归属及可空用法说明 |
| `Parameters` | 上述选择键、`Signature`、`Name`、`Meaning` | 指定重载内每个参数的名称和含义 |
| `Parameters` | `Kind`、`Access`、`DataType` | Tensor/Attribute、读写权限、存储类型或属性类型 |
| `Parameters` | `InternalPrecision`、`Constraints` | 可选内部精度及形状/取值约束 |
| `Signatures` | 选择键、`Signature`、`ArithmeticPrecision`、`AccumulatorPrecision` | 每个签名的算术与累加精度 |

例如 T0 FP16 `core.add` 的 `left/right/output` 存储为 Float16，
内部算术为 Float32，输出再写回 Float16。FP16 存储不意味着原生 FP16 算术。
浮点参数未单独指定内部精度时，参数表采用签名算术精度；整数参数不做此继承。
调用方传入 `PrecisionRequirement`，`Adapt` 选择满足存储、访问与内部精度的签名，
随后实现生成对应源码；不把精度要求编码成一个含糊的指令名称。

以 CPU、T0 FP16 集合、`core.add` 的签名 0 为例：

| 名称 | 含义 | 类型 | 访问 | 存储精度 | 内部精度 |
|---|---|---|---|---|---|
| `left` | 左操作数 | Tensor | Read | Float16 | Float32 |
| `right` | 右操作数 | Tensor | Read | Float16 | Float32 |
| `output` | 结果 | Tensor | Write | Float16 | Float32 |

该签名的 `ArithmeticPrecision` 和 `AccumulatorPrecision` 都是 Float32。
Affine 的 `transpose_left`、`transpose_right` 则是 Boolean Attribute 参数，
不具有 Tensor 读写权限或内部精度。完整参数明细应直接读取 `Parameters`，
并以实现选择键和 `Signature` 关联，不能把多个重载混成一个端口表。

目前 T0 的 FP32 与 FP16 存储集合分别注册；FP32 集合还包含部分混合存储签名。
T1 pointwise 为 FP32。Affine 左输入/输出为 FP32，右矩阵和 bias
分别支持 FP32/FP16，共四个签名，内部算术和累加均为 FP32。
未来按精度动态生成融合可作为 provider 的额外能力，不改变上述选择键或覆盖规则。

## 算子归属表

以下项目名均省略 `SharpInference.`。同一行列出的每个操作均属于该行的层及项目。
这是当前源码的完整通用指令及已实现模型执行指令清单，不将尚未实现的声明冒充实现。

| 分类 / Domain | 操作（精确名称） | 定义项目 | CPU 实现项目 | D3D12 实现项目 |
|---|---|---|---|---|
| T0 / core | `core.copy`, `core.add`, `core.subtract`, `core.multiply`, `core.divide`, `core.maximum` | `Instructions.Tier0` | `Instructions.Tier0.CPU` | `Instructions.Tier0.DirectX` |
| T0 / core | `core.exp`, `core.tanh`, `core.sigmoid`, `core.rsqrt`, `core.square`, `core.relu` | `Instructions.Tier0` | `Instructions.Tier0.CPU` | `Instructions.Tier0.DirectX` |
| T0 / core | `core.reduce-sum`, `core.reduce-mean`, `core.mat-vec`, `core.matrix-multiply`, `core.bias-add`, `core.gather-row` | `Instructions.Tier0` | `Instructions.Tier0.CPU` | `Instructions.Tier0.DirectX` |
| T0 / core | `core.tensor.fill`, `core.tensor.cast-f16-f32`, `core.tensor.reshape`, `core.tensor.slice`, `core.tensor.broadcast` | `Instructions.Tier0` | `Instructions.Tier0.CPU` | `Instructions.Tier0.DirectX` |
| T0 / core | `core.tensor.batched-mat-vec`, `core.tensor.reduce-last-sum`, `core.tensor.reduce-last-mean`, `core.tensor.head-outer` | `Instructions.Tier0` | `Instructions.Tier0.CPU` | `Instructions.Tier0.DirectX` |
| T0 / core | `transformer.argmax`, `transformer.embedding-gather-batch`, `transformer.copy-embedding-rows`, `transformer.batch-shared-matvec` | `Instructions.Tier0` | `Instructions.Tier0.CPU` | `Instructions.Tier0.DirectX` |
| T1 / core | `add-rsqrt`, `maximum-rsqrt`, `multiply-add`, `add-multiply-add`, `multiply-multiply-add`, `multiply-add-multiply` | `Instructions.Tier1` | `Instructions.Tier1.CPU` | `Instructions.Tier1.DirectX` |
| T1 / core | `difference-mix`, `multiply-add-subtract`, `multiply-self-sigmoid`, `add-sigmoid`, `exp-subtract-exp`, `sigmoid-multiply-exp`, `relu-square` | `Instructions.Tier1` | `Instructions.Tier1.CPU` | `Instructions.Tier1.DirectX` |
| T1 / core | `core.affine`（MatrixMultiply + BiasAdd） | `Instructions.Tier1` | `Instructions.Tier1.CPU` | `Instructions.Tier1.DirectX` |
| T1 / core | `transformer.rms-norm`, `transformer.residual-rms-norm`, `transformer.swiglu`, `transformer.scaled-add` | `Instructions.Tier1` | `Instructions.Tier1.CPU` | `Instructions.Tier1.DirectX` |
| T1 / core | `transformer.batch-rms-norm`, `transformer.batch-residual-rms-norm`, `transformer.batch-swiglu` | `Instructions.Tier1` | `Instructions.Tier1.CPU` | `Instructions.Tier1.DirectX` |
| Specialized / phi4 | `transformer.rope-kv-write`, `transformer.gqa-scores`, `transformer.causal-softmax`, `transformer.gqa-values`, `transformer.grouped-query-attention` | `Models.Phi4` | 未实现 | `Models.Phi4.D3D12` |
| Specialized / phi4 | `transformer.batch-rope-kv-write`, `transformer.batch-gqa-scores`, `transformer.batch-causal-softmax`, `transformer.batch-gqa-values`, `transformer.batch-select-last`, `transformer.batch-gather-rows` | `Models.Phi4` | 未实现 | `Models.Phi4.D3D12` |

通用 T0 共 31 个不同操作名（27 个 core + 4 个 tensor/索引操作），
通用 T1 共 21 个（13 个 pointwise + Affine + 7 个归一化/激活融合），
模型执行集合 11 个。T0 FP16 存储集合不重复提供 `cast-f16-f32`；
该转换由 FP32 集合接收 FP16 输入并产生 FP32 输出。

原先混在 GPU Transformer 集合中的 22 项已按 4/7/11 拆分。
通用操作的 CPU/GPU 共享契约及完整清单，模型集合保留位置、cache、control
等 Phi4 执行协议。通用 batch 融合保留 control 参数布局，但不依赖模型位置语义。

### 既有模型声明的边界

下列 23 个逻辑操作仍定义在 `Models.Phi4` 的
[Phi4InstructionContracts.cs](../src/SharpInference.Models.Phi4/Instructions/Phi4InstructionContracts.cs)，
目前没有对应的完整 provider 实现：

| 分类 / Domain | 声明 | 定义项目 | CPU 实现 | D3D12 实现 |
|---|---|---|---|---|
| Specialized / phi4 | `phi4.audio.normalize-features`, `phi4.audio.conv1d`, `phi4.audio.conv2d`, `phi4.audio.flatten-subsampling`, `phi4.audio.layer-norm`, `phi4.audio.bias-activation`, `phi4.audio.swi-glu`, `phi4.audio.residual`, `phi4.audio.relative-attention` | `Models.Phi4` | 无 | 无 |
| Specialized / phi4 | `phi4.vision.patch-embedding`, `phi4.vision.layer-norm`, `phi4.vision.self-attention`, `phi4.vision.gelu-tanh`, `phi4.vision.pool-2x2`, `phi4.vision.hd-gather` | `Models.Phi4` | 无 | 无 |
| Specialized / phi4 | `phi4.text.rms-norm`, `phi4.text.swiglu`, `phi4.text.rope-kv-write`, `phi4.text.gqa-scores`, `phi4.text.causal-softmax`, `phi4.text.gqa-values`, `phi4.text.scaled-add`, `phi4.text.advance-position` | `Models.Phi4` | 无 | 无 |

新模型 provider 不注册 `phi4.text.*` 的 `GraphInstructionBinding`。
名字或公式近似不能证明端口、状态、副作用和图语义等价，因此不能把上面的声明直接
算成那 11 个执行内核已经实现。已有模型缺口不在本轮凭名称补映射。
`Models.Rwkv` 当前没有独立专用指令 provider；已有 pointwise 融合使用通用 T1。

## 保持不变的编译链路

逻辑图仍由模型构造，不查询已安装目录。本轮将 Phi4 的 Linear 局部改为
`core.matrix-multiply → core.bias-add`，保留容器、scope、视图及转置信息。
执行图转换与 VM 编译的公共架构不变，消费者继续使用 provider 查询接口。

优化管线可显式加入 `AffinePostprocessor`，在目标存在 T1 Affine 时，
以单消费者临时量、端口、形状、精度、别名等条件校验后替换执行图中的安全模式。
无支持或不安全时保留原 T0 图；不是按名字盲目替换。VM 的 GPU launch 校验
相应区分 T0 MatrixMultiply 和 T1 Affine。

今后后处理器仍可结合容器、scope、模型元数据等选择预定义专用实现。
任意连续 T0 的主动融合、动态生成新契约/集合，属于后续扩展，不在本轮执行。

## 验证入口

- `InstructionDiscoveryTests`：扫描幂等、构造函数注入、完整所有权、目录参数及独立进程发现。
- `InstructionContractTests`、`PortableTransformerInstructionTests`：契约归属、
  CPU/GPU 清单与签名一致性、形状校验及 CPU 数值基线。
- `AffinePostprocessorTests`：安全替换、转置、混合精度、别名/额外消费者拒绝、
  GPU 源码及 CPU 融合前后数值等价。
- `InstructionPrecisionExecutionTests`：FP16 存储矩阵与 bias 的 CPU 执行、
  GPU 半精度读写源码及 DXC 编译。DXC 编译不等于 GPU 设备数值验证。
- `Instructions.DiscoveryProbe`：只引用 T1.CPU，验证尚未加载的传递依赖能被发现。

不依赖外部 RWKV 模型的回归可运行：

```powershell
dotnet build SharpInference.slnx --no-restore
dotnet test src\SharpInference.Tests\SharpInference.Tests.csproj --no-build --no-restore --filter "FullyQualifiedName!~RwkvPostprocessorTests&FullyQualifiedName!~RwkvTinyExecutionPipelineTests"
dotnet test src\SharpInference.Graphs.Tests\SharpInference.Graphs.Tests.csproj --no-build --no-restore
dotnet test src\SharpInference.Gguf.Tests\SharpInference.Gguf.Tests.csproj --no-build --no-restore
```

完整 RWKV 数值和优化性能矩阵位于被过滤的两个测试类，依赖外部模型，
其中 D3D12 数值测试还需要可用设备；须另行运行，不能用上面的回归结果代替。
