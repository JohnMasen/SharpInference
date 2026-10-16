# Graphs 职责整理计划

状态：设计记录；第一、第二部分已确认，尚未执行代码迁移。

本计划记录后续整理决定，按部分追加。它不替代已经实施的具名建图重构记录，也不表示此前讨论的所有拆分均已确定目标项目。

## 一、第一部分：将七个公共契约迁移至 Instructions

### 1. 已确认的修改

以下类型统一归属 `SharpInference.Instructions` 项目和命名空间，不新建公共契约项目。

| 当前 Graphs 类型 | Instructions 中的用途 | 目标类型 |
|---|---|---|
| `GraphElementType` | 描述指令端口的元素类型和内核运算精度 | `SharpInference.Instructions.ElementType` |
| `TensorDescriptor` | 描述指令张量参数的类型、形状和布局 | `SharpInference.Instructions.TensorDescriptor` |
| `GraphResourceAccess` | 声明参数是只读、只写还是读写 | `SharpInference.Instructions.ResourceAccess` |
| `PrecisionRequirement` | 判断指令能力是否满足调用方的精度要求 | `SharpInference.Instructions.PrecisionRequirement` |
| `NumericTypeCompatibility` | 检查内核精度是否为支持的浮点类型 | `SharpInference.Instructions.NumericTypeCompatibility` |
| `GraphOperationId` | 将图操作映射到具体指令，描述融合阶段 | `SharpInference.Instructions.OperationId` |
| `PrimitiveGraphOperations` | 用标准操作身份描述逐元素融合组合 | `SharpInference.Instructions.PrimitiveOperations` |

源文件当前位于 [Graphs](../src/SharpInference.Graphs/)，目标文件直接放到 [Instructions](../src/SharpInference.Instructions/) 根目录，文件名与目标类型一致。

这是类型所有权、命名空间和部分类型名的迁移，不改变类型的业务含义。既定的不保留旧 API 兼容策略继续适用：旧命名空间不保留包装类型、重复定义、兼容入口或类型转发。

### 2. 依赖方向必须同步反转

当前：

```text
SharpInference.Instructions → SharpInference.Graphs
SharpInference.Graphs       → Abstractions、Graphs.Core
```

目标：

```text
SharpInference.Graphs → SharpInference.Instructions
                     → SharpInference.Abstractions
                     → SharpInference.Graphs.Core

SharpInference.Instructions 不依赖 Graphs 或 Graphs.Core
```

具体修改：

1. 删除 [Instructions 项目文件](../src/SharpInference.Instructions/SharpInference.Instructions.csproj) 中对 Graphs 的引用。
2. 在 [Graphs 项目文件](../src/SharpInference.Graphs/SharpInference.Graphs.csproj) 中增加对 Instructions 的引用。
3. 同一批修改中完成七个类型的迁移及两侧源码引用更新，避免只添加反向引用导致循环依赖。
4. 其他项目直接使用这七个契约时，检查并补齐对 Instructions 的直接项目引用；仍使用图结构的项目保留 Graphs 引用。
5. 不因间接依赖可见就盲目删除现有项目引用，也不批量清理与本部分无关的依赖。

Instructions 中现有的图操作绑定和逐元素融合描述，只使用操作身份、张量及访问/精度契约，不使用具体逻辑图或执行图结构。这些文件本部分保留在原项目；迁移公共契约后移除不再需要的 Graphs 命名空间导入。

### 3. 消费者更新范围

根据已完成的生产消费者统计，逐项目检查：

| 项目 | 本部分要更新的内容 |
|---|---|
| `SharpInference.Graphs` | 公共模型、逻辑/执行节点、构造器、校验、查询、转换/改写及 XML 内部实现的类型引用 |
| `SharpInference.Instructions` | 指令端口、参数、精度适配、图操作映射和逐元素融合目录 |
| `SharpInference.Instructions.Cpu` | 元素类型和张量描述 |
| `SharpInference.Instructions.D3D12` | 元素类型、张量描述、访问方式、操作身份和精度要求 |
| `SharpInference.Instructions.TierZero` | 元素类型、张量描述和访问方式 |
| `SharpInference.Vm` | 元素类型、张量描述、访问方式和精度要求 |
| `SharpInference.Vm.Optimization` | 元素类型、张量描述、访问方式和操作身份 |
| `SharpInference.Models.Phi4` | 构图描述及公共操作/精度契约 |
| `SharpInference.Models.Rwkv` | 构图描述及公共操作/精度契约 |
| `SharpInference.Backends.CpuVm` | 访问方式 |
| `SharpInference.Backends.D3D12Vm` | 访问方式 |

测试项目不计入生产消费者统计，但执行迁移时必须同步更新其编译引用、源码和有关断言。检查范围也包括 README、示例及其他实际使用旧 API 名称的说明。

以上是当前消费者，不是限制扫描范围的白名单。实施时重新搜索完整工作区，兼顾全限定名称、类型别名、泛型约束、目标类型 `new` 和生成源码模板。

### 4. 保持不变的行为

- `ElementType` 的枚举成员、顺序和底层值不变。
- `ResourceAccess` 的 `Read`、`Write`、`ReadWrite` 成员及底层值不变。
- `TensorDescriptor` 的构造参数、默认布局、维度复制、只读暴露及校验规则不变；不顺便加强原有枚举值校验。
- `PrecisionRequirement` 的两个精度约束、校验行为和异常规则不变。
- `NumericTypeCompatibility` 的浮点范围、最大精度选择和空输入/非法输入行为不变；内部辅助方法无需因迁移扩大可见性。
- `OperationId` 保持 `Name`、默认 `Version = 1`、值相等性和 `name@version` 的文本表示。
- `PrimitiveOperations` 的全部 19 个操作常量保持原操作名和版本，包括 `core.*` 标识；只改 C# 容器名及其字段类型。
- 不引入新的注册机制、类型转换层或运行时适配对象。

### 5. 序列化及外部输出

类型名和命名空间变化不应自动变成数据格式变化：

1. [Graphs XML 支持](../src/SharpInference.Graphs/Internal/GraphXmlSupport.cs) 与 [公共关系 XML 编码](../src/SharpInference.Graphs/Internal/ModelGraphXml.cs) 改为读取新的 C# 类型。
2. 保持 XML v3 的元素、属性、枚举文本、操作名/版本和边身份不变；不因 CLR 类型迁移提高 XML 版本，也不恢复 v2 兼容。
3. 保持精度属性、张量描述、Read/Write/ReadWrite 的往返行为不变，使用已有独立手写 XML fixture 验证，而不只测试新写入器配新读取器。
4. 检查 Instructions、VM 和后端的 JSON、代码生成与缓存键是否使用 CLR 类型名或全限定名；若存在，明确记录影响并增加对应回归，不能静默修改输出协议。
5. GraphView 本部分不预设需要改代码；若仅消费未改变的 XML 文本，保持现有实现。

### 6. 实施顺序

下面各阶段是未来代码实施任务，目前均未执行。

#### P1：迁移定义与反转依赖

- [ ] 将七个源文件移动到 Instructions，统一命名空间。
- [ ] 完成四处类型重命名及相应文件名更新。
- [ ] 更新七个类型之间的互相引用与直接相关 XML 文档注释。
- [ ] 同批完成 Instructions 与 Graphs 的项目引用反转和源码更新。
- [ ] 确认旧位置不存在重复类型，Instructions 不再依赖图项目。

#### P2：更新全部消费者

依赖 P1。

- [ ] 更新生产项目中的类型名、命名空间、签名、常量引用及生成模板。
- [ ] 对直接使用迁移契约的项目补齐必要的 Instructions 项目引用。
- [ ] 同步更新测试源码和相关示例。
- [ ] 检查 `ElementType` 与现有同名属性、局部变量或其他命名空间类型的冲突，仅在必要位置使用限定名。
- [ ] 所有项目引用调整完成后执行依赖还原，不新增 NuGet 包或分析工具。

#### P3：验证行为与持久化

依赖 P2。

- [ ] 验证 Instructions 可以独立构建，不需要先构建 Graphs。
- [ ] 验证 Graphs、全部生产消费者和 Phi4 的两个目标框架能够构建。
- [ ] 添加或迁移七个契约的边界回归，覆盖枚举值、张量维度/布局、浮点校验、精度比较、操作身份及 19 个原语常量。
- [ ] 运行相关图构造、转换、后处理、模型适配、VM 发射和 XML 回归。
- [ ] 检查不存在新增循环依赖、旧 API 残留或数据协议变化。

#### P4：文档与消费者交叉表

依赖 P3。

- [ ] 更新中英文 README 中的类型名与命名空间说明。
- [ ] 对已有历史记录保留“当时 API”的语义，必要时加迁移说明，不把历史计划无差别全文替换。
- [ ] 重新生成生产消费者交叉表，测试项目继续排除。
- [ ] 新表区分“七个契约现在属于 Instructions”与“Graphs 仍保留的类型”，不能继续将迁移后的契约算作 Graphs 的类型。
- [ ] 回填本计划的执行结果、实际验证命令及剩余限制。

### 7. 验证命令与验收标准

项目文件调整后还原依赖，再从小范围构建开始：

```powershell
dotnet restore .\SharpInference.slnx
dotnet build .\src\SharpInference.Instructions\SharpInference.Instructions.csproj --no-restore
dotnet build .\src\SharpInference.Graphs\SharpInference.Graphs.csproj --no-restore
dotnet build .\SharpInference.slnx --no-restore
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-build --no-restore --filter "FullyQualifiedName~NamedGraphConstructionTests|FullyQualifiedName~LogicalGraphDefinitionTests|FullyQualifiedName~GraphSerializationTests|FullyQualifiedName~GraphModelAdapterTests|FullyQualifiedName~ExecutionGraphConverterTests|FullyQualifiedName~ExecutionGraphPostprocessorTests|FullyQualifiedName~GraphViewQueryTests|FullyQualifiedName~VmExecutionGraphEmitterTests" --verbosity minimal
```

七个契约新增的独立回归纳入同一次测试选择；若放在新测试类中，扩展上述过滤器。完整解决方案构建用于覆盖本次广泛的编译期 API 迁移，不要求运行无关的完整测试或性能矩阵。真实模型数值测试按相关输出是否变化及本地模型/设备可用性追加，不能将跳过写成通过。

最终验收：

1. 七个类型均只在 Instructions 中定义，命名完全符合已确认表格。
2. Instructions 不引用 Graphs 或 Graphs.Core；Graphs 使用 Instructions 的契约。
3. 生产消费者和测试使用新 API，未遗留旧兼容类型。
4. Graphs.Core 无修改。
5. 枚举值、操作 ID、精度与张量行为、XML v3 格式保持不变。
6. 相关构建和回归通过，生产消费者交叉表更新。

## 二、第二部分：将 Vm.Optimization 专用能力迁至 GraphHelper

### 1. 已确认的归属

将 Graphs 中专供 `SharpInference.Vm.Optimization` 使用的类型和辅助能力迁入：

- 项目：`SharpInference.Vm.Optimization`。
- 目录：`src\SharpInference.Vm.Optimization\GraphHelper`。
- 命名空间：`SharpInference.Vm.Optimization.GraphHelper`。

不创建独立 GraphHelper 项目。本部分是迁移既有能力，不顺带重命名这些类型或重新设计优化管线；第一部分确定的七个类型仍以 Instructions 为唯一归属。

“专用”按实际职责和依赖闭包判断：

1. 测试项目不计入消费者。
2. 唯一库外生产消费者为 Vm.Optimization 的执行图及其辅助类型，是直接迁移对象。
3. 虽没有库外生产消费者、但仅支撑上述执行图和处理流程的类型，一起迁移。
4. 逻辑图本身也需要的公共绑定、作用域、基础查询，不因唯一库外消费者是优化项目就整体搬走。
5. 迁移后 Graphs 不能引用 Vm.Optimization，不能为了修复编译而反向补引用。

### 2. 执行图及处理流程：27 个类型整体迁移

当前 [Execution 目录](../src/SharpInference.Graphs/Execution/) 的全部 27 个顶层类型构成一个完整的迁移单元。它们的库外生产消费者均为 Vm.Optimization 或为空。

#### 直接消费者为 Vm.Optimization 的 21 个类型

| 类型组 | 迁移类型 |
|---|---|
| 图、程序与构造 | `ExecutionGraph`、`ExecutionGraphProgram`、`ExecutionGraphBuilder` |
| 身份 | `ExecutionGraphId`、`ExecutionNodeId` |
| 节点 | `ExecutionGraphStructureNode`、`ExecutionGraphNode`、`ExecutionOperationNode`、`ExecutionGraphCallNode`、`ExecutionResourceNode` |
| 资源与边 | `ExecutionGraphResource`、`ExecutionGraphEdge`、`ExecutionDependencyEdge`、`ExecutionResourceReadEdge`、`ExecutionResourceWriteEdge` |
| 来源 | `ExecutionGraphSource`、`ExecutionSourceMap` |
| 持久化 | `ExecutionGraphXml` |
| 后处理契约 | `IExecutionGraphPostprocessor`、`ExecutionGraphPostprocessContext`、`ExecutionGraphCompilationContext` |

#### 随依赖闭包迁移的 6 个配套类型

| 类型 | 一起迁移的原因 |
|---|---|
| `DefaultExecutionGraphConverter` | 产生已迁移的执行程序，属于逻辑图到执行图的对接能力 |
| `IExecutionGraphConverter` | 上述转换接口 |
| `ExecutionGraphCompiler` | 组织上述转换和后处理流程 |
| `ExecutionGraphRewriter` | 改写已迁移的执行图 |
| `ExecutionGraphCallRelation` | 执行程序的调用点关系 |
| `GraphArchitecture` | 执行图/节点的架构描述元数据 |

因此，这一决定不只是搬走优化器方法，也将当前仅由优化项目直接消费的执行图数据结构、校验、构造和 XML 一起搬走。Graphs 仍保留逻辑图、共享模型接口和基本公共辅助；不再要求保留一套无人独立消费的执行图表示。

### 3. 不按消费者表机械搬迁的共享类型

下列四个类型虽然库外生产消费者只有 Vm.Optimization，但逻辑图本身也使用它们，因此不能整类搬入上层：

| 类型 | 第二部分的处理 |
|---|---|
| `GraphBindingOwner` | 留在 Graphs，逻辑图需要父作用域与调用实例作用域 |
| `ModelResourceNodeId` | 留在 Graphs，逻辑资源节点及绑定需要带作用域的身份 |
| `ModelResourceBinding` | 留在 Graphs，逻辑图绑定查询、构造和校验需要 |
| `ModelGraphQueries` | 保留基础查询与校验必需能力；优化专用分析另行拆入 GraphHelper，不整类移动 |

查询拆分规则：

- 直接读写者、绑定、直接依赖、依赖/数据流视图等基础能力留在 Graphs。
- `ResourceConflicts` 当前被逻辑图校验使用，保留其校验所需实现，不能让逻辑校验转而依赖优化项目。
- `BoundaryResources`、`UnorderedResourceConflicts` 当前没有生产调用，不自动归为“优化项目专用”；本部分不因为测试调用就制造新的归属决定。
- 其他没有生产消费者的遗留清理，继续作为单独事项记录。

### 4. 跨程序集公共辅助的处理

执行图当前通过程序集内部访问复用 [ModelGraphConstruction](../src/SharpInference.Graphs/Internal/ModelGraphConstruction.cs)、[ModelGraphXml](../src/SharpInference.Graphs/Internal/ModelGraphXml.cs) 和 [GraphXmlSupport](../src/SharpInference.Graphs/Internal/GraphXmlSupport.cs)。仅移动目录和命名空间会导致这些调用无法编译。

实施时必须同步完成：

1. 执行图专用的构造、校验、元数据和 XML 处理归 GraphHelper。
2. 两种图共享的绑定展开、结构校验、关系编码及 XML 安全解析保留单一实现，留在低层。
3. 优先通过既有公共模型/查询 API 复用；确需跨程序集的公共辅助，抽出最小的通用入口，内部实现继续隐藏。
4. 不将整套内部工具无差别公开，不通过 `InternalsVisibleTo` 授予优化项目对 Graphs 内部的整体访问。
5. 不复制一套容易分叉的校验/XML 安全解析逻辑，不削弱 DTD 禁用、大小、深度和元素数量限制。
6. 执行节点/图之间原有的内部构造器与方法随整个迁移单元移动，尽量保持现有访问级别。

这些属于保证搬迁完整性的紧密耦合改动，不借此增加独立分析框架或新的项目层次。

### 5. 目标依赖与调用方式

完成第一、第二部分后：

```text
Models.Phi4 / Models.Rwkv → Graphs → Instructions
                                  → Graphs.Core
                                  → Abstractions

Vm.Optimization
  ├─ GraphHelper（执行图、转换、改写及处理管线）
  ├─ 现有优化 Pass、分析与 VM 发射调用者
  └─ 引用 Graphs、Instructions、Vm（保留必要的直接引用）
```

现有优化 Pass 不为了目录整齐全部搬进 GraphHelper。GraphHelper 收纳的是此次从 Graphs 下沉到实际调用者项目的执行图相关辅助；原有调用者通过导入新命名空间使用。

### 6. 实施步骤与验收

依赖第一部分完成；本部分目前未执行。

- [ ] 重新核对生产消费者和引用闭包，确认没有新增的独立执行图消费者。
- [ ] 将 27 个执行图相关类型移入 GraphHelper，保持类型名、业务行为和公开方法语义。
- [ ] 完成跨程序集共享辅助的最小公开入口及调用调整。
- [ ] 更新优化项目、测试和示例中的命名空间、全限定类型名与必要项目引用；不保留旧命名空间兼容壳。
- [ ] 确认 Graphs 中不再残留执行图转换、编译组织和后处理管线，也不依赖 Vm.Optimization。
- [ ] 确认逻辑图的构造、校验、查询及 XML 可在不引用 Vm.Optimization 的情况下独立工作。
- [ ] 验证执行图 XML v3、逻辑到执行转换、内联/融合等改写、后处理校验和 VM 发射行为不变。
- [ ] 构建完整解决方案，运行第一部分的相关回归，并补充跨程序集拆分的针对性回归。
- [ ] 更新 README、归属说明及消费者交叉表：迁出的 27 个类型不再列为 Graphs 所属类型。
- [ ] 确认 Graphs.Core 未修改，记录实际测试结果及限制。

## 三、尚未纳入前两部分的后续整理

以下仍是后续独立事项，不在前两部分中实施：

- 进一步评估未迁移的高阶分析查询，确定删减或归属；保留逻辑校验必需能力。
- 资源/注解/节点层次的进一步合并，以及无消费者类型的清理。
- GraphInstructionBinding 等指令侧对接契约的另行归属调整。

后续决定继续追加至本计划，不能因为前两部分迁移影响到这些文件，就顺带实施尚未确认的职责变更。
