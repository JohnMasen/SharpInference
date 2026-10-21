# Graphs：Kind 与派生属性调整计划

状态：已实施并验证完成。用户确认 Kind 使用 string，提供公开常量；不引入 Kind 泛型参数。用户进一步明确不考虑旧格式兼容，授权对应修改并要求更新架构设计文档。

后续归属修复：执行图组件已迁回 `SharpInference.Graphs.Execution`，
见[执行图依赖修复计划](graphs-optimizer-helper-migration-plan.md)。
本文的 GraphHelper 归属及 40／30 统计保留为 Kind 重构完成时的历史快照，
不代表当前执行图位置。

## 1. 目标与边界

用户已确认的方向：

- 基础节点和边只承载身份、连接与 Kind，不通过业务接口要求派生对象提供张量、端口、精度等属性。
- Kind 表达元素在图中的含义，Graphs 不预定义 Resource、Operation、Call 等封闭类别。
- Kind、具体 CLR 类型、属性和连接是否合法，由具体图的 Validate 规则决定。基础架构不规定一对一映射。
- 派生类直接定义自己的属性，业务查询使用 C# `is`、属性模式、`OfType<T>()` 和现有 Core 查询能力。
- 所有表示类别的属性必须在名称中明确分类对象：节点使用 `NodeKind`，边使用 `EdgeKind`，资源角色使用 `ResourceKind`；不得保留或新增无前缀的单独 `Kind` 属性。
- 不为替代旧接口而引入访问器接口、适配器框架、运行时类型注册表或大量策略对象。
- 常用业务查询允许通过扩展方法复用，例如获取全部资源，但遵循“先写具体代码，再依据实际重复决定辅助函数及归属”的顺序；不预先设计公共查询清单。

本次只授权实现上述边界调整，不是重新设计整个图系统：

- Graphs.Core 不修改；不增加项目，不搬迁已确认的项目归属。
- 逻辑图仍在 Graphs.Logical；执行图仍在 Vm.Optimization.GraphHelper。
- Instructions 归属和 Graphs 对 Instructions 的现有依赖保持，不强行完成长期依赖解耦。
- 保留当前构图、依赖推导、校验、XML、转换、优化及数值执行行为。
- 不顺带删除遗留类型、移除分析功能、合并全部具体节点/边或改变编译边界语义。
- 发现超出范围的架构、访问边界或格式冲突时，暂停受影响部分并询问用户。

## 2. 调整前后

```text
当前：基础身份 + 一组业务接口

IModelGraphNode
  ├─ IModelOperationNode（空标签）
  └─ IModelResourceNode（作用域身份、张量、资源角色）

IModelGraphEdge
  ├─ IDependencyEdge（空标签）
  ├─ IResourceReadEdge（端口、绑定身份）
  └─ IResourceWriteEdge（端口、绑定身份）

目标：最小分类契约 + 派生数据

IInferenceGraphNode : Core.IGraphNode
  └─ NodeKind（不定义业务词汇）

IInferenceGraphEdge : Core.IGraphEdge
  └─ EdgeKind（不定义业务词汇）

逻辑图具体类 / 执行图具体类
  ├─ 自己的属性
  ├─ Kind 的业务含义
  ├─ 类型和属性模式查询
  └─ 对应图的 Validate 与 XML 处理
```

保留两个现有基础接口并增加有实际含义的分类属性，避免再建立与现有逻辑/执行基类重复的新一层公共基类。

| 接口 | 处理 |
|---|---|
| `IModelGraphNode` | 改名为 `IInferenceGraphNode`，增加只读 `NodeKind` 属性 |
| `IModelGraphEdge` | 改名为 `IInferenceGraphEdge`，增加只读 `EdgeKind` 属性 |
| `IModelOperationNode` | 移除，业务查询改用具体节点类型或图自己的 Kind 规则 |
| `IDependencyEdge` | 移除，业务查询改用具体边类型或图自己的 Kind 规则 |
| `IModelResourceNode` | 移除，资源数据直接由具体资源节点及资源声明提供 |
| `IResourceReadEdge` | 移除，直接读取具体读边属性 |
| `IResourceWriteEdge` | 移除，直接读取具体写边属性 |

不新增 `IResourceAccessEdge` 作为替代业务契约。接口数量由 7 个变为 2 个，但数量不是验收目标；收缩基础层知识范围才是。

目标接口及文件名确定为 `IInferenceGraphNode` / `IInferenceGraphNode.cs` 和 `IInferenceGraphEdge` / `IInferenceGraphEdge.cs`，命名空间保持 `SharpInference.Graphs`。两者分别继承 Core 的 `IGraphNode` / `IGraphEdge`，不修改 Core 接口。`Inference` 表达使用领域，不规定 Kind 的业务词汇或派生属性。

重命名时同步更新逻辑图、GraphHelper、公共工具、泛型约束、测试、脚本和文档中的类型引用；不保留旧基础接口别名或兼容包装，不扩大为其他 `Model*` 类型的批量重命名。

## 3. Kind 的表示和业务规则

### 3.1 已确认的表示方案

Kind 使用 string，采用 `NodeKind`、`EdgeKind` 属性名，避免与现有资源角色 `Kind` 冲突。

- 提供公开的 `const string` 常量简化使用；常量不是合法值全集，不限制调用方定义新字符串。
- 逻辑图、执行图的常量放在各自业务命名空间中，不把固定业务词汇写进基础接口。优先每个模型使用一个简单的静态常量类，不建立注册表或分类继承体系。
- 常量覆盖现有业务实际需要的操作、调用、资源节点及依赖、读写边等类别。具体名称和值在实施时沿用现有语义与项目命名习惯，不将图类别与 `OperationId` 或资源角色混淆。
- 字符串比较使用明确的 Ordinal 规则，不按大小写或 CLR 类型自动推断、修复 Kind。
- 底层可承载自定义值；是否接受未知值、空值或某种类型与 Kind 的组合，由具体图明确验证并输出诊断。
- 只读接口不等于类别由 CLR 类型固定：接口不要求硬编码返回值，也不引入可变 setter。具体模型通过自己的构造/快照方式保存 Kind。
- 不自动把某个 Kind 等同某个具体 CLR 类型。业务需要这种对应规则时在 Validate 中实现。

不再将 Kind 承载类型作为实施阻塞项；不得自行改为枚举或添加贯穿图模型的 TKind 泛型层。

### 3.2 现有业务类的处理

- 现有操作、调用、资源、依赖、读写边保留具体类型及属性，增加 Kind 的承载方式。
- Builder 为现有 API 产生保持旧行为的默认 Kind；允许扩展如何提供 Kind，优先采用现有配置模式或可选参数，不设计额外配置框架。
- 需要表达错误组合的测试可以构造对象，然后由 Validate 报错；不能仅为维持架构约束在基础构造器中禁止所有未知组合。
- 资源角色仍表示 Input、Output、Weight、State 等，不改成节点类别；现有相关 `Kind` 属性和构造参数分别改为 `ResourceKind`、`resourceKind`。同步更新逻辑/执行资源声明、节点上保留的角色属性、构造器、查询、校验、转换、优化、XML 实现及测试等全部引用。
- 其他类别属性若存在无前缀 `Kind`，按其实际分类对象命名并更新引用；不将此规则扩大为类型名含 Kind 或普通文本的机械替换。资源角色枚举类型名本轮保持不变。
- 仅为实现旧接口而添加的转发属性，在更新全部消费者后评估移除；不要在保留与否尚有疑问时改变其他业务 API。

## 4. 查询与校验的拆分

| 能力 | 调整方向 |
|---|---|
| 基于身份、端点的相邻查询、遍历 | 继续复用 Core，不加重复包装 |
| 按 Kind 筛选 | 使用普通谓词；筛选值由调用图提供，基础层不内置业务值 |
| 依赖/数据流视图、读者/写者查询 | 由逻辑图和执行图按本模型的类型及 Kind 规则提供便利方法 |
| 端口绑定重建 | 在能访问具体读写边、资源节点的模型侧处理 |
| 资源冲突、区域边界分析 | 保留已有能力与测试，迁移到具体模型查询的实现方式，不趁本轮删功能 |
| 身份重复、端点不存在 | 公共结构检查保留 |
| Kind、类型与连接组合是否合法 | 各图 Validate 负责 |
| 作用域、端口、访问配对、张量和调用接口 | 各模型的校验负责，保留当前规则和诊断信息 |

原 ModelGraphQueries 已拆为 [LogicalGraphQueries](../src/SharpInference.Graphs/Logical/LogicalGraphQueries.cs) 与 [ExecutionGraphQueries](../src/SharpInference.Graphs/Execution/ExecutionGraphQueries.cs)，[ModelGraphConstruction](../src/SharpInference.Graphs/Internal/ModelGraphConstruction.cs) 只保留结构校验等公共能力，不能仅把 `is 接口` 替换为字符串比较。

实施原则：

1. 先在实际调用点和模型实现中使用具体类型、`is`、属性模式及 `OfType<T>()` 写出正确的查询代码，不为预计会复用的场景提前建立查询入口。
2. 两种模型真正相同、且只依赖身份/端点/谓词的算法继续共享。
3. 代码形成后，整理真实重复的调用点，再决定需要哪些扩展方法、签名及放置位置。例如获取全部资源可考虑 `GetResources`；名称、返回类型和按 Kind 还是具体类型筛选，由真实调用语义决定，不预先固定。
4. 辅助函数尽量靠近实际消费者；仅依赖基础结构且确有跨模型复用的才考虑放到 Graphs。不要为复用而让底层认识上层类型，或新增业务接口、适配框架、逐字段 selector。
5. 验证所有旧诊断仍可到达，不出现先强制转换、后检查导致的 InvalidCastException，也不以忽略未知类型替代明确诊断。
6. 在具体实现之后列出公共辅助函数候选、重复证据及建议归属，再制定并确认公共提取方案。不要把本节举例当作已授权新增某个公共 API；现有辅助方法的必要改写与删除旧接口同步进行。

模型侧匹配示意：

```csharp
if (node is LogicalResourceNode { Resource: { Tensor: var tensor } })
{
    // Business logic can inspect the concrete resource data here.
}
```

类型模式负责取得数据，不代替 Kind 语义校验；Kind 筛选也不能代替具体数据类型检查。

## 5. XML 与数据恢复

原 ModelGraphXml 的关系写入已放入各模型 XML；[ModelGraphXml](../src/SharpInference.Graphs/Internal/ModelGraphXml.cs) 保留通过构造回调读取关系记录的公共实现，不认识两种模型的具体节点/边。

- 结构化 XML、安全解析、大小/深度/元素数量限制继续复用现有公开工具。
- 模型侧负责识别具体节点/边，获取业务字段，创建对应派生对象；不采用 CLR 全限定名称反射恢复，不加类型注册框架。
- 根据用户后续授权，逻辑图、执行图和 GraphView 统一使用 XML v4，不读取旧版本。所有节点必须写入 `nodeKind`；依赖必须写入 `edgeKind`；绑定的读写边分别必须写入 `readEdgeKind` / `writeEdgeKind`。手写 fixture 与消费者同步迁移。
- `ResourceKind` 是 C# 属性命名调整，既有 XML 资源角色属性 `kind` 保持不变；其他序列化输出若因属性改名受到影响，显式保持既有字段映射。若必须改变数据协议，先询问用户。
- 自定义类别与默认类别均显式保存，不能静默降级或丢失。此次升级已获用户授权，不保留可选字段回退或旧格式兼容入口。
- 不要求任意用户派生类自动序列化；本轮保证当前具体模型的持久化。扩展模型自行提供其字段读写和校验。
- 保持边身份、绑定身份、声明顺序、ReadWrite 配对、子图共享及来源映射。

## 6. 实施阶段

P0 基线：171 项通过、0 失败、0 跳过。XML v3 无法无损保存自定义类别的问题已向用户说明；用户随后明确授权不考虑旧格式兼容，已解除暂停，实施 XML v4。

| 阶段 | 工作 | 完成依据 |
|---|---|---|
| P0（完成） | 记录现有签名、诊断、XML 和测试基线 | 基线 171 项通过；协议变更已获后续授权 |
| P1（完成） | 基础接口改名，增加 NodeKind / EdgeKind，资源角色改为 ResourceKind，冲突类别改为 ConflictKind | 两个基础接口无业务字段；Core 无修改 |
| P2（完成） | 两套具体查询/业务校验，公共结构校验，更新转换、改写、Pass、Emitter | 不通过类别字符串强制转换；新类别诊断 GR0009 / GR0010 |
| P3（完成） | 模型侧关系写入、公共关系记录读取、XML v4 与 GraphView 同步 | 手写 fixture、自定义类别及安全限制验证通过；旧版本拒绝 |
| P4（完成） | 删除五个业务/标签接口，更新全部消费者 | 旧名称仅在历史文档及类型不存在的断言字符串中保留 |
| P5（完成） | 构建、回归、真实模型、架构文档与交叉表 | 结果见第 9 节 |

依赖关系：P1 依赖 P0；P2、P3 依赖 P1；P4 依赖 P2、P3；P5 依赖 P4。必要的跨文件更新成批进行，不将无法构建的中间状态作为交付。

公共辅助提取另设后续决策点：在 P2 的具体代码形成、调用场景可验证后，再整理候选函数与归属方案并确认；是否在本轮后续实施取决于该方案，不在 P0/P1 预设新的公共扩展 API。

范围包括 Graphs、GraphHelper、现有优化查询与发射调用者、模型构图调用者、测试及文件式基准脚本。项目归属沿用已完成的 [职责整理计划](graphs-responsibility-cleanup-plan.md)。

## 7. 验证与验收

### 7.1 新增针对性回归

- 不依赖五个旧业务接口的自定义派生节点、边，能够参加基础筛选和连接查询。
- 同一派生类型承载不同 Kind，基础层不强制绑定；具体图按自己规则接受或拒绝。
- 同一 Kind 的不同派生类型不会被不安全强制转换，也不会让业务检查被绕过。
- Kind、属性和连接不一致时，具体图 Validate 返回有定位的诊断；合法对象行为不变。
- 资源角色统一通过 `ResourceKind` 访问；本轮涉及的 Graphs、GraphHelper 及相关模型类别属性不存在单独的 `Kind`，XML 原有 `kind` 字段和其他协议映射不受 CLR 属性改名影响。
- 操作与调用的依赖语义、读写查询、绑定顺序和 ReadWrite 配对保持。
- 未知/缺失 Kind 的处理由具体图测试明确；不得通过默认值或静默忽略伪装有效。
- XML v4 的独立 fixture、默认/自定义类别往返、安全限制保持；旧格式明确拒绝。

### 7.2 运行已有验证

优先组合相关测试一次运行，再覆盖消费者构建：

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-restore --filter "FullyQualifiedName~GraphLayeringTests|FullyQualifiedName~NamedGraphConstructionTests|FullyQualifiedName~LogicalGraphDefinitionTests|FullyQualifiedName~GraphSerializationTests|FullyQualifiedName~GraphModelAdapterTests|FullyQualifiedName~ExecutionGraphConverterTests|FullyQualifiedName~ExecutionGraphPostprocessorTests|FullyQualifiedName~GraphViewQueryTests|FullyQualifiedName~VmExecutionGraphEmitterTests" --verbosity minimal
dotnet build .\SharpInference.slnx --no-restore
dotnet build .\scripts\GraphQueryBenchmark.cs
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-build --no-restore --filter "FullyQualifiedName~RwkvTinyExecutionPipelineTests" --verbosity quiet
```

新增测试类加入同一次过滤器。仅在修改依赖清单或遇到缺失依赖时执行 restore；真实模型或设备不可用时如实记录，不把跳过算通过。不默认运行完整性能矩阵。

### 7.3 最终验收

- 只保留含 Kind 的 `IInferenceGraphNode` / `IInferenceGraphEdge` 两个基础接口，不再保留两个旧基础接口名称、五个旧业务接口或替代性业务接口体系。
- Graphs 不定义封闭业务 Kind 词汇，不强制 Kind 与 CLR 类型一一对应。
- 张量、精度、作用域、端口和绑定数据由各具体模型承载。
- 类别属性名称明确区分 `NodeKind`、`EdgeKind`、`ResourceKind`，无含义不明确的单独 `Kind` 属性。
- 逻辑图/执行图原有构造、校验、查询、转换、优化、XML 和数值行为不变。
- 已确认的自定义 Kind 能正确查询和校验；持久化不静默丢失信息。
- Core、项目依赖方向及已确定的归属不变。
- 文档、调用示例和排除测试项目的生产消费者交叉表更新。

## 8. 执行时必须询问的冲突

Kind 使用 string 及提供公开常量已经确认；以下问题发现实际需要时再询问：

1. 是否必须修改 XML schema 才能保存非默认 Kind。
2. 是否需要新公共适配入口或新的访问权限，才能避免查询/编码实现重复。
3. 原有校验规则与允许扩展 Kind 的具体业务需求冲突时，哪些图接受哪些组合。

这些问题不由工具默认选项或“修复编译”的需要自动授权。

### 已确认：自定义类别的 XML 表示

现有 v3 没有保存自定义类别的字段。最初可选属性方案未获授权，不作为实现依据。用户后续明确“不需要考虑向前兼容，做对应修改即可”，据此直接升级 v4，类别字段必填，不读取旧版本。资源角色 `kind` 字段保持独立映射。

实现与架构边界见 [架构设计第 1.2 节](graph-query-design.md#12-推理图分类与具体模型边界)。公共辅助函数提取仍为后续确认事项，本轮不预设或增加 `GetResources` API。

## 9. 实施与验证结果

- 只保留 `IInferenceGraphNode` / `IInferenceGraphEdge`，不增加业务接口、
  类型注册表或适配框架。自定义类别由具体记录保存，并通过具体模型规则校验。
- 原业务查询分别放到 LogicalGraphQueries / ExecutionGraphQueries；
  相邻查询直接复用 Core，保留可见范围、顺序、平行边、自环、绑定配对、
  冲突和区域边界行为。公共工具只做结构检查及现有记录处理。
- 逻辑到执行转换、实例资源复制和内联重映射保留自定义类别；
  模板共享把 NodeKind 纳入分组，避免不同类别被并入同一模板体。
- XML v4 必填类别字段，不保留 v3 读取；资源角色 XML `kind` 独立映射。
  GraphView 同步为 v4，并保留类别到展示详情。
- 全方案构建：0 警告、0 错误；文件式 GraphQueryBenchmark 构建通过。
- 相关回归：212 项通过、0 失败、0 跳过；其中新增 InferenceGraphKindTests
  26 项，覆盖开放类别、最小接口、命名规则、默认类别冲突、未知派生类型、
  XML 必填字段/往返、调用实例、转换、内联与模板分组。
- GraphView 浏览器回归：107 项通过，包括旧版本拒绝和自定义类别详情保留。
- 真实 RWKV：4 项通过、0 跳过，包含 RWKV6/7 CPU 构图发射、
  RWKV7 CPU 及 D3D12 logits 对照。
- 非测试生产消费者语义统计：19 个项目、20 个目标框架配置，编译诊断错误 0；
  当前 77 类型，归属为 Graphs 40、Instructions 7、GraphHelper 30。
  交叉表 77 行 × 11 项目列，共 847 格逐格校验。
  其中 40 个类型存在所属项目之外的直接生产消费者。
  最新会话产物为 `graphs-consumer-matrix-after-kind.csv`，
  证据为 `graph-consumers/consumers-after-kind.json`，旧表作为历史快照保留。
- Graphs.Core、项目依赖与类型归属边界未改；未删除 GraphCallBoundResource。
  公共辅助提取候选与重复证据已写入架构文档，但没有新增 GetResources API。
- 未创建 Git 提交。
