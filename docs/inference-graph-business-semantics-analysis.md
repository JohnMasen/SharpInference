# 推理图业务语义与类型逐项分析

日期：2026-10-18。
状态：基于当前代码的只读分析，不是已批准的重构实施计划。

## 1. 分析目标与范围

分析原则：

> 逻辑图和执行图的结构基础都是节点和边。对于超出结构的业务概念，
> 逐项判断它是两种图共有、部分共有、模型专用，还是图之外的处理流程。
> 先明确语义，再决定公共类型、IInferenceGraph 和通用 Builder 的形状。

本文覆盖以下 **77 个顶层类型**，包括 internal 类型：

| 范围 | 类型数 | 对应章节 |
|---|---:|---|
| Graphs 根命名空间及 Internal 公共支持 | 16 | 第 4 节 |
| Graphs.Logical | 24 | 第 5 节 |
| Graphs.Execution | 23 | 第 6 节 |
| 优化器 GraphHelper 中剩余处理流程 | 7 | 第 7 节 |
| 两种图相关的 Instructions 共享契约 | 7 | 第 8 节 |
| 合计 | 77 | |

Graphs 共 63 个类型、62 个源码文件；GraphDiagnostic 与
GraphDiagnosticSeverity 位于同一文件，其余按当前源码逐项列出。
该范围沿用执行图归属修复后的跟踪口径，不是整个解决方案的类型总数。

不逐项展开 Graphs.Core、VM、后端、具体优化处理器、测试和查看器的所有类型。
它们只作为边界和消费者证据，不意味着这些项目也应接受本轮重构。
IInferenceGraph 和通用 InferenceGraphBuilder 是讨论中的候选设计，当前尚未定义。

### 1.1 阅读约定

- **同一共享实现**：两种图确实使用同一个类型及同一规则，不要求重复给差异例。
- **等价机制、不同类型域**：字段和算法相同，但 CLR 类型或所属模型不同。
  不能称为同一个类型；本文仍给出使用差异例。
- **部分共有**：有共同载荷或规则，也有不能直接合并的差异。
- **专用**：另一种图没有对应类型或没有执行相同处理；用“有／无”的例子说明。
- **处理流程**：消费图，而非图自身必须拥有的能力。
- 表格中的“例”是当前规则的具体场景，不保证每个片段都是独立可运行程序。
  第 9 节的两个 Builder 片段使用当前 API，可以作为对照。
- “建议”与“当前事实”分开表达。分析相同不构成合并授权；
  当前没有检查某条规则，也不表示该规则在理论上不需要。

### 1.2 分析边界

不能根据以下事实单独判断归属：

- 只有一个库外显式消费者；
- 类型名含 Logical、Execution、Model 或 Definition；
- 两个类型字段相似；
- 某段算法使用了搜索；
- 一项业务属性恰好保存在节点或边上。

“公共”也有两层：所有节点和边适用的结构机制，以及本项目两种推理图共有的业务。
后者不应被放进纯结构 Core，也不应自动成为所有自定义推理图的强制契约。

## 2. 当前架构与已完成事项

```text
Graphs.Core：身份、节点、边、视图、遍历、匹配、结构改写
    ↑
Graphs
    ├─ 共享业务支持
    ├─ Logical：逻辑模型及构图、查询、校验、XML
    └─ Execution：执行模型及构图、查询、校验、XML
    ↑
Vm.Optimization：转换、编译组织、优化分析、策略、改写及 VM 发射
```

- 23 个执行图组件已从优化器迁回 Graphs.Execution。
- Graphs 仅直接引用 Graphs.Core 和 Instructions，不引用 Vm 或优化器。
- 三个共享分析结果/类别类型，以及逻辑图的冲突和边界查询，仍在 Graphs。
- 7 个处理流程类型暂留优化器 GraphHelper；“暂留”不等于认定全部是优化算法。
- 本文不重新实施上述迁移，也不变更 XML v4。

依据：[当前架构](graph-query-design.md#12-推理图分类与具体模型边界)、
[执行图归属修复记录](graphs-optimizer-helper-migration-plan.md)。

## 3. 按业务概念的结论

| 概念 | 当前共同部分 | 当前差异 | 结论 |
|---|---|---|---|
| 结构和类别 | 身份、端点、NodeKind、EdgeKind | 各模型支持的 CLR 类型及类别校验 | 公共机制，模型解释 |
| 资源 | Id、Name、ResourceKind、Tensor、BindingKey | 逻辑资源另有声明图 Owner | 公共载荷与声明归属可分开讨论 |
| 资源作用域 | 父图／调用实例 | 共享 Owner 实现绑定逻辑调用身份 | 真正共有，但实现未中性化 |
| 绑定及读写 | 端口、资源、Access、配对身份和边方向 | 具体节点、边类型 | 共有业务，存在重复实现 |
| 操作 | OperationId、属性、最低精度 | 本地身份、注解类型、执行来源追踪 | 载荷共有，整个节点不完全相同 |
| 调用 | 目标图、端口绑定、接口匹配 | 定义集合组织、边界存放位置、来源信息 | 部分共有 |
| 依赖 | 先决节点到后继节点的有向关系 | 逻辑构图推导部分依赖，执行构图不推导 | 关系共有，生成规则专用 |
| 描述性元数据 | 名称、角色、说明、属性 | 目前主要是类型与命名空间不同 | 强共享候选 |
| 编译边界 | 同一枚举及意图 | 图定义／执行图／调用上的位置不同 | 共享元数据，不是执行策略 |
| 来源映射 | 可引用逻辑身份 | 执行侧保存并在改写后累计 | 当前执行侧专用 |
| 校验 | 身份、端口、访问、只读角色、依赖循环等 | 本地图／图集合范围、输出规则及诊断方式 | 共同入口，分层规则 |
| 冲突和区域边界 | 相同资源访问与可达性规则 | 类型识别不同 | 共有分析，不是基础结构 |
| 完成信息 | 可利用依赖图推导 | 逻辑图保存完成汇点和终端写者 | 当前逻辑侧派生结果，不宜强制全图缓存 |
| 存储与指令 | 图提供描述性输入 | VM 发射决定存储和具体实现 | 图外处理，不是通用 Builder 职责 |

## 4. 公共支持：16 个类型

这些类型位于 Graphs 根命名空间或 Internal；“当前位置公共”不保证实现已经中性。

| 编号 | 类型与源码 | 当前职责和共享判断 | 差异例／一致性说明 | 后续判断 |
|---|---|---|---|---|
| G01 | [IInferenceGraphNode](../src/SharpInference.Graphs/IInferenceGraphNode.cs) | 在 Core 节点契约上增加 string NodeKind；同一共享接口 | 一致：两种图都通过该契约暴露身份和类别，不要求 Tensor 或端口 | 保持最小；不能用类别字符串代替业务类型检查 |
| G02 | [IInferenceGraphEdge](../src/SharpInference.Graphs/IInferenceGraphEdge.cs) | 在 Core 边契约上增加 string EdgeKind；同一共享接口 | 一致：端点、边身份和类别契约相同 | 不加入读写绑定字段作为所有边的要求 |
| G03 | [ResourceId](../src/SharpInference.Graphs/ResourceId.cs) | 资源局部名称的值类型，两种图共用 | 一致：`hidden` 都只是资源名，本身不区分父图与调用实例 | 可保留共享；不能承担完整作用域身份 |
| G04 | [GraphBindingOwner](../src/SharpInference.Graphs/GraphBindingOwner.cs) | 父图或直接调用实例；共有语义，但 Call 为逻辑 GraphDefinitionNodeId | 例：逻辑调用 `layer0` 可直接使用其 Id；执行调用 Id 为 ExecutionNodeId，使用公共 Owner 时仍需构造逻辑身份包装 | 优先审查调用实例身份中性化；不要直接改成无约束 object |
| G05 | [ModelResourceNodeId](../src/SharpInference.Graphs/ModelResourceNodeId.cs) | Owner 与 ResourceId 的组合，生成带作用域的结构身份 | 例：两侧都可表示 `resource:instance:layer0:hidden`；但执行侧 Owner 仍间接带入 G04 的逻辑身份类型 | 结构编码共有，作用域类型依赖待修正 |
| G06 | [ModelResourceBinding](../src/SharpInference.Graphs/ModelResourceBinding.cs) | Port、Resource、Access、BindingId；同一数据与配对语义 | 一致：`input → parent/x, Read` 在两种图中都表示该端口读取 x；资源身份的中性化问题来自 G05，而非另有绑定规则 | 是共享业务记录，不等于纯节点/边结构 |
| G07 | [GraphDefinitionResourceKind](../src/SharpInference.Graphs/GraphDefinitionResourceKind.cs) | 七种资源角色，同一枚举供两种图使用 | 一致：Weight/Constant 都是只读，Input/Output/InOut 都用于调用接口；VM 存储映射不是两套图枚举语义 | 名字偏逻辑，概念却共有；是否拆分角色维度需另论，不直接改枚举语义 |
| G08 | [GraphCompileBoundary](../src/SharpInference.Graphs/GraphCompileBoundary.cs) | Auto/Atomic/Transparent 意图，共用枚举 | 例：逻辑 child 的 Atomic 在 child 定义上；转换后同时出现在执行 child 图与父图调用节点上。显式内联目前不以 Atomic 拒绝操作 | 类型共有，放置与消费策略不能混为一谈 |
| G09 | [GraphDiagnosticSeverity](../src/SharpInference.Graphs/GraphDiagnostic.cs) | Warning/Error，同一共享级别 | 一致：模型不同不改变 Error 与 Warning 的含义 | 可作为共同验证结果的一部分 |
| G10 | [GraphDiagnostic](../src/SharpInference.Graphs/GraphDiagnostic.cs) | 级别、代码、消息及图/节点/资源/端口定位，共同承载诊断 | 例：逻辑输出未写会得到 LG0020；执行图当前没有对应输出检查。差异在诊断生产规则，不在该 record 的字段 | 共享结果，不能由此推导两种 Validate 完全一致 |
| G11 | [ModelGraphBoundary](../src/SharpInference.Graphs/ModelGraphBoundary.cs) | 选定操作集的输入/输出资源结果，泛型共享容器 | 一致：两套查询都按接口角色和区域外读写者筛选，保持首访问顺序；TNode 随模型变化 | 保留为可选分析结果，不要求每个图对象缓存 |
| G12 | [ModelResourceConflict](../src/SharpInference.Graphs/ModelResourceConflict.cs) | Resource、First、Second、ConflictKind、Ordered，共同结果 | 一致：First/Second 都按声明顺序，不代表执行顺序；Ordered 都表示任一方向依赖可达 | 不能把声明先后解释为已经安排好的执行顺序 |
| G13 | [ModelResourceConflictKind](../src/SharpInference.Graphs/ModelResourceConflictKind.cs) | RAW/WAR/WAW flags，同一枚举 | 一致：A 写、B 读时相对声明顺序标为 RAW；是否有序另看 Ordered | 分类共有，不是某个优化策略 |
| G14 | [ModelGraphConstruction](../src/SharpInference.Graphs/Internal/ModelGraphConstruction.cs) | 结构验证、绑定展开、错误抛出、张量比较，混合多个层次 | 例：逻辑 AddNode 的配置结果和执行 AddOperation 的绑定列表都经 ExpandBindings；此函数不负责前者的依赖推导，也不负责后者的目标图查找 | 按职责拆看；不能把整个类视为纯结构 Builder |
| G15 | [ModelGraphXml](../src/SharpInference.Graphs/Internal/ModelGraphXml.cs) | 通过构造回调恢复两种图的绑定/依赖边，共享实现 | 例：同一 Binding XML 经逻辑回调得到 LogicalResourceReadEdge，经执行回调得到 ExecutionResourceReadEdge；两者读取 instance 时却都调用逻辑身份及逻辑 RequireId | 公共关系协议与逻辑专用依赖应区分 |
| G16 | [GraphXmlSupport](../src/SharpInference.Graphs/Internal/GraphXmlSupport.cs) | 安全读写、字段与元素检查、张量/精度/属性序列化；同一工具 | 一致：两侧禁用 DTD/外部实体，读取限制为 16 MiB、100000 元素、深度 64；张量、精度的公共 XML 规则相同 | 属于持久化支持，不放进构图核心契约 |

## 5. 逻辑图：24 个类型

### 5.1 模型、身份和元数据

| 编号 | 类型与源码 | 当前职责与执行侧对应 | 差异例／一致性说明 | 分析 |
|---|---|---|---|---|
| L01 | [LogicalGraphDefinition](../src/SharpInference.Graphs/Logical/LogicalGraphDefinition.cs) | 含 Structure、直接子 Definitions、逻辑元数据、诊断及完成信息；执行侧分为 ExecutionGraph 和 Program | 例：逻辑 root 直接持有 child 定义；执行 root 单图不持有 child 对象，Program 存放 root/child 并解析调用。逻辑 root 还保存 TerminalWriters | 图本体、图集合关系与派生缓存混在一个对象中，应分别判断 |
| L02 | [GraphDefinitionId](../src/SharpInference.Graphs/Logical/GraphDefinitionId.cs) | 逻辑定义身份；对应 ExecutionGraphId | 例：逻辑 `block` 是源定义身份；执行优化抽出 `shared-block` 后，执行目标身份可以不同，不要求伪造同名逻辑定义 | 包装结构相同，身份域不同，不应因同为 string 就抹平 |
| L03 | [GraphDefinitionNodeId](../src/SharpInference.Graphs/Logical/GraphDefinitionNodeId.cs) | 逻辑操作/调用的局部身份；对应 ExecutionNodeId | 例：逻辑节点 a、b 融合成执行节点 fused；fused 的 SourceMap 仍可同时保存 a、b 两个逻辑身份 | 源身份与变换后身份有独立价值 |
| L04 | [GraphAnnotation](../src/SharpInference.Graphs/Logical/GraphAnnotation.cs) | 描述性元数据；对应 GraphArchitecture | 字段和当前转换含义一致：Name、Role、Description、Attributes；转换器逐项复制。类型身份仍不同 | 强共享候选；不能仅凭 Architecture 命名假定它包含调度策略 |
| L05 | [GraphDefinitionResource](../src/SharpInference.Graphs/Logical/GraphDefinitionResource.cs) | 资源公共载荷加声明图 Owner；对应 ExecutionGraphResource | 例：child 声明 hidden，实例调用 layer0 引用它时，声明 Owner 仍是 child，作用域 Owner 则是 layer0；执行资源没有声明图 Owner 字段 | 公共描述与逻辑声明归属可分开分析，不能直接删 Owner |
| L06 | [LogicalGraphNode](../src/SharpInference.Graphs/Logical/LogicalGraphNode.cs) | 逻辑操作、调用和资源的基类；对应 ExecutionGraphStructureNode | 例：逻辑结构集合包含 GraphOperationNode 和 LogicalResourceNode；执行集合包含对应执行类型。GraphId/NodeKind 机制相同，但现有验证不接受另一模型类型混入 | 基类机制等价，具体模型域不同 |
| L07 | [GraphDefinitionNode](../src/SharpInference.Graphs/Logical/GraphDefinitionNode.cs) | 逻辑操作和调用共同基类，带 Id、Annotation；对应 ExecutionGraphNode | 例：逻辑 copy 只需源 Id 和可选注解；执行 copy 基类还要求 SourceMap，局部 Id 也换成 ExecutionNodeId | 操作/调用与资源的区分有用，不表示两种基类字段完全相同 |
| L08 | [GraphOperationNode](../src/SharpInference.Graphs/Logical/GraphOperationNode.cs) | 操作身份、属性、最低精度；对应 ExecutionOperationNode | 例：逻辑 a、b 分别声明；执行融合后一个 fused 操作承载组合来源 a、b。共享 Operation/Attributes/Requirements，不共享全部身份与追踪字段 | 优先审查操作载荷，而非强行合并整个节点 |
| L09 | [GraphCallNode](../src/SharpInference.Graphs/Logical/GraphCallNode.cs) | 指向 GraphDefinitionId；对应 ExecutionGraphCallNode | 例：逻辑 call 指向 child，不独立存 CompileBoundary；执行 call 指向执行 child，并显式保存边界和来源 | 调用概念共有，目标域和附加信息不同 |
| L10 | [LogicalResourceNode](../src/SharpInference.Graphs/Logical/LogicalResourceNode.cs) | 作用域资源身份＋逻辑资源声明；对应 ExecutionResourceNode | 例：两个 layer 调用可各自引用 child/hidden；其 scoped Id 不同，底层声明 Owner 都是 child。执行资源节点只保留 scoped Id 与执行资源描述 | 节点机制相似，声明载荷不同 |
| L11 | [LogicalGraphEdge](../src/SharpInference.Graphs/Logical/LogicalGraphEdge.cs) | 逻辑边共同身份、端点、类别；对应 ExecutionGraphEdge | 例：逻辑资源读边放入 Graph<LogicalGraphNode, LogicalGraphEdge>；不能原样作为执行图边加入。字段机制相同，类型域不同 | 可共享机制，但是否保留模型隔离需要权衡 |
| L12 | [LogicalDependencyEdge](../src/SharpInference.Graphs/Logical/LogicalDependencyEdge.cs) | 逻辑操作/调用间先后关系；对应 ExecutionDependencyEdge | 例：A 唯一写 tmp、B 纯读 tmp 时，逻辑 Builder 可自动补 A→B；执行 Builder 不会仅凭两条访问边自动生成该依赖 | 单条边语义相同，生成阶段规则不同，见 9.1 |
| L13 | [LogicalResourceReadEdge](../src/SharpInference.Graphs/Logical/LogicalResourceReadEdge.cs) | 资源→操作/调用，Port、BindingId；对应 ExecutionResourceReadEdge | 例：逻辑 B 读 tmp 的边与其他访问共同参与依赖推导；执行 B 的读边仍只是读取关系，缺少先后边不会被 Builder 自动补全 | 字段及读语义等价，所属构图处理不同 |
| L14 | [LogicalResourceWriteEdge](../src/SharpInference.Graphs/Logical/LogicalResourceWriteEdge.cs) | 操作/调用→资源，Port、BindingId；对应 ExecutionResourceWriteEdge | 例：逻辑 A 写 tmp 可成为后续读取的唯一生产者；执行 A 的写边不单独承诺后续 B 已排序，须显式依赖 | 不把读写边误当先后依赖边 |
| L15 | [LogicalGraphKinds](../src/SharpInference.Graphs/Logical/LogicalGraphKinds.cs) | 六个约定类别常量；对应 ExecutionGraphKinds | 常量值完全一致：operation、call、resource、dependency、resource-read、resource-write；两侧均允许合法自定义字符串 | 值重复是事实；是否统一常量不能代替模型合法性校验 |

### 5.2 构图、校验、分析与持久化

| 编号 | 类型与源码 | 当前职责与执行侧对应 | 差异例／一致性说明 | 分析 |
|---|---|---|---|---|
| L16 | [GraphDefinitionBuilder](../src/SharpInference.Graphs/Logical/GraphDefinitionBuilder.cs) | 声明收集、名称解析、实例资源展开、依赖补全、建图；对应 ExecutionGraphBuilder | 例：使用 AddNode 回调配置端口；Build 会补全某些依赖。执行侧接收已构造的操作及绑定列表，不做同样推导 | 通用收集流程与逻辑规范化应分开；RegisterDefinition 的 XML 比较另有耦合 |
| L17 | [GraphOperationNodeBuilder](../src/SharpInference.Graphs/Logical/GraphOperationNodeBuilder.cs) | 操作/调用的端口、属性、精度、依赖配置；执行侧无同型配置对象 | 例：逻辑通过 node.AddInput 和 node.RequirePrecision 配置；执行侧先 new ExecutionOperationNode，再将绑定列表传入 AddOperation | 是逻辑 DSL/便利入口，不应直接成为所有泛型 Builder 必需的配置模型 |
| L18 | [LogicalGraphValidation](../src/SharpInference.Graphs/Logical/LogicalGraphValidation.cs) | 具体类型/类别、端点角色、作用域、绑定配对等；对应 ExecutionGraphValidation | 例：相同 `operation` 字符串配未知逻辑节点仍报 GR0009；执行侧同规则检查未知执行节点。合法端点分别要求 GraphDefinitionNode 与 ExecutionGraphNode | 规则框架基本一致，CLR 模型识别不同；不能按 Kind 强转 |
| L19 | [LogicalGraphDefinitionValidator](../src/SharpInference.Graphs/Logical/LogicalGraphDefinitionValidator.cs) | 声明归属、调用接口、只读资源、循环及输出写者分析；执行侧由 Graph.Validate 与 Program 分担 | 例：输出 y 未被写入报 LG0020；执行本地 Validate 当前没有对应输出检查。逻辑写入其他调用实例资源还可产生 LGW003 | 共同规则、专用规则、派生分析需拆开，不宜整体提升为公共校验器 |
| L20 | [LogicalGraphAnalysis](../src/SharpInference.Graphs/Logical/LogicalGraphAnalysis.cs) | internal 完成汇点、终端写者和诊断结果；执行侧无对应 record | 例：A→B、B 写 output 时得到 CompletionSinks=[B]、TerminalWriters[output]=[B]；执行图可查询类似事实，但不保存同名结果 | 区分按需分析和验证所需分析，不先要求所有图缓存 |
| L21 | [LogicalGraphQueries](../src/SharpInference.Graphs/Logical/LogicalGraphQueries.cs) | 依赖/数据流视图、绑定、读写者、冲突、边界；对应 ExecutionGraphQueries | 例：查询逻辑资源得到逻辑节点；查询执行资源得到执行节点。相同显式结构下绑定、冲突、边界规则相同；Builder 生成的依赖不同可能使 Ordered 不同 | 是共享算法候选，不是所有方法均适合接口化 |
| L22 | [LogicalGraphXml](../src/SharpInference.Graphs/Logical/LogicalGraphXml.cs) | 根定义和可达 Definitions 的 v4 读写；对应 ExecutionGraphXml | 例：Root 使用 definition，资源写声明 owner，读取时重建直接子 Definitions；执行 XML Root 使用 graph，还包含 SourceMap/Architecture，返回 Program | 安全工具可共用，模型 envelope 和恢复流程不能简单合成同一逻辑 |
| L23 | [LogicalGraphDefinitionRecord](../src/SharpInference.Graphs/Logical/LogicalGraphDefinitionRecord.cs) | internal XML 中间记录，等被引用定义恢复后再创建逻辑图；执行侧无同型记录 | 例：XML 先出现调用 child 的 root，读取先暂存 root 的结构，递归恢复 child 后构造 root；执行侧先得到图集合，再由 Program 解析目标 | 是逻辑 XML 恢复实现，不应暴露到 IInferenceGraph |
| L24 | [GraphCallBoundResource](../src/SharpInference.Graphs/Logical/GraphCallBoundResource.cs) | internal 声明资源＋调用作用域记录；当前生产源码没有使用点 | 例：可表达 Resource.Owner=child、Owner.Instance=layer0；执行侧目前使用 ExecutionResourceNode/ModelResourceNodeId 表达实例资源，没有对应记录 | 当前为未使用类型，不应据此设计公共抽象，也不在本文擅自删除 |

## 6. 执行图：23 个类型

本节逐项列出，避免仅在逻辑图侧提到对应名称而遗漏执行侧职责。

| 编号 | 类型与源码 | 当前职责与逻辑侧对应 | 差异例／一致性说明 | 分析 |
|---|---|---|---|---|
| E01 | [ExecutionGraph](../src/SharpInference.Graphs/Execution/ExecutionGraph.cs) | 单图结构、边界、架构、来源及本地规则；对应逻辑定义的一部分 | 例：单独创建 root 中指向 child 的调用，是否存在 child 要在 Program 中检查；逻辑定义创建时已有直接子 Definitions | 通用图接口需要明确本地验证范围 |
| E02 | [ExecutionGraphProgram](../src/SharpInference.Graphs/Execution/ExecutionGraphProgram.cs) | 图集合、派生调用关系、唯一根、可达调用 DAG 和接口检查；逻辑侧没有独立同型容器 | 例：root/child 两图组成 Program，另加不可达 orphan 可导致多根或不可达错误；逻辑侧从根 Definitions 及 XML 恢复流程组织集合 | 图集合不是一个普通节点/边图的同义词，不能强塞单图接口 |
| E03 | [ExecutionGraphCallRelation](../src/SharpInference.Graphs/Execution/ExecutionGraphCallRelation.cs) | 从执行调用节点派生的调用点记录，含 caller、callee、call；逻辑侧无同型公共记录 | 例：root 的 c1、c2 都调用 child，Program 有两个调用关系；逻辑侧保留两个 GraphCallNode 与一个共享 child 定义，未暴露同型 Calls 表 | 是跨图关系索引，不是每次都需要用户手工添加的结构边 |
| E04 | [ExecutionGraphId](../src/SharpInference.Graphs/Execution/ExecutionGraphId.cs) | Program 内执行图身份；对应 GraphDefinitionId | 例：优化新建 shared-template 图可有新执行身份，而没有对应逻辑定义；Source 可以为空 | 与逻辑源图身份应区分，即使默认转换保留同名字符串 |
| E05 | [ExecutionNodeId](../src/SharpInference.Graphs/Execution/ExecutionNodeId.cs) | 执行操作/调用身份；对应 GraphDefinitionNodeId | 例：内联后产生 c/a，融合后产生 fused；逻辑来源仍可保存原 a、b | 当前身份与来源身份不可互相替代 |
| E06 | [GraphArchitecture](../src/SharpInference.Graphs/Execution/GraphArchitecture.cs) | 描述性元数据；对应 GraphAnnotation | 字段和当前转换语义一致，含 Name、Role、Description、Attributes；它不定义 GPU 分组或指令选择 | 强共享候选，不因命名就认定执行专用业务 |
| E07 | [ExecutionGraphResource](../src/SharpInference.Graphs/Execution/ExecutionGraphResource.cs) | 共享资源字段，不带逻辑声明图 Owner；对应 GraphDefinitionResource | 例：提取执行子图时可为临时资源建立新的资源描述，不需要绑定某个逻辑声明 Owner | 共享资源载荷应与逻辑声明约束分开 |
| E08 | [ExecutionGraphStructureNode](../src/SharpInference.Graphs/Execution/ExecutionGraphStructureNode.cs) | 全部执行节点基类；对应 LogicalGraphNode | 例：同一集合接收 ExecutionOperationNode 和 ExecutionResourceNode，不接收逻辑节点；GraphId/NodeKind 机制等价 | 区分共享结构契约与模型类型域 |
| E09 | [ExecutionGraphNode](../src/SharpInference.Graphs/Execution/ExecutionGraphNode.cs) | 操作/调用基类，带执行 Id、Source、Architecture；对应 GraphDefinitionNode | 例：融合节点能带 a、b 两个逻辑来源；逻辑操作/调用基类没有这种 SourceMap | 附加来源是执行侧能力，不宜强制所有推理节点实现 |
| E10 | [ExecutionOperationNode](../src/SharpInference.Graphs/Execution/ExecutionOperationNode.cs) | 操作载荷加来源及架构；对应 GraphOperationNode | 例：逻辑 Add 与 Multiply 可改写为新执行操作，OperationId/属性/精度随新语义决定，SourceMap 可累计原节点 | 操作核心载荷共享，不保证阶段间操作集合一一对应 |
| E11 | [ExecutionGraphCallNode](../src/SharpInference.Graphs/Execution/ExecutionGraphCallNode.cs) | 执行目标、独立边界、来源及架构；对应 GraphCallNode | 例：模板共享可新建调用 target=template，SourceMap 指向原操作；原逻辑图可能没有这个调用 | 调用机制部分共享，执行调用可由改写产生 |
| E12 | [ExecutionResourceNode](../src/SharpInference.Graphs/Execution/ExecutionResourceNode.cs) | scoped Id＋执行资源描述；对应 LogicalResourceNode | 例：同为 instance/layer0/hidden，执行侧没有逻辑资源声明 Owner，匹配目标资源由 Program 完成 | 区分 scoped 身份与声明身份 |
| E13 | [ExecutionGraphEdge](../src/SharpInference.Graphs/Execution/ExecutionGraphEdge.cs) | 执行边基类；对应 LogicalGraphEdge | 例：执行依赖和执行读边共享该基类；逻辑边即使端点字符串相同也不是此类型 | 机制相同，CLR 模型域不同，不据此强制合并 |
| E14 | [ExecutionDependencyEdge](../src/SharpInference.Graphs/Execution/ExecutionDependencyEdge.cs) | 显式先后关系；对应 LogicalDependencyEdge | 例：手工构图的 A 写 tmp、B 读 tmp 若未传 dependencies，就不会自动多出 A→B | 依赖语义共有，生成策略不同 |
| E15 | [ExecutionResourceReadEdge](../src/SharpInference.Graphs/Execution/ExecutionResourceReadEdge.cs) | 资源→操作/调用的读取关系；对应 LogicalResourceReadEdge | 例：B 读 tmp 的执行边被用于恢复绑定，但不触发逻辑 Builder 的生产者依赖补全 | 读边本身语义一致，构图处理与类型域不同 |
| E16 | [ExecutionResourceWriteEdge](../src/SharpInference.Graphs/Execution/ExecutionResourceWriteEdge.cs) | 操作/调用→资源的写入关系；对应 LogicalResourceWriteEdge | 例：与同 BindingId 读边组合得到 ReadWrite，但执行构图仍需显式提供必要先后依赖 | 配对规则共有，不应从数据访问擅自推导所有调度 |
| E17 | [ExecutionGraphKinds](../src/SharpInference.Graphs/Execution/ExecutionGraphKinds.cs) | 执行常量集合；对应 LogicalGraphKinds | 六个常量值完全相同，均不构成封闭类别全集 | 共享常量候选，不等于统一支持所有派生 CLR 类型 |
| E18 | [ExecutionGraphBuilder](../src/SharpInference.Graphs/Execution/ExecutionGraphBuilder.cs) | 接收资源、具体操作/调用、绑定和显式依赖并建图 | 例：先 new ExecutionOperationNode 再 AddOperation；逻辑侧回调构造操作并自动推导部分依赖 | 共享结构 Builder 可承接物化流程，不能照搬逻辑 DSL |
| E19 | [ExecutionGraphValidation](../src/SharpInference.Graphs/Execution/ExecutionGraphValidation.cs) | 执行具体类型、类别、端点、作用域、绑定结构检查 | 例：依赖两端必须是 ExecutionGraphNode；逻辑校验要求 GraphDefinitionNode。未知派生执行节点即使类别为 operation 仍报错 | 公共规则与模型角色识别分开，不复制所有业务检查到基础层 |
| E20 | [ExecutionGraphQueries](../src/SharpInference.Graphs/Execution/ExecutionGraphQueries.cs) | 执行类型上的依赖/数据流、绑定、冲突和边界查询 | 例：与逻辑图节点/边一一对应且依赖一致时，冲突标志与 Ordered 一致；执行手工漏传依赖时，Ordered 可不同 | 算法共有不代表输入图一定等价，不应添加“执行阶段默认有序”的假设 |
| E21 | [ExecutionGraphXml](../src/SharpInference.Graphs/Execution/ExecutionGraphXml.cs) | v4 Program、Graphs、Source、SourceMap、Architecture 读写 | 例：Root graph=root，返回 Program；逻辑 Root definition=root，返回根定义且恢复子定义集合 | 安全限制与关系协议共有，根对象与恢复职责不同 |
| E22 | [ExecutionGraphSource](../src/SharpInference.Graphs/Execution/ExecutionGraphSource.cs) | 保存来源逻辑定义身份；逻辑侧无对应来源包装 | 例：转换后执行 root 的 Source.Definition=root；新提取的执行图可没有来源定义，原逻辑图不需要“来源于自己”的字段 | 是阶段间溯源，不应机械推广为所有图必需属性 |
| E23 | [ExecutionSourceMap](../src/SharpInference.Graphs/Execution/ExecutionSourceMap.cs) | 多个逻辑节点身份及可选 Transformation | 例：融合 a、b 后 SourceMap=[a,b]，标签说明融合；逻辑节点自身只有它的局部身份 | 是执行来源/变换追踪，区别于当前执行节点 Id |

## 7. 优化器 GraphHelper：7 个处理流程类型

这些类型不组成第三种图模型；它们连接或处理逻辑图和执行图。
下面的差异例说明输入端与输出端的职责，不把“逻辑图没有同名类型”误判为缺失。

| 编号 | 类型与源码 | 职责 | 两种图之间的具体例子 | 边界判断 |
|---|---|---|---|---|
| P01 | [IExecutionGraphConverter](../src/SharpInference.Vm.Optimization/GraphHelper/IExecutionGraphConverter.cs) | 从逻辑根定义与编译上下文得到执行 Program 的契约 | 逻辑 root/child 通过一次 Convert 形成含两个执行图的 Program；不是给逻辑图对象原地加“执行”标志 | 转换流程，不是图的基础方法 |
| P02 | [DefaultExecutionGraphConverter](../src/SharpInference.Vm.Optimization/GraphHelper/DefaultExecutionGraphConverter.cs) | 拷贝结构、资源载荷、类别、注解并建立来源映射 | 逻辑资源的 Owner 不复制到执行资源；逻辑 child 边界复制到执行图及对应调用；单逻辑节点建立单来源映射 | 转换揭示共享字段，也揭示真实差异；默认转换不执行融合优化 |
| P03 | [ExecutionGraphCompiler](../src/SharpInference.Vm.Optimization/GraphHelper/ExecutionGraphCompiler.cs) | 调用转换器，按顺序执行后处理，并重建 Program 验证关系 | 输入逻辑 a、b，转换后后处理器可融合成执行 fused；零后处理器也能得到执行 Program | 编译组织流程，不应成为构造普通执行图的必经入口 |
| P04 | [ExecutionGraphCompilationContext](../src/SharpInference.Vm.Optimization/GraphHelper/ExecutionGraphCompilationContext.cs) | 模型/后端身份、属性、选项的只读快照 | 同一逻辑图可用 cpu-vm 或 d3d12-vm 上下文；当前默认转换的拷贝过程并不因此自动生成 CPU/GPU 专用指令 | 编译输入，不属于纯图结构；不应因存在 Backend 字段声称默认转换已做后端选择 |
| P05 | [ExecutionGraphPostprocessContext](../src/SharpInference.Vm.Optimization/GraphHelper/ExecutionGraphPostprocessContext.cs) | 原逻辑根、编译上下文、阶段序号/名称 | 第二个处理器看到的执行 Program 可能已改写，但 SourceGraph 仍是最初逻辑根，不是前一阶段执行图 | 阶段上下文，不能放进两种图的共同必需状态 |
| P06 | [IExecutionGraphPostprocessor](../src/SharpInference.Vm.Optimization/GraphHelper/IExecutionGraphPostprocessor.cs) | 当前执行 Program 到下一执行 Program 的变换契约 | 原逻辑图不变，处理器删除执行图冗余依赖或复用临时资源，返回新 Program | 优化/后处理流程，不要求逻辑图实现 Process |
| P07 | [ExecutionGraphRewriter](../src/SharpInference.Vm.Optimization/GraphHelper/ExecutionGraphRewriter.cs) | 调用替换、融合、提取、内联等模型级改写机制 | 逻辑 root 保留原 child 调用；执行内联后调用消失，出现 c/a 等节点，保留来源并重映射身份 | 暂留处理层；后续区分公共改写机制与优化策略，不在本文整体搬迁 |

## 8. Instructions：7 个共享契约

这些契约已在独立的 Instructions 程序集中。两种图直接使用同一定义，
不是各自实现两套契约；不因新增 IInferenceGraph 而迁回或复制。

| 编号 | 类型与源码 | 当前语义 | 两种图是否一致 | 分析 |
|---|---|---|---|---|
| C01 | [ElementType](../src/SharpInference.Instructions/ElementType.cs) | Byte、Int32、UInt32、Float16、Float32 标量表示 | 一致：Float32 在两侧都不因阶段改变含义 | 是张量/数值业务，不是所有节点必备属性 |
| C02 | [TensorDescriptor](../src/SharpInference.Instructions/TensorDescriptor.cs) | 元素类型、正维度列表、布局文本 | 一致：`Float32 [4] dense` 的描述含义相同；默认转换保留描述 | 构造器检查形状/布局，但不验证 ElementType 枚举合法性；模型另有检查 |
| C03 | [ResourceAccess](../src/SharpInference.Instructions/ResourceAccess.cs) | Read、Write、ReadWrite | 一致：读写绑定合并规则相同 | 不等于 VM 存储作用域，也不表示依赖顺序 |
| C04 | [PrecisionRequirement](../src/SharpInference.Instructions/PrecisionRequirement.cs) | 最低算术与累加精度 | 一致：两者都独立要求 Float16/Float32，不强制二者的大小关系；默认转换保留 | 共同操作载荷，不是基础图类别接口的成员 |
| C05 | [NumericTypeCompatibility](../src/SharpInference.Instructions/NumericTypeCompatibility.cs) | 浮点类型判断、非空序列最高精度等 | 一致：Float16 与 Float32 求最高均得 Float32；空序列或非浮点按同一规则拒绝 | 公共数值工具，不属于 Builder 本体 |
| C06 | [OperationId](../src/SharpInference.Instructions/OperationId.cs) | 操作契约名称和版本 | 一致：`core.copy@1` 在两侧指同一操作契约；值类型自身不验证名称 | 操作契约身份不同于图节点身份，不能混用 |
| C07 | [PrimitiveOperations](../src/SharpInference.Instructions/PrimitiveOperations.cs) | 后端中性的标准操作身份集合 | 一致：Copy、Add 等常量含义相同，默认转换不换成具体 CPU/GPU 指令 | 图描述操作，发射流程选择实现；常量不是封闭操作白名单 |

## 9. 关键差异的展开例子

### 9.1 相同读写声明不等于相同最终依赖

逻辑图：

```csharp
var tensor = new TensorDescriptor(ElementType.Float32, [4]);
var logical = LogicalGraphDefinition.Define("copy-chain", builder =>
{
    var input = builder.AddInput("input", tensor);
    var temporary = builder.AddResource("tmp", "tmp", tensor);
    var output = builder.AddOutput("output", tensor);
    builder.AddNode("a", PrimitiveOperations.Copy, node =>
    {
        node.AddInput("input", input);
        node.AddOutput("output", temporary);
    });
    builder.AddNode("b", PrimitiveOperations.Copy, node =>
    {
        node.AddInput("input", temporary);
        node.AddOutput("output", output);
    });
});
```

虽然未显式 DependsOn，逻辑 Builder 发现 tmp 的唯一其他写入者是 a、
b 是不带配对写边的读取者，会补 a→b 依赖。

执行图对照：

```csharp
var builder = new ExecutionGraphBuilder(new("copy-chain"));
builder.AddResource(new(new("input"), "input", GraphDefinitionResourceKind.Input, tensor));
builder.AddResource(new(new("tmp"), "tmp", GraphDefinitionResourceKind.Temporary, tensor));
builder.AddResource(new(new("output"), "output", GraphDefinitionResourceKind.Output, tensor));
var a = new ExecutionOperationNode(new("a"), PrimitiveOperations.Copy,
    new Dictionary<string, string>(), new(ElementType.Float32, ElementType.Float32), new([]));
var b = new ExecutionOperationNode(new("b"), PrimitiveOperations.Copy,
    new Dictionary<string, string>(), new(ElementType.Float32, ElementType.Float32), new([]));
builder.AddOperation(a,
[
    new("input", new(GraphBindingOwner.Parent, new("input")), ResourceAccess.Read, "a:input"),
    new("output", new(GraphBindingOwner.Parent, new("tmp")), ResourceAccess.Write, "a:output"),
]);
builder.AddOperation(b,
[
    new("input", new(GraphBindingOwner.Parent, new("tmp")), ResourceAccess.Read, "b:input"),
    new("output", new(GraphBindingOwner.Parent, new("output")), ResourceAccess.Write, "b:output"),
], dependencies: [a.Id]);
var execution = builder.Build();
```

这里显式传入 a.Id 才生成相同的依赖。若省略该参数，执行 Builder 不会自动推导。
这不是推荐省略依赖；恰恰说明当前执行校验不负责证明所有数据访问都已安全排序。
默认转换已有逻辑图时会复制其依赖边，不会丢掉前一侧已经推导出的 a→b。

影响类型：L12–L14、L16、L17、L21、E14–E16、E18、E20。

### 9.2 资源声明身份与调用作用域不是同一个 Owner

```text
逻辑定义 child 声明资源 hidden：
    Resource.Owner = GraphDefinitionId("child")

父图两次调用 child：
    layer0 的资源身份 = instance/layer0/hidden
    layer1 的资源身份 = instance/layer1/hidden
    两个节点底层声明的 Resource.Owner 仍然都是 child

执行侧：
    保留两个不同 scoped 身份
    ExecutionGraphResource 没有逻辑声明 Owner
    Program 通过调用目标核对被引用资源
```

因此不能为了共享资源描述，直接把逻辑声明 Owner 等同于父图/实例作用域。
公共 GraphBindingOwner 使用逻辑调用身份包装，则是另一个可分开修复的问题。

影响类型：G04、G05、L05、L10、L24、E07、E12。

### 9.3 本地图验证不等于跨图验证

```text
逻辑 root：
    Definitions 包含 child
    校验 root 时能检查 child 的公开端口与张量

执行 root：
    单图有调用 Target=child，但没有 child 对象
    root.Validate() 不能核实 child 的接口
    Program([root, child]) 才具有目标集合及跨图验证上下文
```

如果 child 要求 Float32[4]、父图绑定 Float32[8]：
逻辑侧可在逻辑定义校验时报 LG0025；
执行侧在 Program 的接口校验时报 EP0005。

影响类型：L01、L09、L19、E01–E03、E11、P01、P03。

### 9.4 相同输出声明，当前业务检查不同

两侧均声明 Output y，但没有写 y 的节点：

- 逻辑图验证会给出 LG0020，严格 Build 拒绝。
- 执行图本地当前没有对应“所有输出都必须被写”的规则；
  在没有其他错误时，可以通过其现有检查。

这是当前实现差异，不是新的设计建议。
引入共同 Validate 入口时，必须明确是否保持、补齐或改变该规则，
不能以统一签名为由悄悄改变行为。

影响类型：G10、L19、L20、E01、E19。

### 9.5 执行身份与逻辑来源可以多对一

```text
逻辑节点：a、b
默认转换：执行 a → 来源[a]，执行 b → 来源[b]
融合之后：执行 fused → 来源[a,b]，可附变换标签
```

不能以 OperationId 或相同名称代替节点身份，也不能把执行 Id 当成逻辑来源。
执行端可以产生没有对应单一逻辑节点的新调用或新执行子图。

影响类型：L02、L03、L07、L08、E04、E05、E09–E11、E22、E23、P07。

### 9.6 描述性元数据相同，编译边界存放位置不同

GraphAnnotation 与 GraphArchitecture 都可表示：

```text
Name=Layer 0, Role=attention, Description=..., Attributes={layer:0}
```

当前转换逐项复制这四项，没有在此加入调度或设备决策。
它们是共享候选。

但 GraphCompileBoundary 的对象位置不同：

```text
逻辑 child.CompileBoundary = Atomic
    ↓ 默认转换
执行 child.DefaultCompileBoundary = Atomic
父执行图中的 call.CompileBoundary = Atomic
```

Atomic 当前是保留的意图，不是显式内联必须拒绝的权限标志。

影响类型：G08、L04、L09、E06、E11、P02。

### 9.7 冲突算法相同，Ordered 依赖输入图而非阶段名称

A 写 tmp、B 读 tmp，两者声明顺序为 A、B：

- 两种图都产生 RAW 冲突。
- 有 A→B 或 B→A 的可见依赖路径时，两种图都令 Ordered=true。
- 没有路径时，两种图都令 Ordered=false。
- 逻辑 Builder 恰好推导了 A→B，而手工执行图没有添加时，
  两种查询结果不同是输入结构不同，不是算法语义不同。

同样，边界资源分析不是任意“跨选区边”：
选区外生产/消费关系和 Input/InOut、Output/InOut 角色都参与判断。
不能直接用纯 Core 割边查询替换而改变现有规则。

影响类型：G11–G13、L21、E20。

### 9.8 XML 安全机制相同，根对象和业务字段不同

```xml
<!-- 逻辑图 envelope 示意，不是完整可反序列化 fixture -->
<Graph kind="LogicalGraphDefinition" version="4">
  <Root definition="root" />
  <Definitions>...</Definitions>
</Graph>

<!-- 执行图 envelope 示意，不是完整可反序列化 fixture -->
<Graph kind="ExecutionGraphProgram" version="4">
  <Root graph="root" />
  <Graphs>...</Graphs>
</Graph>
```

逻辑侧保存资源声明 owner、重建直接子定义并返回根定义。
执行侧保存 Source、SourceMap、Architecture 等，返回 Program。
两侧都使用共同的关系读取工具和受限 XML 基础工具。

因此应共享协议中真正相同的部分，而不是要求所有图实现同一个固定 envelope。
公共关系读取中调用逻辑 RequireId，是实现依赖，不是 XML 必须属于逻辑图的理由。

影响类型：G15、G16、L22、L23、E21。

### 9.9 共有资源角色不等于共有存储分配算法

两侧资源角色都可以是 State。图记录“这是状态资源”；
目前 [VM 发射器](../src/SharpInference.Vm.Optimization/VmExecutionGraphEmitter.cs)
再决定：

```text
Weight / Constant                  → Global
State / Input / Output / InOut      → Session
Temporary                          → Local
```

该映射是当前 VM 发射规则，不应倒过来要求通用节点/边 Builder 理解 VM 存储。
资源复用、安全性证明和具体指令选择同样属于图的消费者。

影响类型：G07、L05、E07、C01–C07。

### 9.10 未使用类型与专用中间记录不应成为抽象设计依据

- GraphCallBoundResource 当前没有生产使用点；不能因为它能组合资源与 Owner，
  就先设计公共“绑定资源接口”。
- LogicalGraphDefinitionRecord 服务于 XML 前向引用恢复；
  执行侧没有同型 record，并不意味着应补一个。
- LogicalGraphAnalysis 包含诊断和派生信息；
  执行侧没有缓存同名字段，并不意味着图对象缺少基本结构能力。

先确认真实需求，再决定共享或删除，不按对称性补类型。

## 10. 对 IInferenceGraph 和通用 Builder 的约束

本节是由分析导出的设计约束，不是最终 API。

### 10.1 可以先统一的职责

1. 强类型节点和边集合的收集与快照。
2. 通用身份、重复项、端点等结构检查。
3. 通过具体模型创建入口构造完整对象。
4. 明确调用该模型的验证入口，并按约定处理错误与警告。

当前 Core Graph 是 sealed，具体图通过组合保存结构图。
统一 Builder 不应以修改 Core 继承结构为前提。

### 10.2 不能直接塞进公共 Builder 的职责

- 逻辑定义注册与重复定义一致性判断。
- 子定义查找和被引用实例资源展开。
- 逻辑图当前的隐式依赖推导。
- 完成汇点、终端写入者等派生结果的自动缓存。
- 转换、优化策略、模板提取、VM 参数或存储分配。
- 固定的逻辑/执行 XML envelope。

其中有些能力两种图都需要，不代表应该由基础 Builder 自动执行。
专用声明 DSL 可以建立在通用结构构建之上，但不能把复杂性隐藏在模型类型分支中。

### 10.3 Validate 的语义必须先明确

- Validate 检查和报告，不修改图，不暗中补依赖。
- 区分物化前结构检查、单图业务检查、图集合/调用关系检查。
- 明确是 Builder 内部调用，还是调用方显式调用；未决前不新增两套混乱入口。
- 当前逻辑 Validate 返回构造时保存的诊断，执行 Validate 重新执行部分检查；
  统一入口前要明确快照、缓存及重新验证语义。
- 当前构造器、Builder 和校验器有重复调用；共享流程不应增加重复扫描。
- 图结构集合快照不代表派生记录里的任意字典或对象状态都深度不可变；
  验证缓存是否可靠需要结合实际属性和集合持有方式判断。

### 10.4 不应由小接口强制的业务

即使本项目两种图都拥有资源和调用，也不能据此要求所有
IInferenceGraph 实现必须提供固定资源枚举、来源映射、冲突分析或 XML。
先确定共同消费需求，结构访问与验证可以优先讨论；
遍历、筛选、可达性仍复用 Core，不由每种图重复实现。

## 11. 后续优先级建议

| 优先级 | 分析/设计项 | 理由 |
|---|---|---|
| 1 | 资源作用域身份中性化 | G04/G05/G15 明确存在共享实现依赖逻辑身份的问题 |
| 2 | 资源公共载荷、操作公共载荷、描述性注解 | 已有字段和转换证据，不靠猜测新增抽象 |
| 3 | 构图阶段拆分及本地/跨图验证边界 | 决定统一 Builder 和 IInferenceGraph 是否真实简化职责 |
| 4 | 重复的绑定、校验、冲突、边界算法 | 在明确数据语义之后提取，不先引入标签接口或 selector 框架 |
| 5 | XML 比较耦合、分析缓存及遗留类型 | 单独确认行为与消费者，不随迁移顺手改动 |

最终目标不是让逻辑图和执行图拥有完全相同的类型清单，
而是让公共层仅表达真正共有的机制和语义，让模型专用规则与图外处理流程显式归属。

## 12. 覆盖与验证说明

- 编号清单：G01–G16、L01–L24、E01–E23、P01–P07、C01–C07。
- 每个类型都有独立表项、源码链接、职责及共享判断。
- 非完全一致、仅单侧存在或共享实现含专用依赖的条目均给出具体差异例。
- 字段一致的不同 CLR 类型明确标注为等价或共享候选，不声称它们已经是同一类型。
- 本次只新增分析文档；未修改代码、接口、Builder 或现有迁移计划。
- 校验文档类型清单与源码、相对链接及例子覆盖；不因文档变更重复代码构建和测试。
