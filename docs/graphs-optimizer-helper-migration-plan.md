# 执行图归属与优化器依赖修复计划

状态：执行图归属修复已于 2026-10-18 实施并验证完成，结果见第 11 节。

当前首要任务是将执行图本体从优化器迁回图层，
恢复“优化器依赖逻辑图和执行图，图层不依赖优化器”的边界。
原“3 个分析结果/类别类型＋3 个逻辑图分析方法迁入优化器”的计划暂缓，
不与本轮执行图归属修复同时实施。
原 24／34 类型整体迁入优化器的方案继续撤销。

## 1. 目标与判定原则

- 图模型表达结构和语义；优化器读取、分析并改写图，不拥有基础图模型。
- 执行图的构造、基础查询、合法性校验和 XML 读写不要求引用优化器。
- 按职责和实际依赖决定归属，不以“唯一库外显式消费者”决定归属。
- 区分两种图共享的组件、各自专用模型、转换流程及优化策略。
- 不增加生产项目、业务接口、适配框架、友元关系或兼容包装。
- 不修改 Graphs.Core，不重写算法，不改变 XML v4 或执行语义。
- 本文件描述目标结构；在实施完成前，不把现有源码或消费者表标记为已迁移。

## 2. 目标结构和依赖方向

执行图迁入现有 `SharpInference.Graphs` 程序集：

```text
SharpInference.Graphs
    ├─ 共享基础组件
    ├─ Logical：逻辑图模型及构图、查询、校验、XML
    ├─ Execution：执行图模型及构图、查询、校验、XML
    └─→ Graphs.Core / Instructions

SharpInference.Vm.Optimization
    ├─→ Graphs.Logical
    ├─→ Graphs.Execution
    ├─ 图转换与编译处理流程
    └─ 优化分析、策略与改写处理

执行图的普通构建、查询、读写或展示调用方
    └─→ Graphs
```

以上箭头表示代码/程序集引用方向；Logical、Execution 是同一程序集下的命名空间，
不新增两个项目。优化器已有的 Graphs 项目引用承载两种图依赖。
Graphs 不增加对 Vm、Vm.Optimization、后端或模型项目的引用。
模型、Vm、后端及展示工具不因使用执行图而新增优化器引用。

## 3. 第一优先级：执行图组件迁回图层

### 3.1 本轮迁移的 23 个类型

源目录：[优化器 GraphHelper](../src/SharpInference.Vm.Optimization/GraphHelper)。
目标目录：`src/SharpInference.Graphs/Execution`。
目标命名空间：`SharpInference.Graphs.Execution`。

| 职责 | 类型 | 数量 |
|---|---|---:|
| 图与图间调用结构 | ExecutionGraph、ExecutionGraphProgram、ExecutionGraphCallRelation | 3 |
| 节点模型 | ExecutionGraphStructureNode、ExecutionGraphNode、ExecutionOperationNode、ExecutionGraphCallNode、ExecutionResourceNode | 5 |
| 边模型 | ExecutionGraphEdge、ExecutionDependencyEdge、ExecutionResourceReadEdge、ExecutionResourceWriteEdge | 4 |
| 执行图与节点身份 | ExecutionGraphId、ExecutionNodeId | 2 |
| 资源描述 | ExecutionGraphResource | 1 |
| 来源与描述性元数据 | ExecutionGraphSource、ExecutionSourceMap、GraphArchitecture | 3 |
| 类别常量 | ExecutionGraphKinds | 1 |
| 构图、校验、查询、XML | ExecutionGraphBuilder、ExecutionGraphValidation、ExecutionGraphQueries、ExecutionGraphXml | 4 |
| **合计** | | **23** |

迁移规则：

- 保留文件名、类型名、成员、可见性、泛型约束和实现，只调整归属及必要引用。
- ExecutionGraphValidation 继续作为模型内部校验实现，不扩大公共 API。
- 不在优化器保留这些类型的副本、转发或兼容入口。
- 两种图继续共享基础接口、资源身份与绑定、诊断、结构检查及 XML 公共工具。
- ExecutionGraphQueries 整体迁回，包括其现有冲突和边界查询；
  本轮不同时拆方法、合并两种图算法或增加通用 selector。
- ExecutionGraphSource、ExecutionSourceMap 对逻辑图身份的引用用于来源追踪，
  可以继续保留；这不构成对优化器的依赖。
- 不借迁移改造 GraphBindingOwner 的身份类型；其跨模型中性化另行分析。

### 3.2 本轮留在优化器的 7 个 GraphHelper 类型

| 职责 | 类型 | 本轮处理 |
|---|---|---|
| 逻辑图到执行图转换 | IExecutionGraphConverter、DefaultExecutionGraphConverter | 暂留现有处理流程，改为引用迁回图层的执行图 |
| 编译流程与配置 | ExecutionGraphCompiler、ExecutionGraphCompilationContext | 暂留优化器，不迁入基础图模型 |
| 后处理契约与上下文 | IExecutionGraphPostprocessor、ExecutionGraphPostprocessContext | 留在优化器，消费者依赖图层类型 |
| 模型级改写支持 | ExecutionGraphRewriter | 本轮暂留优化器，后续再区分公共改写机制与优化策略 |

“暂留”不等于认定这些类型全部为优化算法专用。
本轮优先修复图本体的归属，不扩大为编译/发射流程重组。
这些类型不得成为迁回图层的 23 个类型所依赖的反向入口。
优化处理器、ExecutionDependencyAnalysis、VM 发射及其选项保持现有归属。

### 3.3 同步调用面

- 优化器中的转换器、编译器、重写器、各后处理器和 VM 发射器。
- 模型、Vm、CPU/D3D12 后端及工具中实际引用执行图类型的生产调用。
- 序列化、适配、查询、转换、改写、类别及分层测试。
- 文件式 GraphQueryBenchmark 与文档中的执行图示例。
- 项目引用只在有直接使用需要时调整；不新增优化器依赖，
  不顺带删除与本轮无关的既有引用。

实施时沿实际符号引用核查，不只做命名空间字符串替换。

## 4. 原三类型、三方法迁移计划的处理

以下三个类型本轮保留在 Graphs：

- [ModelGraphBoundary](../src/SharpInference.Graphs/ModelGraphBoundary.cs)。
- [ModelResourceConflict](../src/SharpInference.Graphs/ModelResourceConflict.cs)。
- [ModelResourceConflictKind](../src/SharpInference.Graphs/ModelResourceConflictKind.cs)。

[LogicalGraphQueries](../src/SharpInference.Graphs/Logical/LogicalGraphQueries.cs)
中的 ResourceConflicts、UnorderedResourceConflicts、BoundaryResources 保持原位。
本轮不创建 LogicalGraphResourceAnalysis。

原因是这三个类型当前是两种图共享的分析结果，并非优化器策略本身。
先把执行图及其查询迁回图层，再根据真实分析消费者决定：
哪些查询属于图的通用能力，哪些分析真正为优化决策服务。
如果后续仍需要迁出，应作为独立功能单元重新设计并确认，
不能沿用“只有优化器显式引用”的旧依据自动实施。

其他逻辑图模型、Builder、校验、XML、公共辅助组件及遗留类型保持现状。

## 5. 本轮不实施的后续职责清理

完成依赖修复后，才继续逐项分析：

1. Builder 中的声明记录、名称解析和实例资源展开边界。
2. InferDependencies 的语义规范化职责及现有默认行为。
3. 合法性校验与完成汇点、终端写入者等派生结果缓存的边界。
4. RegisterDefinition 通过 XML 比较重复定义的耦合。
5. 公共资源分析、优化专用分析及两种图查询重复实现的归属。
6. 转换、公共改写机制、优化策略与 VM 发射的职责。

这些是后续分析项，不是本轮代码修改授权，不影响当前执行图迁移的先后次序。

## 6. 保持的行为

- 节点、边和绑定声明顺序，ReadWrite 配对及资源作用域。
- 可见视图、隐藏节点、平行边、自环及未知身份的既有语义。
- NodeKind、EdgeKind、ResourceKind、ConflictKind 及自定义类别。
- 冲突分类、排序判断、区域边界规则、结果顺序及异常行为。
- 图间调用关系、根图识别、来源追踪和编译边界元数据。
- 逻辑图到执行图转换、内联、融合、资源复用、依赖削减及模板共享结果。
- CPU/GPU VM 发射的参数、依赖、调用及同步行为。
- XML v4 的字段、版本、顺序和安全限制；不因 CLR 命名空间变化修改格式。
- 不要求保留旧命名空间兼容，不新增旧版本 XML 读取。

## 7. 实施阶段与顺序

| 阶段 | 工作 | 完成依据 |
|---|---|---|
| P0 | 核对 23 个模型组件的依赖、生产消费者及相关测试基线 | 迁回组件不依赖剩余处理流程，范围闭合 |
| P1 | 将 23 个类型同批迁到 Graphs.Execution，更新全部必要引用 | 图层独立构建，优化器引用新归属，无双份定义 |
| P2 | 更新分层断言及独立消费者验证，运行相关行为测试 | 不引用优化器也能构造、查询、校验及读写执行图 |
| P3 | 完整构建与集成回归，更新架构文档和消费者统计 | 模型/后端无新增优化器依赖，行为保持，文档与源码一致 |
| 后续 | 重新分析可选分析能力与 Builder 的职责 | 单独确认后续范围，不与本轮迁移绑定 |

P1 依赖 P0，P2 依赖 P1，P3 依赖 P2。
不以不可构建的中间状态交付，不自动实施后续职责重构。

## 8. 验证与验收

### 8.1 归属与依赖

- 23 个类型属于 Graphs 程序集及 Graphs.Execution 命名空间；
  优化器不再定义执行图模型、Builder、校验、基础查询和 XML 类型。
- GraphLayeringTests 更新早期“执行图必须位于优化器”的旧断言，
  增加执行图属于图层、处理流程仍属于优化器的正向断言。
- Graphs 仅保留 Core、Instructions 直接项目引用，
  不引用 Vm、优化器或后端；Graphs.Core 无修改。
- 复用或增加只引用 Graphs 的独立消费者构建验证，
  覆盖执行图构造、查询、校验和 XML 往返。
  不把已有优化器引用的测试项目编译通过作为独立性证明。
- 两个基础图接口继续唯一地定义在 Graphs；共享结果类型仍在 Graphs。
- 若出现边界问题，按类型/成员职责消除反向依赖，
  不通过新增 Graphs→优化器引用或公开内部实现绕过。

### 8.2 行为与集成

- 组合运行 GraphLayeringTests、InferenceGraphKindTests、
  InstructionContractTests、LogicalGraphDefinitionTests、
  NamedGraphConstructionTests、GraphSerializationTests、
  GraphModelAdapterTests、GraphViewQueryTests、
  ExecutionGraphConverterTests、ExecutionGraphPostprocessorTests、
  VmExecutionGraphEmitterTests。
- 覆盖逻辑与执行两套查询、冲突/边界分析、自定义类别、来源映射、
  XML 往返、转换、改写、优化结果及发射。
- 构建完整方案和文件式 GraphQueryBenchmark，覆盖模型全部目标框架。
- 运行现有 RWKV6/7 真实模型构图与 CPU 发射冒烟；
  数据或环境缺失时如实记录，不把跳过当成通过。
- GraphView 页面及 XML 格式不变，默认不重复浏览器回归和 D3D12 logits 长测；
  实际触及对应行为或已有结果出现回归时升级验证。
- 只在依赖清单变更或选定验证命令报告缺失依赖时 restore，不新增包。
- 运行 diff --check，不自动创建 Git 提交。

## 9. 文档与消费者统计

- 实施后更新 [图架构设计](graph-query-design.md)、中英文 README、
  执行图示例及直接相关工具说明，明确执行图归图层而非优化器。
- 原历史计划保留历史事实；必要时标明已被本计划替代，不伪造历史状态。
- 重新生成非测试生产消费者 CSV、Markdown 和逐文件证据，
  区分所属程序集、模型类别、内部引用和库外显式消费者。
- 不把“只有某项目显式消费”标记成“该项目专用”。
- 当前同口径基线为 Graphs 40、优化器 GraphHelper 30、
  Instructions 契约 7，共 77 个跟踪类型。
  仅完成本计划的 23 类型迁移且不增删类型时，
  预期为 Graphs 63、优化器 GraphHelper 7、Instructions 7，合计仍为 77。
  这些不是各程序集所有类型的总数，最终以实际语义统计为准。
- 不提前改写 CSV；原三类型迁移方案的 37／34／7 及总计 78 预测已失效。

## 10. 历史方案处理

此前把执行图整体放入优化器、再依据库外显式消费者进一步迁入图组件，
混淆了数据模型归属与处理流程归属。
虽然没有要求图层项目反向引用优化器，却让普通执行图消费者被迫引用优化器。

本计划首先纠正这一边界：执行图本体迁回图层，优化器向下引用两种图。
原 24／34 类型整体迁移方案继续撤销；
原三类型和三方法方案暂缓，不能作为本轮实施依据。

## 11. 实施与验证记录

### 11.1 实际改动

- 23 个执行图类型已迁到 Graphs.Execution；优化器 GraphHelper 保留 7 个处理流程类型。
- 转换器、编译器、优化处理器、VM 发射器、测试和基准已同步引用新命名空间。
- Graphs 项目引用保持 Core、Instructions；没有新增生产项目或优化器反向引用。
- 分层测试覆盖全部 23 个类型的归属、内部校验可见性、7 个处理流程类型归属，
  以及三个共享分析结果类型继续留在 Graphs。
- 内联改写原先调用 ExecutionGraphNode 的内部身份映射方法。
  跨程序集后改为复用已经重映射的节点，通过公开 GraphId 建立映射；
  没有公开内部方法、重复身份编码或新增友元。
- GraphQueryBenchmark 已移除优化器项目引用，执行图构造与查询直接引用图层。
- 架构设计、中英文 README、GraphView 说明及相关历史计划的现行归属提示已同步。
- 三类型/三方法迁出、Builder 清理、身份中性化及处理流程重组均未实施。

### 11.2 验证结果

| 验证 | 结果 |
|---|---|
| 迁移前相关测试基线 | 212 项通过，0 失败，0 跳过 |
| 迁移后相关测试 | 214 项通过，0 失败，0 跳过，包含新增的 2 项分层测试 |
| 仅引用 Graphs 的独立消费项目 | 构造、校验、依赖/绑定查询、XML 往返及边顺序检查通过 |
| 独立消费项目的运行依赖 | deps.json 不包含 Vm 或优化器；图程序集不引用 Vm |
| 完整方案构建 | 0 警告、0 错误 |
| 文件式 GraphQueryBenchmark 构建 | 按变更的项目指令重新 restore 后成功，deps.json 不包含 Vm 或优化器 |
| RWKV6/7 真实模型 CPU 构图与发射 | 2 项通过，0 失败，0 跳过 |
| 非测试生产项目语义分析 | 19 项目、20 框架配置，0 编译错误 |
| 消费者矩阵 | 77 类型、11 消费项目列、847 格逐格校验通过 |
| Graphs.Core 与补丁空白检查 | Core 无改动，diff --check 通过 |

RWKV 验证曾误选额外的全优化组合性能长测，已停止，不计入通过结果；
随后按计划单独重跑 CPU 冒烟，两项均通过。
本次未改 XML 格式、浏览器实现或数值算法，未重复浏览器及 D3D12 logits 长测。

### 11.3 当前消费者快照

- 归属实测为 Graphs 63、优化器 GraphHelper 7、Instructions 契约 7，共 77 个跟踪类型。
- 新快照为 `consumers-after-execution-return.json`，
  矩阵为 `graphs-consumer-matrix-after-execution-return.csv` 及同名 Markdown。
- 原 after-kind / after-cleanup 快照保留，不覆盖历史数据。
- 分析器及矩阵生成器支持指定输出快照，保留生产源码逐文件符号证据；
  不把显式消费者数量解释为职责或归属结论。
- 独立消费项目保留在会话产物中供重复验证，不新增生产或方案项目。
- 未创建 Git 提交。
