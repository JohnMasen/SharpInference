# SharpInference.Graphs 薄辅助层重构计划

状态：已执行并通过最终验收；实际结果见第 10 节。
制定日期：2026-10-16。
执行基线：提交 `338003b`（`re-org SharpInference.Graphs project`）。
约束更新：不要求旧 API 或旧 XML 格式兼容；仓库调用方、测试和样例统一迁移。

## 1. 目标与不可变边界

将 [SharpInference.Graphs](../src/SharpInference.Graphs) 定位为薄的模型图辅助层：

1. 定义逻辑图和执行图共用的数据类型。
2. 提供少量可复用的泛型构造辅助。
3. 提供模型关系的公共查询及校验能力。
4. 支持基于名字的后期绑定，以及 XML 序列化、反序列化。

不是重做底层图引擎，也不是建设新的强类型端口框架。

### 1.1 必须保持

- [SharpInference.Graphs.Core](../src/SharpInference.Graphs.Core) 源码、公开 API 和行为不变。
- 继续使用 Core 的图、只读视图、模式匹配、可达性和改写能力。
- 保留逻辑图和执行图各自的节点类型、资源语义及必要元数据。
- 保留操作 ID、操作版本、端口名、张量描述、精度要求和 bindingKey。
- 保留定义与调用的区别、资源作用域、显式依赖及现有依赖推导语义。
- 不改变模型公式、指令实现、优化器注册顺序、资源槽 ABI 和执行入口。
- 编译边界继续表达元数据意图；不新增自动内联或禁止显式内联的规则。

### 1.2 本次不做

- 不拆分或新建项目，不重组后端、运行时和模型架构。
- 不引入 CLR 派生类恢复、反射注册、源生成器或类型化端口视图框架。
- 不顺带修改 Core 已有实现问题。
- 不新增动态控制流、形状推导器、指令注册中心或融合算子。
- 不以文件数或类型数为唯一精简指标；优先删除机制和重复职责。
- 不保留旧 API 重载、兼容包装、旧 XML reader 或自动格式升级路径。

本次允许破坏 API 和 XML 格式兼容性。保留计算语义、资源 ABI 和有效图的执行结果，
不等于保留旧构造语法、诊断代码、序列化文本或旧格式文件的可读性。

## 2. 重构前实现与迁移影响

用户已提交之前的工作区改动，当前基线为 `338003b`，核对时工作区干净。
实施从该提交的 Logical、Execution 子目录布局出发，不恢复已删除的旧根目录图文件。
若执行时已有新提交或新改动，重新记录实际起点并保留无关工作。

| 当前区域 | 现状 | 重构方向 |
|---|---|---|
| [GraphDefinitionBuilder](../src/SharpInference.Graphs/Logical/GraphDefinitionBuilder.cs) | 泛型端口契约、资源句柄、调用端口重建、绑定展开、依赖推导集中在一起 | 非泛型的具名构造入口，复用公共机制 |
| [GraphOperationNodeBuilder](../src/SharpInference.Graphs/Logical/GraphOperationNodeBuilder.cs) | 绑定必须取得当前图拥有的泛型句柄 | 记录具名资源引用，延后解析 |
| 旧 IGraphPortContract、InstancePortFactory | 为定义和每次调用创建同形状 C# 端口对象 | 删除，不保留兼容入口 |
| [ExecutionGraphBuilder](../src/SharpInference.Graphs/Execution/ExecutionGraphBuilder.cs) | 已按集合构造，但绑定展开与逻辑构造方有重复 | 统一具名引用和公共展开逻辑 |
| [LogicalGraphDefinitionValidator](../src/SharpInference.Graphs/Logical/LogicalGraphDefinitionValidator.cs) | 在完整结构上分析，部分无效输入会先抛集合异常 | 先检查声明完整性，再进行领域分析 |
| [ExecutionGraph](../src/SharpInference.Graphs/Execution/ExecutionGraph.cs)、[ExecutionGraphProgram](../src/SharpInference.Graphs/Execution/ExecutionGraphProgram.cs) | 构造时校验，图的 Validate 为私有，程序另做调用 DAG 校验 | 提供一致的公开校验体验，保留图级/程序级边界 |
| [ModelGraphQueries](../src/SharpInference.Graphs/ModelGraphQueries.cs) | 已基于 Core 视图提供模型查询 | 保留，必要时消除重复，不重写查询引擎 |
| [LogicalGraphXml](../src/SharpInference.Graphs/Logical/LogicalGraphXml.cs)、[ExecutionGraphXml](../src/SharpInference.Graphs/Execution/ExecutionGraphXml.cs) | 已使用 v2 通用节点/边格式，不恢复端口契约 | 按新声明模型简化格式，统一校验路径，不兼容旧格式 |
| [RWKV 构造适配](../src/SharpInference.Models.Rwkv/Architecture/PortableRwkvGraphDefinitionComposer.cs) | 使用 RwkvGraphPorts、泛型句柄及 AsyncLocal 描述符 | 直接声明 token、logits 等资源，移除契约专用环境状态 |
| [Phi4 构造适配](../src/SharpInference.Models.Phi4/Architecture/Phi4GraphDefinitionBuilder.cs) | 先收集具名声明，再经 DynamicPorts 和句柄字典转换 | 直接接入具名构造 API，移除契约中转 |

## 3. 目标设计与行为约定

### 3.1 公共数据与泛型辅助

优先保留现有 ResourceId、GraphOperationId、TensorDescriptor、PrecisionRequirement、
GraphResourceAccess、GraphBindingOwner、ModelResourceNodeId、ModelResourceBinding 等必要类型。
ID 的字符串值用于名字绑定，不要求把轻量 ID 类型全部替换为裸 string。

逻辑节点和执行节点继续分开；不建立包含双方全部字段的万能节点。
公共诊断统一表示严重程度、代码、图身份及可选的节点、资源、端口位置。

泛型只用于两种构造方实际共享的处理，例如不同节点/边类型的绑定展开。
首先对照两个构造方提取最小公共实现；最多引入一层构造辅助，不为每种节点创建工厂接口。
公共机制不依赖模型、优化器或后端，不使用运行时类型反射。
仅被一方使用的步骤留在对应构造方；简单内部泛型辅助足够时，不公开新的泛型基类。

### 3.2 构建状态与成品图

采用现有 Builder 到成品图的边界，不额外创建另一套用户可见 Draft 图体系：

```text
代码构造                         XML 解析
    |                              |
    v                              v
具名构建声明                 具名节点/边记录
    |                              |
    +---- 公共结构及领域校验 -------+
                   |
                   v
        已验证的逻辑图 / 执行图
                   |
                   v
        现有 Core 结构及其查询视图
```

- Builder 允许尚未声明的资源、节点依赖和目标图名字。
- 参数为空、非法枚举、同名重复添加可以立即报错；前向引用不能立即报错。
- Validate 无副作用，不补资源、不改名字、不添加到持久构建集合。
- Build 使用同一校验流程，发现 Error 时不返回成品图，明确抛出带诊断的异常。
- Builder.Validate 和成品图.Validate 返回公共诊断集合；程序级引用由程序级 Validate 校验。
- 保留成品图只读快照和 Structure 查询入口，不让不完整声明进入 Core。
- 每次 Build 产生独立快照；之后修改 Builder 不影响旧图。
- 不在第一版引入可变图缓存或版本框架。重复 Validate/Build 不累积边或依赖。
- XML 和改写产生的完整节点/边也走公共校验，不能通过直构 Structure 绕过领域规则。

名字查找在构建阶段可以报告未声明；需要完整关系的遍历只针对物化图。
不把无效引用静默跳过，也不返回成功形状的空图。

### 3.3 名字、绑定与调用

- 名字比较统一使用 StringComparer.Ordinal，不自动改大小写或去空格。
- 图 ID 在图集合内唯一；节点 ID、资源 ID 按当前各自命名空间唯一。
- 显示名称不作为引用身份，现有结构 ID 前缀继续作为内部表示细节。
- 操作绑定保存端口名、资源引用和 Read/Write/ReadWrite。
- 子图调用保存调用 ID、目标图 ID，以及子图端口到调用方资源的映射。
- 目标定义可在调用声明之后注册；注册表属于当前构造上下文，不是全局状态。
- 保留结构化 Parent/Instance 作用域；同一子图多次调用的绑定互不覆盖。
- 不用斜杠、点号解析作用域。现有带这些字符的名字仍视为普通名字。
- 新 API 默认显式绑定子图所有端口，包括输出和 InOut。
- 原自动输出映射迁移为显式资源声明和绑定，保留原有输出 ID 及资源角色。
- 不依赖字典覆盖解决重复定义。同 ID 内容冲突必须明确失败；合法的同一定义复用保留。

示意用法如下，方法名允许在 P1 根据现有命名作小幅调整，行为约定不变：

```csharp
var builder = new GraphDefinitionBuilder("residual");

builder.AddNode("add", PrimitiveGraphOperations.Add, node =>
{
    node.AddInput("left", new ResourceId("activated"));
    node.AddInput("right", new ResourceId("input"));
    node.AddOutput("output", new ResourceId("output"));
});
builder.AddNode("relu", PrimitiveGraphOperations.Relu, node =>
{
    node.AddInput("input", new ResourceId("input"));
    node.AddOutput("output", new ResourceId("activated"));
});

builder.AddResource("input", vector, GraphDefinitionResourceKind.Input);
builder.AddResource("activated", vector, GraphDefinitionResourceKind.Temporary);
builder.AddResource("output", vector, GraphDefinitionResourceKind.Output);

var diagnostics = builder.Validate();
var graph = builder.Build();
```

新逻辑图入口直接返回 LogicalGraphDefinition，不再要求从 GraphDefinition<T> 取 Definition。
便利方法不得要求端口契约、环境静态变量或资源泛型标签。

### 3.4 校验与依赖

按顺序执行：声明检查、引用解析、绑定结构检查、Core 物化、领域分析。
领域分析复用 Core 查询；只有全部检查通过才公开返回成品图。
声明检查失败时，跳过依赖该无效项的后续分析，但继续收集独立错误；
不能在 ToDictionary、Single 或 GetNode 处先抛无上下文异常。

规则分层：

1. 公共：唯一性、引用存在、资源作用域、重复端口、成对读写一致性。
2. 逻辑：只读资源写入、输出写入者、子图接口访问及张量兼容、依赖分析。
3. 执行：本地图绑定和依赖 DAG；程序层目标存在、唯一根、调用 DAG 和可达性。

构造方可以在本地 Validate 时不解析外部图目标，但必须明确校验范围；
全图集合 Build/Validate 必须解析目标，不允许把“未提供上下文”解释为目标合法。
执行图现有程序校验不完整检查被调接口。本次按统一接口规则补全目标端口、
访问方式与张量兼容校验，并迁移暴露问题的构造方和测试；
不为兼容旧无效数据保留跳过校验的开关。P0 记录新增拒绝条件，
与有效模型的计算行为回归分开说明。

逻辑依赖推导必须保持当前规则，而不是简单按添加顺序串行化：

- 读取实例作用域资源时，保留对拥有该资源的调用的依赖。
- 成对 ReadWrite 绑定不按普通纯读推导自己的生产者依赖。
- 普通读取只在存在唯一、不同于读取节点的生产者时推导依赖。
- 多写入者不因声明顺序被自动串行化。
- 显式依赖保留；自依赖保留到环检查；重复非自依赖去重。
- 校验依赖子图是否有环，不对包含状态读写边的整个结构套用 DAG 要求。

CompletionSinks、TerminalWriters 和既有 LG 错误/警告语义保持。
跨实例写入的 LGW003、无序终端写入者的 LGW002 不能未经授权升级为 Error。
公共诊断代码可以统一整理，不要求旧代码兼容；新代码及对应语义在测试和文档中固定。
既有错误/警告严重程度的保留不要求保留原诊断类型。

校验结构和已知接口不等于证明任意操作数学等价，也不等于后端支持检查。
未知自定义操作不能仅因未注册后端实现就在通用建图层被拒绝。

### 3.5 查询、转换与改写

- 保留 ModelGraphQueries 对 IGraphView 的扩展及当前可见范围语义。
- 保留 Readers、Writers、Bindings、依赖视图、数据流视图、边界及冲突分析。
- 不在构造辅助中复制 Core 的查找、拓扑排序、匹配或可达性实现。
- 转换器继续保留定义复用、调用、作用域、节点和边身份、精度及来源映射。
- 优化器继续基于节点、边、操作及绑定工作，不判断用户定义的 CLR 图类。
- 改写结果经过相应图级、程序级校验；不从绑定重新推导并覆盖优化后的执行依赖。

### 3.6 XML：直接表达新模型，不保留旧格式

当前 XML 没有序列化 IGraphPortContract，但仍携带旧构造和结构表示的历史选择。
本次以新公共模型为依据确定唯一格式，不以旧 v2 可读性约束设计，也不双写新旧格式。

1. XML 保存图、资源、操作、调用、名字绑定和必要依赖；访问边由绑定展开，
   不同时保存一份可独立修改的等价访问边集合。
2. 非派生的关系、优化后依赖以及实际需要的身份必须无损表达；
   不能只为简化 XML 而丢弃执行结构。P1 固定关系和身份的保存规则。
3. 解析先收集名字记录，节点、资源和定义的出现顺序不影响引用解析。
4. 使用公共声明/结构校验，随后构造成品图；公开反序列化入口返回已验证结果。
5. 执行图导入后保留执行依赖，不能按逻辑图推导规则重新生成。
6. 保留资源声明归属和绑定实例作用域，两者不能混同。
7. 保留来源映射、架构注解、边界、属性、bindingKey 和精度。
8. 固定新的确定性排序和规范化输出规则，不要求与旧 XML 文本一致。
9. 保留 DTD 禁用、外部解析禁用、字符数、元素数、深度及枚举值限制。
10. 不保存或恢复用户 CLR 类型名；只分派受支持的逻辑/执行节点种类。

新格式使用明确的新版本标识，不把旧 v2 标识复用于不同 schema。
旧格式和未知版本直接报出不支持的格式诊断，不尝试兼容读取、猜测或自动升级。
仓库 XML 样例和消费者在同次重构中更新；不要为已有样例保留旧模型旁路。
格式错误明确失败，不兜底返回空图。

## 4. 文件与 API 迁移清单

| 区域 | 主要操作 |
|---|---|
| Logical/GraphDefinitionBuilder.cs、GraphOperationNodeBuilder.cs | 替换为名字引用，提供 Validate/Build，移除端口契约依赖 |
| Logical/LogicalGraphDefinition.cs、LogicalGraphDefinitionValidator.cs、LogicalGraphAnalysis.cs | 保留成品结构和分析结果，公开校验入口，接入公共诊断与前置校验 |
| Execution/ExecutionGraphBuilder.cs、ExecutionGraph.cs、ExecutionGraphProgram.cs | 共享绑定展开及诊断，统一本地与程序级校验边界 |
| ModelResourceBinding.cs、ModelResourceNodeId.cs、GraphBindingOwner.cs | 复用结构化引用，避免新旧两套资源身份 |
| ModelGraphQueries.cs 及公共模型接口 | 保持查询契约，适配真正变化的类型 |
| 两套 GraphXml、LogicalGraphDefinitionRecord、Internal/GraphXmlSupport.cs | 新格式读写、名字记录装载及公共校验，保留安全限制 |
| Execution/DefaultExecutionGraphConverter.cs、ExecutionGraphCompiler.cs、ExecutionGraphRewriter.cs | 适配构造/校验入口，保留转换及改写行为 |
| 模型构造方、VM 优化及发射消费者 | 仅替换构造/绑定/校验调用，不修改算法和 ABI |

旧机制与对应调用方按功能切片同步替换，不增加兼容入口；
最终必须删除以下机制，而不是保留空壳兼容层：

- IGraphPortContract、IGraphPortFactory、DefinitionPortFactory、InstancePortFactory。
- GraphDefinition<TContract>、GraphDefinitionBuilder<TContract> 的泛型入口。
- GraphInput<T>、GraphOutput<T>、GraphInOut<T>、GraphResource<T> 的标签式绑定路径。
- GraphCallBindings<TContract>、GraphCallInstance<TContract> 的端口重建路径。
- PendingBinding 对已解析泛型句柄的依赖；若名字绑定记录仍需要，改造后保留其单一职责。
- 模型端仅为上述机制存在的端口类型、DynamicPorts、描述符环境作用域。

轻量调用身份对象如果仍有实际消费者，可以保留非泛型版本；
删除的是端口契约机制，不是为了删类型而删除有用的调用身份。
CreateListGraph 如有需要保留为非泛型便利函数，输入输出显式提供，
不扩大原有一输入一输出、固定 input/output 端口的适用范围。

## 5. 分阶段执行计划

依赖关系：

```text
P0 基线与固定契约
  -> P1 名字绑定与公共校验辅助
       -> P2 逻辑图迁移
       -> P3 执行图迁移
       P2 + P3
          -> P4 XML、转换与改写接入
               -> P5 模型及消费者迁移
                    -> P6 删除旧机制与最终验证
```

P2、P3 可以逻辑独立推进；若共享文件交叉，按顺序执行，不为并行产生冲突。
每阶段先完成目标实现及最小验证，再推进依赖阶段。

### P0：固定当前基线

操作：

1. 记录当前工作区改动，不覆盖、撤回或暂存不属于本次重构的工作。
2. 重新扫描旧泛型入口及构造/反序列化/转换的全部调用方。
3. 运行第 7 节的图相关基线；失败与跳过分别记录，不借本次任务修复无关问题。
4. 固定代表性图的语义预期，包含逻辑子图、状态和执行来源映射；
   不把旧 v2 文件可读性作为验收目标。
5. 将预期资源、调用、绑定和依赖作为测试数据或代码断言保存，
   不只存于临时目录或聊天记录，不以旧格式输出字符串作为唯一预期。
6. 固定依赖推导、警告、输出名字、结构身份和编译边界行为。

交付：可重跑的基线、语义回归样例和迁移清单；P1 前明确公共 API 签名。
执行时已重新运行基线，结果记录于第 10 节。

### P1：公共名字引用、构造辅助与校验

操作：

1. 复用现有 ID、作用域、绑定类型，必要时新增最小的未解析声明记录。
2. 统一公共诊断类型和 Error/Warning 判断，保留旧诊断语义。
3. 提取两种构造方共享的绑定检查及读写边展开。
4. 实现前向引用、重复声明拒绝、顺序保留和稳定的内部身份生成。
5. 提供局部校验、外部定义上下文及 Build 失败的明确边界。
6. 固定新 XML schema、版本、排序规则以及结构关系和身份的无损表示，
   不设计旧格式兼容或升级分支。

验收：无需端口契约即可记录声明；前向引用与实例引用可验证；
Validate 无副作用；同样输入重复物化结果稳定；Core 无改动。

### P2：逻辑图构造迁移

操作：

1. 增加非泛型 GraphDefinitionBuilder 和具名操作绑定入口。
2. 输入、输出、InOut 与临时/权重/状态资源均直接声明。
3. 子图调用使用目标名字及显式端口映射，注册时机与引用顺序解耦。
4. 在完整声明上复用现有依赖推导规则。
5. 提供 Builder.Validate、LogicalGraphDefinition.Validate 和直接返回成品图的便利入口。
6. 迁移逻辑图测试和受该 API 变更直接影响的调用方，移除旧入口；
   不通过兼容重载维持旧调用。P5 再集中清理模型侧冗余适配。

验收：Copy/ReLU 链、残差、状态、复用子图和跨实例资源行为保持；
缺失目标、端口或资源返回具名诊断；警告仍为警告。

### P3：执行图构造迁移

操作：

1. 执行图构造使用相同的名字引用和绑定展开辅助。
2. 保留直接接收完整节点/边的用途，使优化器无需逆向重建 Builder。
3. 公开一致的图级 Validate，并明确程序级调用关系校验入口。
4. 保留执行图与程序的已验证快照，以及图/调用顺序。
5. 不为执行图新增逻辑图式的隐式依赖推导。
6. 同步迁移受影响的执行图构造方，不新增旧签名兼容重载。

验收：前向依赖可构造；绑定错误和依赖环明确报告；
程序仍拒绝缺失目标、多根、调用环及不可达定义；
来源映射、架构及边界不丢失。

### P4：XML、转换和改写接入

操作：

1. 两套 XML 使用 P1 固定的新格式，先装载记录，后运行公共校验；
   移除旧格式 reader/writer，更新仓库内相关样例。
2. 对 P0 固定的语义预期和独立手写的新格式样例执行往返测试，
   不只测试新 writer 配新 reader；验证旧格式被明确拒绝。
3. 转换器移除对契约包装的依赖，保持定义复用和冲突检测。
4. 编译器、后处理器和改写器调用统一校验入口。
5. 验证图导入后的内联、提取、融合、依赖缩减和资源重用。
6. 保留 XML 后处理器对模型、后端和逻辑来源的匹配检查。

验收：新格式语义完整、规范化往返稳定；XML 图和代码图进入同一优化路径；
旧格式被明确拒绝；无 CLR 类型恢复，无必要身份丢失或执行依赖重算。

### P5：模型及消费者迁移

操作：

1. RWKV composer 直接声明 token/logits 和资源；移除 RwkvGraphPorts、
   RwkvTensor 标签及仅服务契约的 AsyncLocal 描述符作用域。
2. RWKV7 elementwise builder 改用 ResourceId 等非标签式引用；
   保持输出命名、算式、张量类型及既有显式排序。
3. Phi4 builder 的 ResourceSpec/StepSpec 直接接入名字构造入口；
   删除 DynamicPorts 和句柄中转，保留 alias、视图属性、注解和调用排序。
4. 适配 VmExecutionGraphEmitter、依赖缩减、临时资源重用、模板共享、
   XmlExecutionGraphProgramPostprocessor 的构造入口。
5. 扫描脚本和工具中的调用；仅修改受 API 变化影响的内容。
6. 补充无需真实模型文件的 RWKV 构造回归和 Phi4 适配回归。

验收：模型项目可构建；消费者没有契约依赖；资源槽名字、bindingKey、
状态读写、VM 调用及 CPU/D3D12 生成语义保持。
重构前主测试项目未直接覆盖 Phi4；实施中已增加项目引用和独立的合成目录回归，
不能用 RWKV 测试代替其构建和回归。

### P6：旧机制清理与最终验收

操作：

1. 删除第 4 节列出的旧机制及仅为其存在的模型辅助代码。
2. 搜索源码、测试、脚本和文档，清理旧入口及错误示例。
3. 更新相关 README 与建图说明，给出简单图、子图调用和 XML 往返示例。
4. 运行分层验证；检查旧 API 残留及 Core 零改动。
5. 记录通过、失败、跳过、外部条件和未解决阻塞，不将跳过写成通过。

验收：新入口独立可用，旧路径无调用方；全部必需验证通过；
无兼容层或旧格式读取分支、无静态描述符旁路、无未说明行为变化。

## 6. 必须覆盖的测试矩阵

| 类别 | 用例 | 要求 |
|---|---|---|
| 基本构造 | Copy/ReLU、残差、多输入、InOut | 不定义契约类型即可构造和查询 |
| 后期绑定 | 资源、依赖、子图目标在引用后声明 | 与先声明后引用具有相同语义，不要求声明顺序文本相同 |
| 校验 | 未知资源/节点/目标、重复名字/端口、错误作用域/访问/张量 | 明确诊断位置，不由集合异常先中断 |
| 调用复用 | 同一定义两次调用、显式输出映射、跨实例资源 | 输出身份、作用域及调用依赖正确 |
| 依赖 | 唯一生产者、ReadWrite、多写入、自依赖、依赖环 | 保留既有推导与 LGW002/LGW003 |
| 生命周期 | 重复 Validate/Build、Builder 后续修改 | 不累积关系、不修改旧快照 |
| XML 往返 | P0 语义预期、手写新格式样例、代码图往返、非规范声明顺序 | 语义、必要身份、元数据、作用域完整 |
| XML 版本 | 旧格式及未知版本 | 明确拒绝，不自动升级，不保留兼容分支 |
| XML 失败 | 缺失根/目标、重复身份、坏枚举、超限、DTD | 明确失败，安全限制不退化 |
| 转换与优化 | 代码图及 XML 图分别转换、内联/提取/融合及现有后处理 | 同等输入下绑定、来源和执行语义一致 |
| VM | CPU 源码、D3D12 源码及调用/调度/屏障生成 | 不支持操作仍明确报错 |
| 模型 | RWKV6/7 构造、Phi4 适配、可用条件下实际推理 | ABI 和数值容差遵循现有测试 |

XML 比较区分语义等价与规范化字符串稳定。
不能只比较节点数量，也不能只依赖新序列化器与新反序列化器相互配合。
旧格式字符串及旧诊断代码不作为兼容性断言；计算结果与必要元数据继续独立断言。
需要数值验证时使用现有测试的容差，不能为通过重构而放宽。

## 7. 验证命令与外部条件

以下命令在仓库根目录通过 PowerShell 运行。具体新增测试类加入同一过滤器。
先尝试 --no-restore；仅在依赖缺失或清单变化时恢复，不能默认安装新工具。

### 7.1 P0 与结构/API 修改后的最小测试

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-restore --filter "FullyQualifiedName~LogicalGraphDefinitionTests|FullyQualifiedName~ExecutionGraphConverterTests|FullyQualifiedName~ExecutionGraphPostprocessorTests|FullyQualifiedName~GraphViewQueryTests|FullyQualifiedName~VmExecutionGraphEmitterTests" --verbosity minimal
```

P1 添加公共辅助/校验测试后，将对应类一并加入过滤器。
后续单阶段只运行覆盖实际改动的子集，最终再运行合并选择器。

### 7.2 模型适配与下游构建

```powershell
dotnet build .\src\SharpInference.Models.Rwkv\SharpInference.Models.Rwkv.csproj --no-restore
dotnet build .\src\SharpInference.Models.Phi4\SharpInference.Models.Phi4.csproj --no-restore
dotnet build .\src\SharpInference.Vm.Optimization\SharpInference.Vm.Optimization.csproj --no-restore
```

同时运行 P5 新增的不依赖真实模型和 GPU 的适配回归。

### 7.3 最终无外部模型的回归

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-restore --filter "FullyQualifiedName~GraphCoreTests|FullyQualifiedName~GraphViewQueryTests|FullyQualifiedName~LogicalGraphDefinitionTests|FullyQualifiedName~ExecutionGraphConverterTests|FullyQualifiedName~ExecutionGraphPostprocessorTests|FullyQualifiedName~VmExecutionGraphEmitterTests|FullyQualifiedName~NamedGraphConstructionTests|FullyQualifiedName~GraphSerializationTests|FullyQualifiedName~GraphModelAdapterTests" --verbosity minimal
```

已包含新增测试选择器。GraphCoreTests 用于确认 Core 未改动后的集成行为，
不是授权修改 Core；无需重复整个无关解决方案的所有测试。

### 7.4 条件允许时的模型及 GPU 验证

- 使用既有 RwkvTestModels 配置方式和已存在的模型文件，不下载或提交模型。
- 模型目录可用时运行 RwkvTinyExecutionPipelineTests 的 CPU 路径，
  并按改动需要运行 RwkvPostprocessorTests 的既有组合验证。
- D3D12 源码生成测试与实际 GPU 执行分开记录；前者通过不能代替后者。
- 模型目录、驱动或硬件不可用时明确标记条件性验证未执行，不伪造成功。
- 必需的无外部依赖构造、XML、查询及 VM 生成回归不能因此跳过。

## 8. 风险与停止条件

| 风险 | 应对 |
|---|---|
| 执行起点偏离已提交基线 | 核对当前提交和工作区；不恢复旧布局，不覆盖新增无关改动 |
| 泛型辅助再次膨胀为框架 | 只提取两方实际重复逻辑，一层辅助，不引入新项目或类型注册 |
| 后期绑定改变依赖语义 | 固定唯一写入者、ReadWrite、多写入者和实例资源测试 |
| 直接构造或 XML 绕过校验 | 共享前置校验和领域分析，所有成品入口维持有效性 |
| XML 简化丢失图语义及来源映射 | 以语义预期和手写新格式样例验证必要身份、依赖与元数据 |
| 新执行接口校验暴露无效绑定 | 修正构造方和测试，不保留跳过校验的兼容开关 |
| 模型测试被外部条件跳过 | 提供无外部依赖适配测试，单独记录条件性数值验证 |

遇到资源 ABI、计算结果或依赖语义差异时，先补充设计与影响说明，
不得用静默默认值、忽略引用或放宽测试掩盖问题。

## 9. 完成定义

- [x] Core 源码和公开契约没有本次引入的改动。
- [x] 逻辑图和执行图都使用具名构造及明确的校验入口。
- [x] 两种图共用必要模型类型和薄的辅助，没有第二套通用图引擎。
- [x] 简单图、状态图和子图调用无需端口契约或泛型资源标签。
- [x] 字典/ID 查找和既有 Core 查询能力可用，前向引用有正式回归。
- [x] 新 XML 格式往返通过，旧格式被明确拒绝，图语义、依赖、来源及作用域不丢失。
- [x] 所有模型和下游调用方完成迁移，旧契约和静态描述符旁路清理。
- [x] 必需构建和测试通过，条件性验证状态明确。
- [x] 示例与文档更新，无未说明的执行行为变化。

## 10. 实施记录

### 10.1 已落地的阶段

- P0：从 `338003b` 开始，图、查询、后处理及 VM 基线 112 项通过，零失败、零跳过。
- P1：新增公共 GraphDiagnostic 和内部泛型 ModelGraphConstruction；
  两种图共享结构校验和绑定展开，不增加项目、端口工厂或公开泛型框架。
- P2：非泛型 GraphDefinitionBuilder、ResourceId/ModelResourceNodeId 名字绑定、
  前向引用、显式子图绑定及 RegisterDefinition；保留逻辑依赖推导与输出分析。
- P3：执行 Builder 共享结构辅助，图和程序公开 Validate；
  增加调用接口与实例资源匹配检查，拒绝写只读资源的无效执行图。
- P4：XML v3 采用 Bindings/Dependencies，不读取 v2；
  保留边 ID、读写关系顺序、作用域、精度、来源与架构信息。
  反序列化不执行逻辑依赖重推导。
- P5：RWKV、Phi4 移除泛型句柄、端口契约和 AsyncLocal 描述符中转；
  增加无需实际模型文件的公开模型模块构造回归。
- P6：删除旧机制，更新中英文 README；
  GraphView 查看器与浏览器自测一并迁移至 v3。

### 10.2 关键实现与回归入口

- [公共结构辅助](../src/SharpInference.Graphs/Internal/ModelGraphConstruction.cs)：
  泛型绑定展开和端点、身份、端口、作用域检查。
- [公共 XML 关系编码](../src/SharpInference.Graphs/Internal/ModelGraphXml.cs)：
  ReadWrite 仅保存一次，通过 readEdge/writeEdge 及 readOrder/writeOrder 保留结构身份和顺序。
- [具名构造测试](../src/SharpInference.Tests/NamedGraphConstructionTests.cs)：
  前向引用、显式绑定、多错误诊断、重复校验/构建与快照隔离。
- [序列化与执行测试](../src/SharpInference.Tests/GraphSerializationTests.cs)：
  独立手写 v3 样例、版本拒绝、文档限制、来源映射、状态调用及 CPU 实际数值执行。
- [模型适配测试](../src/SharpInference.Tests/GraphModelAdapterTests.cs)：
  公开 RWKV7、Phi4 Speech/Vision 模块，合成 FP16/FP32 张量目录，不读取权重数值。

### 10.3 验证与限制

- Phi4 项目两个目标框架构建通过，零警告、零错误。
- 新增 Phi4 测试项目引用后执行依赖恢复；没有新增包或安装工具。
- 已运行真实 tiny RWKV6/7 CPU 构造/发射与 RWKV7 CPU logits 回归。
- 已运行真实 tiny RWKV7 D3D12 编译和 logits 回归，不以源码生成替代 GPU 执行。
- GraphView 集成浏览器自测 99 项通过，未安装浏览器依赖。
- 第 7.3 节最终合并回归：174 项通过，零失败、零跳过，包含实际 CPU 残差数值执行。
- 真实模型最终合并回归：RwkvTinyExecutionPipelineTests 共 4 项通过，零失败、零跳过；
  包含 RWKV6/7 CPU 构造发射、RWKV7 CPU logits 和 D3D12 logits，耗时约 3 分 10 秒。
- Core 源码无本次差异；差异格式检查和本文相对链接检查通过。
- 曾暴露旧后处理测试向 Weight 写入的无效夹具；已改为只读 Weight，
  并新增 EG0004 写入拒绝回归，不放松生产校验。
- 本次不重新测量完整的 8 种 Pass 开关与正反顺序性能矩阵；
  保留其测试入口，不据本次构造层重构更新 README 中的性能结论。

真实模型回归的重跑命令（要求既有本地模型目录可用）：

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-build --no-restore --filter "FullyQualifiedName~RwkvTinyExecutionPipelineTests" --verbosity quiet
```
