# SharpInference.Graphs 薄辅助层重构计划

状态：待执行。本文是后续实施依据，不代表代码已经完成迁移。
制定日期：2026-10-16。

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

## 2. 当前实现与迁移影响

实施以执行时工作区为准，不能根据旧提交覆盖现有改动。
制定本计划时，工作区正在使用 Logical、Execution 子目录，存在大量未提交改动及未跟踪文件；
旧根目录图文件已经标记删除。以下链接指向当前工作区，而不是要求恢复旧布局。

| 当前区域 | 现状 | 重构方向 |
|---|---|---|
| [GraphDefinitionBuilder](../src/SharpInference.Graphs/Logical/GraphDefinitionBuilder.cs) | 泛型端口契约、资源句柄、调用端口重建、绑定展开、依赖推导集中在一起 | 非泛型的具名构造入口，复用公共机制 |
| [GraphOperationNodeBuilder](../src/SharpInference.Graphs/Logical/GraphOperationNodeBuilder.cs) | 绑定必须取得当前图拥有的泛型句柄 | 记录具名资源引用，延后解析 |
| [IGraphPortContract](../src/SharpInference.Graphs/Logical/IGraphPortContract.cs)、[InstancePortFactory](../src/SharpInference.Graphs/Logical/InstancePortFactory.cs) | 为定义和每次调用创建同形状 C# 端口对象 | 消除这条必经路径 |
| [ExecutionGraphBuilder](../src/SharpInference.Graphs/Execution/ExecutionGraphBuilder.cs) | 已按集合构造，但绑定展开与逻辑构造方有重复 | 统一具名引用和公共展开逻辑 |
| [LogicalGraphDefinitionValidator](../src/SharpInference.Graphs/Logical/LogicalGraphDefinitionValidator.cs) | 在完整结构上分析，部分无效输入会先抛集合异常 | 先检查声明完整性，再进行领域分析 |
| [ExecutionGraph](../src/SharpInference.Graphs/Execution/ExecutionGraph.cs)、[ExecutionGraphProgram](../src/SharpInference.Graphs/Execution/ExecutionGraphProgram.cs) | 构造时校验，图的 Validate 为私有，程序另做调用 DAG 校验 | 提供一致的公开校验体验，保留图级/程序级边界 |
| [ModelGraphQueries](../src/SharpInference.Graphs/ModelGraphQueries.cs) | 已基于 Core 视图提供模型查询 | 保留，必要时消除重复，不重写查询引擎 |
| [LogicalGraphXml](../src/SharpInference.Graphs/Logical/LogicalGraphXml.cs)、[ExecutionGraphXml](../src/SharpInference.Graphs/Execution/ExecutionGraphXml.cs) | 已使用 v2 通用节点/边格式，不恢复端口契约 | 优先保持格式，统一校验路径 |
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

按顺序执行：声明检查、引用解析、绑定检查、领域分析、Core 物化。
声明检查失败时，跳过依赖该无效项的后续分析，但继续收集独立错误；
不能在 ToDictionary、Single 或 GetNode 处先抛无上下文异常。

规则分层：

1. 公共：唯一性、引用存在、资源作用域、重复端口、成对读写一致性。
2. 逻辑：只读资源写入、输出写入者、子图接口访问及张量兼容、依赖分析。
3. 执行：本地图绑定和依赖 DAG；程序层目标存在、唯一根、调用 DAG 和可达性。

构造方可以在本地 Validate 时不解析外部图目标，但必须明确校验范围；
全图集合 Build/Validate 必须解析目标，不允许把“未提供上下文”解释为目标合法。
执行图现有程序校验不完整检查被调接口，不能在迁移中无说明地增加更严格默认规则；
如共享接口校验暴露现有不一致，在 P0 记录并单独确定行为，不伪装成等价迁移。

逻辑依赖推导必须保持当前规则，而不是简单按添加顺序串行化：

- 读取实例作用域资源时，保留对拥有该资源的调用的依赖。
- 成对 ReadWrite 绑定不按普通纯读推导自己的生产者依赖。
- 普通读取只在存在唯一、不同于读取节点的生产者时推导依赖。
- 多写入者不因声明顺序被自动串行化。
- 显式依赖保留；自依赖保留到环检查；重复非自依赖去重。
- 校验依赖子图是否有环，不对包含状态读写边的整个结构套用 DAG 要求。

CompletionSinks、TerminalWriters 和既有 LG 错误/警告语义保持。
跨实例写入的 LGW003、无序终端写入者的 LGW002 不能未经授权升级为 Error。
公共诊断迁移尽量保留已有代码；执行诊断新增代码要在测试和文档中固定。

校验结构和已知接口不等于证明任意操作数学等价，也不等于后端支持检查。
未知自定义操作不能仅因未注册后端实现就在通用建图层被拒绝。

### 3.5 查询、转换与改写

- 保留 ModelGraphQueries 对 IGraphView 的扩展及当前可见范围语义。
- 保留 Readers、Writers、Bindings、依赖视图、数据流视图、边界及冲突分析。
- 不在构造辅助中复制 Core 的查找、拓扑排序、匹配或可达性实现。
- 转换器继续保留定义复用、调用、作用域、节点和边身份、精度及来源映射。
- 优化器继续基于节点、边、操作及绑定工作，不判断用户定义的 CLR 图类。
- 改写结果经过相应图级、程序级校验；不从绑定重新推导并覆盖优化后的执行依赖。

### 3.6 XML：保持 v2，简化构造路径

当前两套 XML 已保存通用结构，没有序列化 IGraphPortContract。
因此本次默认保留现有 v2 schema、根标识和节点/边表示，不另建更简洁但不兼容的格式。

1. 解析阶段先收集名字记录，节点、资源和定义的出现顺序不影响引用解析。
2. 使用公共声明/结构校验，随后构造成品图；公开反序列化入口仍返回已验证结果。
3. XML 以现有节点和边为权威结构，不再同时增加一套可独立修改的 Bindings 节点。
4. 构建时从绑定展开访问边；XML 导入时保留已有边 ID、顺序和成对绑定身份。
5. 导入后再导出不能重新生成并覆盖显式边 ID，不能重新推导执行依赖。
6. 保留逻辑资源的声明 owner 与绑定 instance 作用域，两者不能混同。
7. 保留来源映射、架构注解、边界、属性、bindingKey 和精度。
8. 保留图定义排序、图内声明顺序及已有规范化输出规则。
9. 保留 DTD 禁用、外部解析禁用、字符数、元素数、深度及枚举值限制。
10. 不保存或恢复用户 CLR 类型名；只分派受支持的逻辑/执行节点种类。

如果测试证明 v2 无法无损表达必要数据，暂停格式改动并补充版本迁移设计；
不能在实施中悄悄把 v2 改成另一种含义。格式错误仍明确失败，不兜底返回空图。

## 4. 文件与 API 迁移清单

| 区域 | 主要操作 |
|---|---|
| Logical/GraphDefinitionBuilder.cs、GraphOperationNodeBuilder.cs | 替换为名字引用，提供 Validate/Build，移除端口契约依赖 |
| Logical/LogicalGraphDefinition.cs、LogicalGraphDefinitionValidator.cs、LogicalGraphAnalysis.cs | 保留成品结构和分析结果，公开校验入口，接入公共诊断与前置校验 |
| Execution/ExecutionGraphBuilder.cs、ExecutionGraph.cs、ExecutionGraphProgram.cs | 共享绑定展开及诊断，统一本地与程序级校验边界 |
| ModelResourceBinding.cs、ModelResourceNodeId.cs、GraphBindingOwner.cs | 复用结构化引用，避免新旧两套资源身份 |
| ModelGraphQueries.cs 及公共模型接口 | 保持查询契约，适配真正变化的类型 |
| 两套 GraphXml、LogicalGraphDefinitionRecord、Internal/GraphXmlSupport.cs | 名字记录装载及公共校验，保留 v2 格式和限制 |
| Execution/DefaultExecutionGraphConverter.cs、ExecutionGraphCompiler.cs、ExecutionGraphRewriter.cs | 适配构造/校验入口，保留转换及改写行为 |
| 模型构造方、VM 优化及发射消费者 | 仅替换构造/绑定/校验调用，不修改算法和 ABI |

全部调用方完成迁移后，删除以下旧机制，而不是保留空壳兼容层：

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
4. 从当前代码构造固定 v2 XML 测试样例，包含逻辑子图、状态和执行来源映射。
5. 样例作为测试数据或代码常量保存，不只存于临时目录或聊天记录。
6. 固定依赖推导、警告、输出名字、结构身份和编译边界行为。

交付：可重跑的基线、兼容样例和迁移清单；P1 前明确公共 API 签名。
本计划制定时未重新运行基线，不能把此前会话测试结果当作执行时基线。

### P1：公共名字引用、构造辅助与校验

操作：

1. 复用现有 ID、作用域、绑定类型，必要时新增最小的未解析声明记录。
2. 统一公共诊断类型和 Error/Warning 判断，保留旧诊断语义。
3. 提取两种构造方共享的绑定检查及读写边展开。
4. 实现前向引用、重复声明拒绝、顺序保留和稳定的内部身份生成。
5. 提供局部校验、外部定义上下文及 Build 失败的明确边界。

验收：无需端口契约即可记录声明；前向引用与实例引用可验证；
Validate 无副作用；同样输入重复物化结果稳定；Core 无改动。

### P2：逻辑图构造迁移

操作：

1. 增加非泛型 GraphDefinitionBuilder 和具名操作绑定入口。
2. 输入、输出、InOut 与临时/权重/状态资源均直接声明。
3. 子图调用使用目标名字及显式端口映射，注册时机与引用顺序解耦。
4. 在完整声明上复用现有依赖推导规则。
5. 提供 Builder.Validate、LogicalGraphDefinition.Validate 和直接返回成品图的便利入口。
6. 迁移逻辑图测试；暂留旧入口仅用于未迁移调用方，不扩展其功能。

验收：Copy/ReLU 链、残差、状态、复用子图和跨实例资源行为保持；
缺失目标、端口或资源返回具名诊断；警告仍为警告。

### P3：执行图构造迁移

操作：

1. 执行图构造使用相同的名字引用和绑定展开辅助。
2. 保留直接接收完整节点/边的用途，使优化器无需逆向重建 Builder。
3. 公开一致的图级 Validate，并明确程序级调用关系校验入口。
4. 保留执行图与程序的已验证快照，以及图/调用顺序。
5. 不为执行图新增逻辑图式的隐式依赖推导。

验收：前向依赖可构造；绑定错误和依赖环明确报告；
程序仍拒绝缺失目标、多根、调用环及不可达定义；
来源映射、架构及边界不丢失。

### P4：XML、转换和改写接入

操作：

1. 两套 XML 先装载记录，后运行公共校验；保留 v2 读取和输出。
2. 对 P0 固定样例执行往返测试，不只测试新 writer 配新 reader。
3. 转换器移除对契约包装的依赖，保持定义复用和冲突检测。
4. 编译器、后处理器和改写器调用统一校验入口。
5. 验证图导入后的内联、提取、融合、依赖缩减和资源重用。
6. 保留 XML 后处理器对模型、后端和逻辑来源的匹配检查。

验收：旧 v2 样例可读、规范化往返稳定；XML 图和代码图进入同一优化路径；
无 CLR 类型恢复；无隐式边 ID 重写或执行依赖重算。

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
Phi4 未被当前主测试项目直接覆盖，不能用 RWKV 测试代替其构建和回归。

### P6：旧机制清理与最终验收

操作：

1. 删除第 4 节列出的旧机制及仅为其存在的模型辅助代码。
2. 搜索源码、测试、脚本和文档，清理旧入口及错误示例。
3. 更新相关 README 与建图说明，给出简单图、子图调用和 XML 往返示例。
4. 运行分层验证；检查旧 API 残留及 Core 零改动。
5. 记录通过、失败、跳过、外部条件和未解决阻塞，不将跳过写成通过。

验收：新入口独立可用，旧路径无调用方；全部必需验证通过；
无长驻兼容层、无静态描述符旁路、无未说明行为变化。

## 6. 必须覆盖的测试矩阵

| 类别 | 用例 | 要求 |
|---|---|---|
| 基本构造 | Copy/ReLU、残差、多输入、InOut | 不定义契约类型即可构造和查询 |
| 后期绑定 | 资源、依赖、子图目标在引用后声明 | 与先声明后引用具有相同语义，不要求声明顺序文本相同 |
| 校验 | 未知资源/节点/目标、重复名字/端口、错误作用域/访问/张量 | 明确诊断位置，不由集合异常先中断 |
| 调用复用 | 同一定义两次调用、显式输出映射、跨实例资源 | 输出身份、作用域及调用依赖正确 |
| 依赖 | 唯一生产者、ReadWrite、多写入、自依赖、依赖环 | 保留既有推导与 LGW002/LGW003 |
| 生命周期 | 重复 Validate/Build、Builder 后续修改 | 不累积关系、不修改旧快照 |
| XML 兼容 | P0 固定 v2 样例、新代码图往返、非规范声明顺序 | 语义、边 ID、元数据、作用域完整 |
| XML 失败 | 缺失根/目标、重复身份、坏枚举、超限、DTD | 明确失败，安全限制不退化 |
| 转换与优化 | 代码图及 XML 图分别转换、内联/提取/融合及现有后处理 | 同等输入下绑定、来源和执行语义一致 |
| VM | CPU 源码、D3D12 源码及调用/调度/屏障生成 | 不支持操作仍明确报错 |
| 模型 | RWKV6/7 构造、Phi4 适配、可用条件下实际推理 | ABI 和数值容差遵循现有测试 |

XML 比较区分语义等价与规范化字符串稳定。
不能只比较节点数量，也不能只依赖新序列化器与新反序列化器相互配合。
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
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj --no-restore --filter "FullyQualifiedName~GraphCoreTests|FullyQualifiedName~GraphViewQueryTests|FullyQualifiedName~LogicalGraphDefinitionTests|FullyQualifiedName~ExecutionGraphConverterTests|FullyQualifiedName~ExecutionGraphPostprocessorTests|FullyQualifiedName~VmExecutionGraphEmitterTests" --verbosity minimal
```

加入新增测试选择器。GraphCoreTests 用于确认 Core 未改动后的集成行为，
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
| 现有工作区正在重组文件 | 以实际文件为准，不恢复已删除旧文件，不覆盖他人改动 |
| 泛型辅助再次膨胀为框架 | 只提取两方实际重复逻辑，一层辅助，不引入新项目或类型注册 |
| 后期绑定改变依赖语义 | 固定唯一写入者、ReadWrite、多写入者和实例资源测试 |
| 直接构造或 XML 绕过校验 | 共享前置校验和领域分析，所有成品入口维持有效性 |
| XML 简化破坏外部图及来源映射 | 默认保留 v2，固定旧样例，保持边身份和元数据 |
| 新执行接口校验意外拒绝旧图 | P0 记录差异，不暗中加强默认行为 |
| 模型测试被外部条件跳过 | 提供无外部依赖适配测试，单独记录条件性数值验证 |

遇到无法保持的 XML 兼容、资源 ABI 或依赖语义差异时，先补充设计与影响说明，
不得用静默默认值、忽略引用或放宽测试掩盖问题。

## 9. 完成定义

- [ ] Core 源码和公开契约没有本次引入的改动。
- [ ] 逻辑图和执行图都使用具名构造及明确的校验入口。
- [ ] 两种图共用必要模型类型和薄的辅助，没有第二套通用图引擎。
- [ ] 简单图、状态图和子图调用无需端口契约或泛型资源标签。
- [ ] 字典/ID 查找和既有 Core 查询能力可用，前向引用有正式回归。
- [ ] XML v2 兼容样例通过，图语义、依赖、来源及作用域不丢失。
- [ ] 所有模型和下游调用方完成迁移，旧契约和静态描述符旁路清理。
- [ ] 必需构建和测试通过，条件性验证状态明确。
- [ ] 示例与文档更新，无未说明的执行行为变化。

执行完毕时回填阶段结果及验证证据；未执行的项目保持未勾选。
