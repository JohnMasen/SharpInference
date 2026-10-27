# SharpInference

[English](./README.MD) | 简体中文

SharpInference 是一个实验性的 .NET 推理编译器，提供 CPU 与 Direct3D 12 VM
后端。项目仍在持续开发中。

## 当前架构

```text
模型张量目录 + 模型图提供器
  -> LogicalGraphPackage
  -> ExecutionGraphPackage
  -> VmExecutionGraphEmitter
  -> VmProgram
  -> CpuVmCompiler / D3D12VmCompiler
  -> CPU 程序集 / GPU DXIL
```

计算源模型分为三个层次：`Graph` 定义资源、操作／调用、命名绑定和显式
`GraphFlow`；`LogicalGraph` 增加编译意图；`ExecutionGraph` 增加选定实现、
存储安排及来源。通用结构投影和算法继续使用未改动的 `SharpInference.Graphs.Core`。

- 实验时可显式插入 `XmlExecutionPackagePostprocessor`，用经过校验的预写
  执行图 XML 替换转换结果。

旧图与旧运行时执行路径已经删除，不保留兼容执行路径，也不会静默回退。

## 具名建图、校验与 XML

`SharpInference.Graphs` 提供可编辑的计算定义；构造登记声明，显式验证只读报告问题。
Package 自身就是同阶段图的集合，通过 `MainGraphId` 明确入口。
下面描述 `output = ReLU(input)`：

```csharp
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

var vector = new TensorDescriptor(ElementType.Float32, [4]);
var logical = InferenceGraphBuilder.Builder<LogicalGraph>("relu", graph =>
{
    var input = graph.Resource("input", vector, GraphIO.In);
    var output = graph.Resource("output", vector, GraphIO.Out);
    // output = ReLU(input)
    var relu = graph.Operation("relu", Operations.ReLU);
    relu.Bind("input", input, ResourceAccess.Read);
    relu.Bind("output", output, ResourceAccess.Write);
});
var package = new LogicalGraphPackage(logical);
var diagnostics = package.Validate();
foreach (var diagnostic in diagnostics)
    Console.WriteLine($"{diagnostic.Code}: {diagnostic.Message}");
var analyses = package.Analyze(); // 图 ID -> GraphAnalysis，不补边。
var analysis = package.Analyze(logical);
LogicalGraphPackage snapshot = package.Snapshot();
var xml = LogicalGraphPackageXml.Serialize(package);
var restored = LogicalGraphPackageXml.Deserialize(xml);
```

名字区分大小写，绑定支持资源对象和字符串身份。`Validate()` 不改源图；
构造不自动验证或补边。转换／编译先捕获快照，验证整个 Package，再分析值来源与依赖，
不以声明顺序强制串行化。`CompileBoundary` 仅表达逻辑作者意图；
Execution 不保留该字段，也不将它作为强制内联规则。

`Graph.Validate()` 为 public virtual，各阶段 override 负责自身专属检查。
阶段包的 `Validate()` 包含只读的完整值／依赖检查；`ValidateStructure()` 仅检查
声明、引用和结构，不要求已有完整值方案，供优化器和显式图操作处理未完成源。
阶段包支持 `new(main, helpers)`，`Snapshot()` 保持具体包类型。
`Analyze()` 返回所有定义的 `IReadOnlyDictionary<string, GraphAnalysis>`，
`Analyze(graph)` 返回单图结果；分析和验证都不补写依赖边。

子图调用显式绑定接口；同一定义可复用，调用点各有身份和实参：

```csharp
var parent = InferenceGraphBuilder.Builder<LogicalGraph>("model", graph =>
{
    var input = graph.Resource("input", vector, GraphIO.In);
    var output = graph.Resource("output", vector, GraphIO.Out);
    // output = relu(input)
    var call = graph.Call("activation", logical);
    call.Bind("input", input, ResourceAccess.Read);
    call.Bind("output", output, ResourceAccess.Write);
});
var model = new LogicalGraphPackage(parent, logical);
var modelXml = LogicalGraphPackageXml.Serialize(model);
```

资源用 `GraphIO`、`Scope`（Call／Session／Global）、可选 `Usage`、独立的
`IsState`、权限及可选 `BindingKey` 描述数据。资源和 Action 的 `Category`
为可扩展字符串。`first.Then(second)` 将完成到开始的约束写入 `GraphFlow`，
允许分支和汇合，不等于把所有动作按登记顺序串行执行。

XML 保留 **version 1** 外形，`stage` 为 business／logical／execution。
公共 `GraphPackageXml.Serialize`／`Deserialize<Graph>` 仅处理 business 公用格式；
逻辑包使用 `LogicalGraphPackageXml.Serialize/Deserialize`，
执行包使用 `ExecutionGraphPackageXml.Serialize/Deserialize`，不通过公共入口保存阶段字段。
形状保存为 `Tensor/Dimension`，元数据为 `Metadata/Entry`，Action 内保存 `Binding`，
`GraphFlow/Then` 保存显式偏序。执行包另有 `Target` 子节点、资源 `Storage`、
动作 `Implementation`／`Dispatch`／`Sources`。旧图 XML 和候选 draft XML 不兼容。
反序列化恢复声明，业务验证由显式 `Validate()` 或编译入口执行。
XML 禁止 DTD 和外部实体，并限制文档大小、元素数量和深度。
XML 只包含图描述，不包含实际权重、输入或运行中状态的数值。
查看器能力见 [GraphView](tools/GraphView/README.md)，与源包协议分别说明。

## 可选执行图 Postprocessor

三个 Pass 都不依赖模型名称，需要显式启用；默认 Converter 和 Emitter 不自动
优化。输入输出都是 `ExecutionGraphPackage`：

```csharp
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

IInstructionCollectionProvider[] providers = [new CpuFloat32InstructionCollection()];
var context = new ExecutionGraphCompilationContext("example", "cpu-vm");
var compiler = new ExecutionGraphCompiler(providers,
[
    new DependencyReductionPostprocessor(),
    new TemporaryResourceReusePostprocessor(),
    new OperationTemplateSharingPostprocessor(),
]);
var execution = compiler.Compile(package, context);
// package 为 LogicalGraphPackage，execution 为 ExecutionGraphPackage。
var program = execution.Compile(VmTarget.Cpu, providers, options: null);
```

Converter 和 Compiler 都需要 providers。逻辑转换或直接执行包 Compile 的准备阶段
选择实现与精度；Emitter 消费已准备好的执行包，不负责挑选实现。
Compiler 统一构造器为 `(providers, postprocessors: null, converter: null)`；
自定义转换器使用 `new ExecutionGraphCompiler(providers, converter: customConverter)`。
转换后及每步后处理后，Prepare 统一重算依赖、补齐存储、选择并校验实现；
Emitter 只读校验实现合同，不选择实现。独立执行包应用后处理时使用
`new ExecutionPassContext(context, index, processor.Name)`，逻辑 Source 可空，
无需伪造 LogicalGraph。

执行项目提供只读诊断入口 `ExecutionPlanValidation.ValidatePlan(package)`，
以及解析存储别名的 `ExecutionPlanValidation.StorageSlot(graph, id)`。
`ExecutionGraphRewriter` 同样归执行项目，提供显式 replace／fuse／extract／inline；
是否应用由外部优化器决定，不在图层选择指令或编译 VM。
执行包通过实例方法 `SetTarget`／`GetTarget` 和常量 `BackendMetadataKey` 保存目标，
原 `ExecutionPackage` 辅助类已移除。

- 依赖精简保持所有依赖可达性，正确处理平行边，不改变资源绑定。
- Temporary 复用要求 TensorDescriptor 完全一致，并由依赖偏序证明生命周期
  不重叠，不能只根据某次拓扑排序的位置判断。仅处理无外部绑定、父图拥有、
  由已知完整写入输出的 version-one primitive 初始化的临时资源。GraphCall
  访问、被调用方对外绑定的资源、未知或部分写入、同 operation 输入输出别名
  都保守排除。
- 模板共享将重复 operation 契约替换为共享的单 operation Atomic 子图调用，
  每次调用仍执行一次。签名包含版本、属性、精度、有序端口、访问模式和
  TensorDescriptor；实例作用域或带资源别名的绑定不提取。

三个 Pass 可独立省略。Temporary 复用应先于模板共享，以便分析 primitive
初始化。共享的性能收益依赖后端，CPU 和 D3D12 必须分别评估。不会修改逻辑图、
VM、后端编译器或 instruction 契约。
本轮 RWKV-6/7 tiny 实测中，模板共享显著降低 GPU 准备成本；Temporary 复用减少
slot，但 CPU 稳态执行回退。不能将三个 Pass 全开视为通用的加速配置。

### Postprocessor 收益评分

下列缩写明确对应具体处理器：

| 缩写 | 处理器 | 功能 |
|---|---|---|
| D | `DependencyReductionPostprocessor` | 依赖精简 |
| T | `TemporaryResourceReusePostprocessor` | 临时资源复用 |
| S | `OperationTemplateSharingPostprocessor` | 操作模板共享 |

各维度采用定性的 1-5 分收益评分，分数越高表示降低效果越好：
1 分表示无明显收益或存在回退，2 分表示轻微或不稳定收益，3 分表示中等收益，
4 分表示明显收益，5 分表示显著收益。

- 降低推理时间：包含准备好执行上下文后，每次推理的验证、执行包装和计算。
- 降低准备时间：考虑 Pass 自身开销、Emit、Compile 和执行器创建，
  不只看编译时间。
- 降低推理内存：评价张量和临时缓冲，不把编译缓存或生成代码的减少
  等同于推理内存减少。

| 后端 | 处理器 | 降低推理时间 | 降低准备时间 | 降低推理内存 |
|---|---|---:|---:|---:|
| CPU | D | 1 | 1 | 1 |
| CPU | T | 1 | 2 | 4 |
| CPU | S | 4 | 2 | 1 |
| D3D12 | D | 1 | 1 | 1 |
| D3D12 | T | 2 | 2 | 4 |
| D3D12 | S | 1 | 5 | 1 |

评分针对当前实现单独开启时的 RWKV-6/7 tiny 实验，不保证其他模型或硬件
得到相同收益。T 指当前最大复用处理器，不包含实验性的有限复用执行图。
内存评分为定性评价：尚未测量完整峰值内存，不能将 slot 数量下降比例
直接当作内存下降比例。组合方案需要独立评估，分数不能相加。

## 当前验证范围

- 从兼容 rwkv.cpp 的 GGML 模型构建 RWKV-6、RWKV-7 图；
- CPU VM 编译与执行；
- Direct3D 12 VM shader 编译与执行；
- Phi-4 图构建以及 tokenizer/GGUF 组件。

测试包含 RWKV-6/7 tiny 模型在 CPU、D3D12 上的全部八种 Pass 开关组合。
逐一验证三个 token 的 logits 与未优化配置逐位一致，并记录 Pass、Emit、
编译、Executor 创建时间，以及 slot、kernel、dependency 和稳态 forward。
外部模型文件不包含在仓库中。

复现独立矩阵（每种配置预热 30 次、采样 200 次，正反两种配置顺序）：

```powershell
$env:DOTNET_TieredCompilation = "0"
dotnet test src\SharpInference.Tests\SharpInference.Tests.csproj -c Release `
  -p:RwkvTestModelDirectory="D:\GithubRoot\rwkvcpp\tests" `
  --filter FullyQualifiedName~RwkvPostprocessorTests `
  --logger "console;verbosity=detailed"
```

开关 bitmask：`1` 为依赖精简，`2` 为 Temporary 复用，`4` 为模板共享；
`0` 全关，`7` 全开。GPU forward 不包含上传和读回；编译时间包含后端源码生成。
性能数据用于诊断，不作为测试通过阈值，应比较重复实验，避免冷启动和单次噪声。
未设置模型目录属性时，外部模型测试使用下述默认目录；默认目录不可用时，
明确显示跳过及配置原因，不再空跑通过。

## 环境要求

- .NET 10 SDK；
- 完整解决方案和 D3D12 项目需要 Windows；
- GPU 测试需要兼容的 Direct3D 12 适配器。

## 构建与测试

```powershell
dotnet build .\SharpInference.slnx
dotnet test .\SharpInference.slnx
```

运行外部 tiny RWKV 推理测试：

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj `
  -p:RwkvTestModelDirectory="D:\path\to\rwkvcpp\tests"
```

也可以在 Visual Studio 测试资源管理器中直接运行或调试，无需传入属性。
共享[模型配置](src/SharpInference.Tests/RwkvTestModels.cs)中的默认目录为
`D:\GithubRoot\rwkvcpp\tests`，显式构建属性 `RwkvTestModelDirectory` 优先。
未配置属性且默认目录不存在时，模型测试显示跳过及原因；显式路径无效、
模型文件或参考 logits 缺失时测试失败，不回退到默认目录。
修改构建属性或默认目录后，请重新构建并刷新测试发现。

指定目录应包含 `RwkvTinyExecutionPipelineTests` 使用的 tiny RWKV 模型和
expected-logits 文件。

## 项目结构

| 区域 | 职责 |
| --- | --- |
| `src/SharpInference.Graphs.Core` | 通用强类型图、View、算法、模式与重写 |
| `src/SharpInference.Graphs` | 公共业务图、张量／操作／访问／精度值合同、构图、查询、校验与公用 XML |
| `src/SharpInference.Graphs.Logical` | 逻辑图／包、编译边界意图、分析与逻辑 XML |
| `src/SharpInference.Graphs.Execution` | 执行图／包、执行值合同、方案校验、显式改写与执行 XML |
| `src/SharpInference.Models.Rwkv` | RWKV 逻辑图提供器 |
| `src/SharpInference.Models.Phi4` | Phi-4 图与模型组件 |
| `src/SharpInference.Instructions*` | 指令合同、选择支持及 CPU/D3D12 实现；复用图层值合同 |
| `src/SharpInference.Vm` | VM 程序契约 |
| `src/SharpInference.Vm.Optimization` | 转换、具体指令选择、编译、优化策略与 VM 发射；消费两种阶段图 |
| `src/SharpInference.Backends.CpuVm` | CPU VM 编译器与执行器 |
| `src/SharpInference.Backends.D3D12Vm` | D3D12 VM 编译器与执行器 |
| `src/SharpInference.Tests` | 图、Emitter 与端到端推理测试 |
| `tools/GraphView` | 逻辑图和执行图 XML 查看器 |

### 图领域命名空间

图领域命名空间按职责组织：

- `SharpInference.Graphs`（[项目](src/SharpInference.Graphs)）：`ElementType`、
  `TensorDescriptor`、`ResourceAccess`、`PrecisionRequirement`、`NumericTypeCompatibility`、
  `OperationId`、可编辑 Graph、资源／Action、绑定、公共包、诊断、查询及 business GraphPackageXml。
- `SharpInference.Graphs.Logical`（[项目](src/SharpInference.Graphs.Logical)）：
  `LogicalGraph`、`GraphCompileBoundary`、`LogicalGraphPackage` 和 `LogicalGraphPackageXml`。
- `SharpInference.Graphs.Execution`（[项目](src/SharpInference.Graphs.Execution)）：
  `ExecutionGraph`、`ExecutionGraphPackage`、`ExecutionGraphPackageXml`、选定实现、存储及来源；
  `ExecutionTarget`（原 `InstructionTarget`）、`ExecutionPrecision`（原 `KernelPrecisionProfile`）、
  `ExecutionDispatch`（原 `GraphInstructionDispatch`）。
- `SharpInference.Instructions`：具体指令合同和 `PrimitiveOperations`，直接复用执行值合同。
- `SharpInference.Vm.Optimization.GraphHelper`（[GraphHelper](src/SharpInference.Vm.Optimization/GraphHelper)）：
  转换、编译和后处理契约，不定义执行图模型或显式改写。
- `SharpInference.Graphs.Internal`：共享构造、校验、XML 辅助和无阶段识别的固定点分析机制。
  GraphXmlSupport 等 XML 实现细节保持 internal。

分析公开入口在阶段包，共享引擎不识别阶段。
`GraphPackage<TGraph>` 保留为公共基类，不再作为逻辑／执行阶段的生产包 API。
公共 `GraphPackageXml.Serialize(GraphPackage<Graph>)` 为非泛型入口。
`Deserialize<Graph>(xml)` 仍可用；任意派生 `TGraph` 明确抛出 `NotSupportedException`，
避免旧调用经 business 门面丢失阶段字段。阶段专属入口不变。
旧 GraphXml、LogicalGraphXml、ExecutionGraphXml 入口仍已移除。
消费者引用对应 Logical 或 Execution 独立项目并导入其命名空间。
构造、查询、校验和读写执行图不需要引用优化器；优化器依赖两种图模型，不能反向引用。
Graphs 仅依赖 Graphs.Core。Logical 和 Execution 是独立项目，均仅引用 Graphs 和
Graphs.Core，彼此不引用，也不引用 Instructions、VM、优化器或模型。
指令系统引用图层，不能反向引用。模型构造、具体指令选择、转换／编译及 VM
`Compile` 扩展保留在原消费者项目。

## 许可证

项目使用 [MIT License](./LICENSE)。第三方依赖和资源见
[THIRD-PARTY-NOTICES.txt](./THIRD-PARTY-NOTICES.txt)。模型权重和外部数据集
仍受其各自许可证约束。
