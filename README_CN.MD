# SharpInference

[English](./README.md) | 简体中文

SharpInference 是一个实验性的 .NET 推理编译器，提供 CPU 与 Direct3D 12 VM
后端。项目仍在持续开发中。

## 当前架构

```text
模型张量目录 + 模型图提供器
  -> LogicalGraphDefinition
  -> ExecutionGraphProgram
  -> VmExecutionGraphEmitter
  -> VmProgram
  -> CpuVmCompiler / D3D12VmCompiler
  -> CPU 程序集 / GPU DXIL
```

图层统一使用强类型通用图库：

- 成品图以 `Graph<TNode, TEdge>` 表示结构；
- Resource、Operation 和 GraphCall 是节点；
- Dependency、Read 和 Write 是边；
- 逻辑图与执行图共享遍历、View、模式匹配和批量重写能力。
- 实验时可显式插入 `XmlExecutionGraphProgramPostprocessor`，用经过校验的预写
  执行图 XML 替换转换结果。

旧图与旧运行时执行路径已经删除，不保留兼容执行路径，也不会静默回退。

## 具名建图、校验与 XML

`SharpInference.Graphs` 是逻辑图和执行图的薄辅助层；底层查询和改写继续使用
`SharpInference.Graphs.Core`。建图通过名字后期绑定，不再需要端口契约类或泛型资源标签。
下面描述 `output = ReLU(input)`，操作可以先引用资源，再声明资源：

```csharp
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

var vector = new TensorDescriptor(GraphElementType.Float32, [4]);
var builder = new GraphDefinitionBuilder("relu");
builder.AddNode("relu", PrimitiveGraphOperations.Relu, node =>
{
    node.AddInput("input", new ResourceId("input"));
    node.AddOutput("output", new ResourceId("output"));
});
builder.AddInput("input", vector);
builder.AddOutput("output", vector);

var diagnostics = builder.Validate();
foreach (var diagnostic in diagnostics)
    Console.WriteLine($"{diagnostic.Code}: {diagnostic.Message}");
var logical = builder.Build(); // 有 Error 则明确失败，不返回部分图。
```

名字区分大小写，资源和节点各有命名空间；资源访问与执行依赖是不同关系。
逻辑图在完整声明上推导唯一生产者依赖，不以添加顺序串行化多写入者。
`Validate()` 不修改 Builder；每次 `Build()` 返回独立图快照。
执行图构造也支持名字引用，但保留显式执行依赖，不套用逻辑图的隐式推导。
执行程序的 `Validate()` 另外检查调用目标、端口兼容性和单根调用 DAG。

子图调用显式绑定所有输入、输出和 InOut。调用与子图定义是不同身份，
同一定义可以复用，调用绑定互不覆盖：

```csharp
var parent = new GraphDefinitionBuilder("model");
parent.AddInput("input", vector);
parent.AddOutput("output", vector);
parent.AddCall("activation", logical, call =>
{
    call.AddInput("input", new ResourceId("input"));
    call.AddOutput("output", new ResourceId("output"));
});
var model = parent.Build();

var xml = LogicalGraphXml.Serialize(model);
var restored = LogicalGraphXml.DeserializeDefinition(xml);
```

`LogicalGraphXml` 和 `ExecutionGraphXml` 仅接受 XML **v3**，旧格式不兼容。
XML 保存普通资源、操作、调用、`Bindings` 和 `Dependencies`，不恢复用户 CLR 类型。
ReadWrite 绑定只保存一次，必要的读写边身份和关系顺序仍保留；
执行图的来源映射、架构注解、显式依赖及精度不会在往返时被重新推导。
读取后统一校验，禁止 DTD 和外部实体，并限制文档大小、元素数量和深度。
XML 只包含图描述，不包含实际权重、输入或运行中状态的数值。
查看器同样使用 v3，见 [GraphView](tools/GraphView/README.md)。

## 可选执行图 Postprocessor

三个 Pass 都不依赖模型名称，需要显式启用；默认 Converter 和 Emitter 不自动
优化。输入输出都是使用现有节点、边和 GraphCall 的执行图程序：

```csharp
var compiler = new ExecutionGraphCompiler(
[
    new DependencyReductionPostprocessor(),
    new TemporaryResourceReusePostprocessor(),
    new OperationTemplateSharingPostprocessor(),
]);
var execution = compiler.Compile(logical, context);
```

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
| `src/SharpInference.Graphs` | 逻辑图和执行图领域模型、XML、转换与重写 |
| `src/SharpInference.Models.Rwkv` | RWKV 逻辑图提供器 |
| `src/SharpInference.Models.Phi4` | Phi-4 图与模型组件 |
| `src/SharpInference.Instructions*` | VM 指令契约及 CPU/D3D12 实现 |
| `src/SharpInference.Vm` | VM 程序契约 |
| `src/SharpInference.Vm.Optimization` | 执行图到 VM 的发射与模型绑定 |
| `src/SharpInference.Backends.CpuVm` | CPU VM 编译器与执行器 |
| `src/SharpInference.Backends.D3D12Vm` | D3D12 VM 编译器与执行器 |
| `src/SharpInference.Tests` | 图、Emitter 与端到端推理测试 |
| `tools/GraphView` | 逻辑图和执行图 XML 查看器 |

### 图领域命名空间

图领域项目按一个类型一个文件组织，包含 record、接口、枚举和内部分析记录：

- `SharpInference.Graphs`：共享模型契约、图查询、张量与精度基础类型。
- `SharpInference.Graphs.Logical`（[Logical](src/SharpInference.Graphs/Logical)）：
  逻辑图定义、节点、边、Builder、端口、诊断与校验。
- `SharpInference.Graphs.Execution`（[Execution](src/SharpInference.Graphs/Execution)）：
  执行图程序、节点、边、Builder、转换、编译与重写。
- `SharpInference.Graphs.Internal`：两种序列化器共用的内部 XML 辅助实现。

逻辑图使用 `LogicalGraphXml.Serialize` / `DeserializeDefinition`，
执行图程序使用 `ExecutionGraphXml.Serialize` / `DeserializeProgram`。
两者替代原 `GraphXml` 入口，XML 格式保持不变。
消费者需要导入对应的逻辑图或执行图命名空间，项目和程序集边界不变。

## 许可证

项目使用 [MIT License](./LICENSE)。第三方依赖和资源见
[THIRD-PARTY-NOTICES.txt](./THIRD-PARTY-NOTICES.txt)。模型权重和外部数据集
仍受其各自许可证约束。
