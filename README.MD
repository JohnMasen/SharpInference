# SharpInference

English | [简体中文](./README_CN.MD)

SharpInference is an experimental .NET inference compiler with CPU and
Direct3D 12 VM backends. The repository is under active development.

## Current architecture

```text
model tensor catalog + model graph provider
  -> LogicalGraphDefinition
  -> ExecutionGraphProgram
  -> VmExecutionGraphEmitter
  -> VmProgram
  -> CpuVmCompiler / D3D12VmCompiler
  -> CPU assembly / GPU DXIL
```

The graph layers use the shared strongly typed graph library:

- Built graphs represent their structure with `Graph<TNode, TEdge>`.
- Resources, operations, and graph calls are nodes.
- Dependencies, reads, and writes are edges.
- Logical and execution graphs share traversal, views, pattern matching, and
  batch rewrite primitives.
- Experiments may explicitly insert `XmlExecutionGraphProgramPostprocessor` to
  replace a converted program with a validated, prewritten execution-graph XML.

Previous graph and runtime execution paths have been removed. There is no
compatibility execution path or silent fallback.

## Named construction, validation, and XML

`SharpInference.Graphs` provides shared model types and thin construction/query
helpers above the unchanged `SharpInference.Graphs.Core`. Port-contract classes
and generic resource tags are no longer required:

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
var logical = builder.Build(); // Rejects errors; never returns a partial graph.
var restored = LogicalGraphXml.DeserializeDefinition(LogicalGraphXml.Serialize(logical));
```

Names use ordinal comparison and may refer to later declarations. Validation is
non-mutating; each build produces a separate snapshot. Calls bind every public
port explicitly. Execution programs also validate callee interfaces and the
single-rooted call DAG. Execution dependencies are not implicitly reinferred.

Logical and execution XML now use **v3** only, with named `Bindings` and
`Dependencies`. Earlier formats are rejected rather than upgraded. Round trips
preserve necessary edge identities, ordering, resource scopes, precision, and
provenance without restoring user CLR classes. XML contains graph descriptions,
not tensor contents. DTD/external entity restrictions and document limits remain.
The [GraphView tool](tools/GraphView/README.md) also consumes v3.

## Optional execution-graph postprocessors

Postprocessors are model-independent and explicitly enabled; the default
converter and emitter remain unoptimized. Each pass returns an execution graph
program using the existing nodes, edges, and graph calls:

```csharp
var compiler = new ExecutionGraphCompiler(
[
    new DependencyReductionPostprocessor(),
    new TemporaryResourceReusePostprocessor(),
    new OperationTemplateSharingPostprocessor(),
]);
var execution = compiler.Compile(logical, context);
```

- Dependency reduction preserves dependency reachability, including parallel
  edges, without changing resource bindings.
- Temporary reuse requires identical tensor descriptors and dependency-proven
  disjoint lifetimes, not merely nonoverlapping positions in one topological
  order. Only unbound parent-owned temporaries initialized by known complete
  version-one primitive output contracts are eligible. Calls, externally bound
  resources, unknown/partial writes, and same-operation aliases are excluded.
- Template sharing replaces repeated operation contracts with calls to shared
  single-operation atomic graphs. Every invocation still executes. Signatures
  include version, attributes, precision, ordered ports, access, and tensor
  descriptors. Instance-scoped and alias-bearing bindings are excluded.

The passes are independent and can be omitted individually. Run temporary reuse
before template sharing so it can inspect primitive initialization. Sharing
benefits depend on the backend; CPU and D3D12 need separate measurements.
No VM, backend compiler, instruction contract, or logical graph is changed.
In the tiny RWKV-6/7 measurements, template sharing substantially reduced GPU
preparation cost, while temporary reuse reduced slots but regressed CPU
steady-state execution. Do not treat enabling all passes as a universal speedup.

### Postprocessor benefit ratings

The abbreviations below identify the concrete postprocessors:

| Abbreviation | Postprocessor | Purpose |
|---|---|---|
| D | `DependencyReductionPostprocessor` | Dependency reduction |
| T | `TemporaryResourceReusePostprocessor` | Temporary resource reuse |
| S | `OperationTemplateSharingPostprocessor` | Operation template sharing |

Each dimension uses a qualitative 1-5 benefit score; higher is better:
1 = no clear benefit or a regression, 2 = slight or inconsistent benefit,
3 = moderate benefit, 4 = clear benefit, and 5 = substantial benefit.

- Inference-time reduction includes per-invocation validation, execution
  wrappers, and computation with a prepared execution context.
- Preparation-time reduction considers pass overhead, emission, compilation,
  and executor creation rather than compilation alone.
- Inference-memory reduction concerns tensor and temporary buffers, not
  compilation caches or generated code.

| Backend | Postprocessor | Inference-time reduction | Preparation-time reduction | Inference-memory reduction |
|---|---|---:|---:|---:|
| CPU | D | 1 | 1 | 1 |
| CPU | T | 1 | 2 | 4 |
| CPU | S | 4 | 2 | 1 |
| D3D12 | D | 1 | 1 | 1 |
| D3D12 | T | 2 | 2 | 4 |
| D3D12 | S | 1 | 5 | 1 |

These ratings describe the current implementations enabled individually in the
tiny RWKV-6/7 experiments, not guarantees for other models or hardware. T refers
to the current maximal-reuse pass, not the experimental bounded-reuse graphs.
Memory ratings are qualitative: complete peak memory was not measured, and slot
count reductions must not be interpreted as equal percentage memory reductions.
Combinations require separate evaluation; scores are not additive.

## Supported validation scope

- RWKV-6 and RWKV-7 graph construction from compatible rwkv.cpp GGML models.
- CPU VM compilation and execution.
- Direct3D 12 VM shader compilation and execution.
- Phi-4 graph construction and tokenizer/GGUF components.

The test suite includes end-to-end RWKV-6/7 tiny-model comparisons on CPU and
D3D12 for all eight postprocessor switch combinations. It checks bitwise-equal
logits after each of three tokens and records pass/emission/compilation/executor
times, slots, kernels, dependencies, and resident steady-state forward timings.
External model files are not included in the repository.

To reproduce the isolated matrix (30 warmups and 200 samples per configuration,
both configuration orders), supply the rwkv.cpp test-model directory:

```powershell
$env:DOTNET_TieredCompilation = "0"
dotnet test src\SharpInference.Tests\SharpInference.Tests.csproj -c Release `
  -p:RwkvTestModelDirectory="D:\GithubRoot\rwkvcpp\tests" `
  --filter FullyQualifiedName~RwkvPostprocessorTests `
  --logger "console;verbosity=detailed"
```

Switch masks are `1` for dependency reduction, `2` for temporary reuse, and `4`
for template sharing; `0` disables all and `7` enables all. GPU forward excludes
uploads/readbacks. Compilation includes backend source generation. Timings are
diagnostic, not test thresholds; compare repeated runs rather than cold-start
or noisy single samples. External-model tests use the default directory described
below when the model-directory property is not supplied; unavailable defaults
are reported as skipped tests rather than passing without executing inference.

## Requirements

- .NET 10 SDK.
- Windows for D3D12 projects and the complete solution.
- A compatible Direct3D 12 adapter for GPU tests.

## Build and test

```powershell
dotnet build .\SharpInference.slnx
dotnet test .\SharpInference.slnx
```

To run the external tiny RWKV inference tests:

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj `
  -p:RwkvTestModelDirectory="D:\path\to\rwkvcpp\tests"
```

The expected directory contains the tiny RWKV model and expected-logits files
used by `RwkvTinyExecutionPipelineTests`.

Visual Studio can run or debug these tests directly without supplying a property.
The shared [model configuration](src/SharpInference.Tests/RwkvTestModels.cs)
uses `D:\GithubRoot\rwkvcpp\tests` as its default directory. An explicit
`RwkvTestModelDirectory` build property takes priority. If no property is supplied
and the default directory is absent, model tests are reported as skipped with a
configuration message. Invalid explicit paths or missing model/reference files
fail the tests instead of silently falling back. Rebuild and refresh test
discovery after changing the build property or the default directory.

## Project layout

| Area | Responsibility |
| --- | --- |
| `src/SharpInference.Graphs.Core` | Generic strongly typed graph, views, algorithms, patterns, and rewrites |
| `src/SharpInference.Graphs` | Logical and execution graph domain models, XML, conversion, and rewrites |
| `src/SharpInference.Models.Rwkv` | RWKV logical graph providers |
| `src/SharpInference.Models.Phi4` | Phi-4 graph and model components |
| `src/SharpInference.Instructions*` | VM instruction contracts and CPU/D3D12 implementations |
| `src/SharpInference.Vm` | VM program contracts |
| `src/SharpInference.Vm.Optimization` | Execution graph to VM emission and model bindings |
| `src/SharpInference.Backends.CpuVm` | CPU VM compiler and executor |
| `src/SharpInference.Backends.D3D12Vm` | D3D12 VM compiler and executor |
| `src/SharpInference.Tests` | Graph, emitter, and end-to-end inference tests |
| `tools/GraphView` | Logical and execution graph XML viewer |

### Graph domain namespaces

The graph domain project uses one type per file, including records, interfaces,
enums, and internal analysis records:

- `SharpInference.Graphs`: shared model contracts, graph queries, tensor and
  precision primitives.
- `SharpInference.Graphs.Logical` ([Logical](src/SharpInference.Graphs/Logical)):
  logical definitions, nodes, edges, builders, ports, diagnostics, and validation.
- `SharpInference.Graphs.Execution` ([Execution](src/SharpInference.Graphs/Execution)):
  execution programs, nodes, edges, builders, conversion, compilation, and rewrites.
- `SharpInference.Graphs.Internal`: shared internal XML support.

Use `LogicalGraphXml.Serialize` / `DeserializeDefinition` for logical graphs and
`ExecutionGraphXml.Serialize` / `DeserializeProgram` for execution programs.
These replace the former `GraphXml` entry points; the XML format is unchanged.
Namespace consumers must import the corresponding logical or execution namespace.
The project and assembly boundaries remain unchanged.

## License

The project is licensed under the [MIT License](./LICENSE). See
[third-party notices](./THIRD-PARTY-NOTICES.txt) for bundled dependencies and
assets. Model weights and external datasets remain subject to their own
licenses.
