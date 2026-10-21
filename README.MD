# SharpInference

English | [简体中文](./README_CN.MD)

SharpInference is an experimental .NET inference compiler with CPU and
Direct3D 12 VM backends. The repository is under active development.

## Current architecture

```text
model tensor catalog + model graph provider
  -> GraphPackage<LogicalGraph>
  -> GraphPackage<ExecutionGraph>
  -> VmExecutionGraphEmitter
  -> VmProgram
  -> CpuVmCompiler / D3D12VmCompiler
  -> CPU assembly / GPU DXIL
```

The computation source model has three layers: `Graph` defines resources,
operations/calls, named bindings, and explicit `GraphFlow`; `LogicalGraph` adds
compilation intent; `ExecutionGraph` adds selected implementations, storage, and
source origins. Structure/algorithm projections use the unchanged
`SharpInference.Graphs.Core` library.

- Experiments may explicitly insert `XmlExecutionPackagePostprocessor` to
  replace a converted program with a validated, prewritten execution-graph XML.

Previous graph and runtime execution paths have been removed. There is no
compatibility execution path or silent fallback.

## Named construction, validation, and XML

`SharpInference.Graphs` provides editable computation definitions. Construction
registers declarations; explicit validation reports problems without changing the
source. The package is the collection itself, with an explicitly selected main graph:

```csharp
using SharpInference.Instructions;
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
var package = new GraphPackage<LogicalGraph>(logical);
var diagnostics = package.Validate();
var xml = GraphPackageXml.Serialize(package);
var restored = GraphPackageXml.Deserialize<LogicalGraph>(xml);
```

Names use ordinal comparison. Objects and string IDs are both accepted by binding
helpers. Calls explicitly bind the callee interface; definitions can be reused.
`first.Then(second)` records a completion-to-start constraint, not an insertion-order
schedule. Conversion/compilation captures a snapshot, validates the entire package,
and analyzes value dependencies. `CompileBoundary` is logical author intent; it is
not retained as an execution field or an enforced inlining restriction.

Resources declare `GraphIO`, `Scope` (`Call`, `Session`, `Global`), optional `Usage`,
independent `IsState`, permission, and optional `BindingKey`. Resources and actions
have extensible string `Category` values.

`GraphPackageXml` is the single **version 1** source format, with
`stage="business|logical|execution"`. Tensor shapes use `Tensor/Dimension`,
metadata uses `Metadata/Entry`, and actions own `Binding` entries. Explicit
ordering uses `GraphFlow/Then`. Execution packages additionally use a `Target`
child, resource `Storage`, and action `Implementation`/`Dispatch`/`Sources`.
Earlier graph XML and candidate draft XML are incompatible. XML stores declarations,
not tensor contents; parsing is separate from package validation. DTD/external
entity restrictions and document limits remain. See the
[GraphView tool](tools/GraphView/README.md) for viewer-specific capabilities.

## Optional execution-graph postprocessors

Postprocessors are model-independent and explicitly enabled; the default
converter and emitter remain unoptimized. Each pass returns an execution graph
program using the existing nodes, edges, and graph calls:

```csharp
using SharpInference.Instructions.Cpu;
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
// package is GraphPackage<LogicalGraph>; execution is GraphPackage<ExecutionGraph>.
var program = execution.Compile(VmTarget.Cpu, providers, options: null);
```

The converter and compiler require providers. Conversion and direct execution-package
`Compile` select/prepare implementations and precision; `VmExecutionGraphEmitter`
consumes an already prepared execution package and does not select implementations.
The compiler constructor is `(providers, postprocessors: null, converter: null)`;
custom conversion uses `new ExecutionGraphCompiler(providers, converter: customConverter)`.
After conversion and every postprocessor, preparation recomputes dependencies,
fills missing storage, and selects/validates implementations. The emitter performs
read-only implementation-contract validation; it never selects implementations.
For standalone execution-package postprocessing, use
`new ExecutionPassContext(context, index, processor.Name)`; its logical `Source`
is nullable, so no synthetic logical graph is needed.

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
| `src/SharpInference.Graphs` | Logical and execution graph models, construction, queries, validation, and XML support |
| `src/SharpInference.Models.Rwkv` | RWKV logical graph providers |
| `src/SharpInference.Models.Phi4` | Phi-4 graph and model components |
| `src/SharpInference.Instructions*` | Tensor, operation, access, and precision contracts; VM instructions and CPU/D3D12 implementations |
| `src/SharpInference.Vm` | VM program contracts |
| `src/SharpInference.Vm.Optimization` | Conversion, compilation, rewrites, optimization, and VM emission; consumes both graph models |
| `src/SharpInference.Backends.CpuVm` | CPU VM compiler and executor |
| `src/SharpInference.Backends.D3D12Vm` | D3D12 VM compiler and executor |
| `src/SharpInference.Tests` | Graph, emitter, and end-to-end inference tests |
| `tools/GraphView` | Logical and execution graph XML viewer |

### Graph domain namespaces

The graph domain namespaces are organized by responsibility:

- `SharpInference.Instructions`: `ElementType`, `TensorDescriptor`, `ResourceAccess`,
  `PrecisionRequirement`, `NumericTypeCompatibility`, `OperationId`, and `PrimitiveOperations`.
- `SharpInference.Graphs`: editable `Graph`, resources/actions, bindings, packages, diagnostics, queries, and `GraphPackageXml`.
- `SharpInference.Graphs.Logical` ([Logical](src/SharpInference.Graphs/Logical)):
  `LogicalGraph`, carrying logical compilation-boundary intent.
- `SharpInference.Graphs.Execution` ([Execution](src/SharpInference.Graphs/Execution)):
  `ExecutionGraph`, selected implementations, storage, targets, and source origins.
- `SharpInference.Vm.Optimization.GraphHelper` ([GraphHelper](src/SharpInference.Vm.Optimization/GraphHelper)):
  conversion, compilation, postprocessing contracts, and rewrites; not execution graph models.
- `SharpInference.Graphs.Internal`: shared construction, validation, and XML support.
  XML implementation details, including `GraphXmlSupport`, are internal.

Use `GraphPackageXml.Serialize(package)` and
`GraphPackageXml.Deserialize<TGraph>(xml)`, with `TGraph` equal to `Graph`,
`LogicalGraph`, or `ExecutionGraph`. Older `GraphXml`, `LogicalGraphXml`, and
`ExecutionGraphXml` entry points are removed.
Consumers import the corresponding Logical or Execution namespace and reference Graphs.
Constructing, querying, validating, or serializing an execution graph does not require
Vm.Optimization. The optimizer references both graph models, never the reverse.
Graphs depends on Graphs.Core and Instructions, not Abstractions, Vm, or
Vm.Optimization; Instructions does not depend on either graph project.

## License

The project is licensed under the [MIT License](./LICENSE). See
[third-party notices](./THIRD-PARTY-NOTICES.txt) for bundled dependencies and
assets. Model weights and external datasets remain subject to their own
licenses.
