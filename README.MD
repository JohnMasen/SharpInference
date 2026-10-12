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

- `Graph<TNode, TEdge>` is the only structural source.
- Resources, operations, and graph calls are nodes.
- Dependencies, reads, and writes are edges.
- Logical and execution graphs share traversal, views, pattern matching, and
  batch rewrite primitives.

Previous graph and runtime execution paths have been removed. There is no
compatibility execution path or silent fallback.

## Supported validation scope

- RWKV-6 and RWKV-7 graph construction from compatible rwkv.cpp GGML models.
- CPU VM compilation and execution.
- Direct3D 12 VM shader compilation and execution.
- Phi-4 graph construction and tokenizer/GGUF components.

The test suite includes an end-to-end RWKV-7 tiny-model logits comparison on
both CPU and D3D12. External model files are not included in the repository.

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

## License

The project is licensed under the [MIT License](./LICENSE). See
[third-party notices](./THIRD-PARTY-NOTICES.txt) for bundled dependencies and
assets. Model weights and external datasets remain subject to their own
licenses.
