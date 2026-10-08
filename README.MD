# SharpInference

English | [简体中文](./README_CN.MD)

> **AI authorship and project status:** Most of this project's code and documentation
> was written by AI. The author is currently reviewing, organizing, and refining
> both the code and the documentation. Treat the project as work in progress;
> verify behavior and documentation against your intended use.

SharpInference is a .NET local model inference project with compiled CPU and
Direct3D 12 execution paths. It explores separating model-specific computation
from instruction implementations, executable programs, and resource ownership.

## Supported models and scope

| Model | Model files | Execution paths |
| --- | --- | --- |
| RWKV-6 / RWKV-7 | Compatible rwkv.cpp GGML x060 / x070 FP32 or FP16 weights | Logical graph frontend with compiled CPU or D3D12 VM; prefill and token decoding |
| Phi-4-multimodal-instruct | Converted F16 GGUF components and tokenizer | Native CPU multimodal processor; dedicated D3D12 components and GPU-resident decoder sessions |

- RWKV sessions support recurrent state, reset, branching, and state snapshots.
- Phi-4 supports text generation from text, decoded images, and decoded audio.
  File/container decoding is the caller's responsibility.
- The bundled Web API serves RWKV text models; it is not a Phi-4 multimodal endpoint.
- FP16 weight storage does not imply FP16 arithmetic. Precision requirements are
  explicit; RWKV state and the current T0 arithmetic/accumulation are FP32.
- This is not a general-purpose loader for arbitrary GGUF architectures or
  quantization formats. Backend registration is not a guarantee of correctness
  or performance for every model size, input, or device.
- The current VM has no general program-level dynamic branching, loops, or
  recursion. Fixed-program execution and operator-local loops are distinct.

Model weights and large external validation datasets are not included.

## Architecture

The RWKV path is:

```text
model metadata / tensor catalog + architecture-specific graph provider
  -> LogicalGraph
  -> target optimization, ExecutionGraph, and local-slot reuse planning
  -> VmProgram: fixed slots, explicit State, definitions, calls, entry points
  -> generated C# / HLSL
  -> CPU assembly / GPU DXIL
  -> reusable VM instances
```

Metadata supplies dimensions and weight descriptions; the graph provider supplies
the model's computation. The optimizer decides how logical resources map to
fixed slots and which intermediate resources can safely share storage.

At runtime, `VmResourceManager` allocates backing storage, shares global resources,
and binds session resources. Slots have `Global`, `Session`, or `Local` scope.
`LogicalVmResourceManager` handles component data references, storage-domain
replicas/transfers, and input binding; the D3D12 resource pool supplies GPU storage.
Compile-time scratch reuse is different from runtime sharing of weights or state.

Not every frontend follows the same route: Phi-4's dedicated D3D12 decoder
constructs a `VmProgram` directly rather than using the RWKV logical-graph lowering
path. Its KV caches are explicit session resources.

The backend executes the supplied program; it does not silently replan the graph
or fall back to a legacy executor. See the
[current VM contract](./doc/vm-execution-program.md) for compilation, artifacts,
state, queues, and resource ownership.

## Requirements and build

- .NET 10 SDK.
- Windows for the full solution, Web API, and D3D12 paths: several projects target
  `net10.0-windows10.0.19041.0`. Some platform-neutral libraries target `net10.0`;
  this does not establish cross-platform support for the complete application.
- A compatible D3D12 adapter and driver for GPU execution, with enough memory
  for weights, state, and temporary resources.
- Access to NuGet packages when building for the first time.

Run these PowerShell commands from the repository root:

```powershell
dotnet build .\SharpInference.slnx -c Release
```

Source compilation and imported precompiled artifacts have different deployment
requirements; see [VM deployment](./doc/vm-execution-program.md).

## Quick start: RWKV Web API

Prepare a compatible RWKV-6 or RWKV-7 GGML model. Do not substitute a `.pth` file
or an arbitrary GGUF file.

Start with an explicit CPU backend and a loopback-only address:

```powershell
dotnet run --project .\src\SharpInference.WebApi -c Release -- `
  --model-path "D:\Models\your-rwkv-model.bin" `
  --runtime-kind cpu `
  --urls "http://127.0.0.1:9841"
```

Replace the model path with your own. The server validates the model header at
startup; full model loading and program preparation occur when the model host
is first used, so the first generation request can take longer.

In another PowerShell terminal, query the advertised model ID and request a reply:

```powershell
$baseUrl = "http://127.0.0.1:9841"
Invoke-RestMethod "$baseUrl/health"
$models = Invoke-RestMethod "$baseUrl/v1/models"
$body = @{
    model = $models.data[0].id
    messages = @(@{ role = "user"; content = "Hello!" })
    max_tokens = 32
    stream = $false
} | ConvertTo-Json -Depth 5
Invoke-RestMethod "$baseUrl/v1/chat/completions" `
    -Method Post -ContentType "application/json" -Body $body
```

The API exposes OpenAI-style model and chat-completion routes, including streaming
responses; this does not imply full compatibility with every OpenAI API feature.

### Backend and server configuration

| Option | Purpose |
| --- | --- |
| `--model-path` | Required RWKV GGML model path |
| `--runtime-kind cpu` | Compiled CPU VM |
| `--runtime-kind d3d12` | Compiled Direct3D 12 VM; `vortice` is an alias |
| `--config` | Additional JSON configuration; command-line values override it |
| `--urls` | Complete listen URL |
| `--prefill-instances` / `--inference-instances` | Reusable VM counts |
| `--prefill-queue-capacity` / `--inference-queue-capacity` | Bounded queue capacities |
| `--api-key` | Require an API key |

The checked-in [appsettings.json](./src/SharpInference.WebApi/appsettings.json)
selects `vortice`, so keep `--runtime-kind cpu` for the CPU example. GPU adapter
selection is configured with `Rwkv:Runtime:Vortice:AdapterIndex`.
Use `--help` for the complete CLI options:

```powershell
dotnet run --project .\src\SharpInference.WebApi -c Release -- --help
```

Keep the service local until you have configured authentication and an appropriate
deployment environment. The checked-in settings have an empty API key and an
all-interface listen address; the quick-start command overrides the address.
Request logging can include prompt content.

## Phi-4 quick start

Prepare a converted model directory containing:

```text
Phi-4-multimodal-text-f16.gguf
Phi-4-multimodal-omni-f16.gguf
Phi-4-multimodal-vision-lora-f16.gguf
Phi-4-multimodal-speech-lora-f16.gguf
tokenizer.json
```

Run the text decoder benchmark:

```powershell
dotnet run --project .\tools\SharpInference.VmBenchmark -c Release -- phi4 `
  --model "D:\Models\Phi4" --tokens 16 --samples 3
```

For the GPU-resident text session, add `--backend d3d12-session --context 4096`.
This is a benchmark entry point, not the RWKV Web API. See
[Phi-4 multimodal inference](./doc/phi4-multimodal-inference.md) for processor
examples, conversion tooling, D3D12 components, and external validation.

## Project layout

| Area | Responsibility |
| --- | --- |
| `src/SharpInference.Abstractions`, `src/SharpInference.Core` | Public contracts, model loading, sessions, tokenization, and sampling |
| `src/SharpInference.Architectures.*` | Model-specific frontends and processors |
| `src/SharpInference.Graphs` | Graph IR, validation, optimization, and resource planning |
| `src/SharpInference.Instructions*` | Instruction contracts and CPU/GPU implementations |
| `src/SharpInference.Vm*` | VM program contracts, resource ownership, optimization, templates, and tools |
| `src/SharpInference.Backends.*Vm` | Compiled CPU and D3D12 execution |
| `src/SharpInference.Runtime` | Model/backend integration and VM runtime configuration |
| `src/SharpInference.WebApi` | RWKV HTTP service |
| `src/SharpInference.Tests`, `src/SharpInference.Gguf.Tests` | Tests |
| `tools/SharpInference.VmBenchmark`, `scripts`, `doc` | Benchmarks, supporting scripts, and detailed documentation |

## Tests and performance

```powershell
dotnet test .\src\SharpInference.Tests\SharpInference.Tests.csproj -c Release
dotnet test .\src\SharpInference.Gguf.Tests\SharpInference.Gguf.Tests.csproj -c Release
```

Some tests require a GPU, external models, or explicit opt-in settings. A passing
run with skipped tests is not evidence that all large-model and GPU paths were
validated. Phi-4 external tests use `PHI4_TEST_MODEL_DIRECTORY`,
`PHI4_RUN_GOLDEN_MODEL`, and `PHI4_RUN_GOLDEN_VISION`; see its detailed guide.

RWKV timing and numerical-output benchmark:

```powershell
dotnet run --project .\tools\SharpInference.VmBenchmark -c Release -- run `
  --target cpu --model "D:\Models\RWKV-x060-model.bin" `
  --output ".\benchmark-results\rwkv-cpu.json"
```

This command writes a report plus logits and state sidecars. The RWKV benchmark
chooses its frontend from the model filename, so retain an architecture-identifying
name (`060` / `rwkv-6` for RWKV-6). Use `--target gpu` for GPU measurement.
Compare timings only with matching hardware, models, precision, warmup, and
workload conditions; historical numbers are not performance guarantees.

## Documentation

- [Compiled VM contract and deployment](./doc/vm-execution-program.md)
- [VM operator tiers and precision](./doc/vm-operator-tiers.md)
- [Queued VM execution](./doc/queued-vm-execution.md)
- [Phi-4 multimodal inference](./doc/phi4-multimodal-inference.md)
- [GPU MatVec performance](./doc/gpu-matvec-performance.md)
- [Tier-one performance](./doc/tier-one-performance.md)
- [Architecture history](./doc/arch-design.md) — includes historical designs;
  use the current VM contract and source for current runtime behavior.

## License

The project is licensed under the [MIT License](./LICENSE).
See [third-party notices](./THIRD-PARTY-NOTICES.txt) for bundled dependencies and
assets. Model weights and external datasets remain subject to their own licenses.
