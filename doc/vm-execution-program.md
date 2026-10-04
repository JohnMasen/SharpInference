# Compiled inference VM

The default `Processor.Load` and Runtime `cpu` paths now execute compiled CPU
programs. Runtime `d3d12` and its `vortice` alias execute compiled Direct3D12
programs. RWKV-6/7 model readers and logical graph providers remain the frontend:

```text
model + logical graph
  -> external target optimizer
  -> editable VmProgram XML
  -> complete generated C# / HLSL
  -> CPU assembly / DXIL artifact
  -> fixed-slot VM instances
```

The backend validates and implements the supplied program; it does not silently
replan storage, fuse nodes, invent dispatch boundaries, or fall back to a legacy
executor. Legacy graph types remain optimizer/model-binding intermediates and
standalone numerical reference tools, not runtime execution contracts.
Runtime no longer accepts legacy backends, execution-graph XML, or primitive
debug plans. Old `Graph kind="ExecutionGraph"` XML is not VM XML.

## Program contract

`SharpInference.Vm` is platform-neutral and independent of models, optimizers,
source compilers, and Web API. `VmProgram` contains fixed slots, State, reusable
parameterized node definitions, ordered calls, and entry points:

| Definition kind | Meaning |
| --- | --- |
| `Function` | CPU function or GPU device helper |
| `Kernel` | Explicit GPU compute entry and thread-group shape |
| `Orchestration` | Host-side method/command composition |

Nested device helpers do not create dispatches. GPU orchestration explicitly
uses `VmDispatch` and `VmBarrier`. Dependencies refer to preceding nodes.
Layer/Region semantics belong to the logical frontend, not the VM.
Programs have no general graph-level recursion, dynamic branching, or loops;
operator-generated local loops and scalar variables do not require slots.
The [T0 profile and InstructionCollections](./vm-operator-tiers.md)
also describe a later control-flow redesign; branching is not implemented.

Arguments refer to a caller parameter or physical slot with an explicit byte
offset. Parameters declare dense tensor shape/type and read-only/read-write
access. Typed views can change shape within compatible capacity without copying
storage; aligned slices of byte workspaces can provide typed views. Bindings
cannot widen access permissions.

`VmProgramXml` uses independent version-2 `VmProgram` XML, deterministic ordering,
strict fields/enums, and DTD protection. Limits are 16 MiB of characters, 100,000
elements, and depth 64. Structural validation checks calls, recursion, domains,
capacity, offsets, access, and grid dimensions. Target compilers additionally
validate IC GUID/name, shapes/precision, initialization, and supported
execution mappings. Parsing alone does not certify arbitrary GPU dataflow:
cross-thread dependencies require explicit dispatch boundaries/barriers.
Unsupported contracts produce diagnostics, not successful fallback execution.
Version-1 XML is rejected; regenerate execution graphs and compiled artifacts.
An `Operator` carries `collection` (GUID) and `name`, without an instruction
version. File format, State schema and backend ABI versions remain independent.
Legacy source constructors are retained only as a migration convenience and
reject versions other than 1; new callers should use the GUID constructor.

### Shared T0 profile

`TierZeroOperationContracts` defines the logical frontend's common T0 profile:
23 required RWKV operation IDs, plus optional Divide and whole-tensor ReduceSum.
T0 FP32 IC has GUID zero; T0 FP16 IC has GUID ending in `0001`.
The FP32 IC includes FP16-weight/FP32-activation signatures for MatVec,
BatchedMatVec and GatherRow.
`VmGraphOptimizer` validates logical graphs against this catalog before lowering.
IC implementations apply type, geometry, attribute, alias and access
contracts during CPU and GPU recording. MatVec accepts the existing
`weight` port as an alias for `matrix`, but not both at once.

Arithmetic, accumulation and model State are FP32. The separate FP16 IC supports
FP16 input/output tensors on CPU and Direct3D12 with FP32 accumulation and FP16
output rounding. Reshape retains copy semantics. Runtime tensor-slot allocation,
in-place operator outputs, implicit broadcasting and silent fallback are not
introduced. Existing numerical kernels are reused rather than adding another
interpreter. Tier-1/2 implementations and automatic extension discovery are not
implemented. Trusted provider instances are supplied by the host.

### IC assembly boundary

`CpuVmCompiler` and `D3D12VmCompiler` require an
`IEnumerable<IInstructionCollectionProvider>` constructor argument; they have
no default ICs or concrete IC dependencies. The host may load assemblies before
injecting provider instances. Architecture matching uses extensible identifiers
(`cpu.managed`, `direct3d12`), not a generic GPU category.

`QueryInstructionCollection` lists GUID/name/Tier/architecture;
`QueryInstruction(Guid, string)` exposes supported typed parameter signatures.
Instructions adapt one parameter array and call a backend-provided recorder.
Recording collects source, helpers, assembly references and synchronization for
one compilation, never a runtime reflection call.

`VmExecutionGraphGenerator` takes an independent catalog. Runtime factories
accept separate backend `instructionCollections` and `generatorCollections`;
defaults are installed only at this host composition boundary. A metadata-
selected `RwkvExecutionGraphGenerator` can produce portable XML without a
backend. CPU-target XML is not automatically a Direct3D12 execution plan.
Dynamic instruction adaptation errors are explicit; missing target/IC support
never falls back to another implementation.

See [the operator profile](./vm-operator-tiers.md#implemented-t0-baseline-and-tests)
for numerical policy and conformance tests. Regenerate existing compiled
artifacts to obtain the new T0 numerical fixes; validation of a graph does not
retroactively certify previously compiled machine code.

## Slots, owners, and physical sharing

| Scope | Owner | Typical use |
| --- | --- | --- |
| Global | VM/resource manager | Shared immutable model weights |
| Session | Session manager | Persistent State and session-private IO |
| Local | VM worker | Private, explicitly reusable workspace |

Ownership and access permissions are independent. Every registered State slot
must be Session scope; not every Session slot must be State.
Managers allocate fixed descriptors during engine/session/worker initialization.
Generated code receives existing buffers/views, not allocation or rebinding
APIs: it cannot create, remove, resize, or rebind slots.
CPU kernels may allocate bounded conversion scratch/local implementation data;
this does not create new VM-visible slots.

`VmResource`, `VmResourceLease`, and `VmBindings` separate physical storage,
ownership, and bindings. Releasing one owner does not destroy resources still
leased elsewhere. Writable physical resources have exclusive reservations across
VMs; read-only bindings can coexist. Local storage is private to each worker.
Bindings cannot change while an execution lease is active.
Managers bind compatible session resources only while workers are idle, and
unbind them after use. An execution lease lasts through actual device completion.

`VmResourceManager` initializes globals once by binding key/slot ID. Independent
Prefill and Inference programs share registered State by stable entry name,
schema/version, type, and shape, even if their State slot IDs differ. Compatible
global resources share physical storage. Each program retains its own private
local workspace and non-State session resources.
`VmModelBindings.InitializeGlobal` strictly validates model binding, type, shape,
and byte capacity before writing raw weights.

### Direct3D12 coherence and lifetime

`D3D12VmResourcePool.Allocate` implements the `IVmStorage`/`IVmManagedStorage` seam.
Workers share actual GPU allocations for globals and session State, not just
copies of CPU arrays; local allocations remain worker-private. Globals upload
once per shared allocation. Host writes invalidate the corresponding upload.
Mutable CPU shadows are authoritative between managed execution calls:
uploads and readbacks complete before returning. This is an explicit correctness
baseline, not a promise of optimal device residency or transfer performance.

Executors borrow core-owned storage. Idle command caches retain allocation IDs,
not closed-session arrays or resource owners. Released/stale allocations are
rejected before submission. Disposal follows engine/workers -> pool -> device.
The shared pool currently serializes GPU execution; increasing instance counts
does **not** guarantee concurrent GPU kernels or increased throughput.

## State transport

`VmStateStream` exports/imports registered State only, leaving the external stream
open. Its versioned representation includes model/context identity,
schema/version, ordered names, types, dimensions, payload lengths, and per-entry
SHA-256 hashes. Non-seekable import is supported with an explicit staging budget.
All metadata/payloads are checked before committing under an exclusive lease.
The supported numerical byte representation is dense little-endian data;
pointers and backend-private layouts are never serialized.

Malformed input before commit leaves State unchanged. Failed commits invalidate
the physical State, including other VMs bound to it, until successful restoration.
The host can restore State exposed as read-only to generated code without
widening that code's permissions. Integrity checks are not snapshot authentication.
External caches, network transport, and distributed scheduling are outside the
graph; compatible VM instances on separate hosts can exchange this representation.
Existing `ProcessorSession` GGUF snapshots/fork/reset remain supported through the
runtime State adapter.

## Engine, queues, and sessions

`VmInferenceEngine` combines two independent bounded FIFO queues and VM pools.
The default is **2 Prefill workers + 2 Inference workers**, with queue capacities
16 each in Runtime. Full queues immediately throw `VmQueueFullException`; there
is no unbounded producer waiting room. Worker count bounds private workspace,
not resident session State, queued inputs, or global weights.

Sessions own State, not permanently assigned VMs:

1. `PrefillAsync` submits bounded token input and binds an idle Prefill worker
   to the session's State. Inputs are copied before waiting.
2. Declared `prefill.N` entries process the largest fitting chunks. Sparse
   variants are supported; the default optimizer emits hierarchical doubling
   calls rather than duplicating every slot argument for every token.
3. `BeginGenerationAsync` leases one Inference worker for the **whole reply**.
   All token steps use that lease, without nested queue submissions.
4. Scope disposal, cancellation, exceptions, and early enumeration exit release
   the worker only after in-flight execution finishes.

Session operations serialize access to shared State. Reset/snapshot/fork/direct
forward are rejected while a queued/scoped Processor operation owns the session.
Partial execution failure/cancellation may invalidate State rather than roll back;
restore/reset is required before further execution. Invalid input is rejected
before State modification. Shutdown rejects new requests, cancels pending work,
waits for active work, and reports cleanup failures. Failed workers are disposed
and replaced; failed replacement stops admission with explicit errors.

`RwkvTextGenerator` uses async prefill and one whole-generation scope.
Web API uses the same engine queues, with no additional generation throttle.
GPU resident-session admission remains separately bounded and immediately rejects
excess residency. Busy errors before response start map to HTTP 503 and
`Retry-After`; after streaming begins they become explicit SSE errors.
Headers are deferred until the first iterator result so initial busy errors can
still be ordinary HTTP responses.

## Runtime configuration

```json
{
  "Rwkv": {
    "Runtime": {
      "Kind": "d3d12",
      "Vortice": { "AdapterIndex": 0 },
      "Vm": {
        "PrefillInstances": 2,
        "InferenceInstances": 2,
        "PrefillQueueCapacity": 16,
        "InferenceQueueCapacity": 16,
        "MaximumPrefillTokens": 65536,
        "PrefillCapacity": 64,
        "ThreadsPerGroup": 64,
        "ReuseLocalStorage": true,
        "NativeHalfWeights": true,
        "WeightViews": true
      }
    }
  }
}
```

Instance counts/queue capacities must be positive. Prefill capacity and thread
group dimensions are in 1..1024. The baseline GPU lowering uses 1-D groups and
explicitly rejects grids exceeding 65,535 groups. Immutable weight views and
direct FP16 MatVec/Gather/BatchedMatVec avoid full FP32 matrix conversion storage;
accumulation and recurrent State remain FP32.

Optional `ProgramPath` supplies edited inference XML for compilation.
Alternatively, `ArtifactDirectory` imports a prebuilt artifact without compiling
source. `PrefillProgramPath` / `PrefillArtifactDirectory` independently select
the Prefill program; without them one artifact supplies both entry sets.
For each role, program path and artifact directory are mutually exclusive.
Model target/ABI, registered State, token/logits, and required entries are checked
before execution.

The old `EnableCommandReplay` setting and `--enable-command-replay` CLI flag
are rejected; command caching belongs to the compiled backend.
`--max-in-flight-generation-batches` is replaced by `--inference-instances`.
The legacy scheduler option does not impose an extra generation throttle in VM
mode. Web CLI also exposes `--prefill-instances`, `--prefill-queue-capacity`, and
`--inference-queue-capacity`. `d3d12` and `vortice` select the same backend.

`Processor.InferenceProgram` / `PrefillProgram` expose actual execution programs.
`ExportExecutionGraph(stream, kind)` always exports VM XML;
`ExportCompiledArtifact(directory, kind)` exports the chosen artifact.
Logical graphs and retained legacy frontend metadata are not compiled programs.
Legacy primitive-step debugging is unsupported by compiled VMs; use generated
source, source mappings/PDBs, and backend diagnostics.

### Breaking runtime API migration

`Processor.LoadGraph` and pipeline `UseBackend` now require `VmGraphBackend`;
`RwkvRuntimeSelection.CreateBackend` returns that type. It is not an
`IExecutionGraphBackend` or primitive backend. `VmGraphBackend.Prepare` accepts
a logical model graph and returns a typed `VmCompiledPlan`, whose public
execution surface is `VmProgram`, not an old `ExecutionGraph`.
The internal binding graph only describes model/State metadata.

```csharp
using var processor = Processor.LoadGraph(
    weightsPath,
    RwkvRuntimeFactory.CreateGraphProvider("rwkv-7"),
    VmBackendFactory.CreateCpu(new VmRuntimeConfig
    {
        ArtifactDirectory = "inference-package",
        PrefillArtifactDirectory = "prefill-package",
        PrefillInstances = 2,
        InferenceInstances = 2,
    }));
using var session = processor.CreateSession();
await session.PrefillAsync(new int[] { 1, 2, 3 });
await using var generation = await session.BeginGenerationAsync();
var logits = generation.Session.ForwardToken(4);
```

Removed surfaces include `UseExecutionGraph`, `UseExecutionXml`,
`UseXmlExecutionGraph`, `UsePrefillOptimizers`, execution-graph metadata readers,
`Processor.ExecutionGraph`/`InferenceExecutionGraph`/`PrefillExecutionGraph`,
legacy prepared-plan aliases, and primitive debug mode/events/Resume/Stop/LayerTrace.
Logical XML remains a supported model frontend through `UseXmlLogicalGraph`
and the logical-only `XmlArchitectureMetadataReader`.
Select actual execution XML/artifacts through `VmRuntimeConfig`, not the old
pipeline graph source. Independent Prefill programs replace legacy dual-graph
prefill optimizers. Integration/deployment utilities use compiled VMs, and
`--dump-graphs` exports logical JSON plus `cpu.vm.xml` and `d3d12.vm.xml`.
Standalone mathematical reference/debug tools remain outside Runtime; they
cannot be attached to a production `Processor`.

## Artifacts, editing, and tuning

CPU packages contain `manifest.json`, `program.xml`, `contracts.xml`, `options.json`, and
`CpuProgram.g.cs`; binary packages add the paired DLL/PDB. Imports verify hashes
and the generated program/ABI identity, per-instruction read/write contracts
and supported options, without invoking
providers or regenerating source. `CpuVmCompiledArtifact.LoadFromResources`
supports embedded packages without extracting files.
The current generated-source ABI is `cpu-vm-ic-call-frame-v3`. Generated methods
take a context and immutable call frame rather than passing every slot Span by
value, keeping stack usage bounded for large models. Binding tables retain
offset/access contracts without copying tensors or allocating bindings per node.
Packages and static deployments generated with the previous ABI must be
regenerated/rebuilt; incompatible generated-code fingerprints are rejected.
D3D12 packages are `program.vm.zip`, containing execution metadata, generated
HLSL and DXIL. ABI `SharpInference.D3D12Vm.raw-uav.ic.v2` includes hashed
`contracts.xml` with generic index bounds and port access emitted by the selected IC.
Import verifies contracts against program structure, module membership and
hashes without IC providers, source regeneration or DXC.
These integrity checks do not prove arbitrary source/binary semantic equivalence
and are not authentication. Execute only trusted compiled packages.
CPU generated code may retain IC-declared runtime library references; the host
must deploy those dependencies. The CPU static template references the CPU IC
project explicitly. Plugin unload must wait until dependent executables close.

The Windows CLI builds artifacts or measures **externally supplied** candidate
XML. It does not promise a global optimum or silently search/rewrite candidates.
Examples run from the repository root:

```powershell
dotnet run --project src\SharpInference.Vm.Tool -- build cpu weights.bin cpu-package --source-only
dotnet run --project src\SharpInference.Vm.Tool -- build d3d12 weights.bin gpu-package
dotnet run --project src\SharpInference.Vm.Tool -- build cpu weights.bin edited-package candidate.xml
dotnet run --project src\SharpInference.Vm.Tool -- tune d3d12 weights.bin report.json hardware-id driver-id compiler-id inference candidate1.xml candidate2.xml
```

Tuning currently uses tokens `1,2,3,2` and zero initial State. It validates **all
logits and all registered State** against the portable CPU reference, using
explicit absolute/relative tolerance, before accepting timings.
Every warmup/measured run restores the initial State. Reports separate compile,
initialize, first execution, warmup, and steady-state median times, with rejected
candidate diagnostics and caller-supplied hardware/driver/backend/compiler
fingerprints. No accepted candidates yields no winner and a nonzero CLI exit.
Use accurate fingerprints and the intended workload (`prefill` or `inference`);
this small workload is an entry point, not a complete production benchmark.

After selecting a candidate on the intended device, build once and deploy the
same artifact to matching hardware/driver/runtime configurations. Binary fidelity
does not guarantee equal performance across changed drivers, compiler versions,
power/thermal conditions, or operating systems.

## Compiler-free deployment

### CPU static source / Native AOT

The CPU template links generated C# at build time and embeds the source-only
package. Runtime uses registered static code, not dynamic assembly compilation.

```powershell
.\scripts\publish-cpu-vm.ps1 -ArtifactDirectory .\cpu-package -OutputDirectory .\cpu-deploy
.\cpu-deploy\SharpInference.Vm.Template.exe weights.bin --tokens 1,2,3,2,4 logits.bin
```

The script uses Native AOT by default (Windows x64 needs Visual Studio C++
build tools at publication time). `-Managed` selects trimmed, self-contained
single-file publication. `RuntimeIdentifier` selects another supported CPU
publication target where its toolchain is available. Templates are excluded from
normal solution builds because they require `VmArtifactDirectory`.
Virtual source mappings are retained; embedding untracked sources is disabled
because mapped VM documents are not physical files.

### GPU embedded DXIL

The Direct3D12 template embeds `program.vm.zip` in a self-contained single-file
Windows x64 executable:

```powershell
.\scripts\publish-d3d12-vm.ps1 -ArtifactDirectory .\gpu-package -OutputDirectory .\gpu-deploy
.\gpu-deploy\SharpInference.Vm.D3D12Template.exe weights.bin --tokens 1,2,3,2,4 logits.bin
```

It imports DXIL, creates a shared GPU resource pool, and does not compile HLSL.
Publication still needs .NET; execution needs Windows/Direct3D12 and a compatible
driver, not an SDK or DXC. PDB sidecars are optional debugging files; external
weights remain separate. Both templates also support text input when the model's
vocabulary covers the bundled tokenizer. Tiny fixtures have vocabulary 256 and
must use raw token mode; pretending they support full-vocabulary generation would
be an invalid acceptance test.

## Verified boundary

The migration has exercised RWKV-6 FP32 and RWKV-7 FP32/FP16 on CPU and
AMD Radeon RX 7900 XTX, including Prefill/Inference handoff, complete logits/State,
snapshots, fork/reset, and shared resources. Current Native AOT execution reports
`dynamic code supported: False`; current embedded GPU execution reports
`DXC loaded: False`, both with only System32 on PATH.
All 256 raw-token deployment logits agree within `0.0003 + abs(expected)*0.0003`;
With the current call-frame ABI, observed maximum CPU-AOT/GPU difference for
tokens `1,2,3,4` was approximately `4.17e-7`.

The 7.2B FP16 metadata/XML/source generation and local-workspace threshold below
1% of global weight bytes passed. Full-vocabulary RWKV-6 7B and RWKV-7 7.2B FP16
models also completed CPU/GPU ten-question Prefill/generation benchmarks.
Each main/VM pair produced identical token IDs for all ten measured replies.
This is large-model execution and sampled greedy-output parity, **not** a full
large-model golden-logits/State validation. The external rwkv.cpp tiny-model
golden-logits fixture was located and
both FP32/FP16 golden cases passed in the final regression.
Mobile/console/MCU backends, network scheduling, a visual
editor, advanced autotuning, and globally optimal kernels are not implemented.

Post-call-frame Release validation passed 660 inference/runtime tests and 17 GGUF tests,
with zero skips. The command excluded full-execution `ModelSize=Large` cases;
the VM large-model metadata/source/workspace
test was included. Release solution build completed with zero warnings/errors.
Runtime cutover and removal of its legacy pipeline/debug/XML execution surfaces
are complete. Retained frontend/oracle types are not an alternate Runtime path.

## Main versus compiled VM performance

Measured on Ryzen 9 3900X (24 logical processors), 63.93 GiB RAM and Radeon
RX 7900 XTX (driver 32.0.31041.1004), with baseline commit
`36fd4b04c10b6fc783605f2dc7a526fa48583a01`. Models were
`RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin` and
`rwkv7-g1j-7.2b-20260831-ctx16384-FP16.bin`.

Both branches used the same ten Chinese questions, World tokenizer, prompt
format and temperature zero. Two warmup questions were excluded. Each measured
question generated exactly 16 tokens, ignoring EOS/stops: 167 prompt tokens and
160 output tokens per configuration. Configurations ran serially in Release
.NET 10 with tiered compilation disabled and one active session. The VM used
its default two Prefill and two Inference workers; main GPU used its default
command-replay=false. Rates are total stage tokens divided by total stage time.
Initialization, warmup and session construction are excluded.

| Model | Backend | Main decode token/s | VM decode token/s | Decode change | Main Prefill token/s | VM Prefill token/s |
|---|---|---:|---:|---:|---:|---:|
| RWKV-6 7B FP16 | CPU | 0.858 | 1.114 | +29.8% | 0.857 | 0.249 |
| RWKV-6 7B FP16 | GPU | 7.749 | 1.726 | -77.7% | 7.477 | 2.632 |
| RWKV-7 7.2B FP16 | CPU | 1.243 | 1.522 | +22.5% | 1.240 | 1.078 |
| RWKV-7 7.2B FP16 | GPU | 9.034 | 2.541 | -71.9% | 8.680 | 4.890 |

Output tokens divided by combined Prefill/decode time were respectively
main/VM: V6 CPU `0.420/0.197`, V6 GPU `3.722/1.025`, V7 CPU `0.607/0.615`,
V7 GPU `4.330/1.648` token/s. Faster CPU decode does not imply faster overall
replies: V6 CPU Prefill regressed substantially. GPU decode and Prefill both
regressed; this migration is not yet performance-equivalent to main.
These are one-pass, short-reply measurements, not confidence intervals,
quality evaluation, queue-saturation throughput or globally optimal graphs.
Startup includes compilation/model preparation and is page-cache sensitive.

The first V6 CPU run exposed large by-value Span call signatures overflowing
ThreadPool stacks. The current call-frame ABI fixes that failure. CPU candidate
measurements use the corrected ABI; a main V7 CPU measurement overlapping a
build/test was discarded and rerun in isolation. GPU measurements were
unaffected by that CPU-only fix. The regression cause still requires profiling;
these measurements alone do not identify the bottleneck.
