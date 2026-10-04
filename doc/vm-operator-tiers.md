# VM instruction collections and operator tiers

Status: T0 FP32/FP16 InstructionCollections and constructor-injected providers
are implemented for managed CPU and Direct3D12. Tier-1/2 implementations,
annotation discovery and control flow remain future work. The runtime contract is
[VM execution programs](./vm-execution-program.md).
VM operators now carry explicit arithmetic/accumulator precision requirements;
IC signatures declare their actual internal precision. Tier-1 implementation,
catalog additions and fusion-specific rounding metadata are deferred.

## Scope and measured Tier-0 inventory

The first common profile covers the existing full-vocabulary RWKV-6 7B and
RWKV-7 7.2B FP16 models. It is not a complete inventory for every LLM.
Counts are expanded, unfused single-token logical operator instances, before
weight views, native-half loads, cast caching, copy elision or fusion.

CPU and Direct3D12 require the **same 23 stable instruction names** in T0 FP32:

| Group | Operation | V6 instances | V7 instances |
|---|---|---:|---:|
| Elementwise binary | `core.add` | 2788 | 738 |
| Elementwise binary | `core.subtract` | 194 | 417 |
| Elementwise binary | `core.multiply` | 644 | 803 |
| Elementwise binary | `core.maximum` | 0 | 32 |
| Elementwise unary | `core.exp` | 64 | 32 |
| Elementwise unary | `core.tanh` | 64 | 32 |
| Elementwise unary | `core.sigmoid` | 64 | 127 |
| Elementwise unary | `core.rsqrt` | 98 | 130 |
| Elementwise unary | `core.square` | 130 | 162 |
| Elementwise unary | `core.relu` | 32 | 32 |
| Reduction | `core.reduce-mean` | 132 | 132 |
| Reduction | `core.tensor.reduce-last-sum` | 0 | 64 |
| Reduction | `core.tensor.reduce-last-mean` | 64 | 64 |
| Matrix/vector | `core.mat-vec` | 353 | 447 |
| Matrix/vector | `core.tensor.batched-mat-vec` | 32 | 64 |
| Matrix/vector | `core.tensor.head-outer` | 32 | 64 |
| Lookup | `core.gather-row` | 1 | 1 |
| Data | `core.copy` | 97 | 97 |
| Data | `core.tensor.fill` | 130 | 194 |
| Data | `core.tensor.cast-f16-f32` | 258 | 448 |
| Data | `core.tensor.reshape` | 2914 | 1247 |
| Data | `core.tensor.slice` | 2208 | 192 |
| Data | `core.tensor.broadcast` | 292 | 292 |
| **Total** | **23 common IDs** | **10591** | **5811** |

V6 uses 21 of these IDs; V7 uses all 23. The existing registered catalog has
25 IDs: `core.divide` and `core.reduce-sum` are also registered but are not
required by these workloads. Retain them as optional general-purpose coverage,
not as measured RWKV requirements.

Evidence was checked in three independent representations:

1. The unfused logical graph, expanded without optimization.
2. Every main CPU/GPU fused expression expanded back to its Tier-0 operations,
   including cached/alias nodes. Counts match the logical inventory exactly.
3. CPU/GPU lowered VM forward sequences. Operation order, counts, typed
   parameter shapes and attributes match each other exactly. Both CPU C# and
   GPU HLSL source generation succeeded.

Lowered VM totals are 10011 for V6 and 4788 for V7. Explicit casts disappear
under native-half weight views; reshape counts become 2592 and 672.
An eliminated operation remains part of the common semantic catalog.
The original inventory is structural/source-generation evidence. Numerical
conformance tests for the implemented baseline are described below.

## Tier-0: fixed, shared semantic instruction set

An operation ID identifies math/data semantics, not a particular kernel.
Its common contract includes:

- IC GUID and stable exported instruction name. There is no instruction version;
  incompatible contracts receive a different GUID.
- Named input/output ports and read/write permissions.
- Element types, rank/shape constraints, dense layout and permitted views.
- Static attributes versus runtime scalar inputs.
- Arithmetic/accumulator type and numerical policy.
- Valid-input domain, errors, alias restrictions and explicit side effects.

CPU/GPU expose the same contracts and the same required signature variants.
SIMD width, cooperative reduction, workgroup shape, tiling, register use and
native-half loads belong to an **implementation**, not new operation IDs.
They must still be explicit target choices in the execution program where
they affect generated code.

### Initial type and shape profile

- Math and State use FP32; State entries remain Session resources.
- Matrix/table storage may be FP32 or FP16 with FP32 activation/output and
  FP32 accumulation. This mixed signature applies to MatVec, BatchedMatVec
  and GatherRow, on both targets.
- Gather index is Int32. Scalars can be represented as one-element tensors;
  generated internal scalar temporaries do not require slots.
- Explicit CastFp16ToFp32 remains available when a materialized conversion
  is needed; accepting FP16 storage does not imply FP16 arithmetic.
- Elementwise operands have equal shapes; implicit broadcasting is not part
  of Add/Multiply. Broadcast is explicit.
- MatVec: `[R,C] x [C] -> [R]`.
- BatchedMatVec: `[B,R,C] x [B,C] -> [B,R]`; it is not general batched GEMM.
- HeadOuter: `[H,R] x [H,C] -> [H,R,C]`.
- GatherRow: `[V,C], Int32[1] -> [C]`.
- ReduceMean reduces the whole input to `[1]`; ReduceLast removes the last
  axis, retaining the leading axes.
- Fill has a finite FP32 static value. Slice has explicit axis/start/length.
  Broadcast aligns trailing axes and expands singleton axes.

The current `core.tensor.reshape` copies flat element order into a distinct
output. Do not silently redefine it as a zero-copy view. A future graph can
represent typed static views separately, or prove and explicitly lower a copy
to compatible physical aliasing. Copying registered State is not dead scratch.

### Implemented T0 baseline and tests

`TierZeroOperationContracts` is the logical frontend's shared T0 semantic catalog.
It exposes 25 operation IDs and 28 signatures: the 23 measured requirements,
two optional general-purpose operations, and three mixed-storage variants.
The contracts and signature lists are read-only. Concrete IC providers expose
signatures through `QueryInstruction`, not compiler-owned `TierZeroOperators`.
Each IC signature separately declares FP32 arithmetic/accumulation, including
FP16 input/output signatures. Storage types are per-port and must not be used to
infer compute precision. Explicit FP16 minima may be satisfied by these FP32
implementations without changing their existing numerical semantics.
Logical lowering validates geometry; each IC adapts and validates its typed
parameter array before recording any source.

The implementations reuse existing CPU numerical kernels and explicit tensor
loops, and generate one unfused serial-output GPU kernel per lowered operation.
GPU MatVec/BatchedMatVec and whole/last-axis reductions use compensated FP32
summation. Neumaier residuals are marked `precise` to prevent reassociation from
discarding compensation; non-finite sums explicitly discard the residual so
an isolated infinity does not incorrectly become NaN.
This is a correctness baseline, not a new optimizer or an interpreter.
The separate FP16 IC supports FP16 tensor inputs/outputs on both targets, with
FP32 intermediate arithmetic/accumulation and FP16 output rounding. Packed GPU
half stores use masked atomic compare/exchange to preserve adjacent elements,
including odd-length buffers. This is not a promise of native FP16 arithmetic.

Baseline numerical policy:

- Arithmetic, reduction accumulators and State are FP32. FP16 matrix/table
  values are converted to FP32 when loaded.
- CPU SIMD/FMA and GPU FP32 reductions may have different accumulation order.
  Reduction and transcendental results are not promised to be bit-identical;
  graph-level algebraic reassociation/fusion is not added.
- NaN propagates through Maximum and Relu. Maximum prefers positive zero when
  comparing positive and negative zero, retaining negative zero when both are
  negative. NaN payload preservation is not required.
- Copy, cast and zero-valued Tanh preserve the sign of zero. Relu maps negative
  zero to positive zero. Tanh saturates at +/-1 for infinities; Sigmoid maps
  negative/positive infinity to 0/1.
- Rsqrt returns NaN for negative nonzero inputs, signed infinity for signed
  zero, and positive zero for positive infinity. FP32 overflow can yield
  infinity; Fill still rejects non-finite static attributes.
- General subnormal behavior is not guaranteed across devices. FP16 cast tests
  include the smallest FP16 subnormal, which is a normal FP32 value.
- Finite conformance fixtures use an elementwise tolerance of
  `0.00003 + abs(reference) * 0.00003`; NaN/infinity classification and specified
  signed zeros are checked separately. This fixture tolerance is not a universal
  error bound for arbitrary ill-conditioned reductions or model inputs.

`TierZeroOperationTests` contains 94 tests. Every one of the 28 signatures runs
through logical lowering, native CPU assembly or actual GPU DXIL execution,
and repeated fixed-buffer invocation. Additional tests cover NaN, infinities,
signed zero, odd FP16 buffer lengths, malformed versions/types/shapes/attributes,
access and alias violations, and read-only input preservation. Model-width
4096-element cancellation and non-finite compensated sums have dedicated GPU
regressions. Hardware tests
must pass on each supported target device; the current host is not proof for
every future mobile/console device.

The related 264-test regression selection also validates RWKV-6/7 CPU/GPU
Prefill, inference, logits, State continuation/transport and artifact reload,
including the existing rwkvcpp golden-logits fixture. All passed without skips.
Full-model V6/V7 graph validation and CPU/GPU source generation were rerun.
GPU Maximum/Relu explicitly implement NaN/signed-zero behavior; CPU and GPU
Tanh correct signed-zero handling where existing approximations differ.

Regenerate pre-existing compiled artifacts to use these numerical fixes.
Tier-1/2 implementations and control-flow proposals below remain unimplemented.

### Actual tiny and full-model acceptance

The T0 implementation was executed, not just source-generated, with all six
available model fixtures. Each was tested serially against an unfused CPU
primitive reference, a compiled CPU VM and a compiled D3D12 VM:

| Model | Storage | CPU VM | GPU VM | GPU maximum logit error | GPU maximum State error |
|---|---|---|---|---:|---:|
| Tiny V6 3M | FP32 | Pass | Pass | 2.86102e-6 | 6.19888e-6 |
| Tiny V6 3M | FP16 | Pass | Pass | 7.62939e-6 | 9.05991e-6 |
| Tiny V7 834K | FP32 | Pass | Pass | 4.76837e-7 | 3.57628e-7 |
| Tiny V7 834K | FP16 | Pass | Pass | 2.38419e-7 | 4.76837e-7 |
| V6 World 7B | FP16 | Pass | Pass | 0.00317383 | 0.000854492 |
| V7 G1J 7.2B | FP16 | Pass | Pass | 0.0000343323 | 0.000106812 |

CPU VM logits and State matched the primitive reference exactly for these
fixtures. GPU comparisons retained the existing model acceptance gate:
`abs(actual-reference) <= 0.0003 + abs(reference)*0.0003`, with finite-value
checks and identical greedy argmax at every checked logit checkpoint. The
absolute maxima above are not the acceptance thresholds; relative scaling
matters for larger-magnitude values.

Inputs were raw token IDs `[1,2,3,2,4,3]`. Checks included three-token Prefill,
two successive inference calls, every registered State element after five
tokens, stream export/import, continued inference versus restored Prefill, and
reset-to-fresh-State output. Every large-model State snapshot contained
8,650,752 FP32 values. Runtime configuration used one Prefill and one Inference
worker with Prefill capacity four, and AMD Radeon RX 7900 XTX for hardware GPU
execution. This is short-sequence numerical acceptance, not a long-context
stress test, text-generation quality evaluation or warmed throughput benchmark.

The initial V6 7B GPU run exceeded this unchanged State tolerance with naive
serial sums. Compensated FP32 accumulation corrected the failure; all 12
CPU/GPU VM configurations were rerun successfully afterwards.

## Tier-1: basic merged operators (short, semantics-preserving fusion)

This section is a future design, not current implementation scope. CPU/GPU may
provide different Tier-1 subsets and mixed-storage signatures. Future fusion
must also specify intermediate rounding: removing an FP16 intermediate store
does not authorize removing its FP16 rounding boundary, even with FP32 internal
arithmetic. Typed reference expansions and alternate numerical contracts remain
later work.

Tier-1 fuses a subgraph of **2-3 Tier-0 stages** without changing its algorithm.
Use maximum Tier-0 dependency depth, not the number of tensor nodes or model
layers. A branched subgraph may contain more than three nodes at depth three.
Every Tier-1 definition exposes its complete Tier-0 expansion and ports.

Examples:

- `rsqrt(a + b)` (depth 2).
- `a * b + c` (depth 2).
- `(x - previous) * scale + previous` (depth 3).
- A depth-three branched multiply/add chain with several read-only inputs.

Main's existing fused expressions fit this boundary:

| Model | Depth 2 invocations | Depth 3 invocations | Total |
|---|---:|---:|---:|
| V6 | 322 | 322 | 644 |
| V7 | 257 | 417 | 674 |

These counts are identical on CPU/GPU. Main's 11 typed fused definitions per
model are the initial design reference; this does not assert 11 distinct
mathematical formulas.

Fusion must preserve external consumers, visible State writes and ordering.
Cross-thread dependencies require a legal cooperative kernel/synchronization
implementation; placing operations in one NodeDefinition is not sufficient.
Longer pointwise chains can remain compositions of Tier-1 calls. They do not
become Tier-2 merely because they contain four operations.

## Tier-2: semantic blocks and algorithmic rewrites

Tier-2 represents higher-level semantic blocks with deliberate algorithmic
rewrites, not just larger textual fusion. Candidate areas include normalization
blocks and RWKV recurrent/State updates that avoid expanded broadcasts,
slices or intermediate tensor materialization.

Each definition needs:

- A precise semantic operation and typed input/output/State contract.
- A reference expansion or other explicit reference oracle.
- Rewrite preconditions and permitted numerical deviations.
- State effects and failure/initialization behavior.
- Target implementations and measured selection evidence.

A faster cooperative MatVec is still a Tier-0 implementation. A LayerNorm
block using a different reduction algorithm is a Tier-2 candidate.
Tier-2 can decompose into Tier-0/1, and Tier-1 can decompose into Tier-0.
Reference expansion must be acyclic, terminating and tied to an IC identity.

## Implemented InstructionCollection provider boundary

The same IC API supports T0 and future Tier-1/2 extension packages.
There are three independent classifications:

| Classification | Meaning | Example |
|---|---|---|
| Tier | Semantic/rewrite strength | Tier-1 short fusion; Tier-2 recurrent block |
| Collection | GUID-identified exported capability set | T0 FP32, T0 FP16, future recurrent IC |
| Target feature | Implementation hardware requirement | AVX2/FMA; a particular GPU wave capability |

An extension may contain several instructions and both Tier-1 and Tier-2.
Installing an extension is not proof that its AVX2 or GPU implementation is
usable. Feature availability must be checked separately.

### Identity, architecture and packaging

- T0 FP32: `00000000-0000-0000-0000-000000000000`.
- T0 FP16: `00000000-0000-0000-0000-000000000001`.
- An incompatible precision/semantic contract uses another GUID, not a version.
- `InstructionCollectionDescription.Architecture` is an extensible target
  string, currently `cpu.managed` and `direct3d12`. It participates in matching.
  CUDA, Vulkan, native x64 and ARM can use distinct future identifiers; they
  are not currently implemented and are not all collapsed into "GPU".
- Same architecture and semantics with different precision may share one
  project, exposing separate provider classes. CPU and Direct3D12 live in
  separate `SharpInference.Instructions.Cpu` / `.D3D12` projects.
- `SharpInference.Instructions` contains only contracts/registry/recorder.
  `.TierZero` shares T0 adaptation logic. Neither compiler references a
  concrete IC project or the legacy CPU numerical backend.

`IInstructionCollectionProvider.QueryInstructionCollection()` returns
GUID/name/Tier/architecture descriptions. `QueryInstruction(Guid, string)`
returns implementations and their supported named parameter signatures.
`InstructionSignature` declares named ports, attributes and an actual
`KernelPrecisionProfile`. `GetSignature(parameters, precision)` requires the
caller's `PrecisionRequirement`; incompatible precision and ambiguous matches
fail explicitly.
`Instruction.Invoke(IInstructionRecorder, InstructionParameter[], PrecisionRequirement)` runs once
during assembly. It records source, named helpers, compile-reference assemblies
and synchronization; it does not execute tensor math.
Unsupported adaptation throws `InstructionAdaptationException`. T0 validates
before recording. Invocation is a nonvirtual validation/generation/recording
entry; providers implement protected validation and return a complete
`InstructionRecording` from `Generate`. Recorder helper collisions are atomic
and reported explicitly. No recording is submitted on validation/generation
failure.

### Independent graph generation and backend assembly

`VmExecutionGraphGenerator` owns its own injected IC catalog. Runtime's
`RwkvRuntimeFactory.CreateGraphGenerator` selects RWKV-6/7 from model/GGUF
metadata and the requested architecture; custom catalogs may be supplied.
Backend compilers independently receive their provider instance collections
in their constructors. Runtime factories are host composition conveniences,
with separate `instructionCollections` and `generatorCollections` parameters.
`VmGraphBackend` itself never installs defaults.

The host may load trusted provider assemblies and then inject instances.
Backends do not scan directories, reflectively discover providers or load
plugins. A generator may carry only its CPU catalog while the receiving host
supplies all installed CPU/Direct3D12 providers. XML transfers GUID/name,
static parameters and target declarations, never class instances. There is no
per-token registry lookup or reflection.

The current recorder supports source/helper emission and GPU group-memory
synchronization. Separate GPU dispatches and host UAV barriers must already
be explicit in the execution program; automatic multi-dispatch instruction
expansion and network transport are not implemented.

### IC extraction verification

- 377 directly related tests passed across regression selections, with no skips.
  Coverage includes every FP32 signature, all 24 FP16 instructions on CPU and
  real Direct3D12 hardware, odd-length packed stores, independent catalogs,
  host-side assembly loading, custom non-T0 instructions with nonstandard port
  names, read/write State, helper conflicts and source/binary reload.
- All six tiny/full RWKV-6/7 fixtures above were rerun with extracted ICs:
  six independent references and twelve compiled CPU/Direct3D12 configurations
  passed the unchanged logits/State tolerance. Numerical maxima were unchanged.
- Full Release solution build passed with zero warnings/errors.
- A source-only tiny V7 FP16 package was embedded and published as Windows x64
  Native AOT. It executed with dynamic code disabled and passed all 256 logits
  against the independent reference, with maximum absolute error `2.38419e-7`.
  Native publication still reports existing logical-graph JSON trimming/AOT
  warnings; this does not certify those unused frontend JSON paths for AOT.

### Future annotated-method authoring

A provider base class can discover specially annotated methods and expose
their declared signatures. This is an authoring/discovery mechanism, not the
whole instruction ABI.

Method types/names can supply port types, scalar types and overload identity.
Supplementary metadata must supply IC GUID/name, Tier, symbolic shape
constraints, layout, alias/access rules, numerical policy, State effects,
reference expansion and target features. Tensor shape cannot be reliably
inferred from a `Span<float>` method parameter.

A method might declare semantics, implement a CPU kernel, emit target code, or
provide a graph rewrite. Those roles must be distinct. Reflection does not turn
an arbitrary C# implementation into a GPU implementation.
Annotated declarations may be nonpublic; discovery must be deterministic,
handle inheritance explicitly, reject ambiguous/duplicate contracts, and not
depend on reflection enumeration order.

For SDK/tooling, reflection can discover trusted providers once. For Native AOT,
prefer a build-time generated registry/manifest, preserving annotated private
methods explicitly. Deployment must not rely on runtime private-method scanning
or dynamic assembly loading.

### Freeze requirements into the execution program

The logical graph may be target-independent. A selected execution graph must
name the exact instruction/contract, target implementation and code-generation
configuration. Preserve GUID/name and architecture in program requirements;
artifact format/ABI fingerprints are separate from instruction identity.

The separation is:

1. Semantic instruction provider and contract catalog.
2. Pattern/rewrite provider used by the external optimizer.
3. CPU/GPU implementation and code-emission provider.
4. Target capability/feature catalog used by selection and validation.

They can be delivered in one package without becoming one undifferentiated
runtime callback.

Missing/incompatible required instructions fail explicitly. A caller may choose
a lower-tier expansion during external optimization, but the VM must not
silently substitute one or re-optimize an imported execution graph.
Prebuilt C#/DXIL/native artifacts should not require rediscovering the authoring
plugin to execute. Dynamic extension discovery belongs primarily to preparation,
not the per-token hot path.

Plugins and generated kernels remain subject to fixed-slot rules: they cannot
create/delete/resize/rebind tensor slots at execution time. Optimizer rewrites
may request explicit temporary storage before final allocation. Local scalar
variables and statically planned scratch remain legal.
These are trusted extension contracts, not an isolation guarantee for arbitrary
untrusted native code.

### Remaining extension-authoring decisions

- Are annotations semantic declarations, implementation exports, or separate
  annotation kinds for each role?
- Is expansion required for every extension, or can a non-decomposable block
  provide an explicitly identified oracle?
- Which scalar/shape expression language becomes part of the stable signature?
- Which CPU/GPU target requirements and static-deployment registries are needed?

The implemented registry rejects identity/architecture conflicts and ambiguous
implementations. Hardware-feature selection and annotation discovery are not
yet an implemented extension-authoring API.

## Future graph control flow: separate from operator tiers

The current runtime explicitly lacks general dynamic branches/loops.
The later graph redesign must support more than a static token sequence;
this proposal does not claim those nodes are implemented.

Separate concepts:

- **Data fan-out**: several consumers of the same value; already expressible as
  dependencies. It does not guarantee concurrent execution.
- **Fork**: explicit branch/task creation with declared resources and ownership.
- **Join**: wait for all selected parallel branches and merge dependencies;
  it is not numerical addition/concatenation.
- **If/Condition**: evaluate a predicate and execute only the selected branch.
- **Phi/conditional merge**: select the value from the branch actually taken;
  it must not read an uninitialized output from an untaken branch.

A tensor Select/Where computing both alternatives is different from If.
Bool/comparison/predicate-producing operations may later extend the common
Tier-0 profile on both targets; routing belongs to graph control flow.

All branches still use initialization-time fixed slots. Allocation/liveness
must become path-aware; mutually exclusive branches may alias storage only
with a proof, while concurrent branches may not alias conflicting writes.
Shape-changing branch outputs need a declared bounded representation or
separate precompiled variants, not runtime slot resizing.

Conditionally modified State remains explicit and Session-owned. Parallel
branches must declare disjoint writes or an explicit serialized/reduction merge.
Only the selected branch may commit its State effects, and failure/partial-write
semantics must be specified rather than assumed transactional.

GPU data-dependent branching needs a declared device-control mechanism or
an explicit host synchronization boundary. Do not implicitly read back a
predicate per node. Backend capability checks and CPU/GPU lowering support
must be designed with that execution-model choice.

## Next design sequence

1. Freeze the shared Tier-0 semantic/type/numeric profile and conformance tests.
2. Preserve explicit VM precision requirements and IC capabilities throughout
   preparation, serialization and static deployment.
3. Later agree on extension discovery and typed expansion, and consider
   Tier-1 patterns in the external optimizer, preserving explicit decisions.
4. Design the control-flow/value/State ABI and path-aware storage model.
5. Add measured Tier-2 rewrites and additional model profiles.

This document adds no operator/plugin/control-flow implementation.
