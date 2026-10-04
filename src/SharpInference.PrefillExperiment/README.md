# RWKV-6 prefill experiment

Historical measurements below used the former model-specific production
backend and are archival, not expected performance for the compiled VM
backend now used by the live comparisons.

This executable evaluates RWKV-6 layer-major prefill on CPU and
chunked WKV plus tiled projection kernels on Direct3D 12. Its independent
experiments do not replace the production backends; separate comparison modes
exercise the integrated production GPU prefill implementation. The CPU experiment computes **full-model
logits and state** for every token sequence. DirectX experiments include
isolated kernels, an integrated **first-layer projection-to-WKV stage**, an
experimental **full-model hybrid CPU/DirectX** evaluator, and a separate
**GPU-resident full-model** candidate. The hybrid keeps
the original RWKV-6 model and output semantics but is not GPU-resident:
normalization, dynamic TimeMix and elementwise operations still run on the CPU,
with each matrix projection and chunked WKV stage returning to host memory.
Only the complete-model runs should be compared against production `Prefill`.

Run from the workspace root:

```powershell
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --tokens 1024 --chunk 64 --repeats 3
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --cpu-tensor-primitives --tokens 256 --projection-width 256 --repeats 5
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --tokens 1024 --chunk 64
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --model "F:\RWKV\RWKVModels\tiny-rwkv-6v0-3m-FP32.bin"
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --model "F:\RWKV\RWKVModels\tiny-rwkv-6v0-3m-FP32.bin" --layer-major --model-tokens 1024
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --model "F:\RWKV\RWKVModels\tiny-rwkv-6v0-3m-FP16.bin"
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --model "F:\RWKV\RWKVModels\RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin" --model-tokens 16
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --model "F:\RWKV\RWKVModels\RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin" --attention-pipeline --pipeline-tokens 256 --pipeline-repeats 3 --model-tokens 256
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --full-gpu-experiment --model "F:\RWKV\RWKVModels\tiny-rwkv-6v0-3m-FP16.bin" --model-tokens 65 --chunk 64
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --full-gpu-experiment --full-gpu-nonzero --model "F:\RWKV\RWKVModels\tiny-rwkv-6v0-3m-FP16.bin" --model-tokens 65 --chunk 64
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --full-gpu-experiment --model "F:\RWKV\RWKVModels\RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin" --model-tokens 256 --chunk 64
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --full-gpu-resident --model "F:\RWKV\RWKVModels\tiny-rwkv-6v0-3m-FP16.bin" --model-tokens 65 --chunk 64
```

The test-model directory in this workspace is `F:\RWKV\RWKVModels`. It contains
the tiny RWKV-6 FP32/FP16 GGML models and a 7B FP16 GGML model; the `.pth` file
there is a training checkpoint, not a GGML inference model. Pass another GGML
path with `--model` as needed. The `--gpu` model baseline requires a Direct3D 12
adapter and sufficient VRAM; the compiled VM executes FP32 operators
and accepts FP16 model weights. `--model-tokens` sets
the full-model baseline length (default 32); `--model-repeats` (default 2)
alternates measurement order and reports averages. Synthetic dimensions can be adjusted
with `--heads` and `--head-size`. `--projection-width` sets the width of the
separate CPU projection microbenchmark (default 256).

`--cpu-tensor-primitives` runs only the CPU primitive migration microbenchmark.
It compares direct scalar loops, the production CPU backend, and applicable
`TensorPrimitives` candidates for FP32/FP16 elementwise operations, FP16
reduction, and FP16 MatVec. `--tokens` selects the element count and
`--projection-width` selects the square MatVec width. For stable optimized-JIT
comparisons, set `DOTNET_TieredCompilation=0`.

The same mode validates and measures custom CPU kernels: FP16-to-FP32 cast,
multi-axis broadcast, per-head outer product, and branched fused expressions.
The custom comparisons use the old scalar kernel as a reference; the new graph
timing also includes dispatch and output allocation, so these ratios are not
an isolated before/after graph benchmark. No model files are required.

### CPU Tensor migration coverage

| Operation | CPU implementation |
| --- | --- |
| FP32/FP16 elementwise primitives | `TensorPrimitives`; short Half Sigmoid/Rsqrt explicitly promote to FP32 to avoid intermediate Half rounding |
| FP32 reductions and MatVec | `TensorPrimitives.Sum` and per-row `Dot` |
| FP16 reductions | Blocks of at most 1024 values, `ConvertToSingle` then FP32 `Sum`, FP32 block accumulator |
| Half matrix / FP32 vector | Serial block conversion plus FP32 `Dot`, or parallel row conversion for large matrices |
| Half matrix / Half vector | Convert the vector once, then row/block conversion and FP32 `Dot`; output is rounded to Half |
| Cast | `TensorPrimitives.ConvertToSingle` |
| Broadcast | Zero-copy `ReadOnlyTensorSpan` / `TensorSpan` views and `Tensor.BroadcastTo` |
| HeadOuter | Scalar-by-span `TensorPrimitives.Multiply` for each row |
| BatchedMatVec and last-axis reductions | Reuse the primitive backend |
| Straight-line fused expression | Reuse the primitive backend |
| Branched fused expression | Blocks of at most 256 values, reuse the primitive backend, retain referenced earlier steps in bounded scratch |
| Copy/GatherRow/Reshape/Slice | Reuse Span copy; Reshape/Slice call the primitive backend |
| Fill | `Span.Fill` |

Migration is not restricted to operations with a speedup. Scalar exceptions
are retained only for measured significant regressions:

- Fewer than 8 Half reduction elements and fewer than 8 columns in Half/Half
  MatVec retain FP32 scalar arithmetic. The measured 7-element reduction and
  3x3 MatVec were approximately 1.7x and 2x slower after conversion.
- Contiguous TensorSpan copy/fill candidates were approximately 245x/93x slower
  than Span copy/fill at length 4097 on the measured x64 machine. These candidates
  remain in the benchmark, not in the production memory kernels.

FP16 data remains stored as Half; no full FP32 weight matrix is materialized.
SIMD reduction order can change floating-point rounding. Tests cover block tails,
short vectors, non-finite inputs, branching, repeated pooled workspace use,
weight residency, and tiny RWKV-6/7 fusion parity. The implementation uses
portable .NET APIs; ARM64 hardware performance must be measured separately.

`--layer-major` requires `--model` without `--gpu`. The CPU 7B FP16 model
can require tens of GiB of host RAM when its weights expand to FP32; use the
tiny FP32 or FP16 model for full-model CPU tests.

`--attention-pipeline` requires `--gpu --model`; its token count defaults to
16 and is independent of `--model-tokens`. The stage benchmark requires a
Direct3D 12 adapter and keeps its buffers on the device between kernels.

`--full-gpu-experiment` requires `--gpu --model`. It first measures production
portable-graph `Prefill`, then **disposes** the production GPU backend before uploading the
experimental model weights to a separate GPU projector. It checks **all**
final logits and state values against production, warms the independent
evaluator and its device-weight cache, and times complete prompt prefill.
`--full-gpu-nonzero` starts the measured prompt from a production state
created by three earlier tokens; without it, the starting state is zero.
Logits are checked elementwise with tolerance `0.0002 + 0.001 * |reference|`;
the hybrid recurrent state uses `0.0005 + 0.002 * |reference|`, and the
GPU-resident recurrent state uses `0.001 + 0.003 * |reference|` to account for
the different FP32 reduction orders through the entire model. The resident
run additionally reports how many state entries exceed the **stricter**
hybrid threshold, so small outliers are not hidden. Neither check is an
exact-equality guarantee. `--full-gpu-diagnostics` prints out-of-tolerance
counts and worst normalized errors without stopping before the timed run;
it still sets a nonzero exit code if the comparison fails.
Times include CPU preparation, activation transfers, GPU execution and
readback on every projection/WKV stage, but exclude model load and GPU
weight-cache warmup. These results must not be confused with the
GPU-resident first-layer stage's kernel-only measurements.

`--full-gpu-resident` is a separate candidate for the performance question.
It requires `--gpu --model` and cannot be combined with
`--full-gpu-experiment`. It retains all per-token intermediates and the
recurrent state on the device across layers, with only initial input/state
upload and final logits/state readback; the same production comparison,
`--full-gpu-nonzero`, `--full-gpu-diagnostics`, `--model-tokens` and
`--model-repeats` options apply. Its initial run uploads/cache-creates device
weights; subsequent timed runs reuse them. Treat a build as insufficient:
only a passing full-model numerical comparison and a measured end-to-end run
can establish feasibility or performance.

The sequential WKV implementation repeatedly updates every state element for
every token. The chunked implementation computes an affine summary `(a,b)` per
chunk and state element (`state_out = a * state_in + b`) independently across
chunks, then composes summaries in token order. DirectX dispatches one kernel
per token in the final-state baseline versus summary and merge kernels for
the chunk path. The **all-output** version
first summarizes each chunk and computes its starting state, then processes
chunks in parallel, replaying the tokens within a chunk to produce each
token's attention output and the final state. It uses scratch proportional
to `chunk count * state size`, not `token count * state size`. Both paths must
match a CPU reference. Checks include lengths around chunk boundaries,
nonzero initial state, output at every token, and a non-multiple-of-16
projection shape.

With `--layer-major`, a separate CPU evaluator loads GGML tensors and
processes all prompt tokens through each RWKV-6 layer before moving to the
next. The evaluator computes dynamic TimeMix projections, chunked WKV,
GroupNorm, output projection, ChannelMix and the final logits; it preserves
the production state layout. It compares final state and logits to an
independent production `ProcessorSession.Prefill` on a **nonzero imported
state**, and times three ablations: layer-major with sequential WKV,
layer-major with chunked WKV, and layer-major with chunked WKV plus 16-token
SIMD-batched rectangular projections. The optional ordinary model baseline
checks that production `Prefill` and repeated `ForwardToken` agree.

Timings are wall-clock durations after warmup, not GPU timestamp-query
times. The isolated DirectX kernel measurements include recording, submission
and the fence wait, but exclude input uploads, initial-state reset and final-state
readback. The complete-model hybrid measurements **include** activation
transfers and readback at every operator boundary. Compare only within the
same device/run; shader compilation and model loading are excluded.
CPU chunk summaries allocate intermediate buffers and parallelize across
chunks. Do not extrapolate a synthetic WKV-only speedup to a 7B model speedup.
The separate synthetic CPU projection experiment compares per-token
matrix-vector products with reusing each weight for a 16-token block; it
also benchmarks a transposed-input `Vector<float>` implementation against
`TensorPrimitives.Dot`. Neither is a tuned BLAS GEMM. The DirectX projection
experiment compares per-token dispatches to a 16-by-16 shared-memory tiled
FP32 or packed-FP16-weight matrix product. When a FP16
GGML model is supplied, it also tests the model's actual first-layer
`att.key.weight` tensor against CPU FP32 calculations on 16 synthetic
activations. The GPU stage kernels do not reproduce the entire dynamic
TimeMix/ChannelMix graph.

### GPU-resident first-layer stage

[The pipeline benchmark](./GpuAttentionPipelineBenchmark.cs) uses the first
layer's real FP16 key, value and receptance matrices. Its input generator
reads GGML embeddings, applies input and attention LayerNorm, dynamic MAA
W1/W2 and TimeMix for the three distinct projection inputs, and computes
per-token decay with the real decay weights. These particular preparatory
operations currently run **on the CPU before measurement**; they are not
part of the GPU-resident benchmark. The starting WKV matrix contains
deterministic small nonzero values; the previous-token TimeMix input starts
at zero. The resulting stage sequence is valid for the supplied inputs,
but is **not** a captured intermediate sequence from a complete 7B session.

One command list records three FP16 tiled matrix projections into persistent
K/V/R buffers, then chunk summary, chunk-state fold and all-token WKV output;
the intermediate projections do **not** go back to the CPU. The comparison
path records three per-token FP16 matvec dispatches followed by a per-token
WKV dispatch, with the state reset before (not during) timing. The outputs
of all three projections, every WKV token and final state are checked against
an independent FP32 CPU calculation. Input upload, initial-state reset, CPU
preparation, readback and shader compilation are excluded from the timing;
command recording, submission and GPU fence wait are included. This tests
one layer's **projection-to-WKV** schedule, not GroupNorm, attention output,
FFN, the remaining layers, or full-model logits. Its serial path is an
experimental baseline, **not the production fused projection shader**.

## Initial final-state-only measurements

Release build, AMD Radeon RX 7900 XTX, RWKV-6 7B dimensions (32 layers,
4096 embedding, 64 heads of size 64), 1024 synthetic tokens, chunk size 64,
three timed runs. Times are averages and may vary with driver load:

| Isolated operation | Sequential | Experimental | Ratio |
| --- | ---: | ---: | ---: |
| CPU WKV final state | 223.388 ms | 203.438 ms | 1.10x |
| DirectX WKV final state | 3.458 ms | 0.350 ms | 9.89x |
| CPU projection, width 256 | 49.729 ms | 119.020 ms | 0.42x |
| DirectX projection, width 256, FP32 | 1.865 ms | 0.191 ms | 9.77x |

The **unchanged** production DirectX FP16 backend with the 7B GGML model
processed 512 tokens in 11,996.919 ms via repeated `ForwardToken`, versus
12,094.849 ms via `Prefill` (two runs with alternating order, ratio 0.99x).
These initial numbers measured only the final state. They cannot indicate
whether a full layer can consume the output of every prompt token.

## All-output and full-model CPU experiment

Later measurements with 1024 synthetic tokens and a 7B-shaped state (64
heads of size 64), chunk size 64, three runs:

| Isolated operation | Sequential | Chunked | Ratio |
| --- | ---: | ---: | ---: |
| CPU WKV all token outputs and final state | 867.694 ms | 381.178 ms | 2.28x |
| DirectX WKV all token outputs and final state | 9.504 ms | 2.083 ms | 4.56x |
| DirectX synthetic packed-FP16 projection, width 256 | 1.354 ms | 0.289 ms | 4.69x |

On the actual 7B FP16 model's **first-layer key weight** (4096x4096,
16 synthetic activations), DirectX per-token projection took 8.921 ms
versus 0.289 ms for the FP16 tiled kernel (30.89x); both matched the CPU
reference within 1.20e-7 maximum absolute error. This is *one projection*,
not the model's complete TimeMix or prefill path.

An independent full-model CPU experiment using the tiny FP32 GGML model
(12 layers, embedding 128) and a nonzero state after a 3-token prefix
gave these **1024-token** averages over three alternating-order runs:

| CPU complete-model path | Time | Ratio to production Prefill |
| --- | ---: | ---: |
| Existing `Prefill` | 1,794.215 ms | 1.00x |
| Layer-major, sequential WKV | 553.781 ms | 3.24x |
| Layer-major, chunked WKV | 468.289 ms | 3.83x |
| Layer-major, chunked WKV and SIMD-batched projections | 915.524 ms | 1.96x |

Final logits differed by at most 3.82e-6, and the flat recurrent state
by at most 8.55e-4; every element passed a combined absolute/relative
tolerance. The FP16 tiny model also passed a 1024-token full-model comparison.
At this size, the bulk of the CPU gain comes from reorganizing execution and
avoiding per-token plan/parallel-dispatch overhead, **not** the chunk scan
alone. SIMD batching the many small real-model projections *regressed*
despite its isolated synthetic projection improvement.

The production 7B DirectX path remains **unchanged**. The separate
layer-major hybrid evaluator now executes projections and WKV on DirectX,
but its CPU-side preparation and readback between operators make it
fundamentally different from an optimized GPU-resident graph. Production
still uses token-scoped scratch; integrating layer-major execution into it
would require chunk-wide activations and GPU-resident state and intermediate
values. RWKV-6 TimeMix depends on the previous token's layer input, so all
inputs for one layer must be available before processing its block; the next
layer then consumes all of that layer's per-token outputs. The CPU tiny-model
speedup must not be extrapolated to GPU 7B prefill.

## GPU-resident first-layer measurements

Release build on the AMD Radeon RX 7900 XTX, real 7B FP16 GGML weights,
three timed runs per stage case:

| Stage inputs | Per-token stage reference | Tiled projections + chunked WKV | Stage ratio |
| --- | ---: | ---: | ---: |
| 16 tokens, chunk 64 | 5.194 ms | 0.915 ms | 5.68x |
| 65 tokens, chunk 32 | 21.211 ms | 3.203 ms | 6.62x |
| 256 tokens, chunk 64 | 83.592 ms | 9.848 ms | 8.49x |
| 512 tokens, chunk 64 | 164.704 ms | 19.346 ms | 8.51x |

For the 256-token run, the maximum absolute differences from CPU across
the three projections, all WKV outputs, and final state were respectively
1.91e-6, 2.45e-4 and 6.87e-5. The estimated upload, default-heap and
readback allocations were 285,245,440 bytes (approximately 272 MiB).
For the 512-token probe the maximum WKV-output difference was 3.06e-4,
with approximately 348 MiB of estimated allocations. For comparison only,
the **unchanged** production 7B FP16 `Prefill` took 6,047 ms for 256 tokens
and 12,093 ms for 512 tokens (two alternating-order runs). These stage
figures use CPU-prepared TimeMix inputs; comparing them directly to the
full-model timings or multiplying by 32 does **not** establish a full-model
GPU speedup.

## Complete-model hybrid CPU/DirectX measurements

Release build, AMD Radeon RX 7900 XTX, `--chunk 64`, actual GGML FP16
weights, warm experimental GPU weight cache, complete RWKV-6 prefill and
independent final-logit/**entire-state** comparison against unchanged
production DirectX FP16 `Prefill`. Production model/device resources are
disposed **before** the experimental GPU weight cache is created; the two
model copies do not occupy VRAM simultaneously. Each 7B row below is one
timed run (warmup and shader compilation excluded):

| Model and tokens | Production `Prefill` | Hybrid prefill | Production / hybrid | Largest logit difference | Largest state difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny, 65, nonzero starting state | 45.103 ms | 206.931 ms | 0.22x | 6.68e-6 | 1.22e-4 |
| 7B, 16 | 378.694 ms | 3,412.124 ms | 0.11x | 0.005127 | 0.002625 |
| 7B, 256 | 6,040.371 ms | 55,676.412 ms | 0.11x | 0.006226 | 0.003052 |
| 7B, 512 | 12,096.559 ms | 107,211.429 ms | 0.11x | 0.003662 | 0.013672 |

The tiny row uses two timed hybrid runs, and a separately timed production
prefill from the same nonzero state after a three-token prefix. The 7B
rows start from zero state. The 512-token run verifies 8,650,752 state
entries and all logits with **zero** entries outside the documented
elementwise tolerances. At the earlier, tighter state-only threshold
`0.0002 + 0.001 * |reference|`, 30 of those 8,650,752 state entries
exceeded the threshold; the worst error was 2.25 times that tighter
threshold, though all logits still passed. The documented state tolerance
`0.0005 + 0.002 * |reference|` accommodates the observed different
FP32 reduction order without claiming bitwise equivalence. The worst
normalized state error at that tolerance was 0.92 for 512 tokens.

The hybrid runs **all 32 layers**, including dynamic TimeMix, each matrix
projection, chunked WKV output for every token, GroupNorm, ChannelMix and
the final vocabulary head. It issues 353 separate GPU projections per
run. At 512 tokens those projections (including transfers and
fence/readback waits) account for about 98.0 seconds of the 107.2-second
experimental prefill. The independent model is therefore roughly **8.9x
slower** than production at 512 tokens, despite the earlier 8.51x gain
inside the isolated GPU-resident first-layer stage. This path lacks a
GPU-resident layer schedule and pays CPU/GPU round trips between operators.
These hybrid measurements establish feasibility and correctness of
layer-major prefill on DirectX, **not** its achievable GPU speedup. The
separate resident candidate below tests that question end to end.

## GPU-resident complete-model candidate

The former `--main-prefill-compare` mode compared the removed legacy
per-token backend against its dedicated prefill graph and is no longer
available. Use the independent hybrid or GPU-resident evaluator to compare
experimental approaches with the portable-graph production baseline.

Unlike the hybrid correctness bridge above, the
[resident evaluator](./GpuResidentRwkv6Experiment.cs) uploads and caches
the GGML model weights in GPU memory and records the entire layer-major
RWKV-6 graph on the device. It uploads prompt token IDs and the initial
recurrent state, then runs embedding lookup, layer and group normalization,
dynamic TimeMix, all matrix projections, chunked WKV with all-token
outputs, FFN and final vocabulary projection **without intermediate CPU
readback**. Only the final logits and complete recurrent state are
downloaded. A reusable instance keeps FP16-packed matrix weights across
timed runs; prompts can start from an imported nonzero state. The
production backend is released before creating this instance, avoiding
simultaneous residency of two 7B models.

Release build, AMD Radeon RX 7900 XTX, actual FP16 GGML models, chunk size
64. Baseline and experimental numbers below are complete prefill times,
including per-run GPU synchronization and final readback but **excluding**
model loading, shader compilation and the initial experimental weight
upload. The 65- and 512-token rows average two timed runs; other rows
are one run each:

| Model | Tokens | Production `Prefill` | GPU-resident prefill | Speedup | Max logit difference | Max state difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny, nonzero state | 65 | 46.750 ms | 24.445 ms | 1.91x | 8.58e-6 | 4.58e-5 |
| 7B | 16 | 377.928 ms | 235.788 ms | 1.60x | 0.002441 | 0.002319 |
| 7B | 256 | 6,029.566 ms | 1,547.005 ms | 3.90x | 0.003296 | 0.001953 |
| 7B | 512 | 12,080.311 ms | 2,947.625 ms | 4.10x | 0.002930 | 0.014282 |
| 7B | 1024 | 24,179.701 ms | 5,684.435 ms | 4.25x | 0.003906 | 0.011841 |

All final logits and all 8,650,752 recurrent-state entries of the 7B
model passed the stated resident tolerances for 16, 256, 512 and 1024
tokens; 512 tokens had five state entries outside the earlier *stricter*
hybrid tolerance, and 1024 tokens had one. The 1024-token row was also run
with `--full-gpu-diagnostics`; it reported **zero** entries outside the
resident tolerance and exited successfully. These are numerical
comparisons, not bit-for-bit reproduction. The full-model result supports
an end-to-end long-prompt speedup for this GPU and model; it does **not**
isolate how much is due specifically to chunked WKV versus layer-major
batched projections, elimination of per-token graph overhead, and avoiding
host/device round trips. An ablation disabling chunked WKV inside the
same resident evaluator would be needed to quantify that share.

As a limited chunk-size sensitivity check on 7B / 512 tokens, one timed
resident run each took 2,937.508 ms with chunk 32, 2,947.625 ms with
chunk 64 (two-run average), and 2,949.624 ms with chunk 128; each passed
the resident numerical tolerance. These differences are comparable to
run-to-run timing variation and do not identify a best chunk size or the
independent contribution of chunked WKV.

To reproduce the 7B long-prompt result:

```powershell
dotnet run --project src\SharpInference.PrefillExperiment -c Release -- --gpu --full-gpu-resident --model "F:\RWKV\RWKVModels\RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin" --model-tokens 512 --model-repeats 2 --chunk 64
```
