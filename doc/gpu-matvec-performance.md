# Cooperative GPU MatVec: implementation and measurements

Measured on 2026-10-05 using the implementation worktree, Release .NET 10,
AMD Ryzen 9 3900X and hardware AMD Radeon RX 7900 XTX, driver
32.0.31041.1004. This compares two implementations in the same current VM,
not the historical main backend.

## Implemented scope

- Existing T0 `core.mat-vec`, including native FP16 matrix/FP32 activation
  and legacy all-FP16 storage signatures; internal arithmetic remains FP32.
- One 64-thread group per output row. Adjacent lanes read adjacent columns.
- Per-lane Neumaier accumulation and fixed shared-memory merging of both
  sum and correction. No discarded residuals, whole-weight casts, wave-size
  assumption or extra partial-sum dispatch.
- Balanced two-dimensional row grids above 65535 rows; explicit group/lane
  compiler context, typed offset views and atomic packed-half output stores.
- Shared launch descriptor and validation of isolated kernels, thread size,
  row grid, alias restrictions, XML and artifact round trips.
- Explicit serial/cooperative overrides and offline, exact-domain profile
  selection. New GPU graphs now default to cooperative; profiles are never
  auto-loaded or timed during inference. The measurements below used an
  explicitly controlled serial/cooperative comparison before that default
  change.

CPU, BatchedMatVec, other reductions, activation reshape elision and managed
GPU entry upload/readback policy are unchanged.

## Resident GPU kernel measurements

The offline tool discovers actual post-weight-view MatVec domains from all
four models. It uploads deterministic finite synthetic matrices/vectors,
checks every output, warms four batches and alternates nine paired samples.
Each sample explicitly records eight executions in one command list. GPU
compute-queue timestamps exclude upload, result readback and CPU recording/
submission, but include dispatches and the scheduled output UAV barrier.

An atomic-counter regression verifies exactly eight executions. An initial
instrumentation attempt submitted duplicate references to the same cached
command list in one queue submission; this host executed only one instance.
That method was corrected and all resident measurements regenerated. Those
initial times are not used.

All 27 measured shape/storage domains had positive conservative savings:
`Q25(serial)*0.98 - Q75(cooperative)*1.02 > 0`. This is a selection margin,
not a statistical confidence interval or universal performance guarantee.

Representative medians, microseconds per MatVec:

| Matrix shape | Matrix/input/output storage | Serial | Cooperative | Kernel speedup |
|---|---|---:|---:|---:|
| 4096 x 4096 | FP16/FP32/FP32 | 117.230 | 40.830 | 2.87x |
| 4096 x 14336 | FP16/FP32/FP32 | 1123.360 | 197.390 | 5.69x |
| 4096 x 16384 | FP16/FP32/FP32 | 1558.945 | 218.665 | 7.13x |
| 65536 x 4096 | FP16/FP32/FP32 | 5919.415 | 641.800 | 9.22x |

All synthetic outputs matched serial exactly in these fixtures. Repeated
use warms caches; these are not cold-cache timings, and synthetic weights
do not replace numerical testing with real model weights.

## End-to-end GPU model comparison

The same freshly regenerated T1 profile is enabled in both groups:
serial MatVec versus independently profile-selected cooperative MatVec.
All 37 T1 domains were eligible. Both groups retain identical T1 calls,
dispatch counts and local capacities; only MatVec execution changes.

There are 16 separate process runs: four models, two variants and two rounds.
Order is serial/profile in round zero and profile/serial in round one.
Large models are loaded and released sequentially, never simultaneously.
Each process uses one Decode/Prefill instance, capacity eight, both paths
warmed for at least two cycles and two seconds, and seven samples of eight
tokens `[1,2,3,2,4,5,1,3]`. Setup/compilation and numerical validation are
excluded from timing.

Summary values below are the arithmetic mean of the two per-process
medians, not pooled-sample medians. Lower latency is better.

| Model | Decode serial -> cooperative, ms/token | Decode latency reduction | Prefill serial -> cooperative, ms/token | Prefill latency reduction |
|---|---:|---:|---:|---:|
| Tiny V6 3M FP32 | 112.8242 -> 110.9007 | 1.70% | 16.8129 -> 15.8300 | 5.85% |
| Tiny V7 834K FP32 | 105.0464 -> 104.8438 | 0.19% | 15.3668 -> 15.1096 | 1.67% |
| V6 7B FP16 | 320.0075 -> 260.1431 | 18.71% | 127.8864 -> 70.5996 | 44.80% |
| V7 7.2B FP16 | 336.6340 -> 258.4414 | 23.23% | 138.5940 -> 65.8835 | 52.46% |

Per-process medians are retained to expose variation:

| Model | Serial Decode r0/r1 | Cooperative Decode r0/r1 | Serial Prefill r0/r1 | Cooperative Prefill r0/r1 |
|---|---:|---:|---:|---:|
| Tiny V6 | 111.9087 / 113.7396 | 110.8040 / 110.9975 | 16.7025 / 16.9233 | 15.8477 / 15.8122 |
| Tiny V7 | 105.8106 / 104.2822 | 104.7818 / 104.9058 | 15.4887 / 15.2448 | 15.0932 / 15.1260 |
| V6 7B | 319.0396 / 320.9754 | 260.7029 / 259.5833 | 128.3162 / 127.4567 | 70.8358 / 70.3634 |
| V7 7.2B | 339.6501 / 333.6179 | 257.4260 / 259.4569 | 139.8200 / 137.3680 | 65.7820 / 65.9850 |

Tiny V7 Decode differences change sign between rounds: do not claim a
stable Decode improvement for that case. Large-model improvements occur in
both orders. Two rounds are still limited evidence, not a device-wide
guarantee or formal significance test.

| Model | Cooperative MatVec calls | T1 calls | Total dispatches | Local bytes |
|---|---:|---:|---:|---:|
| Tiny V6 | 133 | 244 | 1389 | 34988 |
| Tiny V7 | 167 | 254 | 1397 | 57648 |
| V6 7B | 353 | 644 | 9045 | 5166892 |
| V7 7.2B | 447 | 674 | 3697 | 4151468 |

Large-model cooperative Decode rates are approximately 3.844 and 3.869
token/s; Prefill rates are 14.164 and 15.178 token/s. They must not be
treated as same-protocol comparisons with historical main measurements.

## Numerical validation

Every noninitial run checks every Decode token's complete logits and every
named final State tensor against its model's first serial run. There are
804 successful comparison records, including serial repeatability checks;
536 records are the eight cooperative runs. The unchanged model gate is:

```text
abs(actual-reference) <= 0.0003 + abs(reference)*0.0003
```

All values are finite and State tensor names/shapes agree.

| Model | Maximum absolute error, logits/State | Maximum scaled error relative to model gate |
|---|---:|---:|
| Tiny V6 | 0 | 0 |
| Tiny V7 | 0 | 0 |
| V6 7B | 0.00024414062 | 0.026733566 |
| V7 7.2B | 0.000015258789 | 0.010861715 |

The V7 maximum logit error is 0.000005722046; its larger absolute maximum
comes from State. Cooperative reductions need not be bit-identical to the
serial implementation. Fixture tests retain the stricter operator tolerance
`0.00003 + abs(reference)*0.00003`, with separate NaN/Inf classification
checks, cancellation within/across lanes, tails, padded grids, FP16 packed
stores, views, repeated invocation, input preservation, invalid launch
selection and artifact reload coverage.

Final validation ran 586 directly related VM, instruction, T0/T1, GPU and
external-model regressions with zero failures or skips. The full Release
solution build completed with zero warnings and errors.

## Using the implementation

Normal Runtime now defaults to cooperative GPU MatVec without costs.
CPU behavior and old unconfigured XML/artifacts remain unchanged. Explicit
`GpuMatVecMode.Serial` retains the baseline. Optionally enable trusted costs:

```csharp
var config = new VmRuntimeConfig
{
    GpuMatVecMode = GpuMatVecMode.Profile,
    GpuMatVecCostProfile =
        TierOneCostProfile.Deserialize(File.ReadAllText(profilePath)),
};
using var backend = SharpInference.Runtime.D3D12.D3D12VmBackendFactory.Create(config);
```

`GpuMatVecMode.Cooperative` explicitly selects the same implementation as
the new GPU default.
`MatVecOptimizationReport` exposes actual selected calls and reasons for
serial retention. Regenerate profiles after code/runtime/device changes.
The existing sample schema is reused, but MatVec profile selection is not
T1 fusion selection.

Reproduce with the [benchmark tool](../tools/SharpInference.VmBenchmark/README.md)
and its [comparison script](../tools/SharpInference.VmBenchmark/Compare-MatVec.ps1).
Raw JSON, logits and State files are retained in the session's
`files\matvec-performance` directory. The separately supplied
`matvec-performance-raw-measurements.zip` contains 19 verified parseable JSON
files: 16 model runs, the MatVec profile/protocol file pair, and the fresh T1
profile.

## Interpretation and remaining limits

This improves single-call hardware mapping without reducing graph nodes.
Large-model gains establish practical value beyond fusion-count proxies.
The CPU-authoritative managed entry bridge, other serial reductions and
reshape copies remain unchanged and can still limit end-to-end performance.
Their individual costs have not been attributed by this experiment.

No wave-specialized reduction, split-K, multirow tile, automatic tuning,
BatchedMatVec rewrite or MatVec/activation fusion is included. Additional
implementations require their own numerical validation and hardware/shape
measurements.

## Default validation with the original ten questions

The default change was measured separately using the original Chinese
question benchmark, not the eight-token protocol above. Normal Runtime and
direct GPU graph generation now select cooperative MatVec without a MatVec
cost profile. CPU defaults are unchanged. Existing XML/artifacts with no
MatVec execution configuration still compile as serial; explicit Serial
and Profile modes remain available. Execution failures do not trigger a
silent serial retry.

### Protocol

- Same hardware as above, Release, `DOTNET_TieredCompilation=0`.
- World tokenizer and prompt `User: {question}\n\nAssistant:`.
- Warm questions 1 and 9, generating four tokens each, excluded from results.
- Fresh session per measured question, exactly 16 greedy tokens, temperature
  zero and empty stop-token/string sets; EOS does not shorten the run.
- Two Prefill and two Inference workers, Prefill capacity 64, one active
  session at a time, matching the original benchmark.
- Eight sequential processes: V6 7B and V7 7.2B, serial/default, two rounds.
  Round zero runs serial then default; round one reverses that order.
  The default runs omit the MatVec option entirely.
- Identical freshly regenerated GPU T1 profile in both variants, with
  tiered compilation disabled; all 17 measured domains are eligible.
  The environment fingerprint now includes `DOTNET_TieredCompilation`.
- Each process measures 167 prompt tokens and 160 generated tokens.
  Prefill and Decode include the original generation path, including
  sampling/text decoding in Decode. Initialization, warmup, session
  construction, snapshots and validation are outside the stage timers.

The exact original Chinese questions are preserved verbatim in the
[question harness](../tools/SharpInference.VmBenchmark/QuestionBenchmark.cs).
Their topics, in order, are artificial intelligence, the blue sky,
recursion, photosynthesis, learning efficiency, database indexes,
Solar System planets, TCP versus UDP, Python addition, and exercise.

### Actual generation performance

Rates pool both rounds: 334 prompt tokens divided by total Prefill seconds,
320 generated tokens divided by total Decode seconds, and 320 generated
tokens divided by total Prefill plus Decode seconds for Combined. They are
not averages of per-question or per-process rates. Higher is better;
percentages below are throughput increases, not latency reductions.

| Model | Stage | Serial, token/s | Default cooperative, token/s | Throughput increase |
|---|---|---:|---:|---:|
| V6 7B FP16 | Prefill | 2.615 | 3.062 | 17.08% |
| V6 7B FP16 | Decode | 1.708 | 1.895 | 10.95% |
| V6 7B FP16 | Combined | 1.016 | 1.151 | 13.35% |
| V7 7.2B FP16 | Prefill | 4.804 | 7.225 | 50.41% |
| V7 7.2B FP16 | Decode | 2.456 | 2.988 | 21.64% |
| V7 7.2B FP16 | Combined | 1.601 | 2.087 | 30.31% |

Per-process results expose order variation:

| Model | Variant | Prefill r0/r1, token/s | Decode r0/r1, token/s | Combined r0/r1, token/s |
|---|---|---:|---:|---:|
| V6 7B | Serial | 2.560 / 2.674 | 1.693 / 1.723 | 1.001 / 1.030 |
| V6 7B | Default | 3.040 / 3.084 | 1.885 / 1.904 | 1.144 / 1.158 |
| V7 7.2B | Serial | 4.819 / 4.789 | 2.455 / 2.458 | 1.603 / 1.600 |
| V7 7.2B | Default | 7.191 / 7.260 | 2.987 / 2.988 | 2.084 / 2.090 |

Both stages improve in both run orders for both models. Default selection
is verified as 353/447 cooperative MatVec calls for V6/V7, versus zero
cooperative calls in serial runs. Both variants retain 644/674 selected T1
calls and the same operator graph counts.

These results must not be mixed with the earlier one-worker, capacity-eight
protocol. The historical original-main Decode rates of 7.749/9.034 token/s
are reference measurements, not a contemporaneous rerun. This change
improves the current VM but does not establish parity with that backend or
attribute the remaining gap to individual components.

### Answer and numerical agreement

All 80 measured answers have identical question text, prompt token IDs and
generated token IDs to the original-main historical records. Current
default and serial runs therefore also agree exactly on generated tokens.
The fixed 16-token outputs are intentionally short answer prefixes.

Every noninitial run compares full Prefill logits, final Decode logits and
every named final State tensor with that model's first serial run. This
checks 5,880 comparison records: 3,920 from default runs and 1,960 from
serial repeat runs. All pass the existing model gate
`abs(actual-reference) <= 0.0003 + abs(reference)*0.0003`, with finite
values and matching State names/shapes. Unlike the earlier eight-token
experiment, it does not snapshot every intermediate Decode logit vector.

| Model | Maximum absolute Prefill logit error | Maximum absolute final Decode logit error | Maximum absolute State error | Maximum scaled error relative to gate |
|---|---:|---:|---:|---:|
| V6 7B | 0.00048828125 | 0.0007324219 | 0.00024414062 | 0.09523176 |
| V7 7.2B | 0.000011444092 | 0.000009536743 | 0.000045776367 | 0.033340894 |

Reproduce using the [ten-question comparison script](../tools/SharpInference.VmBenchmark/Compare-Questions.ps1)
and the [benchmark instructions](../tools/SharpInference.VmBenchmark/README.md).
Raw question reports and snapshots are retained in the session's
`files\matvec-ten-questions` directory. The separately supplied
`matvec-ten-questions-raw-measurements.zip` contains all eight process
reports and their shared T1 profile, nine verified parseable JSON files;
large binary snapshots are retained separately.

Final default-change validation passed 596 directly related tests with
zero failures or skips. The full Release solution build completed with
zero warnings and errors.
