# FP32 Tier-1 implementation and measured comparison

## Scope

The implementation is on `agents/tier1-operator-implementation`, based on fixed
main `6270ff399e7eeb81fa8b07132c47eef191a6a3a5`. Main sources were not modified.
The same [benchmark harness](../tools/SharpInference.VmBenchmark/README.md) was
compiled against main and the implementation worktree. No legacy optimized
execution was migrated.

The implemented scope is the 13 canonical CPU/D3D12 FP32 pointwise candidates,
immutable target execution configurations, capability metadata, dependency
matching, offline cost gating, global candidate selection and physical
replanning. FP16-weight T0 matrix operations remain available; T1 inputs,
outputs and canonical intermediate arithmetic are FP32. FP16/mixed T1,
reduction/MatVec/State-update fusion and T2 are not implemented.

## Machine and protocol

- AMD Ryzen 9 3900X, 12 cores / 24 logical processors, approximately 64 GiB RAM.
- Actual hardware adapter: AMD Radeon RX 7900 XTX, driver 32.0.31041.1004.
  No Remote Display Adapter or WARP was used.
- .NET SDK 10.0.401, Release build.
- Tiny V6 3M / V7 834K use FP32 weights. V6 World 7B / V7 G1J 7.2B use
  their existing FP16 weights and FP32 activation/State contracts.
- One inference and one prefill instance; prefill capacity eight, group size
  64, native-half weights, weight views and local-storage reuse enabled.
- All models/backends/variants were loaded and disposed serially. CPU and GPU
  were not timed concurrently. System caches/clocks and other users' processes
  were not changed.
- `main`, branch `off`, and branch `on` are distinct groups. `on` receives an
  explicitly supplied offline profile; `off` receives none.

Offline profiling discovered 37 operation/shape domains per target from the
four real models. It numerically checked each T0/T1 pair, then timed 128 segment
invocations per VM entry, 64 warmup batches and nine alternating paired samples
of eight batches. Reported costs are wall-clock microseconds per scheduled
segment, with host overhead amortized; they are not GPU timestamp measurements.
CPU admitted 32/37 domains and GPU 37/37 under the conservative Q25/Q75 margin.
The remaining CPU domains stayed T0 rather than being forced into fusion.

There are two retained model experiments:

1. **Rotating exploratory matrix:** 72 process runs, three rounds with group
   order `main/off/on`, `off/on/main`, `on/main/off`; three samples of four
   tokens per process. This checked repeatability and numerical behavior but
   revealed insufficient CPU JIT warmup and first-Prefill-call noise.
   Its timing samples are retained, not presented as the final warmed result.
2. **Corrected warmed follow-up:** 24 process runs, seven samples of eight
   tokens `[1,2,3,2,4,5,1,3]`. Each process warms both Decode and Prefill for at
   least two cycles and two seconds before measuring either path. Tiny CPU
   required seven or eleven cycles. Reset happens outside timing; validation
   logit copies and State export happen afterwards. Load/compile and warmup
   durations are separate fields.

The corrected follow-up uses one serial `main/off/on` group order, not three
rotated rounds. Small percentage differences must not be treated as statistically
proven speedups. Cache/load variability and CPU tiered compilation are not
completely eliminated.

## Warmed latency results

Median milliseconds per token; lower is better. Delta is `on/main - 1` or
`on/off - 1`, not a throughput percentage. Prefill is an eight-token window,
not a parallel-sequence training or large-batch throughput measurement.

| Model | Backend | Decode main | Decode off | Decode on | On/main | On/off |
|---|---|---:|---:|---:|---:|---:|
| Tiny V6 FP32 | CPU | 4.1681 | 4.1098 | 2.8562 | -31.47% | -30.50% |
| Tiny V7 FP32 | CPU | 2.7923 | 2.7623 | 2.0768 | -25.62% | -24.82% |
| V6 7B FP16 weights | CPU | 834.4574 | 835.8828 | 832.1311 | -0.28% | -0.45% |
| V7 7.2B FP16 weights | CPU | 640.0558 | 653.2273 | 642.7870 | +0.43% | -1.60% |
| Tiny V6 FP32 | D3D12 | 118.3305 | 116.5154 | 116.7293 | -1.35% | +0.18% |
| Tiny V7 FP32 | D3D12 | 117.3470 | 112.9933 | 110.0836 | -6.19% | -2.58% |
| V6 7B FP16 weights | D3D12 | 322.6803 | 323.4000 | 321.8264 | -0.26% | -0.49% |
| V7 7.2B FP16 weights | D3D12 | 335.4325 | 338.5968 | 335.1182 | -0.09% | -1.03% |

| Model | Backend | Prefill main | Prefill off | Prefill on | On/main | On/off |
|---|---|---:|---:|---:|---:|---:|
| Tiny V6 FP32 | CPU | 22.6977 | 22.7444 | 10.2230 | -54.96% | -55.05% |
| Tiny V7 FP32 | CPU | 16.5101 | 15.7947 | 10.0810 | -38.94% | -36.17% |
| V6 7B FP16 weights | CPU | 1795.5451 | 1759.7125 | 1796.2326 | +0.04% | +2.08% |
| V7 7.2B FP16 weights | CPU | 733.1128 | 731.1557 | 689.9412 | -5.89% | -5.64% |
| Tiny V6 FP32 | D3D12 | 17.5531 | 17.4678 | 17.0571 | -2.83% | -2.35% |
| Tiny V7 FP32 | D3D12 | 17.0447 | 16.6313 | 16.1361 | -5.33% | -2.98% |
| V6 7B FP16 weights | D3D12 | 128.6958 | 128.3672 | 127.9850 | -0.55% | -0.30% |
| V7 7.2B FP16 weights | D3D12 | 138.2921 | 138.0466 | 136.8519 | -1.04% | -0.87% |

Representative Decode sample middle ranges (sorted sample indices 1..4 of
seven; descriptive, not confidence intervals):

- Tiny V6 CPU: main 3.9617..4.2485, on 2.7288..2.8926.
- Tiny V7 CPU: main 2.7180..2.8126, on 1.9050..2.0844.
- V6 7B CPU: main 832.4747..835.5842, on 831.2288..833.4775.
- V7 7.2B CPU: main 639.0041..641.6897, on 641.3447..653.6339.
- V6 7B GPU: main 321.8128..323.2088, on 320.9315..323.5600.
- V7 7.2B GPU: main 334.9917..337.4382, on 332.8333..336.1293.

The small-model CPU benefit survives adequate warmup and comparison with off.
The large-model Decode differences are generally small and/or overlapping:
this experiment does **not** establish a stable large-model Decode speedup.
V7 CPU Prefill has a larger observed benefit; V6 CPU Prefill does not.
The GPU V7 tiny comparison against off is more modest than its comparison
against main, demonstrating why the off control and raw samples matter.
No claim is made that all 13 candidates accelerate every shape or that reduced
dispatch count translates proportionally to end-to-end speedup.

## Actual selected plans and resources

These count forward invocations, not reusable definitions. For D3D12 the call
count is also dispatch count and one final-output barrier is emitted per
dispatch. All on preparations reported `ModelOptimal` **under the stated
additive measured-segment cost model**, not hardware-global optimality.

| Model/backend | Actual T1 calls | T0 baseline calls | Selected-plan calls | Local bytes before | Local bytes after |
|---|---:|---:|---:|---:|---:|
| Tiny V6 CPU | 244 | 1755 | 1427 | 34988 | 34988 |
| Tiny V6 GPU | 244 | 1755 | 1389 | 34988 | 34988 |
| Tiny V7 CPU | 254 | 1808 | 1397 | 57904 | 57648 |
| Tiny V7 GPU | 254 | 1808 | 1397 | 57904 | 57648 |
| V6 7B CPU | 612 | 10011 | 9367 | 5166892 | 5166892 |
| V6 7B GPU | 644 | 10011 | 9045 | 5166892 | 5166892 |
| V7 7.2B CPU | 642 | 4788 | 3859 | 4167852 | 4151468 |
| V7 7.2B GPU | 674 | 4788 | 3697 | 4167852 | 4151468 |

The allocator already reuses physical scratch, so deleting many logical
intermediates need not lower peak allocated bytes. T1 intermediate resources
are genuinely removed and allocation rerun; equal byte totals are reported
as equal, not as fabricated memory savings. The CPU and GPU subsets differ
because their measured profitable domains differ.

## Numerical and build acceptance

- 544 directly related tests passed, zero failures/skips. Coverage includes
  all 13 kernels on real CPU/D3D12, special values, signed zero, tails,
  FP32 intermediate-rounding/FMA-sensitive cases, artifact reload, immutable
  configurations, strict XML, cost rejection, weighted-interval optimality,
  budget labels, external consumers, joint contraction cycles, real tiny
  model Prefill/Decode/State and existing T0/resource contracts.
- Release solution build passed with zero warnings/errors.
- Every-token full logits and every final named State tensor were compared
  against the corresponding main backend, with finite-value checks and the
  unchanged gate `abs(actual-reference) <= 0.0003 + abs(reference)*0.0003`.
- All 96 model runs succeeded. The 72 exploratory runs produced 4,288
  nonempty comparison records; the warmed follow-up produced another 1,072.
  Every recorded maximum absolute error and scaled error was **zero**.
  This is exact finite-value agreement for these tokens, not proof for all
  possible inputs, NaN payloads or long-context sequences.

## Reproduction and retained raw evidence

See the tool README for building the same harness against fixed main and for
creating profiles explicitly. `Compare-TierOne.ps1` selects all four models
and both targets, disposes each process before loading the next and fails on
any numerical disagreement.

The session delivery retains both experiments' JSON timing samples, positive
and rejected offline cost domains, environment fingerprints, selection
diagnostics, per-token logits and complete named State snapshots. The compact
raw-measurement archive contains the JSON samples/profiles; the approximately
1.73 GB binary snapshots remain alongside them rather than being silently
discarded or included in the compact archive.

The appropriate conclusion is conditional: T1 provides measurable execution/
dataflow benefit for some workloads, while the optimizer must keep a T0
baseline, reject unsupported or uncertain evidence, and distinguish model
optimality from observed hardware performance.
