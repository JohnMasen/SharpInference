# Phi-4 automatic UAV barrier experiment

## Decision

The slot-level automatic barrier experiment did not demonstrate a stable
runtime improvement on the validation GPU. Its implementation is not included
in main; only this report is retained. Production continues to use explicit
`VmBarrier` nodes and the existing pending-write validation.

The experimental snapshot is commit `852bc73` (`Plan and batch D3D12 UAV
barriers`), based on foundation commit `6796859`. The snapshot was validated
in an independent worktree, not on the subsequently merged main tree.

## Scope

The experiment flattened orchestration calls and tracked pending writes by
logical slot. Instead of eagerly executing each explicit barrier, it placed
one combined barrier immediately before the first RAW or WAW consumer.
Remaining writes were conservatively synchronized at the end of each entry.

Command recording additionally tracked physical D3D12 resources to account
for pooled aliases. Overlapping writable parameters within one dispatch
remained invalid. Multiple resource-specific UAV barriers at one hazard
point were submitted through a single D3D12 `ResourceBarrier` call.

This was not byte-range-aware analysis or cross-entry lifetime analysis.
Dispatch ordering, shader algorithms, model weights, and entry boundaries
were unchanged.

## Machine and protocol

- AMD Radeon RX 7900 XTX, driver `32.0.31041.1004`.
- Windows, .NET SDK `10.0.401`, Release builds.
- Phi-4 Multimodal Golden fixtures: Audio 351 frames (bucket 352), Vision
  seven crops and 1,841 projected image tokens.
- Foundation and experimental builds used the same model and inputs.
- GPU workloads were executed serially, without concurrent benchmark jobs.
- After a reboot, the initial sustained workload completed ten Audio and ten
  Vision encodes using reusable components without a GPU fault. Earlier
  attempts had been stopped following a reported graphics fault.

Two measurement methods were used:

1. Component wall clock: create and compile each component once, warm up,
   then run ten encodes with final output readback. Model loading and shader
   compilation are outside the timed region; uploads, CPU scheduling, and
   readback are included. This exploratory measurement preceded the final
   batched D3D12 barrier implementation and is not its final performance result.
2. GPU timestamps: initialize resident inputs, then use the existing
   `D3D12VmExecutor.MeasureGpuMicroseconds` API. Each sample measures three
   executions per entry, including its scheduled barriers, but excluding
   upload, readback, component creation, and shader compilation. Vision
   numbers sum the entry timings; they are not end-to-end request latency.

The final comparison has three samples per variant, with baseline measured
before the experimental variant rather than alternating paired samples.
GPU clocks, thermals, caches, and unrelated system activity were not
controlled. Small percentage differences are not statistically proven.

## Scheduled barrier counts

Counts cover the Vision audio projector entry and all Vision encoder/projector
entries respectively. Resource counts are logical scheduled barrier targets,
not a capture of all driver operations or physical-alias compensation.

| Component | Baseline commands | Experiment commands | Command reduction | Baseline resource targets | Experiment resource targets |
|---|---:|---:|---:|---:|---:|
| Audio | 780 | 732 | 6.15% | 780 | 780 |
| Vision | 2,373 | 2,009 | 15.34% | 2,373 | 2,373 |

Fewer scheduled commands did not remove resource-specific synchronization.
The final implementation batched those targets into fewer D3D12 API calls.

## Final GPU timestamp results

Milliseconds per component; lower is better. Delta is
`experiment / baseline - 1`, so a negative delta indicates lower latency.

| Component | Baseline mean | Experiment mean | Mean delta | Baseline median | Experiment median | Median delta |
|---|---:|---:|---:|---:|---:|---:|
| Audio | 38.2588 | 40.0942 | +4.80% | 38.2448 | 38.1529 | -0.24% |
| Vision | 1,822.9506 | 1,823.4835 | +0.03% | 1,821.2473 | 1,825.2385 | +0.22% |

All measured samples are retained below:

| Sample | Baseline Audio ms | Experiment Audio ms | Baseline Vision ms | Experiment Vision ms |
|---|---:|---:|---:|---:|
| 1 | 39.166280 | 44.370053 | 1,818.014893 | 1,816.959440 |
| 2 | 37.365213 | 37.759707 | 1,821.247320 | 1,825.238520 |
| 3 | 38.244800 | 38.152920 | 1,829.589733 | 1,828.252480 |

The first experimental Audio sample was slower, but the cause was not
measured. It is included in the mean, not discarded as a confirmed outlier.
The mixed directions of the mean and median do not establish an Audio
speedup. Vision differences are similarly too small to establish a benefit.

## Correctness validation of the experimental snapshot

- Schedule tests: 3/3 passed (RAW, WAW, branch merging, terminal writes).
- D3D12 VM tests: 58/58 passed.
- Tier Zero and GPU MatVec tests: 135/135 passed.
- Phi-4 Audio projector, Vision projector, and full Image+Audio Golden
  tests passed after batched barrier recording was implemented.
- Full Release solution build passed with zero warnings and zero errors.
- `git diff --check` passed.

Correctness validation does not imply a performance improvement or exhaustive
coverage of every possible physical alias arrangement.

## Interpretation

Slot-level sinking and merging reduced synchronization command groups, but
retained the same resource targets and showed no stable runtime benefit.
Driver coalescing or shader execution dominating barrier cost are possible
explanations, not conclusions established by this experiment.

The experiment is therefore not promoted to production. A future investigation
would need paired, adequately warmed measurements and a separately validated
way to remove unnecessary synchronization, such as byte-range-aware hazards
or non-escaping scratch lifetime analysis. Those approaches were not tested.

For the retained runtime architecture, see
[Phi-4 multimodal inference](./phi4-multimodal-inference.md).
