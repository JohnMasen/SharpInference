# Queued VM execution

## Execution contract

Sessions own persistent State and operation serialization, never VM instances.
Each token or prefill batch submits an immutable input packet, State ownership
and a cancellation source to one bounded FIFO queue. The queue owns completion
through a Task completion source; only its Worker acquires an execution lease.
Default runtime sizing remains two Prefill and two Inference instances.
Queue saturation is an explicit error, not an unbounded waiting producer.

A generation scope reserves the Session, not a Worker. Its asynchronous token
steps still enter the Inference queue. Synchronous compatibility APIs wait for
the same queued execution, rather than bypassing it. Text generation awaits
the asynchronous capability when available.

Before each task, the Worker binds compatible State and overwrites the entry's
input parameters. Persistent State is not zeroed by this execution-context
reset: Prefill State must survive into Inference. Explicit reset/restore remains
the Session manager's responsibility. Logits returned by a completed request
are independent of the next request's output buffer.

## Fixed GPU windows and coherence

Each GPU instance has fixed execution resources and lazily records each selected
compute entry once. Immutable globals share the model allocation; Local scratch
belongs to the Worker. Session resources use private execution windows, with
GPU-to-GPU load/commit copies at task boundaries. Switching Sessions does not
rebind or re-record the compute command list.

Globals upload once. Initial Local scratch is initialized once per used instance.
New or host-modified State uploads before its next GPU task; subsequent State
steps remain device-resident. CPU shadows become current only on requested reads,
including snapshot/export/fork, while host writes invalidate the GPU copy.
Partial host writes first preserve untouched bytes through coherence.
Required token inputs and logits still cross the host/device boundary.

The pool serializes GPU work. Four instances are a scheduling/resource contract,
not four-way GPU concurrency. State is copied at every task boundary in this
first implementation; there is no Session-affinity cache or skipped commit.
Kernels, XML format, mathematical operators and cooperative MatVec are unchanged.
Existing compiled graph layouts remain supported by the new execution path.
Legacy managed/GPU-only APIs remain separate; a single executor cannot switch
from queued windows back to managed execution.

## Cancellation and lifetime

Requests carry a `CancellationTokenSource`; public token-based APIs create linked
sources with operation and Session lifetime cancellation. Generation steps share
the generation cancellation signal. Prefill checks cancellation between selected
batches, and Inference checks it for every token.

A canceled pending request does not execute. Cancellation during submitted GPU
work cannot abort the command list safely: the Worker waits for actual completion
before releasing reservations or completing a canceled result. Partially executed
or failed State is conservatively invalidated, requiring explicit reset/restore.
Generation disposal and Session shutdown wait for active work; closing a Session
cancels its outstanding operations. Cancellation does not mean State rollback.

## Validation and comparison

Targeted regression coverage includes pending/running cancellation, batch
cancellation, logical scopes without Worker reservation, independent outputs,
GPU window reuse across Sessions, CPU State edits, reset/import/export and
Prefill-to-Inference continuity. Existing VM, operator, publication, resource
ownership, runtime and generator tests are also exercised.

The reproducible [comparison script](../tools/SharpInference.VmBenchmark/Compare-QueuedTasks.ps1)
uses the unmodified main worktree and this worktree's separately built harness.
GPU uses the same original ten questions, tokenizer/prompt, two warmups,
16 greedy tokens per question, 2+2 instances, capacity 64, and fresh
worktree-specific T1 profiles. CPU uses seven eight-token raw samples, 1+1
instances, capacity eight, and no T1 profile. Two rounds reverse variant order.
Tiered compilation is disabled and all large-model processes run sequentially.

GPU numerical checks compare full prefill/final logits and final State with main
round zero, plus every generated answer token ID exactly. CPU checks compare
every raw-token logit and final State. The numerical gate is
`abs(error) <= 0.0003 + 0.0003 * abs(reference)`.
The task counters include warmup, validation and explicit control reads.
`InitialStateUploadBytes` counts host-initialized or host-modified State uploads,
not steady-state State movement. `StateLoadBytes`/`StateCommitBytes` describe
GPU-to-GPU copies. `CpuReadbackBytes` includes logits and requested State snapshots.
Preparation/compute/commit milliseconds are host wall-clock intervals including
submission/fence waits; first-entry compute also includes lazy recording.

## Measured results

Baseline: main `85a7f01fb7ddc87979c3f57c634756276bce0ef1`.
Candidate: `agents/queue-vm-task-execution`, branched directly from that main.
Hardware: Ryzen 9 3900X, Radeon RX 7900 XTX, driver 32.0.31041.1004,
.NET 10.0.401 SDK. Models are the World 7B V6 FP16 and G1J 7.2B V7 FP16
files used in the preceding MatVec comparison.

Both worktrees generated fresh 17-domain GPU T1 profiles. V6 selected 644 T1
candidates and 353 cooperative MatVec calls, with 9,045 forward nodes; V7
selected 674 and 447 respectively, with 3,697 nodes. These graph counts match
across variants: the speedup is from the execution/transfer path, not a changed
operator graph or another MatVec implementation.

GPU throughput pools token counts and elapsed time across two reversed-order
rounds, rather than averaging question rates. All rates below are tokens/second.
Combined means output tokens divided by Prefill plus Decode time, excluding
model loading/compilation, Session creation, tokenization and validation.

| GPU model | Stage | main | Queued VM | Speedup |
| --- | --- | ---: | ---: | ---: |
| V6 7B | Prefill | 3.081950 | 22.407371 | 7.27x |
| V6 7B | Decode | 1.877470 | 20.763531 | 11.06x |
| V6 7B | Combined | 1.147714 | 10.554979 | 9.20x |
| V7 7.2B | Prefill | 7.306060 | 25.812008 | 3.53x |
| V7 7.2B | Decode | 2.964401 | 23.251797 | 7.84x |
| V7 7.2B | Combined | 2.082478 | 11.984080 | 5.75x |

Round-level GPU rates:

| Model | Variant | Round | Prefill | Decode | Combined |
| --- | --- | ---: | ---: | ---: | ---: |
| V6 | main | 0 | 3.091 | 1.879 | 1.150 |
| V6 | queued | 0 | 22.339 | 20.651 | 10.510 |
| V6 | queued | 1 | 22.476 | 20.877 | 10.600 |
| V6 | main | 1 | 3.073 | 1.876 | 1.146 |
| V7 | main | 0 | 7.425 | 2.973 | 2.097 |
| V7 | queued | 0 | 25.780 | 23.303 | 11.990 |
| V7 | queued | 1 | 25.844 | 23.201 | 11.978 |
| V7 | main | 1 | 7.191 | 2.956 | 2.069 |

CPU rates use the median of all 14 per-token latency samples across two rounds;
for the even sample count, the middle pair is averaged before taking its reciprocal.
CPU T1 remained off, with matching forward call counts of 10,011 (V6) and
4,788 (V7), and unchanged workspace sizes.

| CPU model | Stage | main | Queued VM | Throughput change |
| --- | --- | ---: | ---: | ---: |
| V6 7B | Prefill | 0.372022 | 0.371179 | -0.23% |
| V6 7B | Decode | 1.077585 | 1.065403 | -1.13% |
| V7 7.2B | Prefill | 1.217935 | 1.212146 | -0.48% |
| V7 7.2B | Decode | 1.490293 | 1.478397 | -0.80% |

CPU arithmetic is unchanged and no CPU acceleration is claimed. The measured
small negative differences are reported, not discarded as noise; two rounds
cannot isolate scheduling overhead from environmental drift.

Across eight GPU processes, all 80 answers matched main round zero exactly.
The 5,880 GPU numerical comparison groups and 582 CPU groups all passed, with
maximum absolute error zero. GPU compares full Prefill/final logits and final
State; CPU additionally captures all eight raw-token logits.

### Residency and recording evidence

Every candidate GPU run completed 195 queued tasks, across 12 fresh Sessions
(two warmups plus ten questions), but recorded only 12 compute entries across
the four instances. No Session-switch recording growth occurred.

Both models have 34,603,008 bytes of registered State. Each run recorded:

- State host uploads: 415,236,096 bytes, exactly one initial upload per Session.
- State GPU load and commit: 6,747,586,560 bytes each, one State copy per task.
- CPU readback: 397,148,160 bytes, exactly ten final-State snapshots plus
  195 full 65,536-element FP32 logits arrays. There is no per-token State readback.
- Token/input uploads: 49,920 bytes.
- One-time Local initialization: 20,667,440 bytes for V6 and 16,605,744 for V7.

These counters cover warmup and validation as well as timed execution. They
verify residency and bounded recording; they are not a pure measured-stage
GPU-time attribution.

### Validation outcome and remaining boundaries

The expanded regression run passed 631 tests with no skips or failures.
After the final diagnostic-field changes, 18 queue/coherence/generator tests
passed again. The full Release solution build had zero warnings and errors.

The original Session/queue/Worker boundary is restored. Remaining optimization
opportunities are explicit: serialized GPU submissions, per-task State device
copies, and required token/logit transfers. Those do not prevent the verified
execution contract, and this change does not claim multi-request GPU throughput
scaling or immediate interruption inside a submitted batch.
