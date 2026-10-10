# Offline T1 profiling and model comparison

This Windows tool measures the actual compiled CPU VM or hardware D3D12 VM.
It never installs profiles implicitly and is not invoked by normal model load.
Use a Release build and keep unrelated workloads consistent between groups.

```powershell
dotnet build tools\SharpInference.VmBenchmark\SharpInference.VmBenchmark.csproj -c Release
$tool = 'tools\SharpInference.VmBenchmark\bin\Release\net10.0-windows10.0.19041.0\SharpInference.VmBenchmark.dll'
$models = 'F:\models\tiny-rwkv-6v0-3m-FP32.bin|F:\models\tiny-rwkv-7v0-834K-FP32.bin'
dotnet $tool profile --models $models --target cpu --output C:\results\cpu-profile.json
dotnet $tool profile --models $models --target gpu --output C:\results\gpu-profile.json
dotnet $tool run --model F:\models\tiny-rwkv-7v0-834K-FP32.bin --target cpu `
  --variant off --output C:\results\off
dotnet $tool run --model F:\models\tiny-rwkv-7v0-834K-FP32.bin --target cpu `
  --variant on --profile C:\results\cpu-profile.json --reference C:\results\off `
  --output C:\results\on
```

`profile` discovers actual candidate names/shapes from the listed models and
measures the canonical T0 segment versus its T1 implementation. It currently
measures distinct-input alias classes only; unmeasured repeated-input classes
remain T0. Inputs are finite deterministic FP32 values. Numerical validation
precedes timing. Each VM entry repeats its segment 128 times, amortizing host
submission/context cost; 64 batches warm up each implementation, then nine
alternating paired samples measure eight batches each. Costs are wall-clock
microseconds per segment, including scheduled calls/dispatches/barriers, not
GPU timestamp-query measurements. A final-output barrier is retained in both
programs. This does not price every whole-model interaction or certify cold
cache performance.

Profiles record positive finite raw samples, exact dimensions, implementation
fingerprints, alias classes, thread-group size and storage-reuse policy.
Runtime factories recompute CPU/device/driver and code/runtime fingerprints.
Foreign/stale/expired profiles are reported and retain T0. The default profile
age limit is 30 days; direct optimizer settings can explicitly change it.
Regenerate profiles after compiler, runtime or kernel changes.

`run` uses one inference and one prefill instance, prefill capacity eight,
at least two warmup cycles and two seconds of warmup, seven samples and eight
raw tokens `[1,2,3,2,4,5,1,3]` by default. Each warmup cycle executes both Decode
and Prefill so neither measured path includes its first invocation.
`--samples`, `--tokens` and `--warmups` override these positive counts.
`--minimum-warmup-ms` overrides the positive minimum warmup duration. Decode
and Prefill are measured separately after reset, excluding load/compile and
validation; setup/warmup time is reported separately. Timing does not include
logit-array copies for validation. Every checked logit at every token and
every final State value must be finite and satisfy the existing model gate
`abs(actual-reference) <= 0.0003 + abs(reference)*0.0003`.

`--export-logical <file.xml>` exports the actual model-owned logical graph as
indented UTF-8 XML. `--logical-xml <file.xml>` loads that graph with the supplied
model rather than regenerating it, so `--reference` can verify serialization/load
and compiled-execution equivalence. This is logical XML, not VmProgram XML.

Outputs are raw samples/selection/resource metadata in `.json`, every-token
logits in `.logits.bin`, and a named GGUF State snapshot in `.state.bin`.
`--reference` compares with another output prefix and fails explicitly on
incompatible shapes or numerical disagreement. `TierOneCalls` counts actual
forward invocations, not just reusable definitions. A profile-supplied run
with zero T1 calls prints a warning: it is not evidence of T1 acceleration.

For the historical pre-T1 main comparison, build the **same harness** against main sources
without editing them:

```powershell
dotnet build tools\SharpInference.VmBenchmark\SharpInference.VmBenchmark.csproj -c Release `
  -p:SharpInferenceSourceRoot=G:\GitRoot\SharpInference `
  -p:TierOneApi=false -p:OutputPath='bin\MainBaseline\'
dotnet tools\SharpInference.VmBenchmark\bin\MainBaseline\SharpInference.VmBenchmark.dll `
  run --model F:\models\tiny-rwkv-7v0-834K-FP32.bin --target cpu `
  --variant main --output C:\results\main
```

Rebuild normally afterwards to restore the tool's project-reference assets.
Compare fixed main, branch T1-off and branch T1-on separately. Rotate group
order across rounds, retain all raw samples and report noise as well as
medians. Load/dispose large models serially; do not keep multiple GPU weight
copies resident. Do not equate the cost-model optimum, reduced call count or
smaller workspace with measured end-to-end speedup.

For queued-task changes, main already contains T1. Build the benchmark project
normally in both worktrees, then run:

```powershell
.\tools\SharpInference.VmBenchmark\Compare-QueuedTasks.ps1 `
  -MainDirectory G:\GitRoot\SharpInference -ModelDirectory F:\RWKV\RWKVModels `
  -OutputDirectory C:\results\queued-vm
```

This uses each worktree's own harness, regenerates its GPU T1 profile, and
alternates main/queued order over two rounds. GPU uses the original ten-question
protocol, 2+2 workers, capacity 64 and 16 greedy output tokens. CPU uses seven
eight-token raw samples, 1+1 workers and capacity eight, with T1 disabled in both
groups. All subsequent runs numerically compare against main round zero.
Do not compare CPU raw-token rates directly with GPU question rates.
Queued GPU JSON includes aggregate task counters covering warmup, validation,
and measured execution; wall-clock compute time is not a GPU timestamp.
The script writes `summary.json`; `-SummarizeOnly` regenerates the summary from
completed raw reports without rerunning models. GPU pools elapsed time and token
counts; CPU uses the pooled median of the 14 per-token samples.
See [queued VM execution and measured results](../../doc/queued-vm-execution.md).

The [measured implementation report](../../doc/tier-one-performance.md)
includes all four models on both backends, the initial rotating experiment
and the corrected fully warmed follow-up.

## Cooperative GPU MatVec

```powershell
dotnet $tool matvec-profile --models $models --target gpu --output C:\results\matvec-profile.json
dotnet $tool run --model F:\models\tiny-rwkv-7v0-834K-FP32.bin --target gpu `
  --matvec serial --variant serial --output C:\results\serial
dotnet $tool run --model F:\models\tiny-rwkv-7v0-834K-FP32.bin --target gpu `
  --matvec profile --matvec-profile C:\results\matvec-profile.json `
  --variant measured --reference C:\results\serial --output C:\results\measured
```

`--matvec cooperative` forces the 64-thread row-cooperative variant for
controlled numerical/performance experiments. `--matvec profile` only selects
it in exact trusted domains with conservative positive savings; otherwise
serial is retained with diagnostics. Default runs use cooperative GPU MatVec
without a cost profile; CPU defaults are unchanged. CPU runs
cannot enable GPU MatVec selection. `--profile` still independently enables
T1; regenerate both profiles after code changes.

`matvec-profile` discovers actual MatVec shapes and storage signatures after
native-half weight-view lowering. It uploads finite deterministic synthetic
weights/vectors, checks every output against serial with the existing operator
fixture tolerance, warms four batches, and alternates nine paired timestamp
samples. `--repetitions` defaults to eight invocations per batch. Timestamp
intervals include resident entry dispatch/barriers and command-list execution,
not CPU submission, input upload or output readback. Repeated use warms caches;
these are not cold-cache or whole-model costs.

The output is an explicit reusable cost profile plus
`.measurements.json` containing hardware/protocol and numerical checks.
Profiles record the fixed cooperative group size and a 64-thread serial
reference. Shape keys include matrix/input/output storage types. Only the
distinct-input, local-reuse domain is measured. Models, and then each
serial/cooperative pair of synthetic matrices, are processed sequentially.

`Compare-MatVec.ps1` compares the four reference models serially with the same
fresh T1 profile in both variants. Place `gpu-matvec-profile.json` and
`gpu-t1-profile.json` in its output directory first. Two rounds reverse
`serial/profile` order; defaults are seven samples, eight tokens and two
warmup cycles, retaining the harness's two-second minimum. Every noninitial
run compares logits/State with its model's first serial run. The script
fails if T1 or cooperative MatVec unexpectedly selects no calls.

Timestamp profiling records each repetition's dispatch/barrier sequence
explicitly in a single command list, rather than submitting duplicate
references to a cached command list. A GPU atomic-counter regression checks
the exact number of executions; elapsed time alone is not an execution-count
check.

The [cooperative MatVec report](../../doc/gpu-matvec-performance.md) contains
resident kernel timings, both model rounds, exact selected-call counts,
numerical deviations and the remaining limits.

## Original ten-question protocol

`questions` reproduces the original ten Chinese questions and exact
`User: {question}\n\nAssistant:` World-tokenizer prompts. It checks the
167-token total, excludes warmups on questions 1 and 9 (four generated tokens
each), then generates exactly 16 tokens for each question at temperature zero
without EOS/stops. A fresh session is used per question; the Runtime retains
its two Prefill/two Inference workers and capacity 64. Stage rates are total
tokens divided by total stage time, not averages of per-question rates.

```powershell
$env:DOTNET_TieredCompilation = '0'
dotnet $tool questions --model F:\models\RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin `
  --target gpu --matvec serial --variant serial --profile C:\results\gpu-t1-profile.json `
  --output C:\results\serial-questions
dotnet $tool questions --model F:\models\RWKV-x060-World-7B-v3-20241112-ctx4096-FP16.bin `
  --target gpu --variant default --profile C:\results\gpu-t1-profile.json `
  --reference C:\results\serial-questions --output C:\results\default-questions
```

No `--matvec` option is needed for the new default. Regenerate the T1 profile
under the same tiered-compilation setting before comparison. This command
requires `DOTNET_TieredCompilation=0`, as in the historical test. Prompt and
output token IDs, question text, timings and answers are preserved in JSON.
Prefill/final Decode logits and final named State snapshots are saved outside
the timed stages; `--reference` checks every question's token IDs and numeric
snapshots using the existing model gate. Generation timings include greedy
sampling/text decoding as in the historical harness.

`Compare-Questions.ps1` runs V6/V7 7B GPU models sequentially, with two rounds
reversing serial/default order. It passes `--matvec serial` only to the
baseline; the default group exercises the actual unspecified Runtime
configuration. Both groups use the same freshly generated
`gpu-t1-profile.json` from the output directory. No timing runs overlap.
