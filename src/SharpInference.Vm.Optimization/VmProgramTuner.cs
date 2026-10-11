using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpInference.Vm;

namespace SharpInference.Vm.Optimization;

/// <summary>Identifies the token-processing workload measured during VM tuning.</summary>
public enum VmTuningWorkload
{
    /// <summary>Measures prefill of the supplied token batch.</summary>
    Prefill,
    /// <summary>Measures sequential single-token inference.</summary>
    Inference
}

/// <summary>Configures warmup, measurement counts, and numerical acceptance tolerances for VM tuning.</summary>
/// <param name="WarmupRuns">The number of warmup executions before collecting steady-state samples.</param>
/// <param name="MeasuredRuns">The number of validated execution samples used to compute the median.</param>
/// <param name="AbsoluteTolerance">The maximum absolute logits error accepted during validation.</param>
/// <param name="RelativeTolerance">The maximum relative logits error accepted during validation.</param>
/// <param name="Workload">The prefill or inference workload used for measurements.</param>
public sealed record VmTuningOptions(
    int WarmupRuns = 2, int MeasuredRuns = 5,
    float AbsoluteTolerance = 0.0003f, float RelativeTolerance = 0.0003f,
    VmTuningWorkload Workload = VmTuningWorkload.Prefill);

/// <summary>Identifies the hardware, driver, backend, and compiler used for a tuning run.</summary>
/// <param name="Hardware">The hardware identity of the tuning target.</param>
/// <param name="Driver">The driver identity of the tuning target.</param>
/// <param name="Backend">The execution backend identity.</param>
/// <param name="Compiler">The compiler identity used for candidate preparation.</param>
public sealed record VmTuningTarget(string Hardware, string Driver, string Backend, string Compiler);

/// <summary>Contains expected logits and state values used to validate tuning candidates.</summary>
/// <param name="Logits">The expected output logits used for numerical validation.</param>
/// <param name="State">The expected state payloads keyed by canonical state-entry name.</param>
public sealed record VmTuningReference(float[] Logits, IReadOnlyDictionary<string, byte[]> State);
/// <summary>Associates a unique candidate name with a VM program to compile and measure.</summary>
/// <param name="Name">The unique name identifying this candidate in measurements and winner selection.</param>
/// <param name="Program">The VM artifact compiled and measured for this candidate.</param>
public sealed record VmTuningCandidate(string Name, VmProgram Program);
/// <summary>Reports one candidate's acceptance, phase timings, and failure diagnostic.</summary>
/// <param name="Name">The candidate name associated with these measurements.</param>
/// <param name="Accepted">Whether execution, logits, and state validation succeeded for the candidate.</param>
/// <param name="CompilationMilliseconds">The elapsed candidate compilation time in milliseconds.</param>
/// <param name="InitializationMilliseconds">The elapsed engine and session initialization time in milliseconds.</param>
/// <param name="FirstExecutionMilliseconds">The first execution latency in milliseconds.</param>
/// <param name="WarmupMilliseconds">The total warmup and validation time in milliseconds.</param>
/// <param name="MedianMilliseconds">The median validated steady-state execution latency in milliseconds.</param>
/// <param name="Diagnostic">The failed phase and error, or null for an accepted candidate.</param>
public sealed record VmTuningMeasurement(string Name, bool Accepted, double CompilationMilliseconds,
    double InitializationMilliseconds, double FirstExecutionMilliseconds, double WarmupMilliseconds, double MedianMilliseconds,
    string? Diagnostic);
/// <summary>Records the fastest accepted candidate, execution environment, and all candidate measurements.</summary>
/// <param name="Winner">The name of the fastest accepted candidate, or null if all candidates failed.</param>
/// <param name="Target">The explicit hardware, driver, backend, and compiler identity for the run.</param>
/// <param name="Options">The tuning options used for the run.</param>
/// <param name="OperatingSystem">The operating system description recorded for the run.</param>
/// <param name="Runtime">The .NET runtime description recorded for the run.</param>
/// <param name="Candidates">The acceptance results and phase measurements for all candidates.</param>
public sealed record VmTuningResult(string? Winner, VmTuningTarget Target, VmTuningOptions Options,
    string OperatingSystem, string Runtime, IReadOnlyList<VmTuningMeasurement> Candidates);

/// <summary>Compiles, validates, and benchmarks candidate programs against reference logits and state.</summary>
public static class VmProgramTuner
{
    /// <summary>Compiles and benchmarks candidates, validating logits and state before selecting the fastest accepted median.</summary>
    /// <param name="candidates">The uniquely named VM programs to compile, validate, and measure.</param>
    /// <param name="inputSlot">The Int32 slot receiving input token identifiers.</param>
    /// <param name="outputSlot">The Float32 slot containing output logits.</param>
    /// <param name="tokens">The nonempty vocabulary token sequence to process.</param>
    /// <param name="initialState">The complete state snapshot restored before each workload execution.</param>
    /// <param name="reference">The expected logits and exact state payloads used to accept candidates.</param>
    /// <param name="compile">The asynchronous compiler returning a factory for newly owned executables.</param>
    /// <param name="initializeGlobal">The callback initializing newly allocated global slot storage.</param>
    /// <param name="target">The explicit hardware, driver, backend, and compiler identity for the tuning run.</param>
    /// <param name="options">The workload, run counts, and numerical tolerances; null uses default tuning settings.</param>
    /// <param name="cancellation">The token used to cancel the operation.</param>
    /// <param name="allocate">The optional slot storage allocator; null uses managed byte arrays.</param>
    public static async ValueTask<VmTuningResult> TuneAsync(
        IReadOnlyList<VmTuningCandidate> candidates, string inputSlot, string outputSlot,
        ReadOnlyMemory<int> tokens, IReadOnlyDictionary<string, byte[]> initialState,
        VmTuningReference reference, Func<VmProgram, CancellationToken, ValueTask<Func<IVmExecutable>>> compile,
        Action<VmSlot, IVmStorage> initializeGlobal, VmTuningTarget target, VmTuningOptions? options = null,
        CancellationToken cancellation = default, Func<VmSlot, IVmStorage>? allocate = null)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(initialState);
        ArgumentNullException.ThrowIfNull(reference);
        ArgumentNullException.ThrowIfNull(compile);
        ArgumentNullException.ThrowIfNull(initializeGlobal);
        ArgumentNullException.ThrowIfNull(target);
        if (new[] { target.Hardware, target.Driver, target.Backend, target.Compiler }.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Tuning requires an explicit hardware/driver/backend/compiler fingerprint.", nameof(target));
        options ??= new();
        if (candidates.Count == 0 || candidates.Select(candidate => candidate.Name).Distinct(StringComparer.Ordinal).Count() != candidates.Count)
            throw new ArgumentException("Candidate names must be nonempty and unique.", nameof(candidates));
        if (candidates.Any(candidate => string.IsNullOrWhiteSpace(candidate.Name)) || tokens.IsEmpty)
            throw new ArgumentException("Candidates and input tokens must be nonempty.");
        if (options.WarmupRuns < 0 || options.MeasuredRuns < 1 ||
            !Enum.IsDefined(options.Workload) ||
            !float.IsFinite(options.AbsoluteTolerance) || options.AbsoluteTolerance < 0 ||
            !float.IsFinite(options.RelativeTolerance) || options.RelativeTolerance < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
        var tokenSnapshot = tokens.ToArray();
        var initialSnapshot = initialState.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray(), StringComparer.Ordinal);
        reference = new(reference.Logits.ToArray(),
            reference.State.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray(), StringComparer.Ordinal));
        var measurements = new List<VmTuningMeasurement>();
        foreach (var candidate in candidates)
        {
            cancellation.ThrowIfCancellationRequested();
            var timer = Stopwatch.StartNew();
            double compilation = 0, initialization = 0, firstExecution = 0, warmup = 0;
            var phase = "compilation";
            VmTuningMeasurement measurement;
            try
            {
                var factory = await compile(candidate.Program, cancellation).ConfigureAwait(false);
                compilation = timer.Elapsed.TotalMilliseconds;
                phase = "initialization";
                timer.Restart();
                await using var engine = await VmInferenceEngine.CreateAsync(candidate.Program, inputSlot, outputSlot,
                    initializeGlobal, factory, new(1, 1, 1, 1, Math.Max(1, tokenSnapshot.Length)), allocate).ConfigureAwait(false);
                await using var session = engine.CreateSession();
                initialization = timer.Elapsed.TotalMilliseconds;
                phase = "first execution";
                await session.WriteStateAsync(initialSnapshot, cancellation).ConfigureAwait(false);
                var first = await ExecuteAsync(session).ConfigureAwait(false);
                firstExecution = first.Milliseconds;
                await ValidateAsync(session, first.Logits).ConfigureAwait(false);
                phase = "warmup";
                timer.Restart();
                for (var run = 0; run < options.WarmupRuns; run++)
                    await RunAndValidateAsync(session).ConfigureAwait(false);
                warmup = timer.Elapsed.TotalMilliseconds;
                phase = "measurement";
                var samples = new double[options.MeasuredRuns];
                for (var run = 0; run < samples.Length; run++)
                {
                    await session.WriteStateAsync(initialSnapshot, cancellation).ConfigureAwait(false);
                    var result = await ExecuteAsync(session).ConfigureAwait(false);
                    samples[run] = result.Milliseconds;
                    await ValidateAsync(session, result.Logits).ConfigureAwait(false);
                }
                Array.Sort(samples);
                var median = samples.Length % 2 == 1 ? samples[samples.Length / 2] :
                    (samples[samples.Length / 2 - 1] + samples[samples.Length / 2]) / 2;
                phase = "cleanup";
                measurement = new(candidate.Name, true, compilation, initialization, firstExecution, warmup, median, null);
            }
            catch (OperationCanceledException) when (cancellation.IsCancellationRequested) { throw; }
            catch (Exception error)
            {
                if (phase == "compilation") compilation = timer.Elapsed.TotalMilliseconds;
                if (phase == "initialization") initialization = timer.Elapsed.TotalMilliseconds;
                measurement = new(candidate.Name, false, compilation, initialization, firstExecution, warmup, 0, $"{phase}: {error}");
            }
            measurements.Add(measurement);
        }
        var winner = measurements.Where(measurement => measurement.Accepted)
            .OrderBy(measurement => measurement.MedianMilliseconds).FirstOrDefault()?.Name;
        return new(winner, target, options, RuntimeInformation.OSDescription, RuntimeInformation.FrameworkDescription, measurements);

        async ValueTask RunAndValidateAsync(VmInferenceSession session)
        {
            await session.WriteStateAsync(initialSnapshot, cancellation).ConfigureAwait(false);
            var result = await ExecuteAsync(session).ConfigureAwait(false);
            await ValidateAsync(session, result.Logits).ConfigureAwait(false);
        }

        async ValueTask<(float[] Logits, double Milliseconds)> ExecuteAsync(VmInferenceSession session)
        {
            var watch = new Stopwatch();
            if (options.Workload == VmTuningWorkload.Prefill)
            {
                watch.Start();
                var logits = await session.PrefillAsync(tokenSnapshot, cancellation).ConfigureAwait(false);
                return (logits, watch.Elapsed.TotalMilliseconds);
            }
            await using var generation = await session.BeginGenerationAsync(cancellation).ConfigureAwait(false);
            watch.Start();
            var last = Array.Empty<float>();
            foreach (var token in tokenSnapshot)
                last = await generation.ForwardTokenAsync(token, cancellation).ConfigureAwait(false);
            watch.Stop();
            return (last, watch.Elapsed.TotalMilliseconds);
        }

        async ValueTask ValidateAsync(VmInferenceSession session, float[] logits)
        {
            Compare(reference.Logits, logits, "logits", options);
            var actual = await session.ReadStateAsync(cancellation).ConfigureAwait(false);
            if (actual.Count != reference.State.Count) throw new InvalidDataException("Candidate changed the State contract.");
            var types = session.StateProgram.Slots.ToDictionary(slot => slot.Id, slot => slot.Tensor.ElementType);
            foreach (var (name, expected) in reference.State)
            {
                if (!actual.TryGetValue(name, out var bytes) || bytes.Length != expected.Length)
                    throw new InvalidDataException($"Candidate changed State entry '{name}'.");
                if (types[name] == VmElementType.Float32)
                    Compare(MemoryMarshal.Cast<byte, float>(expected), MemoryMarshal.Cast<byte, float>(bytes), $"state:{name}", options);
                else if (!bytes.AsSpan().SequenceEqual(expected))
                    throw new InvalidDataException($"Candidate changed non-FP32 State entry '{name}'.");
            }
        }
    }

    private static void Compare(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
        string name, VmTuningOptions options)
    {
        if (expected.Length != actual.Length) throw new InvalidDataException($"Candidate changed '{name}' length.");
        for (var i = 0; i < actual.Length; i++)
        {
            var limit = options.AbsoluteTolerance + options.RelativeTolerance * Math.Abs(expected[i]);
            if (!float.IsFinite(expected[i]) || !float.IsFinite(actual[i]) || Math.Abs(expected[i] - actual[i]) > limit)
                throw new InvalidDataException($"Candidate '{name}' differs at {i}: expected {expected[i]}, actual {actual[i]}, tolerance {limit}.");
        }
    }
}
