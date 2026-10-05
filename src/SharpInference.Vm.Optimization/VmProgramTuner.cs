using System.Diagnostics;
using System.Runtime.InteropServices;
using SharpInference.Vm;

namespace SharpInference.Vm.Optimization;

public enum VmTuningWorkload { Prefill, Inference }

public sealed record VmTuningOptions(
    int WarmupRuns = 2, int MeasuredRuns = 5,
    float AbsoluteTolerance = 0.0003f, float RelativeTolerance = 0.0003f,
    VmTuningWorkload Workload = VmTuningWorkload.Prefill);

public sealed record VmTuningTarget(string Hardware, string Driver, string Backend, string Compiler);

public sealed record VmTuningReference(float[] Logits, IReadOnlyDictionary<string, byte[]> State);
public sealed record VmTuningCandidate(string Name, VmProgram Program);
public sealed record VmTuningMeasurement(string Name, bool Accepted, double CompilationMilliseconds,
    double InitializationMilliseconds, double FirstExecutionMilliseconds, double WarmupMilliseconds, double MedianMilliseconds,
    string? Diagnostic);
public sealed record VmTuningResult(string? Winner, VmTuningTarget Target, VmTuningOptions Options,
    string OperatingSystem, string Runtime, IReadOnlyList<VmTuningMeasurement> Candidates);

public static class VmProgramTuner
{
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
