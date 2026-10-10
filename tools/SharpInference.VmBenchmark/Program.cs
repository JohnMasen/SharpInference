using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using SharpInference;
using SharpInference.Architectures.Phi4;
using SharpInference.Architectures.Phi4.D3D12;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Runtime;
using SharpInference.Applications;
using SharpInference.Vm;
#if TIER_ONE
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Instructions;
using SharpInference.Vm.Optimization;
#endif

var arguments = Arguments.Parse(args);
if (arguments.Command == "phi4")
{
    Phi4Benchmark.Run(arguments);
    return;
}
#if TIER_ONE
if (arguments.Command == "profile")
{
    OfflineProfiler.Run(arguments);
    return;
}
if (arguments.Command == "matvec-profile")
{
    MatVecProfiler.Run(arguments);
    return;
}
if (arguments.Command == "questions")
{
    await QuestionBenchmark.Run(arguments);
    return;
}
#endif
if (arguments.Command != "run") throw new ArgumentException("Use 'run', 'questions', 'profile', or 'matvec-profile'.");
var target = arguments.Target;
var path = arguments.Required("model");
var output = Path.GetFullPath(arguments.Required("output"));
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var samples = arguments.Number("samples", 7);
var tokenCount = arguments.Number("tokens", 8);
var warmups = arguments.Number("warmups", 2);
var minimumWarmupMilliseconds = arguments.Number("minimum-warmup-ms", 2000);
int[] pattern = [1, 2, 3, 2, 4, 5, 1, 3];
var tokens = Enumerable.Range(0, tokenCount).Select(index => pattern[index % pattern.Length]).ToArray();
var configuration = new VmRuntimeConfig
{
    InferenceInstances = 1, PrefillInstances = 1, PrefillCapacity = 8,
#if TIER_ONE
    GpuMatVecMode = arguments.MatVecMode,
    GpuMatVecCostProfile = arguments.Optional("matvec-profile") is { } matVecPath
        ? TierOneCostProfile.Deserialize(File.ReadAllText(matVecPath)) : null,
    TierOneCostProfile = arguments.Optional("profile") is { } profilePath
        ? TierOneCostProfile.Deserialize(File.ReadAllText(profilePath)) : null,
#endif
};
#if !TIER_ONE
if (arguments.Optional("profile") is not null) throw new ArgumentException("The main baseline does not expose the T1 API.");
#endif
var setup = Stopwatch.StartNew();
using var backend = target == VmTarget.Cpu ? SharpInference.Runtime.Cpu.CpuVmBackendFactory.Create(configuration) :
    SharpInference.Runtime.D3D12.D3D12VmBackendFactory.Create(configuration);
using var processor = arguments.Optional("logical-xml") is { } logicalXml
    ? Processor.LoadGraph(path, GraphXml.DeserializeLogical(File.ReadAllText(logicalXml)), backend)
    : Processor.Load(path, new GgmlModelReader(), RwkvApplicationComposition.CreateModelModules(), backend);
if (arguments.Optional("export-logical") is { } exportLogical)
{
    var document = System.Xml.Linq.XDocument.Parse(GraphXml.Serialize(processor.LogicalGraph!));
    using var writer = System.Xml.XmlWriter.Create(exportLogical, new System.Xml.XmlWriterSettings
    {
        Indent = true, Encoding = new System.Text.UTF8Encoding(false), NewLineChars = "\r\n",
    });
    document.Save(writer);
}
using var session = processor.CreateSession();
setup.Stop();
var warmup = Stopwatch.StartNew();
var warmupCycles = 0;
while (warmupCycles < warmups || warmup.Elapsed.TotalMilliseconds < minimumWarmupMilliseconds)
{
    session.Reset();
    Decode();
    session.Reset();
    session.Prefill(tokens);
    warmupCycles++;
}
warmup.Stop();
var decode = new double[samples];
var prefill = new double[samples];
for (var index = 0; index < samples; index++)
{
    session.Reset();
    var watch = Stopwatch.StartNew();
    Decode();
    watch.Stop();
    decode[index] = watch.Elapsed.TotalMilliseconds / tokenCount;
    session.Reset();
    watch.Restart();
    session.Prefill(tokens);
    watch.Stop();
    prefill[index] = watch.Elapsed.TotalMilliseconds / tokenCount;
}
session.Reset();
var logits = new List<float>();
foreach (var token in tokens) logits.AddRange(session.ForwardToken(token).ToArray());
File.WriteAllBytes(output + ".logits.bin", MemoryMarshal.AsBytes(CollectionsMarshal.AsSpan(logits)).ToArray());
using (var destination = File.Create(output + ".state.bin")) session.SaveState(destination);
var comparisons = new Dictionary<string, NumericComparison>();
if (arguments.Optional("reference") is { } reference)
{
    comparisons.Add("logits", NumericComparison.Compare(
        MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(reference + ".logits.bin")), CollectionsMarshal.AsSpan(logits)));
    using var expectedFile = File.OpenRead(reference + ".state.bin");
    using var actualFile = File.OpenRead(output + ".state.bin");
    var expected = GgufStateFile.Read(expectedFile);
    var actual = GgufStateFile.Read(actualFile);
    if (expected.Tensors.Count != actual.Tensors.Count) throw new InvalidDataException("State entry count differs.");
    foreach (var tensor in expected.Tensors)
    {
        var other = actual.Tensors.Single(candidate => candidate.Name == tensor.Name);
        if (!tensor.Dimensions.SequenceEqual(other.Dimensions)) throw new InvalidDataException($"State shape differs for {tensor.Name}.");
        comparisons.Add("state:" + tensor.Name, NumericComparison.Compare(
            MemoryMarshal.Cast<byte, float>(tensor.Data.Span), MemoryMarshal.Cast<byte, float>(other.Data.Span)));
    }
}
var program = processor.InferenceProgram!;
var definitions = program.Definitions.ToDictionary(definition => definition.Id);
var forward = definitions[program.Entries.Single(entry => entry.Name == "forward").Definition];
var invocations = forward.Nodes.Select(node => node.Instruction).Select(instruction => instruction switch
{
    VmCall call => definitions[call.Definition],
    VmDispatch dispatch => definitions[dispatch.Definition],
    _ => null,
}).Where(definition => definition is not null).ToArray();
var operations = invocations.SelectMany(definition => definition!.Nodes).Select(node => node.Instruction).OfType<VmOperator>().ToArray();
#if TIER_ONE
var tierOneCount = operations.Count(operation => operation.InstructionCollectionId == InstructionCollectionIds.TierOneFloat32);
if (configuration.TierOneCostProfile is not null && tierOneCount == 0)
    Console.WriteLine("WARNING: profile supplied but no T1 selected; this is a baseline-retained measurement, not evidence of T1 speedup.");
object? decision = backend.OptimizationReport;
object? matVecDecision = backend.MatVecOptimizationReport;
var hardware = (target == VmTarget.Cpu ? SharpInference.Runtime.Cpu.CpuVmEnvironment.HardwareIdentity() : SharpInference.Runtime.D3D12.D3D12VmEnvironment.HardwareIdentity());
#else
var tierOneCount = 0;
object? decision = null;
object? matVecDecision = null;
var hardware = backend.DeviceName;
#endif
var result = new
{
    Variant = arguments.Required("variant"), Model = Path.GetFullPath(path), Target = target.ToString(),
    Device = backend.DeviceName, Hardware = hardware, Runtime = RuntimeInformation.FrameworkDescription,
    Machine = Environment.MachineName, Tokens = tokens, Samples = samples, Warmups = warmups,
    MinimumWarmupMilliseconds = minimumWarmupMilliseconds, WarmupCycles = warmupCycles,
    SetupMilliseconds = setup.Elapsed.TotalMilliseconds, WarmupMilliseconds = warmup.Elapsed.TotalMilliseconds,
    DecodeMillisecondsPerToken = decode, DecodeMedian = Median(decode),
    DecodeTokensPerSecond = 1000 / Median(decode),
    PrefillMillisecondsPerToken = prefill, PrefillMedian = Median(prefill),
    TierOneCalls = tierOneCount, Calls = operations.Length,
    Barriers = forward.Nodes.Count(node => node.Instruction is VmBarrier),
    LocalBytes = program.Slots.Where(slot => slot.Scope == VmSlotScope.Local).Sum(slot => checked((long)slot.Tensor.ByteLength)),
    GlobalBytes = program.Slots.Where(slot => slot.Scope == VmSlotScope.Global).Sum(slot => checked((long)slot.Tensor.ByteLength)),
    Decision = decision, MatVecDecision = matVecDecision, Comparisons = comparisons,
    LogitsSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(output + ".logits.bin"))),
};
File.WriteAllText(output + ".json", JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"{result.Variant}: {Path.GetFileName(path)} {target} {backend.DeviceName}; " +
    $"decode {result.DecodeMedian:F4} ms/token; prefill {result.PrefillMedian:F4} ms/token; T1 {tierOneCount}/{operations.Length}; " +
    $"comparisons {comparisons.Count} passed; setup {result.SetupMilliseconds:F1} ms.");

void Decode()
{
    foreach (var token in tokens) session.ForwardToken(token);
}

static double Median(double[] values) => values.Order().ElementAt(values.Length / 2);

internal static class Phi4Benchmark
{
    private const string Prompt =
        "<|user|>What is the result of 1+1? Explain briefly.<|end|><|assistant|>";

    public static void Run(Arguments arguments)
    {
        var modelDirectory = Path.GetFullPath(arguments.Required("model"));
        var tokenCount = arguments.Number("tokens", 16);
        var samples = arguments.Number("samples", 3);
        if (tokenCount <= 0)
            throw new ArgumentOutOfRangeException("tokens", "Token count must be positive.");
        if (samples <= 0)
            throw new ArgumentOutOfRangeException("samples", "Sample count must be positive.");

        var setup = Stopwatch.StartNew();
        using var package = Phi4ModelPackage.Open(modelDirectory);
        var adapterIndex = arguments.Optional("adapter") is { } adapterText
            ? int.Parse(adapterText, System.Globalization.CultureInfo.InvariantCulture)
            : 0;
        ArgumentOutOfRangeException.ThrowIfNegative(adapterIndex);
        var backendName = arguments.Optional("backend");
        if (backendName == "d3d12-session")
        {
            RunResidentSession(
                arguments, package, adapterIndex, tokenCount, samples, setup);
            return;
        }
        using var projector = backendName switch
        {
            null or "cpu" => null,
            "d3d12" => new Phi4D3D12MatrixProjector(package, adapterIndex),
            var backend => throw new ArgumentException($"Unsupported Phi-4 backend '{backend}'."),
        };
        var model = package.CreateTextModel(projector);
        setup.Stop();
        var encoded = package.Tokenizer.Encode(Prompt);
        var tokens = Enumerable.Range(0, tokenCount)
            .Select(index => encoded[index % encoded.Length])
            .ToArray();

        ForceCollection();
        var allocatedBefore = GC.GetTotalAllocatedBytes(true);
        var collectionsBefore = CollectionCounts();

        var coldSession = model.CreateSession();
        var watch = Stopwatch.StartNew();
        var coldLogits = coldSession.ForwardToken(tokens[0]);
        watch.Stop();
        var coldMilliseconds = watch.Elapsed.TotalMilliseconds;

        var decode = new double[samples];
        for (var sample = 0; sample < samples; sample++)
        {
            watch.Restart();
            coldSession.ForwardToken(tokens[(sample + 1) % tokens.Length]);
            watch.Stop();
            decode[sample] = watch.Elapsed.TotalMilliseconds;
        }

        var prefill = new double[samples];
        for (var sample = 0; sample < samples; sample++)
        {
            var session = model.CreateSession();
            watch.Restart();
            foreach (var token in tokens)
                session.ForwardToken(token);
            watch.Stop();
            prefill[sample] = watch.Elapsed.TotalMilliseconds;
        }

        var allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
        var collectionsAfter = CollectionCounts();
        var decodeMedian = Median(decode);
        var prefillMedian = Median(prefill);
        var result = new
        {
            ModelDirectory = modelDirectory,
            Backend = projector is null ? "cpu" : "d3d12-resident-projections",
            Runtime = RuntimeInformation.FrameworkDescription,
            Machine = Environment.MachineName,
            ProcessorCount = Environment.ProcessorCount,
            Tokens = tokenCount,
            Samples = samples,
            SetupMilliseconds = setup.Elapsed.TotalMilliseconds,
            ColdFirstTokenMilliseconds = coldMilliseconds,
            WarmDecodeMillisecondsPerToken = decode,
            WarmDecodeMedianMillisecondsPerToken = decodeMedian,
            WarmDecodeTokensPerSecond = 1000 / decodeMedian,
            PrefillMilliseconds = prefill,
            PrefillMedianMilliseconds = prefillMedian,
            PrefillTokensPerSecond = tokenCount * 1000 / prefillMedian,
            AllocatedBytes = allocatedBytes,
            Gen0Collections = collectionsAfter[0] - collectionsBefore[0],
            Gen1Collections = collectionsAfter[1] - collectionsBefore[1],
            Gen2Collections = collectionsAfter[2] - collectionsBefore[2],
            ArgMax = Array.IndexOf(coldLogits, coldLogits.Max()),
        };

        var json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        if (arguments.Optional("output") is { } output)
        {
            var path = Path.GetFullPath(output);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, json);
        }
        Console.WriteLine(json);
    }

    private static void RunResidentSession(
        Arguments arguments,
        Phi4ModelPackage package,
        int adapterIndex,
        int tokenCount,
        int samples,
        Stopwatch setup)
    {
        using var session = new Phi4D3D12TextSession(
            package,
            arguments.Optional("context") is { } contextText
                ? int.Parse(contextText, System.Globalization.CultureInfo.InvariantCulture)
                : Math.Max(64, tokenCount + samples + 1),
            adapterIndex);
        setup.Stop();
        var encoded = package.Tokenizer.Encode(Prompt);
        var tokens = Enumerable.Range(0, tokenCount)
            .Select(index => encoded[index % encoded.Length])
            .ToArray();
        ForceCollection();
        var allocatedBefore = GC.GetTotalAllocatedBytes(true);
        var collectionsBefore = CollectionCounts();
        var watch = Stopwatch.StartNew();
        var firstToken = session.ForwardToken(tokens[0]);
        watch.Stop();
        var coldMilliseconds = watch.Elapsed.TotalMilliseconds;
        var decode = new double[samples];
        for (var sample = 0; sample < samples; sample++)
        {
            watch.Restart();
            session.ForwardToken(tokens[(sample + 1) % tokens.Length]);
            watch.Stop();
            decode[sample] = watch.Elapsed.TotalMilliseconds;
        }
        var prefill = new double[samples];
        for (var sample = 0; sample < samples; sample++)
        {
            session.Reset();
            watch.Restart();
            foreach (var token in tokens)
                session.ForwardToken(token);
            watch.Stop();
            prefill[sample] = watch.Elapsed.TotalMilliseconds;
        }
        var allocatedBytes = GC.GetTotalAllocatedBytes(true) - allocatedBefore;
        var collectionsAfter = CollectionCounts();
        var decodeMedian = Median(decode);
        var prefillMedian = Median(prefill);
        var result = new
        {
            ModelDirectory = package.Directory,
            Backend = "d3d12-resident-session",
            Runtime = RuntimeInformation.FrameworkDescription,
            Machine = Environment.MachineName,
            ProcessorCount = Environment.ProcessorCount,
            Tokens = tokenCount,
            ContextCapacity = session.MaximumContext,
            Samples = samples,
            SetupMilliseconds = setup.Elapsed.TotalMilliseconds,
            ColdFirstTokenMilliseconds = coldMilliseconds,
            WarmDecodeMillisecondsPerToken = decode,
            WarmDecodeMedianMillisecondsPerToken = decodeMedian,
            WarmDecodeTokensPerSecond = 1000 / decodeMedian,
            PrefillMilliseconds = prefill,
            PrefillMedianMilliseconds = prefillMedian,
            PrefillTokensPerSecond = tokenCount * 1000 / prefillMedian,
            AllocatedBytes = allocatedBytes,
            Gen0Collections = collectionsAfter[0] - collectionsBefore[0],
            Gen1Collections = collectionsAfter[1] - collectionsBefore[1],
            Gen2Collections = collectionsAfter[2] - collectionsBefore[2],
            ArgMax = firstToken,
        };
        var json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        if (arguments.Optional("output") is { } output)
        {
            var path = Path.GetFullPath(output);
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllText(path, json);
        }
        Console.WriteLine(json);
    }

    private static int[] CollectionCounts() =>
        [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];

    private static void ForceCollection()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    private static double Median(double[] values) =>
        values.Order().ElementAt(values.Length / 2);
}

internal sealed record NumericComparison(int Count, float MaximumAbsoluteError, float MaximumScaledError)
{
    public static NumericComparison Compare(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        if (expected.Length != actual.Length) throw new InvalidDataException("Numerical comparison length differs.");
        var maximumAbsolute = 0f;
        var maximumScaled = 0f;
        for (var index = 0; index < expected.Length; index++)
        {
            if (!float.IsFinite(expected[index]) || !float.IsFinite(actual[index]))
                throw new InvalidDataException($"Nonfinite model value at {index}: {expected[index]:R}/{actual[index]:R}.");
            var error = MathF.Abs(expected[index] - actual[index]);
            var scaled = error / (0.0003f + MathF.Abs(expected[index]) * 0.0003f);
            if (scaled > 1) throw new InvalidDataException($"Numerical gate failed at {index}: {expected[index]:R}/{actual[index]:R}, scaled error {scaled:R}.");
            maximumAbsolute = MathF.Max(maximumAbsolute, error);
            maximumScaled = MathF.Max(maximumScaled, scaled);
        }
        return new(expected.Length, maximumAbsolute, maximumScaled);
    }
}

internal sealed class Arguments
{
    private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
    public string Command { get; private init; } = "";
    public VmTarget Target => Required("target") switch
    {
        "cpu" => VmTarget.Cpu, "gpu" => VmTarget.Direct3D12,
        var value => throw new ArgumentException($"Unknown target '{value}'."),
    };
#if TIER_ONE
    public GpuMatVecMode MatVecMode => Optional("matvec") switch
    {
        null or "default" => GpuMatVecMode.Default,
        "serial" => GpuMatVecMode.Serial,
        "cooperative" => GpuMatVecMode.Cooperative,
        "profile" => GpuMatVecMode.Profile,
        var value => throw new ArgumentException($"Unknown MatVec selection '{value}'."),
    };
#endif
    public string Required(string name) => Optional(name) ?? throw new ArgumentException($"Missing --{name}.");
    public string? Optional(string name) => values.GetValueOrDefault(name);
    public int Number(string name, int defaultValue)
    {
        var value = Optional(name) is { } text ? int.Parse(text, System.Globalization.CultureInfo.InvariantCulture) : defaultValue;
        return value > 0 ? value : throw new ArgumentException($"--{name} must be positive.");
    }
    public static Arguments Parse(string[] args)
    {
        if (args.Length == 0 || args.Length % 2 != 1) throw new ArgumentException("Use command followed by --name value pairs.");
        var result = new Arguments { Command = args[0] };
        for (var index = 1; index < args.Length; index += 2)
            if (!args[index].StartsWith("--", StringComparison.Ordinal) || !result.values.TryAdd(args[index][2..], args[index + 1]))
                throw new ArgumentException($"Invalid or duplicate option '{args[index]}'.");
        return result;
    }
}

#if TIER_ONE
internal static class OfflineProfiler
{
    public static void Run(Arguments arguments)
    {
        var target = arguments.Target;
        var registry = new InstructionRegistry(target == VmTarget.Cpu ? SharpInference.Runtime.Cpu.CpuInstructionCollections.Create() : SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create());
        var capabilities = registry.QueryOptimizationCapabilities();
        var cases = new Dictionary<string, (InstructionOptimizationCapability Capability, int[] Shape)>(StringComparer.Ordinal);
        foreach (var path in arguments.Required("models").Split('|'))
        {
            using var catalog = GgmlModelFile.Open(path);
            var graph = RwkvApplicationComposition.CreateModelModules().Build(catalog).Graph;
            var execution = new GraphOptimizer().Optimize(graph,
                new GraphOptimizationOptions(OptimizationBoundary.Off, DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded));
            foreach (var candidate in TierOnePatternMatcher.Find(execution, capabilities, target))
            {
                if (candidate.InputAliases != string.Join(",", Enumerable.Range(0, candidate.Inputs.Count))) continue;
                var shape = execution.Resources.Single(resource => resource.Id == candidate.Output).Tensor.Dimensions.ToArray();
                cases.TryAdd($"{candidate.Capability.Name}|{string.Join("x", shape)}", (candidate.Capability, shape));
            }
        }
        var measurements = new List<TierOneCostMeasurement>();
        foreach (var item in cases.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            var capability = item.Value.Capability;
            var shape = item.Value.Shape;
            var count = shape.Aggregate(1, (value, dimension) => checked(value * dimension));
            const int batch = 128;
            var reference = Repeat(VmGraphOptimizer.Optimize(TierOneReferencePrograms.CreateGraph(capability.Name, shape), target), batch);
            var fused = Repeat(TierOneReferencePrograms.CreateFused(capability.Name, target, shape), batch);
            var inputs = capability.Definition.Inputs.Select((port, index) => (port, index)).ToDictionary(pair => pair.port,
                pair => MemoryMarshal.AsBytes(Enumerable.Range(0, count).Select(index =>
                    0.5f + MathF.Sin(index * 0.13f + pair.index) * 0.1f).ToArray().AsSpan()).ToArray(), StringComparer.Ordinal);
            double[] baseline;
            double[] optimized;
            if (target == VmTarget.Cpu)
            {
                var compiler = new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create());
                using var oldCode = compiler.Compile(reference).LoadExecutable();
                using var newCode = compiler.Compile(fused).LoadExecutable();
                var oldBuffers = Buffers(reference, inputs);
                var newBuffers = Buffers(fused, inputs);
                oldCode.Execute("forward", oldBuffers);
                newCode.Execute("forward", newBuffers);
                NumericComparison.Compare(Output(reference, oldBuffers), Output(fused, newBuffers));
                (baseline, optimized) = Measure(() => oldCode.Execute("bench", oldBuffers), () => newCode.Execute("bench", newBuffers), 8, batch);
            }
            else
            {
                var compiler = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create());
                using var oldCode = compiler.Compile(reference).CreateExecutor();
                using var newCode = compiler.Compile(fused).CreateExecutor();
                foreach (var (port, input) in inputs) { oldCode.Upload(port, input); newCode.Upload(port, input); }
                oldCode.Execute("forward");
                newCode.Execute("forward");
                NumericComparison.Compare(MemoryMarshal.Cast<byte, float>(oldCode.Readback("output")),
                    MemoryMarshal.Cast<byte, float>(newCode.Readback("output")));
                (baseline, optimized) = Measure(() => oldCode.Execute("bench"), () => newCode.Execute("bench"), 8, batch);
            }
            var measurement = new TierOneCostMeasurement(capability.Name, capability.ImplementationFingerprint,
                string.Join("x", shape), string.Join(",", Enumerable.Range(0, inputs.Count)), 64, true, baseline, optimized);
            measurements.Add(measurement);
            Console.WriteLine($"{target} {item.Key}: conservative delta {measurement.ConservativeSavingMicroseconds:F3} us; " +
                (measurement.ConservativeSavingMicroseconds > 0 ? "eligible" : "T0 retained"));
        }
        var profile = new TierOneCostProfile((target == VmTarget.Cpu ? SharpInference.Runtime.Cpu.CpuVmEnvironment.Fingerprint() : SharpInference.Runtime.D3D12.D3D12VmEnvironment.Fingerprint()), DateTimeOffset.UtcNow, measurements);
        var output = Path.GetFullPath(arguments.Required("output"));
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        File.WriteAllText(output, profile.Serialize());
        Console.WriteLine($"Saved {measurements.Count} measurements; hardware: {(target == VmTarget.Cpu ? SharpInference.Runtime.Cpu.CpuVmEnvironment.HardwareIdentity() : SharpInference.Runtime.D3D12.D3D12VmEnvironment.HardwareIdentity())}.");
    }

    private static VmProgram Repeat(VmProgram program, int repetitions)
    {
        var forward = program.Definitions.Single(definition => definition.Id == "forward");
        var arguments = forward.Parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var nodes = Enumerable.Range(0, repetitions).Select(index => new VmNode($"repeat.{index}", new VmCall("forward", arguments),
            index == 0 ? [] : [$"repeat.{index - 1}"])).ToArray();
        return new(program.Name, program.Abi, program.Target, program.Slots,
            program.Definitions.Append(new("bench", VmDefinitionKind.Orchestration, forward.Parameters, nodes)),
            program.Entries.Append(new("bench", "bench", arguments)), program.State);
    }

    private static (double[] Reference, double[] Optimized) Measure(Action reference, Action optimized, int repetitions, int batch)
    {
        for (var index = 0; index < 64; index++) { reference(); optimized(); }
        var baseline = new double[9];
        var candidate = new double[9];
        for (var index = 0; index < baseline.Length; index++)
        {
            if (index % 2 == 0) { baseline[index] = Time(reference); candidate[index] = Time(optimized); }
            else { candidate[index] = Time(optimized); baseline[index] = Time(reference); }
        }
        return (baseline, candidate);

        double Time(Action action)
        {
            var start = Stopwatch.GetTimestamp();
            for (var index = 0; index < repetitions; index++) action();
            return Stopwatch.GetElapsedTime(start).TotalMicroseconds / repetitions / batch;
        }
    }

    private static byte[][] Buffers(VmProgram program, IReadOnlyDictionary<string, byte[]> inputs) =>
        program.Slots.Select(slot => inputs.TryGetValue(slot.Id, out var bytes) ? bytes.ToArray() :
            new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
    private static float[] Output(VmProgram program, byte[][] buffers) =>
        MemoryMarshal.Cast<byte, float>(buffers[program.Slots.ToList().FindIndex(slot => slot.Id == "output")]).ToArray();
}
#endif
