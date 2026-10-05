#if TIER_ONE
using System.Runtime.InteropServices;
using System.Text.Json;
using SharpInference;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Gguf;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

internal static class MatVecProfiler
{
    public static void Run(Arguments arguments)
    {
        if (arguments.Target != VmTarget.Direct3D12) throw new ArgumentException("MatVec profiling requires --target gpu.");
        var repetitions = arguments.Number("repetitions", 8);
        var domains = new Dictionary<string, VmParameter[]>(StringComparer.Ordinal);
        foreach (var path in arguments.Required("models").Split('|'))
        {
            using var catalog = GgmlModelFile.Open(path);
            var graph = RwkvRuntimeFactory.CreateGraphProvider(Path.GetFileName(path).Contains("060", StringComparison.OrdinalIgnoreCase) ||
                Path.GetFileName(path).Contains("rwkv-6", StringComparison.OrdinalIgnoreCase) ? "rwkv-6" : "rwkv-7").Build(catalog);
            var program = VmGraphOptimizer.Optimize(graph, VmTarget.Direct3D12);
            foreach (var definition in program.Definitions.Where(definition =>
                         definition.Nodes.Any(node => node.Instruction is VmOperator { Operation: "core.mat-vec" })))
                domains.TryAdd(GpuMatVecOptimizationSettings.Shape(definition.Parameters), definition.Parameters.ToArray());
        }
        var measurements = new List<TierOneCostMeasurement>();
        var validations = new List<object>();
        foreach (var (shape, parameters) in domains.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            var matrix = parameters.Single(parameter => parameter.Name is "matrix" or "weight");
            var input = parameters.Single(parameter => parameter.Name == "input");
            var output = parameters.Single(parameter => parameter.Name == "output");
            var rows = matrix.Tensor.Dimensions[0];
            var columns = matrix.Tensor.Dimensions[1];
            var serial = GpuMatVecReferencePrograms.Create(rows, columns, false,
                matrix.Tensor.ElementType, input.Tensor.ElementType, output.Tensor.ElementType);
            var cooperative = GpuMatVecReferencePrograms.Create(rows, columns, true,
                matrix.Tensor.ElementType, input.Tensor.ElementType, output.Tensor.ElementType);
            var compiler = new D3D12VmCompiler(DefaultInstructionCollections.Create());
            using var oldCode = compiler.Compile(serial).CreateExecutor();
            using var newCode = compiler.Compile(cooperative).CreateExecutor();
            var weights = Enumerable.Range(0, checked(rows * columns)).Select(index => (index % 31 - 15) * 0.001f).ToArray();
            var vector = Enumerable.Range(0, columns).Select(index => MathF.Sin(index * 0.13f) * 0.1f).ToArray();
            var weightBytes = Bytes(weights, matrix.Tensor.ElementType);
            var inputBytes = Bytes(vector, input.Tensor.ElementType);
            foreach (var executor in new[] { oldCode, newCode })
            {
                executor.Upload("matrix", weightBytes);
                executor.Upload("input", inputBytes);
                executor.Execute("forward");
            }
            var expected = Values(oldCode.Readback("output"), output.Tensor.ElementType);
            var actual = Values(newCode.Readback("output"), output.Tensor.ElementType);
            var maximumAbsolute = 0f;
            var maximumScaled = 0f;
            for (var index = 0; index < expected.Length; index++)
            {
                var error = MathF.Abs(expected[index] - actual[index]);
                var scaled = error / (0.00003f + MathF.Abs(expected[index]) * 0.00003f);
                if (!float.IsFinite(scaled) || scaled > 1)
                    throw new InvalidDataException($"MatVec numerical gate failed for {shape}, row {index}: {expected[index]}/{actual[index]}.");
                maximumAbsolute = MathF.Max(maximumAbsolute, error);
                maximumScaled = MathF.Max(maximumScaled, scaled);
            }
            for (var warmup = 0; warmup < 4; warmup++)
            {
                oldCode.MeasureGpuMicroseconds("forward", repetitions);
                newCode.MeasureGpuMicroseconds("forward", repetitions);
            }
            var baseline = new double[9];
            var candidate = new double[9];
            for (var sample = 0; sample < baseline.Length; sample++)
                if (sample % 2 == 0)
                {
                    baseline[sample] = oldCode.MeasureGpuMicroseconds("forward", repetitions);
                    candidate[sample] = newCode.MeasureGpuMicroseconds("forward", repetitions);
                }
                else
                {
                    candidate[sample] = newCode.MeasureGpuMicroseconds("forward", repetitions);
                    baseline[sample] = oldCode.MeasureGpuMicroseconds("forward", repetitions);
                }
            var measurement = new TierOneCostMeasurement("core.mat-vec", GpuMatVecExecution.ImplementationFingerprint,
                shape, "0,1", 64, true, baseline, candidate);
            measurements.Add(measurement);
            validations.Add(new { Shape = shape, Count = expected.Length, MaximumAbsoluteError = maximumAbsolute,
                MaximumScaledError = maximumScaled });
            Console.WriteLine($"{shape}: serial {baseline.Order().ElementAt(4):F3} us; cooperative {candidate.Order().ElementAt(4):F3} us; " +
                $"conservative saving {measurement.ConservativeSavingMicroseconds:F3} us.");
        }
        var profile = new TierOneCostProfile(VmTierOneEnvironment.Fingerprint(VmTarget.Direct3D12),
            DateTimeOffset.UtcNow, measurements);
        var pathOutput = Path.GetFullPath(arguments.Required("output"));
        Directory.CreateDirectory(Path.GetDirectoryName(pathOutput)!);
        File.WriteAllText(pathOutput, profile.Serialize());
        File.WriteAllText(pathOutput + ".measurements.json", JsonSerializer.Serialize(new
        {
            Hardware = VmTierOneEnvironment.HardwareIdentity(VmTarget.Direct3D12),
            Timing = "Compute-queue GPU timestamps; resident weights/input; scheduled output barrier included; no upload/readback.",
            Repetitions = repetitions, WarmupBatches = 4, Samples = 9, Validations = validations, Profile = profile,
        }, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static byte[] Bytes(float[] values, VmElementType type) => type == VmElementType.Float16
        ? MemoryMarshal.AsBytes(values.Select(value => (Half)value).ToArray().AsSpan()).ToArray()
        : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static float[] Values(byte[] bytes, VmElementType type) => type == VmElementType.Float16
        ? MemoryMarshal.Cast<byte, Half>(bytes).ToArray().Select(value => (float)value).ToArray()
        : MemoryMarshal.Cast<byte, float>(bytes).ToArray();
}
#endif
