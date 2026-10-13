using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Architectures.Rwkv7;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using Xunit.Abstractions;

namespace SharpInference.Tests;

[CollectionDefinition("RWKV postprocessors", DisableParallelization = true)]
public sealed class RwkvPostprocessorCollection;

[Collection("RWKV postprocessors")]
public sealed class RwkvPostprocessorTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData("rwkv-6", false, false)]
    [InlineData("rwkv-7", false, false)]
    [InlineData("rwkv-6", true, false)]
    [InlineData("rwkv-7", true, false)]
    [InlineData("rwkv-6", false, true)]
    [InlineData("rwkv-7", false, true)]
    [InlineData("rwkv-6", true, true)]
    [InlineData("rwkv-7", true, true)]
    public void AllPassCombinationsPreserveLogitsAndMeasurePerformance(
        string architecture, bool gpu, bool reverse)
    {
        var directory = typeof(RwkvPostprocessorTests).Assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .SingleOrDefault(attribute => attribute.Key == "RwkvTestModelDirectory")?.Value;
        if (string.IsNullOrWhiteSpace(directory))
        {
            return;
        }

        if (!Path.IsPathFullyQualified(directory) || !Directory.Exists(directory))
        {
            throw new InvalidOperationException("RwkvTestModelDirectory must be an existing absolute directory.");
        }

        var model = architecture == "rwkv-6" ? "6v0-3m" : "7v0-834K";
        using var catalog = GgmlModelFile.Open(Path.Combine(directory, $"tiny-rwkv-{model}-FP32.bin"));
        var logical = architecture == "rwkv-6"
            ? new PortableRwkv6GraphProvider().Build(catalog)
            : new PortableRwkv7GraphProvider().Build(catalog);
        var backend = gpu ? "d3d12-vm" : "cpu-vm";
        var compilation = new ExecutionGraphCompilationContext(architecture, backend);
        var original = new DefaultExecutionGraphConverter().Convert(logical, compilation);
        var expected = MemoryMarshal.Cast<byte, float>(
            File.ReadAllBytes(Path.Combine(directory, $"expected-logits-{model}.bin"))).ToArray();
        IInstructionCollectionProvider[] providers = gpu
            ? [new GpuFloat32InstructionCollection(), new GpuFloat16InstructionCollection()]
            : [new CpuFloat32InstructionCollection(), new CpuFloat16InstructionCollection()];
        var emitter = new VmExecutionGraphEmitter(gpu ? VmTarget.Direct3D12 : VmTarget.Cpu, providers);
        float[][]? baseline = null;

        output.WriteLine("BENCH_HEADER,Model,Backend,Reverse,Mask,Slots,Kernels,Dependencies,PassMs,EmitMs,CompileMs,ExecutorMs,ForwardMeanMs,ForwardMedianMs,ForwardP95Ms,LogitsMae");
        var masks = new[] { 0, 1, 2, 4, 3, 5, 6, 7 };
        foreach (var mask in reverse ? masks.Reverse() : masks)
        {
            var passes = new List<IExecutionGraphPostprocessor>();
            if ((mask & 1) != 0)
            {
                passes.Add(new DependencyReductionPostprocessor());
            }

            if ((mask & 2) != 0)
            {
                passes.Add(new TemporaryResourceReusePostprocessor());
            }

            if ((mask & 4) != 0)
            {
                passes.Add(new OperationTemplateSharingPostprocessor());
            }

            var stopwatch = Stopwatch.StartNew();
            var execution = original;
            for (var index = 0; index < passes.Count; index++)
            {
                execution = new(passes[index].Process(execution,
                    new(logical, compilation, index, passes[index].Name)).Graphs);
            }

            stopwatch.Stop();
            var passMs = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Restart();
            var program = emitter.Emit(execution);
            stopwatch.Stop();
            var emitMs = stopwatch.Elapsed.TotalMilliseconds;

            var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
            for (var index = 0; index < program.Slots.Count; index++)
            {
                var slot = program.Slots[index];
                if (slot.Scope != VmSlotScope.Global)
                {
                    continue;
                }

                using var storage = new VmMemoryStorage(buffers[index].Length);
                VmModelBindings.InitializeGlobal(catalog, slot, storage);
                storage.Read(0, buffers[index]);
            }
            var tokenIndex = program.Slots.ToList().FindIndex(slot => slot.Id == "token");
            var logitsIndex = program.Slots.ToList().FindIndex(slot => slot.Id == "logits");
            double compileMs;
            double executorMs;
            double[] samples;
            float[][] logits;
            if (gpu)
            {
                stopwatch.Restart();
                var artifact = new D3D12VmCompiler(providers).Compile(program);
                stopwatch.Stop();
                compileMs = stopwatch.Elapsed.TotalMilliseconds;
                stopwatch.Restart();
                using var executable = artifact.CreateExecutorForExplicitGlobalUploads();
                stopwatch.Stop();
                executorMs = stopwatch.Elapsed.TotalMilliseconds;
                executable.UploadSlots(buffers, Enumerable.Range(0, buffers.Length)
                    .Where(index => program.Slots[index].Scope == VmSlotScope.Global).ToArray());
                logits = RunTokens(token =>
                {
                    MemoryMarshal.Write(buffers[tokenIndex], token);
                    executable.Upload(tokenIndex, buffers[tokenIndex]);
                    executable.Execute("forward");
                    executable.Readback(logitsIndex, buffers[logitsIndex]);
                }, buffers[logitsIndex]);
                samples = Measure(() => executable.Execute("forward"));
            }
            else
            {
                stopwatch.Restart();
                var artifact = new CpuVmCompiler(providers).Compile(program);
                stopwatch.Stop();
                compileMs = stopwatch.Elapsed.TotalMilliseconds;
                stopwatch.Restart();
                using var executable = artifact.CreateExecutor();
                stopwatch.Stop();
                executorMs = stopwatch.Elapsed.TotalMilliseconds;
                logits = RunTokens(token =>
                {
                    MemoryMarshal.Write(buffers[tokenIndex], token);
                    executable.Execute("forward", buffers);
                }, buffers[logitsIndex]);
                samples = Measure(() => executable.Execute("forward", buffers));
            }
            Assert.All(logits.SelectMany(values => values), value => Assert.True(float.IsFinite(value)));
            if (baseline is null)
            {
                baseline = logits;
            }
            else
            {
                for (var token = 0; token < logits.Length; token++)
                {
                    Assert.Equal(baseline[token].Select(BitConverter.SingleToInt32Bits),
                        logits[token].Select(BitConverter.SingleToInt32Bits));
                }
            }

            Assert.Equal(expected.Length, logits[^1].Length);
            var mae = logits[^1].Zip(expected, (actual, reference) => MathF.Abs(actual - reference)).Average();
            Assert.True(mae <= 0.0011f, $"{architecture}/{backend}/{mask} MAE={mae:G9}");
            var kernels = program.Definitions.Count(definition => definition.Kind == VmDefinitionKind.Kernel);
            var dependencies = execution.Graphs.Sum(graph =>
                graph.Nodes.Sum(node => graph.Dependencies(node).Count));
            output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"BENCH,{architecture},{backend},{reverse},{mask},{program.Slots.Count},{kernels},{dependencies},{passMs:F3},{emitMs:F3},{compileMs:F3},{executorMs:F3},{samples.Average():F6},{samples[samples.Length / 2]:F6},{samples[(int)(samples.Length * 0.95)]:F6},{mae:G9}"));
        }
    }

    private static float[][] RunTokens(Action<int> execute, byte[] logits)
    {
        var result = new List<float[]>();
        foreach (var token in new[] { (int)'"', (int)'i', (int)'n' })
        {
            execute(token);
            result.Add(MemoryMarshal.Cast<byte, float>(logits).ToArray());
        }
        return result.ToArray();
    }

    private static double[] Measure(Action execute)
    {
        for (var index = 0; index < 30; index++)
        {
            execute();
        }

        var samples = new double[200];
        var stopwatch = new Stopwatch();
        for (var index = 0; index < samples.Length; index++)
        {
            stopwatch.Restart();
            execute();
            stopwatch.Stop();
            samples[index] = stopwatch.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        return samples;
    }
}
