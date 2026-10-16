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
using SharpInference.Graphs.Execution;

namespace SharpInference.Tests;

public sealed class RwkvTinyExecutionPipelineTests
{
    [RwkvModelTheory]
    [InlineData("rwkv-6", "tiny-rwkv-6v0-3m-FP32.bin")]
    [InlineData("rwkv-7", "tiny-rwkv-7v0-834K-FP32.bin")]
    public void TinyModelBuildsAndEmitsCpuProgram(string architecture, string fileName)
    {
        var directory = RwkvTestModels.GetDirectory();

        using var catalog = GgmlModelFile.Open(Path.Combine(directory, fileName));
        var logical = architecture == "rwkv-6"
            ? new PortableRwkv6GraphProvider().Build(catalog)
            : new PortableRwkv7GraphProvider().Build(catalog);
        var execution = new DefaultExecutionGraphConverter().Convert(
            logical,
            new(architecture, "cpu-vm"));
        IInstructionCollectionProvider[] providers =
        [
            new CpuFloat32InstructionCollection(),
            new CpuFloat16InstructionCollection(),
        ];
        var program = new VmExecutionGraphEmitter(VmTarget.Cpu, providers).Emit(execution);
        var rootGraph = execution.Graphs.Single(graph => graph.Id == execution.RootGraph);
        var rootDefinition = program.Definitions.Single(definition =>
            definition.Id == $"graph.{execution.RootGraph.Value}");

        Assert.Equal("forward", Assert.Single(program.Entries).Name);
        Assert.Equal(
            rootGraph.Nodes.Select(node => node.Id.Value),
            rootDefinition.Nodes.Select(node => node.Id));
        Assert.Contains(program.Slots, slot => slot.BindingKey is not null);
        Assert.Contains(program.Slots, slot => slot.Id == "token");
        Assert.Contains(program.Slots, slot => slot.Id == "logits");
        Assert.NotEmpty(new CpuVmCompiler(providers).GenerateSource(program).Source);
    }

    [RwkvModelFact]
    public void TinyRwkv7MatchesRwkvCppExpectedLogits()
    {
        var directory = RwkvTestModels.GetDirectory();

        using var catalog = GgmlModelFile.Open(
            Path.Combine(directory, "tiny-rwkv-7v0-834K-FP32.bin"));
        var logical = new PortableRwkv7GraphProvider().Build(catalog);
        var execution = new DefaultExecutionGraphConverter().Convert(
            logical,
            new("rwkv-7", "cpu-vm"));
        IInstructionCollectionProvider[] providers =
        [
            new CpuFloat32InstructionCollection(),
            new CpuFloat16InstructionCollection(),
        ];
        var program = new VmExecutionGraphEmitter(VmTarget.Cpu, providers).Emit(execution);
        var artifact = new CpuVmCompiler(providers).Compile(program);
        using var executable = artifact.CreateExecutor();
        var buffers = program.Slots.Select(slot =>
            new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
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
        foreach (var token in new[] { (int)'"', (int)'i', (int)'n' })
        {
            MemoryMarshal.Write(buffers[tokenIndex], token);
            executable.Execute("forward", buffers);
        }

        var logits = MemoryMarshal.Cast<byte, float>(buffers[logitsIndex]).ToArray();
        Assert.Equal(catalog.VocabularySize, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.Contains(logits, value => value != 0f);
        var expectedBytes = File.ReadAllBytes(
            Path.Combine(directory, "expected-logits-7v0-834K.bin"));
        var expected = MemoryMarshal.Cast<byte, float>(expectedBytes).ToArray();
        Assert.Equal(expected.Length, logits.Length);
        var meanAbsoluteError = logits.Zip(expected, (actual, baseline) =>
            MathF.Abs(actual - baseline)).Average();
        Assert.True(
            meanAbsoluteError <= 0.0011f,
            $"Mean absolute error {meanAbsoluteError:G9}; " +
            $"actual=[{string.Join(",", logits.Take(8).Select(value => value.ToString("G9")))}]; " +
            $"expected=[{string.Join(",", expected.Take(8).Select(value => value.ToString("G9")))}].");
    }

    [RwkvModelFact]
    public void TinyRwkv7D3D12MatchesRwkvCppExpectedLogits()
    {
        var directory = RwkvTestModels.GetDirectory();

        using var catalog = GgmlModelFile.Open(
            Path.Combine(directory, "tiny-rwkv-7v0-834K-FP32.bin"));
        var logical = new PortableRwkv7GraphProvider().Build(catalog);
        var execution = new DefaultExecutionGraphConverter().Convert(
            logical,
            new("rwkv-7", "d3d12-vm"));
        IInstructionCollectionProvider[] providers =
        [
            new GpuFloat32InstructionCollection(),
            new GpuFloat16InstructionCollection(),
        ];
        var program = new VmExecutionGraphEmitter(
            VmTarget.Direct3D12, providers).Emit(execution);
        var artifact = new D3D12VmCompiler(providers).Compile(program);
        using var executable = artifact.CreateExecutor();
        var buffers = program.Slots.Select(slot =>
            new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
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
        foreach (var token in new[] { (int)'"', (int)'i', (int)'n' })
        {
            MemoryMarshal.Write(buffers[tokenIndex], token);
            executable.Execute("forward", buffers);
        }

        var logits = MemoryMarshal.Cast<byte, float>(buffers[logitsIndex]).ToArray();
        Assert.Equal(catalog.VocabularySize, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.Contains(logits, value => value != 0f);
        var expectedBytes = File.ReadAllBytes(
            Path.Combine(directory, "expected-logits-7v0-834K.bin"));
        var expected = MemoryMarshal.Cast<byte, float>(expectedBytes).ToArray();
        Assert.Equal(expected.Length, logits.Length);
        var meanAbsoluteError = logits.Zip(expected, (actual, baseline) =>
            MathF.Abs(actual - baseline)).Average();
        Assert.True(
            meanAbsoluteError <= 0.0011f,
            $"Mean absolute error {meanAbsoluteError:G9}; " +
            $"actual=[{string.Join(",", logits.Take(8).Select(value => value.ToString("G9")))}]; " +
            $"expected=[{string.Join(",", expected.Take(8).Select(value => value.ToString("G9")))}].");
    }

}
