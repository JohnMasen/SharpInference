using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Instructions.D3D12;

namespace SharpInference.Tests;

public sealed class InstructionDiscoveryTests
{
    [Fact]
    public void ScanningIsIdempotentAndCatalogExposesParameters()
    {
        var services = new ServiceCollection();
        var assemblies = new[] { typeof(CpuFloat32InstructionCollection).Assembly, typeof(CpuTierOneInstructionCollection).Assembly,
            typeof(GpuFloat32InstructionCollection).Assembly, typeof(GpuTierOneInstructionCollection).Assembly };
        services.AddInstructions(assemblies);
        services.AddInstructions(assemblies);
        services.AddTransient<CpuVmCompiler>();
        using var provider = services.BuildServiceProvider();
        var catalog = provider.GetRequiredService<IInstructionCatalog>();
        Assert.Same(catalog, provider.GetRequiredService<InstructionRegistry>());
        Assert.Equal(catalog.Rows.Count, catalog.Rows.Distinct().Count());
        Assert.Equal(catalog.EnumerateInstructions().Count,
            provider.GetRequiredService<CpuVmCompiler>().InstructionCollections.EnumerateInstructions().Count);
        Assert.Equal(2, catalog.Rows.Count(row => row.Instruction == "core.affine"));
        Assert.All(catalog.Rows.Where(row => row.Instruction == "core.affine"), row => Assert.Equal(InstructionCategory.Tier1, row.Category));
        Assert.DoesNotContain(catalog.Rows, row => row.Instruction == "core.affine" && row.Category == InstructionCategory.Tier0);
        Assert.Contains(catalog.Parameters, row => row.Instruction == "core.add" && row.Name == "left" &&
            row.DataType == "Float16" && row.InternalPrecision == ElementType.Float32);
        Assert.Contains(catalog.Parameters, row => row.Instruction == "core.affine" && row.Name == "transpose_left" && row.DataType == "Boolean");
        var instruction = catalog.Resolve(TierZeroInstructionCollectionIds.Float16, "core.add", ExecutionTarget.Cpu);
        InstructionParameter[] parameters = [
            Tensor("left", ElementType.Float16, ResourceAccess.Read),
            Tensor("right", ElementType.Float16, ResourceAccess.Read),
            Tensor("output", ElementType.Float16, ResourceAccess.Write)];
        var signature = instruction.Adapt(parameters, new(ElementType.Float16, ElementType.Float32));
        Assert.Equal(ElementType.Float32, signature.Precision.ArithmeticType);
        var recorder = new SourceInstructionRecorder(ExecutionTarget.Cpu);
        instruction.Invoke(recorder, parameters, new(ElementType.Float32, ElementType.Float32));
        Assert.Contains("System.Half", recorder.GetSource());
    }

    [Fact]
    public void ProvidersCannotMergeOrOverrideAnInstructionSet()
    {
        var exception = Assert.Throws<InvalidDataException>(() =>
            new InstructionRegistry([new CpuFloat32InstructionCollection(), new CpuFloat32InstructionCollection()]));
        Assert.Contains("cannot be merged or overridden", exception.Message);
        Assert.Throws<InvalidDataException>(() => new InstructionRegistry([new IncompleteProvider()]));
        Assert.Throws<InvalidDataException>(() => new InstructionRegistry([new DuplicateInstructionProvider()]));
    }

    [Fact]
    public void ExplicitDiscoverySupportsConstructorInjectionWithoutDevices()
    {
        var services = new ServiceCollection();
        services.AddSingleton(new ProviderDependency());
        services.AddInstructions(typeof(InstructionDiscoveryTests).Assembly);
        using var provider = services.BuildServiceProvider(new ServiceProviderOptions { ValidateOnBuild = true, ValidateScopes = true });
        Assert.Single(provider.GetServices<IInstructionProvider>());
        Assert.Single(provider.GetRequiredService<IInstructionCatalog>().Rows);
    }

    [Fact]
    public async Task ReferencedButUnloadedPackagesAreDiscoveredInFreshProcess()
    {
        var root = new DirectoryInfo(AppContext.BaseDirectory);
        while (root is not null && !File.Exists(Path.Combine(root.FullName, "SharpInference.slnx")))
        {
            root = root.Parent;
        }
        Assert.NotNull(root);
        var configuration = new DirectoryInfo(AppContext.BaseDirectory).Parent!.Name;
        var probe = Path.Combine(root.FullName, "tools", "Instructions.DiscoveryProbe", "bin", configuration,
            "net10.0", "Instructions.DiscoveryProbe.dll");
        Assert.True(File.Exists(probe), $"Discovery probe was not built: {probe}");
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo("dotnet")
            {
                RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
                ArgumentList = { probe },
            },
        };
        Assert.True(process.Start());
        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        try
        {
            await process.WaitForExitAsync(timeout.Token);
        }
        catch (OperationCanceledException)
        {
            process.Kill(entireProcessTree: true);
            throw;
        }
        var output = await outputTask;
        var error = await errorTask;
        Assert.True(process.ExitCode == 0, output + error);
        Assert.Contains("discovery-ok:", output);
    }

    [Fact]
    public void FrameworkHasNoReferenceToConcreteInstructionPackages()
    {
        Assert.DoesNotContain(typeof(Instruction).Assembly.GetReferencedAssemblies(),
            assembly => assembly.Name!.StartsWith("SharpInference.Instructions.", StringComparison.Ordinal));
        Assert.Empty(new InstructionRegistry([]).Rows);
    }

    private static InstructionTensorParameter Tensor(string name, ElementType type, ResourceAccess access) =>
        new(name, new(type, [4]), access, name, name, "0");

    public sealed class ProviderDependency;
    public sealed class InjectedProvider(ProviderDependency dependency) : IInstructionProvider
    {
        private readonly Instruction instruction = new TestInstruction();
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection()
        {
            ArgumentNullException.ThrowIfNull(dependency);
            return [new(TestInstruction.Id, "test", 0, ExecutionTarget.Cpu, RequiredInstructions: ["test"])];
        }
        public IReadOnlyList<Instruction> EnumerateInstructions() => [instruction];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string name) =>
            collectionId == instruction.CollectionId && name == instruction.Name ? [instruction] : [];
    }
    private sealed class IncompleteProvider : IInstructionCollectionProvider
    {
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(TestInstruction.Id, "test", 0, ExecutionTarget.Cpu, RequiredInstructions: ["test"])];
        public IReadOnlyList<Instruction> EnumerateInstructions() => [];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string name) => [];
    }
    private sealed class DuplicateInstructionProvider : IInstructionCollectionProvider
    {
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(TestInstruction.Id, "test", 0, ExecutionTarget.Cpu)];
        public IReadOnlyList<Instruction> EnumerateInstructions() => [new TestInstruction(), new TestInstruction()];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string name) => [];
    }
    private sealed class TestInstruction : Instruction
    {
        public static readonly Guid Id = new("288cdde7-6586-4cee-bd28-43ae78a5ef4d");
        public override Guid CollectionId => Id;
        public override string Name => "test";
        public override ExecutionTarget Target => ExecutionTarget.Cpu;
        public override IReadOnlyList<InstructionSignature> Signatures => [
            new([new("output", ElementType.Float32, ResourceAccess.Write)], [], new(ElementType.Float32, ElementType.Float32))];
        protected override InstructionRecording Generate(InstructionParameter[] parameters) => new("{}");
    }
}
