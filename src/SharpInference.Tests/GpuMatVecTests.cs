using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class GpuMatVecTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 17)]
    [InlineData(7, 63)]
    [InlineData(7, 64)]
    [InlineData(7, 65)]
    [InlineData(3, 4096)]
    [InlineData(65536, 3)]
    [InlineData(65537, 3)]
    public void CooperativeSignaturesHandleTailsViewsAndRepeatedInvocation(int rows, int columns)
    {
        foreach (var (matrixType, inputType, outputType) in new[]
                 {
                     (VmElementType.Float32, VmElementType.Float32, VmElementType.Float32),
                     (VmElementType.Float16, VmElementType.Float32, VmElementType.Float32),
                     (VmElementType.Float16, VmElementType.Float16, VmElementType.Float16),
                 })
        {
            var matrix = Enumerable.Range(0, checked(rows * columns)).Select(index => (index % 9 - 4) * 0.125f).ToArray();
            var input = Enumerable.Range(0, columns).Select(index => (index % 7 - 3) * 0.25f).ToArray();
            var expected = Enumerable.Range(0, rows).Select(row =>
                (float)Enumerable.Range(0, columns).Sum(column => (double)matrix[row * columns + column] * input[column])).ToArray();
            if (outputType == VmElementType.Float16)
                expected = expected.Select(value => (float)(Half)value).ToArray();
            Execute(rows, columns, matrix, input, expected, matrixType, inputType, outputType);
        }
    }

    [Fact]
    public void CooperativeMergePreservesCancellationAcrossAndWithinLanes()
    {
        foreach (var finalIndex in new[] { 4095, 4032 })
        {
            var input = Enumerable.Repeat(1f, 4096).ToArray();
            input[0] = 100_000_000;
            input[finalIndex] = -100_000_000;
            Execute(3, 4096, Enumerable.Repeat(1f, 3 * 4096).ToArray(), input,
                [4094, 4094, 4094], VmElementType.Float32, VmElementType.Float32, VmElementType.Float32);
        }
    }

    [Fact]
    public void CooperativeNonfiniteSumsRetainClassification()
    {
        foreach (var input in new[]
                 {
                     new[] { float.PositiveInfinity, 1f },
                     new[] { float.NegativeInfinity, 1f },
                     new[] { float.PositiveInfinity, float.NegativeInfinity },
                     new[] { float.NaN, 1f },
                 })
            Execute(3, 2, Enumerable.Repeat(1f, 6).ToArray(), input,
                Enumerable.Repeat(input.Sum(), 3).ToArray(),
                VmElementType.Float32, VmElementType.Float32, VmElementType.Float32);
    }

    [Fact]
    public void ConfigurationAndGridSurviveArtifactReload()
    {
        var program = GpuMatVecReferencePrograms.Create(65536, 3, true);
        var artifact = Compiler().Compile(program);
        using var stream = new MemoryStream();
        artifact.Export(stream);
        stream.Position = 0;
        var restored = D3D12VmArtifact.Import(stream);
        Assert.Equal(VmProgramXml.Serialize(program), VmProgramXml.Serialize(restored.Program));
        Assert.Equal(new VmThreadGroup(32768, 2), Dispatch(restored.Program).Groups);
        using var executor = restored.CreateExecutor();
        executor.Upload("matrix", Bytes(Enumerable.Repeat(1f, 65536 * 3).ToArray(), VmElementType.Float32));
        executor.Upload("input", Bytes([1, 2, 3], VmElementType.Float32));
        executor.Execute("forward");
        Assert.All(Values(executor.Readback("output"), VmElementType.Float32), value => Assert.Equal(6, value));
    }

    [Fact]
    public void CompilerRejectsWrongLaunchUnknownConfigurationAndAliasing()
    {
        var valid = GpuMatVecReferencePrograms.Create(7, 65, true);
        var kernel = valid.Definitions[0];
        var operation = Assert.IsType<VmOperator>(kernel.Nodes[0].Instruction);
        Assert.Throws<InvalidDataException>(() => Clone(valid,
            new(kernel.Id, kernel.Kind, kernel.Parameters, kernel.Nodes, new(128))));
        Assert.Throws<InvalidDataException>(() => Clone(valid, root: new("forward", VmDefinitionKind.Orchestration,
            valid.Definitions[1].Parameters,
            [new("compute", new VmDispatch("matvec", Dispatch(valid).Arguments, new(1)))])));
        var unknown = new VmOperator(operation.Precision, operation.InstructionCollectionId, operation.InstructionName,
            operation.Arguments, executionConfiguration: new D3D12InstructionExecutionConfiguration(new("missing")));
        Assert.Throws<InstructionAdaptationException>(() => Compiler().GenerateSources(Clone(valid,
            new(kernel.Id, kernel.Kind, kernel.Parameters, [new("body", unknown)], kernel.Threads))));
        var aliasedRoot = new VmDefinition("forward", VmDefinitionKind.Orchestration,
            valid.Definitions[1].Parameters.Select(parameter => parameter.Name == "matrix"
                ? parameter with { Access = VmAccess.ReadWrite } : parameter),
            [new("compute", new VmDispatch("matvec", Dispatch(valid).Arguments.Select(argument =>
                argument.Parameter == "output" ? argument with { Source = "matrix" } : argument), Dispatch(valid).Groups))]);
        var aliased = new VmProgram(valid.Name, valid.Abi, valid.Target,
            valid.Slots.Select(slot => slot.Id == "matrix" ? slot with { Access = VmAccess.ReadWrite } : slot),
            [kernel, aliasedRoot], valid.Entries, valid.State);
        Assert.Throws<NotSupportedException>(() => Compiler().GenerateSources(aliased));
        Assert.Throws<InvalidDataException>(() => Clone(valid,
            new(kernel.Id, kernel.Kind, kernel.Parameters, [new("first", operation), new("second", operation)], kernel.Threads)));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuMatVecExecution.Groups(uint.MaxValue));
    }

    [Fact]
    public void ResidentTimestampMeasurementExecutesAndRejectsInvalidRequests()
    {
        using var executor = Compiler().Compile(GpuMatVecReferencePrograms.Create(3, 65, true)).CreateExecutor();
        executor.Upload("matrix", Bytes(Enumerable.Repeat(1f, 195).ToArray(), VmElementType.Float32));
        executor.Upload("input", Bytes(Enumerable.Repeat(1f, 65).ToArray(), VmElementType.Float32));
        var elapsed = executor.MeasureGpuMicroseconds("forward", 4);
        Assert.True(double.IsFinite(elapsed) && elapsed > 0);
        Assert.All(Values(executor.Readback("output"), VmElementType.Float32), value => Assert.Equal(65, value));
        Assert.Throws<ArgumentOutOfRangeException>(() => executor.MeasureGpuMicroseconds("forward", 0));
        Assert.Throws<ArgumentException>(() => executor.MeasureGpuMicroseconds("missing", 1));
    }

    [Fact]
    public void SpanUploadExecutesWithoutOwnedSourceArray()
    {
        using var executor = Compiler().Compile(
            GpuMatVecReferencePrograms.Create(3, 2, true)).CreateExecutor();
        var matrix = Bytes([1, 2, 3, 4, 5, 6], VmElementType.Float32);
        var input = Bytes([2, 3], VmElementType.Float32);

        executor.Upload("matrix", matrix.AsSpan());
        executor.Upload("input", input.AsSpan());
        executor.Execute("forward");

        Assert.Equal([8, 18, 28], Values(executor.Readback("output"), VmElementType.Float32));
    }

    [Fact]
    public void TimestampMeasurementExecutesExactlyTheRequestedNumberOfDispatches()
    {
        VmParameter[] parameters = [new("output", VmAccess.ReadWrite, new(VmElementType.Float32, [1]))];
        var arguments = new[] { new VmArgument("output", "output") };
        var kernel = new VmDefinition("counter", VmDefinitionKind.Kernel, parameters,
            [new("body", new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32),
                Guid.Empty, "benchmark-counter", arguments))], new(64));
        var forward = new VmDefinition("forward", VmDefinitionKind.Orchestration, parameters,
            [new("dispatch", new VmDispatch("counter", arguments, new(1))),
             new("barrier", new VmBarrier(["output"]), ["dispatch"])]);
        var program = new VmProgram("counter", "benchmark", VmTarget.Direct3D12,
            [new("output", VmSlotScope.Local, VmAccess.ReadWrite, parameters[0].Tensor)], [kernel, forward],
            [new("forward", "forward", arguments)], new("none", 1, []));
        using var executor = new D3D12VmCompiler([new CounterProvider()]).Compile(program).CreateExecutor();
        executor.MeasureGpuMicroseconds("forward", 8);
        Assert.Equal(8, BitConverter.ToInt32(executor.Readback("output")));
    }

    private sealed class CounterProvider : IInstructionCollectionProvider
    {
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(Guid.Empty, "counter", 0, InstructionTarget.Direct3D12)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            collectionId == Guid.Empty && instructionName == "benchmark-counter" ? [new CounterInstruction()] : [];
    }

    private sealed class CounterInstruction : Instruction
    {
        public override Guid CollectionId => Guid.Empty;
        public override string Name => "benchmark-counter";
        public override InstructionTarget Target => InstructionTarget.Direct3D12;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("output", GraphElementType.Float32, GraphResourceAccess.ReadWrite)], [],
                new(GraphElementType.Float32, GraphElementType.Float32))];
        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            var output = Assert.IsType<InstructionTensorParameter>(parameters.Single());
            return new($"if(i==0u) {{ uint old; {output.Expression}.InterlockedAdd({output.OffsetExpression},1u,old); }}");
        }
    }

    [Fact]
    public void PlannerSelectsPerKernelGeometryAndKeepsCpuUnchanged()
    {
        var example = TierZeroOperationTests.Example("core.mat-vec", 0);
        var options = new VmOptimizationOptions(PrefillCapacity: 1, WeightViews: false,
            ThreadsPerGroup: 128, GpuMatVec: new(GpuMatVecMode.Cooperative));
        var result = VmGraphOptimizer.OptimizeWithReport(example.Graph(), VmTarget.Direct3D12, options);
        Assert.Equal(new VmThreadGroup(64), result.Program.Definitions[0].Threads);
        Assert.Equal(new VmThreadGroup((uint)example.Output.Dimensions[0]), Dispatch(result.Program).Groups);
        Assert.Equal(1, result.MatVecReport!.CooperativeCalls);
        Assert.Throws<NotSupportedException>(() => VmGraphOptimizer.Optimize(example.Graph(), VmTarget.Cpu, options));
        var baseline = VmGraphOptimizer.Optimize(example.Graph(), VmTarget.Direct3D12,
            options with { GpuMatVec = new(GpuMatVecMode.Serial) });
        Assert.Equal(GpuMatVecExecution.Serial,
            Assert.IsType<VmOperator>(baseline.Definitions[0].Nodes[0].Instruction).ExecutionConfiguration);
        Assert.Equal(new VmThreadGroup(128), baseline.Definitions[0].Threads);
        var defaultProgram = VmGraphOptimizer.Optimize(example.Graph(), VmTarget.Direct3D12,
            options with { GpuMatVec = null });
        Assert.Equal(GpuMatVecExecution.Cooperative,
            Assert.IsType<VmOperator>(defaultProgram.Definitions[0].Nodes[0].Instruction).ExecutionConfiguration);
        Assert.Equal(new VmThreadGroup(64), defaultProgram.Definitions[0].Threads);
        var cpu = VmGraphOptimizer.Optimize(example.Graph(), VmTarget.Cpu, new VmRuntimeConfig().OptimizationOptions());
        Assert.Null(Assert.IsType<VmOperator>(cpu.Definitions[0].Nodes[0].Instruction).ExecutionConfiguration);
        Assert.Null(new VmRuntimeConfig { GpuMatVecMode = GpuMatVecMode.Serial }.OptimizationOptions(target: VmTarget.Cpu).GpuMatVec);
    }

    [Fact]
    public void OldUnconfiguredMatVecRetainsSerialMeaningOnCompileAndArtifactReload()
    {
        var serial = GpuMatVecReferencePrograms.Create(7, 65, false);
        var kernel = serial.Definitions[0];
        var operation = Assert.IsType<VmOperator>(kernel.Nodes[0].Instruction);
        var legacy = Clone(serial, new(kernel.Id, kernel.Kind, kernel.Parameters,
            [new("body", new VmOperator(operation.Precision, operation.InstructionCollectionId,
                operation.InstructionName, operation.Arguments))], kernel.Threads));
        var artifact = Compiler().Compile(VmProgramXml.Deserialize(VmProgramXml.Serialize(legacy)));
        Assert.DoesNotContain("groupshared", artifact.Kernels.Single().Source);
        using var stream = new MemoryStream();
        artifact.Export(stream);
        stream.Position = 0;
        var restored = D3D12VmArtifact.Import(stream);
        Assert.Null(Assert.IsType<VmOperator>(restored.Program.Definitions[0].Nodes[0].Instruction).ExecutionConfiguration);
        Assert.Equal(new VmThreadGroup(1), Dispatch(restored.Program).Groups);
        using var executor = restored.CreateExecutor();
        executor.Upload("matrix", Bytes(Enumerable.Repeat(1f, 7 * 65).ToArray(), VmElementType.Float32));
        executor.Upload("input", Bytes(Enumerable.Repeat(1f, 65).ToArray(), VmElementType.Float32));
        executor.Execute("forward");
        Assert.All(Values(executor.Readback("output"), VmElementType.Float32), value => Assert.Equal(65, value));
    }

    [Fact]
    public void RuntimeRequiresExplicitConsistentMatVecConfiguration()
    {
        Assert.Throws<InvalidOperationException>(() => new VmRuntimeConfig
            { GpuMatVecMode = (GpuMatVecMode)999 }.EngineOptions());
        Assert.Throws<InvalidOperationException>(() => new VmRuntimeConfig
            { GpuMatVecMode = GpuMatVecMode.Profile }.EngineOptions());
        Assert.Throws<ArgumentException>(() => new GpuMatVecOptimizationSettings(GpuMatVecMode.Profile));
        Assert.Throws<ArgumentException>(() => new GpuMatVecOptimizationSettings(GpuMatVecMode.Serial,
            new("host", DateTimeOffset.UtcNow, [])));
        var profile = new TierOneCostProfile("host", DateTimeOffset.UtcNow, []);
        var configuration = new VmRuntimeConfig { GpuMatVecMode = GpuMatVecMode.Profile, GpuMatVecCostProfile = profile };
        configuration.EngineOptions();
        Assert.Same(profile, configuration.OptimizationOptions("host").GpuMatVec!.Profile);
        Assert.Throws<ArgumentException>(() => configuration.OptimizationOptions());
    }

    [Theory]
    [InlineData("valid", true)]
    [InlineData("slower", false)]
    [InlineData("hardware", false)]
    [InlineData("stale", false)]
    [InlineData("future", false)]
    [InlineData("implementation", false)]
    [InlineData("shape", false)]
    [InlineData("threads", false)]
    [InlineData("reuse", false)]
    public void OfflineSelectionRequiresExactTrustedPositiveCosts(string mutation, bool selected)
    {
        var example = TierZeroOperationTests.Example("core.mat-vec", 0);
        var baseline = VmGraphOptimizer.Optimize(example.Graph(), VmTarget.Direct3D12, new(WeightViews: false));
        var measurement = new TierOneCostMeasurement("core.mat-vec",
            mutation == "implementation" ? "obsolete" : GpuMatVecExecution.ImplementationFingerprint,
            mutation == "shape" ? "unknown" : GpuMatVecOptimizationSettings.Shape(baseline.Definitions[0].Parameters),
            "0,1", mutation == "threads" ? 128u : 64u, mutation != "reuse",
            Enumerable.Repeat(10d, 7).ToArray(), Enumerable.Repeat(mutation == "slower" ? 12d : 5d, 7).ToArray());
        var measuredAt = mutation == "stale" ? DateTimeOffset.UtcNow.AddDays(-31) :
            mutation == "future" ? DateTimeOffset.UtcNow.AddHours(1) : DateTimeOffset.UtcNow;
        var profile = new TierOneCostProfile(mutation == "hardware" ? "other" : "host", measuredAt, [measurement]);
        var result = VmGraphOptimizer.OptimizeWithReport(example.Graph(), VmTarget.Direct3D12,
            new(WeightViews: false, GpuMatVec: new(GpuMatVecMode.Profile, profile, "host")));
        Assert.Equal(selected ? 1 : 0, result.MatVecReport!.CooperativeCalls);
        Assert.NotEmpty(result.MatVecReport.Diagnostics);
    }

    private static D3D12VmCompiler Compiler() => new(DefaultInstructionCollections.Create());
    private static VmDispatch Dispatch(VmProgram program) => program.Definitions.SelectMany(definition => definition.Nodes)
        .Select(node => node.Instruction).OfType<VmDispatch>().Single();
    private static VmProgram Clone(VmProgram program, VmDefinition? kernel = null, VmDefinition? root = null) =>
        new(program.Name, program.Abi, program.Target, program.Slots,
            [kernel ?? program.Definitions[0], root ?? program.Definitions[1]], program.Entries, program.State);

    private static void Execute(int rows, int columns, float[] matrix, float[] input, float[] expected,
        VmElementType matrixType, VmElementType inputType, VmElementType outputType)
    {
        const int padding = 3;
        var program = GpuMatVecReferencePrograms.Create(rows, columns, true, matrixType, inputType, outputType,
            padding, weightPort: true);
        using var executor = Compiler().Compile(program).CreateExecutor();
        var buffers = program.Slots.Select(slot =>
        {
            var data = Enumerable.Repeat(13f, checked((int)slot.Tensor.ElementCount)).ToArray();
            if (slot.Id != "output") (slot.Id == "weight" ? matrix : input).CopyTo(data, padding);
            return Bytes(data, slot.Tensor.ElementType);
        }).ToArray();
        for (var index = 0; index < buffers.Length; index++) executor.Upload(program.Slots[index].Id, buffers[index]);
        for (var repeat = 0; repeat < 3; repeat++)
        {
            executor.Execute("forward");
            var actual = Values(executor.Readback("output"), outputType);
            Assert.All(actual.Take(padding).Concat(actual.TakeLast(padding)), value => Assert.Equal(13, value));
            for (var index = 0; index < expected.Length; index++)
                if (float.IsNaN(expected[index])) Assert.True(float.IsNaN(actual[index + padding]));
                else if (float.IsInfinity(expected[index])) Assert.Equal(expected[index], actual[index + padding]);
                else Assert.InRange(MathF.Abs(expected[index] - actual[index + padding]), 0,
                    0.00003f + MathF.Abs(expected[index]) * 0.00003f);
        }
        Assert.Equal(buffers[0], executor.Readback("weight"));
        Assert.Equal(buffers[1], executor.Readback("input"));
    }

    private static byte[] Bytes(float[] values, VmElementType type) => type == VmElementType.Float16
        ? MemoryMarshal.AsBytes(values.Select(value => (Half)value).ToArray().AsSpan()).ToArray()
        : MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static float[] Values(byte[] bytes, VmElementType type) => type == VmElementType.Float16
        ? MemoryMarshal.Cast<byte, Half>(bytes).ToArray().Select(value => (float)value).ToArray()
        : MemoryMarshal.Cast<byte, float>(bytes).ToArray();
}
