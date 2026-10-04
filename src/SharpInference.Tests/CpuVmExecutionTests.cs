using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using SharpInference.Backends.CpuVm;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class CpuVmExecutionTests
{
    private static readonly IReadOnlyDictionary<string, string> EmptyAttributes = new Dictionary<string, string>();
    public static IEnumerable<object[]> Operations()
    {
        foreach (var operation in new[] { "copy", "add", "subtract", "multiply", "divide", "maximum",
                     "exp", "tanh", "sigmoid", "rsqrt", "square", "relu", "reduce-sum", "reduce-mean", "mat-vec", "gather-row" })
            foreach (var type in new[] { VmElementType.Float32, VmElementType.Float16 })
                yield return [operation, type];
        foreach (var operation in new[] { "fill", "cast-f16-f32", "reshape", "slice", "broadcast",
                     "batched-mat-vec", "reduce-last-sum", "reduce-last-mean", "head-outer" })
            yield return ["tensor." + operation, VmElementType.Float32];
    }

    [Theory]
    [MemberData(nameof(Operations))]
    public void ExecutesEveryGraphContract(string operation, VmElementType type)
    {
        var example = Example(operation, type);
        var program = ProgramFor(operation, example.Parameters, example.Attributes);
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        Assert.Contains("private static void D0", artifact.Source);
        Assert.DoesNotContain("CpuPrimitiveGraphExecutor", artifact.Source);
        Assert.Contains("#line 1 \"vm/function/node\"", artifact.Source);
        Assert.NotEmpty(artifact.PortablePdb.ToArray());
        using var executable = artifact.LoadExecutable();
        var buffers = example.Inputs.Append(new byte[(int)example.Parameters[^1].Tensor.ByteLength]).ToArray();
        var context = executable.Prepare(buffers);
        executable.Invoke("run", context);
        var actual = Values(buffers[^1], example.Parameters[^1].Tensor.ElementType);
        Assert.Equal(example.Expected.Length, actual.Length);
        for (var i = 0; i < actual.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - example.Expected[i]), 0, type == VmElementType.Float16 ? .004f : .00001f);
        executable.Invoke("run", context);
        Assert.Equal(actual, Values(buffers[^1], example.Parameters[^1].Tensor.ElementType));
    }

    [Fact]
    public void ReusesDefinitionThroughNestedCallsAndOffsetViews()
    {
        var tensor = new VmTensor(VmElementType.Float32, [2]);
        var parameters = new[] { new VmParameter("input", VmAccess.ReadOnly, tensor), new VmParameter("output", VmAccess.ReadWrite, tensor) };
        var copy = new VmDefinition("copy", VmDefinitionKind.Function, parameters,
            [new VmNode("copy", new VmOperator("core.copy", 1, [new("input", "input"), new("output", "output")]))]);
        var large = new VmTensor(VmElementType.Float32, [4]);
        var root = new VmDefinition("root", VmDefinitionKind.Orchestration,
            [new("input", VmAccess.ReadOnly, large), new("output", VmAccess.ReadWrite, large)],
            [new VmNode("first", new VmCall("copy", [new("input", "input"), new("output", "output")])),
             new VmNode("second", new VmCall("copy", [new("input", "input", 8), new("output", "output", 8)]), ["first"])]);
        var program = new VmProgram("reuse", "cpu", VmTarget.Cpu,
            [new("x", VmSlotScope.Global, VmAccess.ReadOnly, large), new("y", VmSlotScope.Local, VmAccess.ReadWrite, large)],
            [copy, root], [new("run", "root", [new("input", "x"), new("output", "y")])], new VmState("none", 1, []));
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program);
        Assert.Equal(1, artifact.Source.Split("private static void D0").Length - 1);
        using var executable = artifact.LoadExecutable();
        var buffers = new[] { Bytes([1, 2, 3, 4], VmElementType.Float32), new byte[16] };
        executable.Invoke("run", executable.Prepare(buffers));
        Assert.Equal(new float[] { 1, 2, 3, 4 }, Values(buffers[1], VmElementType.Float32));
    }

    [Theory]
    [InlineData(VmElementType.Float32)]
    [InlineData(VmElementType.Float16)]
    public void ExecutesMatVecWithOriginalWeightPort(VmElementType type)
    {
        var example = Example("mat-vec", type);
        var parameters = example.Parameters.Select(p => p.Name == "matrix" ? p with { Name = "weight" } : p).ToArray();
        var program = ProgramFor("mat-vec", parameters, EmptyAttributes);
        Assert.Contains(program.Definitions[0].Parameters, p => p.Name == "weight");
        using var executable = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program).LoadExecutable();
        var buffers = example.Inputs.Append(new byte[(int)parameters[^1].Tensor.ByteLength]).ToArray();
        executable.Invoke("run", executable.Prepare(buffers));
        Assert.Equal(example.Expected, Values(buffers[^1], type));
        var ambiguous = parameters.Append(example.Parameters[0]).ToArray();
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(ProgramFor("mat-vec", ambiguous, EmptyAttributes)));
    }

    [Fact]
    public void ReloadsActualBinaryAndSourcePackagesWithIntegrity()
    {
        var example = Example("add", VmElementType.Float32);
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(ProgramFor("add", example.Parameters, example.Attributes));
        var directory = Path.Combine(Path.GetTempPath(), "cpuvm-" + Guid.NewGuid().ToString("N"));
        try
        {
            artifact.Export(directory);
            var loaded = CpuVmCompiledArtifact.Load(directory);
            Assert.True(loaded.HasBinary);
            Assert.Equal(artifact.Source, loaded.Source);
            using (var executable = loaded.LoadExecutable())
            {
                var buffers = example.Inputs.Append(new byte[8]).ToArray();
                executable.Invoke("run", executable.Prepare(buffers));
                Assert.Equal(example.Expected, Values(buffers[^1], VmElementType.Float32));
            }
            var sourceDirectory = Path.Combine(directory, "source");
            artifact.Export(sourceDirectory, includeBinary: false);
            var source = CpuVmCompiledArtifact.Load(sourceDirectory);
            Assert.False(source.HasBinary);
            Assert.Throws<InvalidOperationException>(() => source.LoadExecutable());
            using (var executable = source.CreateExecutable(new StaticAdd(source.ProgramHash)))
            {
                var buffers = example.Inputs.Append(new byte[8]).ToArray();
                executable.Invoke("run", executable.Prepare(buffers));
                Assert.Equal(example.Expected, Values(buffers[^1], VmElementType.Float32));
            }
            File.AppendAllText(Path.Combine(directory, "CpuProgram.g.cs"), "//tampered");
            Assert.Throws<InvalidDataException>(() => CpuVmCompiledArtifact.Load(directory));
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void RejectsUnknownOperationsVersionsTypesPrecisionAndReadOnlyWrites()
    {
        var example = Example("copy", VmElementType.Float32);
        var compiler = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create());
        Assert.Throws<NotSupportedException>(() => compiler.GenerateSource(ProgramFor("unknown", example.Parameters, EmptyAttributes)));
        Assert.Throws<NotSupportedException>(() => compiler.GenerateSource(ProgramFor("copy", example.Parameters, EmptyAttributes, version: 2)));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => compiler.GenerateSource(ProgramFor("copy",
            example.Parameters.Select(p => p with { Tensor = new VmTensor(VmElementType.Int32, [2]) }).ToArray(), EmptyAttributes)));
        Assert.Throws<NotSupportedException>(() => compiler.GenerateSource(ProgramFor("copy", example.Parameters, EmptyAttributes),
            new CpuVmCompilerOptions { Accumulator = "Float16" }));
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => compiler.GenerateSource(ProgramFor("copy",
            example.Parameters.Select(p => p with { Access = VmAccess.ReadOnly }).ToArray(), EmptyAttributes)));
    }

    [Fact]
    public void RejectsPhysicalAliasingCapacityAndUninitializedReads()
    {
        var example = Example("copy", VmElementType.Float32);
        var standard = ProgramFor("copy", example.Parameters, EmptyAttributes);
        using var executable = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(standard).LoadExecutable();
        var shared = new byte[8];
        Assert.Throws<ArgumentException>(() => executable.Prepare([shared, shared]));
        Assert.Throws<ArgumentException>(() => executable.Prepare([new byte[4], new byte[8]]));
        var local = new VmProgram("local", "cpu", VmTarget.Cpu,
            standard.Slots.Select(s => s with { Scope = VmSlotScope.Local }), standard.Definitions,
            standard.Entries, standard.State);
        using var localExecutable = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(local).LoadExecutable();
        var buffers = new[] { example.Inputs[0], new byte[8] };
        Assert.Throws<InvalidOperationException>(() => localExecutable.Invoke("run", localExecutable.Prepare(buffers)));
        localExecutable.Invoke("run", localExecutable.Prepare(buffers, ["input"]));
        Assert.Equal(example.Expected, Values(buffers[^1], VmElementType.Float32));
        var overlap = new VmProgram("overlap", "cpu", VmTarget.Cpu,
            [new("both", VmSlotScope.Session, VmAccess.ReadWrite, example.Parameters[0].Tensor)],
            standard.Definitions, [new("run", "function", [new("input", "both"), new("output", "both")])], standard.State);
        Assert.Throws<InvalidDataException>(() => new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(overlap));
    }

    [Fact]
    public void PublishesGeneratedSourceForStaticCompilation()
    {
        var example = Example("add", VmElementType.Float32);
        var source = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(ProgramFor("add", example.Parameters, EmptyAttributes));
        Assert.False(source.HasBinary);
        Assert.Contains("public sealed class CpuProgram : ICpuVmCode", source.Source);
        Assert.Throws<InvalidDataException>(() => source.CreateExecutable(new StaticAdd("wrong")));
        var directory = Environment.GetEnvironmentVariable("CPUVM_PUBLICATION_DIRECTORY");
        if (directory is not null) source.Export(directory, includeBinary: false);
    }

    [Fact]
    public void RejectsLegacySpanSourceAbiForSameProgram()
    {
        var example = Example("add", VmElementType.Float32);
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).GenerateSource(ProgramFor("add", example.Parameters, EmptyAttributes));
        var legacyHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(VmProgramXml.Serialize(artifact.Program))));
        var error = Assert.Throws<InvalidDataException>(() => artifact.CreateExecutor(new StaticAdd(legacyHash)));
        Assert.Contains("source ABI", error.Message);
        Assert.NotEqual(legacyHash, artifact.ProgramHash);
    }

    [Fact]
    public void InitializesLocalScratchBeforeReadingItAndRetainsSourceMapOnFailure()
    {
        var tensor = new VmTensor(VmElementType.Float32, [2]);
        var definition = new VmDefinition("pipeline", VmDefinitionKind.Function,
            [new("scratch", VmAccess.ReadWrite, tensor), new("output", VmAccess.ReadWrite, tensor)],
            [new VmNode("fill", new VmOperator("core.tensor.fill", 1, [new("output", "scratch")],
                new Dictionary<string, string> { ["value"] = "3" })),
             new VmNode("square", new VmOperator("core.square", 1, [new("input", "scratch"), new("output", "output")]), ["fill"])]);
        var program = new VmProgram("scratch", "cpu", VmTarget.Cpu,
            [new("scratch", VmSlotScope.Local, VmAccess.ReadWrite, tensor), new("output", VmSlotScope.Local, VmAccess.ReadWrite, tensor)],
            [definition], [new("run", "pipeline", [new("scratch", "scratch"), new("output", "output")])], new VmState("none", 1, []));
        using (var executable = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(program).LoadExecutable())
        {
            var buffers = new[] { new byte[8], new byte[8] };
            executable.Invoke("run", executable.Prepare(buffers));
            Assert.Equal(new float[] { 9, 9 }, Values(buffers[1], VmElementType.Float32));
        }
        var gather = Example("gather-row", VmElementType.Float32);
        var artifact = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create()).Compile(ProgramFor("gather-row", gather.Parameters, gather.Attributes),
            new CpuVmCompilerOptions(Optimize: false));
        using var failing = artifact.LoadExecutable();
        var invalidBuffers = new[] { gather.Inputs[0], Bytes([-1], VmElementType.Int32), new byte[8] };
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => failing.Invoke("run", failing.Prepare(invalidBuffers)));
        Assert.Contains("vm/function/node", error.StackTrace);
    }

    [Fact]
    public void RejectsMalformedPortableShapesAndAttributesBeforeCompilation()
    {
        var compiler = new CpuVmCompiler(SharpInference.Runtime.DefaultInstructionCollections.Create());
        var broadcast = Example("tensor.broadcast", VmElementType.Float32);
        var invalidShape = broadcast.Parameters.Select(p => p.Name == "input"
            ? p with { Tensor = new VmTensor(VmElementType.Float32, [4]) } : p).ToArray();
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => compiler.GenerateSource(ProgramFor("tensor.broadcast", invalidShape, EmptyAttributes)));
        var fill = Example("tensor.fill", VmElementType.Float32);
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => compiler.GenerateSource(ProgramFor("tensor.fill", fill.Parameters,
            new Dictionary<string, string> { ["value"] = "NaN" })));
        var copy = Example("copy", VmElementType.Float32);
        Assert.Throws<SharpInference.Instructions.InstructionAdaptationException>(() => compiler.GenerateSource(ProgramFor("copy", copy.Parameters,
            new Dictionary<string, string> { ["unexpected"] = "value" })));
    }

    private sealed class StaticAdd(string hash) : ICpuVmCode
    {
        public string ProgramHash => hash;
        public void Invoke(string entry, CpuVmContext context) =>
            SharpInference.Backends.Cpu.CpuPrimitiveOperatorBackend.Instance.Add(
                MemoryMarshal.Cast<byte, float>(context.Read(0, 0, 8)),
                MemoryMarshal.Cast<byte, float>(context.Read(1, 0, 8)),
                MemoryMarshal.Cast<byte, float>(context.Write(2, 0, 8)));
    }

    private static VmProgram ProgramFor(string operation, VmParameter[] parameters,
        IReadOnlyDictionary<string, string> attributes, int version = 1)
    {
        var definition = new VmDefinition("function", VmDefinitionKind.Function, parameters,
            [new VmNode("node", new VmOperator("core." + operation, version,
                parameters.Select(p => new VmArgument(p.Name, p.Name)), attributes))]);
        return new VmProgram("test", "cpu", VmTarget.Cpu,
            parameters.Select(p => new VmSlot(p.Name, p.Access == VmAccess.ReadOnly ? VmSlotScope.Global : VmSlotScope.Local,
                p.Access, p.Tensor)), [definition],
            [new VmEntry("run", definition.Id, parameters.Select(p => new VmArgument(p.Name, p.Name)))],
            new VmState("none", 1, []));
    }

    private sealed record OperationExample(VmParameter[] Parameters, byte[][] Inputs, float[] Expected,
        IReadOnlyDictionary<string, string> Attributes);

    private static OperationExample Example(string operation, VmElementType type)
    {
        var parameters = new List<VmParameter>();
        var buffers = new List<byte[]>();
        var attributes = new Dictionary<string, string>();
        void Input(string name, int[] shape, float[] values, VmElementType? elementType = null)
        {
            var actualType = elementType ?? type;
            parameters.Add(new VmParameter(name, VmAccess.ReadOnly, new VmTensor(actualType, shape)));
            buffers.Add(Bytes(values, actualType));
        }
        var outputShape = new[] { 2 };
        float[] expected;
        switch (operation)
        {
            case "tensor.fill": attributes.Add("value", "2.5"); expected = [2.5f, 2.5f]; break;
            case "tensor.cast-f16-f32":
                Input("input", [2], [1.5f, 2.5f], VmElementType.Float16); expected = [1.5f, 2.5f]; break;
            case "tensor.reshape":
                Input("input", [1, 2], [1, 2]); expected = [1, 2]; break;
            case "tensor.slice":
                Input("input", [2, 3], [1, 2, 3, 4, 5, 6]);
                attributes.Add("axis", "1"); attributes.Add("start", "1"); attributes.Add("length", "2");
                outputShape = [2, 2]; expected = [2, 3, 5, 6]; break;
            case "tensor.broadcast":
                Input("input", [2, 1], [1, 2]); outputShape = [2, 3]; expected = [1, 1, 1, 2, 2, 2]; break;
            case "tensor.batched-mat-vec":
                Input("matrix", [2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]); Input("vector", [2, 2], [1, 2, 3, 4]);
                outputShape = [2, 2]; expected = [5, 11, 39, 53]; break;
            case "tensor.reduce-last-sum":
            case "tensor.reduce-last-mean":
                Input("input", [2, 2], [1, 2, 3, 4]); expected = operation.EndsWith("mean") ? [1.5f, 3.5f] : [3, 7]; break;
            case "tensor.head-outer":
                Input("left", [2, 2], [1, 2, 3, 4]); Input("right", [2, 2], [5, 6, 7, 8]);
                outputShape = [2, 2, 2]; expected = [5, 6, 10, 12, 21, 24, 28, 32]; break;
            case "mat-vec":
                Input("matrix", [2, 2], [1, 2, 3, 4]); Input("input", [2], [2, 3]); expected = [8, 18]; break;
            case "gather-row":
                Input("table", [2, 2], [1, 2, 3, 4]); Input("index", [1], [1], VmElementType.Int32); expected = [3, 4]; break;
            case "add": case "subtract": case "multiply": case "divide": case "maximum":
                Input("left", [2], [2, 4]); Input("right", [2], [1, 2]);
                expected = operation switch { "add" => [3, 6], "subtract" => [1, 2], "multiply" => [2, 8], "divide" => [2, 2], _ => [2, 4] }; break;
            default:
                Input("input", [2], [1, 2]);
                expected = operation switch
                {
                    "copy" => [1, 2], "exp" => [MathF.Exp(1), MathF.Exp(2)], "tanh" => [MathF.Tanh(1), MathF.Tanh(2)],
                    "sigmoid" => [1 / (1 + MathF.Exp(-1)), 1 / (1 + MathF.Exp(-2))],
                    "rsqrt" => [1, 1 / MathF.Sqrt(2)], "square" => [1, 4], "relu" => [1, 2],
                    "reduce-sum" => [3], "reduce-mean" => [1.5f], _ => throw new ArgumentException(operation),
                };
                if (operation.StartsWith("reduce-")) outputShape = [1];
                break;
        }
        parameters.Add(new VmParameter("output", VmAccess.ReadWrite, new VmTensor(type, outputShape)));
        return new(parameters.ToArray(), buffers.ToArray(), expected, attributes);
    }

    private static byte[] Bytes(float[] values, VmElementType type) => type switch
    {
        VmElementType.Float16 => MemoryMarshal.AsBytes(values.Select(v => (Half)v).ToArray().AsSpan()).ToArray(),
        VmElementType.Int32 => MemoryMarshal.AsBytes(values.Select(v => (int)v).ToArray().AsSpan()).ToArray(),
        _ => MemoryMarshal.AsBytes(values.AsSpan()).ToArray(),
    };

    private static float[] Values(byte[] values, VmElementType type) => type == VmElementType.Float16
        ? MemoryMarshal.Cast<byte, Half>(values).ToArray().Select(v => (float)v).ToArray()
        : MemoryMarshal.Cast<byte, float>(values).ToArray();
}
