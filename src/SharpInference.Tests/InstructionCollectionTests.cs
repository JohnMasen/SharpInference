using System.Runtime.InteropServices;
using System.Runtime.Loader;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class InstructionCollectionTests
{
    [Fact]
    public void CollectionsExposePrecisionAndExtensibleArchitecture()
    {
        var registry = new InstructionRegistry(DefaultInstructionCollections.Create());
        Assert.Equal(4, registry.QueryInstructionCollection().Count);
        Assert.Equal(2, registry.QueryInstruction(Guid.Empty, "core.add").Count);
        Assert.Equal(GraphElementType.Float16, registry.Resolve(InstructionCollectionIds.TierZeroFloat16,
            "core.add", InstructionTarget.Cpu).Signatures[0].Ports[0].ElementType);
        Assert.Equal(typeof(CpuFloat32InstructionCollection).Assembly, typeof(CpuFloat16InstructionCollection).Assembly);
        var custom = new Plugin(new InstructionTarget("cuda.future"));
        Assert.Same(custom.Value, new InstructionRegistry([custom]).Resolve(Plugin.Id, "test.double", custom.Value.Target));
        Assert.Throws<NotSupportedException>(() => new InstructionRegistry([custom]).Resolve(
            Plugin.Id, "test.double", InstructionTarget.Cpu));
        Assert.Throws<InvalidDataException>(() => new InstructionRegistry([custom, custom])
            .QueryInstruction(Plugin.Id, "test.double"));
    }

    [Fact]
    public void BackendsHaveNoConcreteInstructionReferencesOrDefaultConstructors()
    {
        foreach (var type in new[] { typeof(CpuVmCompiler), typeof(D3D12VmCompiler) })
        {
            Assert.Null(type.GetConstructor(Type.EmptyTypes));
            Assert.DoesNotContain(type.Assembly.GetReferencedAssemblies(), assembly =>
                assembly.Name is "SharpInference.Instructions.Cpu" or "SharpInference.Instructions.D3D12" or
                    "SharpInference.Instructions.TierZero" or "SharpInference.Backends.Cpu");
        }
        Assert.Throws<NotSupportedException>(() => new CpuVmCompiler([]).GenerateSource(CopyProgram(Guid.Empty, "core.copy")));
    }

    [Fact]
    public void GeneratorCatalogIsIndependentFromBackendCatalog()
    {
        var graph = TierZeroOperationTests.Example("core.add", 0).Graph();
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Cpu, [new CpuFloat32InstructionCollection()]);
        var program = generator.Generate(graph);
        var remote = VmProgramXml.Deserialize(VmProgramXml.Serialize(program));
        var backend = new CpuVmCompiler(DefaultInstructionCollections.Create());
        Assert.Single(generator.InstructionCollections.QueryInstructionCollection());
        Assert.Equal(4, backend.InstructionCollections.QueryInstructionCollection().Count);
        Assert.True(backend.Compile(remote).HasBinary);
        Assert.Throws<NotSupportedException>(() => new VmExecutionGraphGenerator(InstructionTarget.Cpu, []).Generate(graph));
        Assert.All(remote.Definitions.SelectMany(definition => definition.Nodes).Select(node => node.Instruction)
            .OfType<VmOperator>(), instruction => Assert.Equal(Guid.Empty, instruction.InstructionCollectionId));
    }

    [Theory]
    [InlineData(false, "rwkv-6")]
    [InlineData(true, "rwkv-7")]
    public void ModelMetadataSelectsGeneratorWithItsOwnCatalog(bool version7, string architecture)
    {
        using var catalog = TestModelLoader.OpenCatalog(version7 ? TestModel.Rwkv7Fp16 : TestModel.Rwkv6);
        var generator = RwkvRuntimeFactory.CreateGraphGenerator(catalog, InstructionTarget.Cpu);
        Assert.Equal(architecture, generator.ModelArchitecture);
        Assert.Equal(InstructionTarget.Cpu, generator.Architecture);
        Assert.Equal(2, generator.InstructionCollections.QueryInstructionCollection().Count);
        Assert.All(generator.InstructionCollections.QueryInstructionCollection(),
            collection => Assert.Equal(InstructionTarget.Cpu, collection.Architecture));
        Assert.Equal(VmTarget.Cpu, generator.Generate(catalog).Target);
        Assert.Throws<NotSupportedException>(() => RwkvRuntimeFactory.CreateGraphGenerator(catalog, new("vulkan")));
    }

    [Fact]
    public void PluginRecordsHelpersAndCompilesOnceWithoutTierZero()
    {
        var plugin = new Plugin(InstructionTarget.Cpu);
        var xml = VmProgramXml.Serialize(PluginProgram(VmTarget.Cpu));
        Assert.Contains($"collection=\"{Plugin.Id:D}\"", xml);
        Assert.DoesNotContain("<Operator name=\"test.double\" version=", xml);
        var artifact = new CpuVmCompiler([plugin]).Compile(VmProgramXml.Deserialize(xml));
        Assert.Equal(1, plugin.Value.Invocations);
        var directory = Path.Combine(Path.GetTempPath(), "ic-" + Guid.NewGuid().ToString("N"));
        try
        {
            artifact.Export(directory);
            using var executor = CpuVmCompiledArtifact.Load(directory).LoadExecutable();
            var buffers = new[] { Bytes([1f, -2f, 3f]), new byte[12] };
            executor.Execute("run", buffers);
            executor.Execute("run", buffers);
            Assert.Equal(new[] { 2f, -4f, 6f }, MemoryMarshal.Cast<byte, float>(buffers[1]).ToArray());
            Assert.Equal(1, plugin.Value.Invocations);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

        [Fact]
        public void PluginWithNonstandardPortNamesCompilesAndReloadsOnGpu()
        {
            var artifact = new D3D12VmCompiler([new Plugin(InstructionTarget.Direct3D12)]).Compile(PluginProgram(VmTarget.Direct3D12));
            using var stream = new MemoryStream();
            artifact.Export(stream);
            stream.Position = 0;
            using var executor = D3D12VmArtifact.Import(stream).CreateExecutor();
            executor.Upload("src", Bytes([1f, -2f, 3f]));
            executor.Execute("run");
            Assert.Equal(new[] { 2f, -4f, 6f }, MemoryMarshal.Cast<byte, float>(executor.Readback("dest")).ToArray());
        }

        [Fact]
        public void ReadWriteInputRequiresInitializationAfterArtifactReload()
        {
            var program = Program(VmTarget.Cpu, Guid.Empty, "core.copy",
                [new("input", VmAccess.ReadWrite, new(VmElementType.Float32, [3])),
                    new("output", VmAccess.ReadWrite, new(VmElementType.Float32, [3]))], new Dictionary<string, string>());
            var artifact = new CpuVmCompiler([new CpuFloat32InstructionCollection()]).Compile(program);
            var directory = Path.Combine(Path.GetTempPath(), "ic-access-" + Guid.NewGuid().ToString("N"));
            try
            {
                artifact.Export(directory);
                using var executor = CpuVmCompiledArtifact.Load(directory).LoadExecutable();
                var buffers = new[] { Bytes([1f, 2f, 3f]), new byte[12] };
                Assert.Throws<InvalidOperationException>(() => executor.Invoke("run", executor.Prepare(buffers)));
                executor.Invoke("run", executor.Prepare(buffers, ["input"]));
                Assert.Equal(buffers[0], buffers[1]);
            }
            finally { Directory.Delete(directory, recursive: true); }
        }

    [Fact]
    public void FailedAdaptationAndHelperConflictsLeaveRecorderUnchanged()
    {
        var instruction = new CpuFloat32InstructionCollection().QueryInstruction(Guid.Empty, "core.add").Single();
        var recorder = new SourceInstructionRecorder(InstructionTarget.Cpu);
        Assert.Throws<InstructionAdaptationException>(() => instruction.Invoke(recorder, []));
        Assert.Empty(recorder.Recordings);
        recorder.Record(new("first", [new("h", "helper")]));
        recorder.Record(new("second", [new("h", "helper")]));
        Assert.Single(recorder.Helpers);
        Assert.Throws<InvalidDataException>(() => recorder.Record(new("bad", [new("new", "other"), new("h", "different")])));
        Assert.Equal(2, recorder.Recordings.Count);
        Assert.Single(recorder.Helpers);
        Assert.Throws<NotSupportedException>(() => recorder.Record(new("bad", synchronization: InstructionSynchronization.GroupMemoryBarrier)));
        var gpu = new SourceInstructionRecorder(InstructionTarget.Direct3D12);
        gpu.Record(new("op;", synchronization: InstructionSynchronization.GroupMemoryBarrier));
        Assert.Contains("GroupMemoryBarrierWithGroupSync();", gpu.GetSource());
    }

    [Fact]
    public void HostCanLoadProviderAssemblyAndInjectItsInstance()
    {
        var context = new AssemblyLoadContext("ic-test", isCollectible: true);
        try
        {
            var assembly = context.LoadFromAssemblyPath(typeof(CpuFloat32InstructionCollection).Assembly.Location);
            var provider = Assert.IsAssignableFrom<IInstructionCollectionProvider>(Activator.CreateInstance(
                assembly.GetType(typeof(CpuFloat32InstructionCollection).FullName!, throwOnError: true)!));
            Assert.NotSame(typeof(CpuFloat32InstructionCollection).Assembly, provider.GetType().Assembly);
            using var executor = new CpuVmCompiler([provider]).Compile(CopyProgram(Guid.Empty, "core.copy")).LoadExecutable();
            var buffers = new[] { Bytes([1f, 2f, 3f]), new byte[12] };
            executor.Execute("run", buffers);
            Assert.Equal(buffers[0], buffers[1]);
        }
        finally { context.Unload(); }
    }

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void PluginCanDeclareReadWriteState(VmTarget target)
    {
        var program = Program(target, Plugin.Id, "test.increment",
            [new("state", VmAccess.ReadWrite, new(VmElementType.Float32, [3]))], new Dictionary<string, string>());
        var architecture = target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12;
        var plugin = new Plugin(architecture);
        byte[] result;
        if (target == VmTarget.Cpu)
        {
            using var executor = new CpuVmCompiler([plugin]).Compile(program).CreateExecutor();
            var buffers = new[] { Bytes([1f, 2f, 3f]) };
            executor.Invoke("run", executor.Prepare(buffers, ["state"]));
            result = buffers[0];
        }
        else
        {
            using var executor = new D3D12VmCompiler([plugin]).Compile(program).CreateExecutor();
            executor.Upload("state", Bytes([1f, 2f, 3f]));
            executor.Execute("run");
            result = executor.Readback("state");
        }
        Assert.Equal(new[] { 2f, 3f, 4f }, MemoryMarshal.Cast<byte, float>(result).ToArray());
    }

    public static IEnumerable<object[]> HalfOperations() =>
        TierZeroOperationContracts.Contracts.Where(contract => contract.Operation.Name != "core.tensor.cast-f16-f32")
            .Select(contract => new object[] { contract.Operation.Name });

    [Theory]
    [MemberData(nameof(HalfOperations))]
    public void HalfCollectionExecutesEveryInstructionOnCpuAndGpu(string operation)
    {
        var example = TierZeroOperationTests.Example(operation, 0);
        var parameters = example.Inputs.Select(input => new VmParameter(input.Port, VmAccess.ReadOnly,
            new(input.Tensor.ElementType == GraphElementType.Int32 ? VmElementType.Int32 : VmElementType.Float16,
                input.Tensor.Dimensions))).Append(new("output", VmAccess.ReadWrite,
                new(VmElementType.Float16, example.Output.Dimensions))).ToArray();
        var inputs = example.Inputs.Select(input => input.Tensor.ElementType == GraphElementType.Int32
            ? MemoryMarshal.AsBytes(input.Values.Select(value => (int)value).ToArray().AsSpan()).ToArray()
            : MemoryMarshal.AsBytes(input.Values.Select(value => (Half)value).ToArray().AsSpan()).ToArray()).ToArray();
        foreach (var target in new[] { VmTarget.Cpu, VmTarget.Direct3D12 })
        {
            var program = Program(target, InstructionCollectionIds.TierZeroFloat16, operation, parameters, example.Attributes);
            byte[] result;
            if (target == VmTarget.Cpu)
            {
                using var executor = new CpuVmCompiler(DefaultInstructionCollections.Create()).Compile(program).LoadExecutable();
                var buffers = inputs.Append(new byte[(int)parameters[^1].Tensor.ByteLength]).ToArray();
                executor.Execute("run", buffers);
                result = buffers[^1];
            }
            else
            {
                using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create()).Compile(program).CreateExecutor();
                for (var i = 0; i < inputs.Length; i++) executor.Upload(parameters[i].Name, inputs[i]);
                for (var i = 0; i < 3; i++) executor.Execute("run");
                result = executor.Readback("output");
            }
            var values = MemoryMarshal.Cast<byte, Half>(result).ToArray();
            Assert.Equal(example.Expected.Length, values.Length);
            for (var i = 0; i < values.Length; i++)
                Assert.True(MathF.Abs((float)values[i] - example.Expected[i]) <= .002f + MathF.Abs(example.Expected[i]) * .002f,
                    $"{target}/{operation}[{i}]: expected {example.Expected[i]}, actual {values[i]}");
        }
    }

    private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static VmProgram CopyProgram(Guid id, string name) => Program(VmTarget.Cpu, id, name,
        [new("input", VmAccess.ReadOnly, new(VmElementType.Float32, [3])),
            new("output", VmAccess.ReadWrite, new(VmElementType.Float32, [3]))], new Dictionary<string, string>());
    private static VmProgram PluginProgram(VmTarget target) => Program(target, Plugin.Id, "test.double",
        [new("src", VmAccess.ReadOnly, new(VmElementType.Float32, [3])),
            new("dest", VmAccess.ReadWrite, new(VmElementType.Float32, [3]))], new Dictionary<string, string>());

    private static VmProgram Program(VmTarget target, Guid id, string name, VmParameter[] parameters,
        IReadOnlyDictionary<string, string> attributes)
    {
        var arguments = parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var kernel = new VmDefinition("op", target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Kernel,
            parameters, [new("body", new VmOperator(id, name, arguments, attributes))], target == VmTarget.Cpu ? null : new(64));
        var definitions = new List<VmDefinition> { kernel };
        if (target != VmTarget.Cpu)
            definitions.Add(new("host", VmDefinitionKind.Orchestration, parameters,
                [new("dispatch", new VmDispatch("op", arguments, new(2)))]));
        return new("ic-test", "ic-test", target, parameters.Select(parameter =>
            new VmSlot(parameter.Name, VmSlotScope.Local, parameter.Access, parameter.Tensor)), definitions,
            [new("run", target == VmTarget.Cpu ? "op" : "host", arguments)], new("none", 1, []));
    }

    private sealed class Plugin(InstructionTarget target) : IInstructionCollectionProvider
    {
        internal static readonly Guid Id = new("5d3483d5-e65d-4343-9aba-33049145e892");
        internal DoubleInstruction Value { get; } = new(target);
        private IncrementInstruction State { get; } = new(target);
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => [new(Id, "test", 1, target)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string name) =>
            collectionId != Id ? [] : name == Value.Name ? [Value] : name == State.Name ? [State] : [];
    }

    private sealed class IncrementInstruction(InstructionTarget target) : Instruction
    {
        public override Guid CollectionId => Plugin.Id;
        public override string Name => "test.increment";
        public override InstructionTarget Target => target;
        public override bool RequiresDispatchIsolation => false;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("state", GraphElementType.Float32, GraphResourceAccess.ReadWrite)], [])];
        public override void Invoke(IInstructionRecorder recorder, InstructionParameter[] parameters)
        {
            var state = parameters.OfType<InstructionTensorParameter>().Single();
            recorder.Record(new(Target == InstructionTarget.Cpu
                ? $"{{ var values=MemoryMarshal.Cast<byte,float>({state.Expression}); for(int i=0;i<values.Length;i++) values[i]+=1; }}"
                : $"if(i<3u) {state.Expression}.Store({state.OffsetExpression}+i*4u,asuint(asfloat({state.Expression}.Load({state.OffsetExpression}+i*4u))+1.0f));"));
        }
    }

    private sealed class DoubleInstruction(InstructionTarget target) : Instruction
    {
        public override Guid CollectionId => Plugin.Id;
        public override string Name => "test.double";
        public override InstructionTarget Target => target;
        public int Invocations { get; private set; }
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("src", GraphElementType.Float32, GraphResourceAccess.Read),
                new("dest", GraphElementType.Float32, GraphResourceAccess.Write)], [])];
        public override void Invoke(IInstructionRecorder recorder, InstructionParameter[] parameters)
        {
            var input = parameters.OfType<InstructionTensorParameter>().Single(parameter => parameter.Name == "src");
            var output = parameters.OfType<InstructionTensorParameter>().Single(parameter => parameter.Name == "dest");
            if (Target == InstructionTarget.Cpu)
                recorder.Record(new($"Double({input.Expression}, {output.Expression});",
                    [new("test.double", "private static void Double(ReadOnlySpan<byte> input, Span<byte> output) { var a=MemoryMarshal.Cast<byte,float>(input); var b=MemoryMarshal.Cast<byte,float>(output); for(int i=0;i<a.Length;i++) b[i]=a[i]*2; }")]));
            else if (Target == InstructionTarget.Direct3D12)
                recorder.Record(new($"if(i<3u) {output.Expression}.Store({output.OffsetExpression}+i*4u, asuint(asfloat({input.Expression}.Load({input.OffsetExpression}+i*4u))*2.0f));"));
            else throw new InstructionAdaptationException(CollectionId, Name, "No test emitter for this architecture.");
            Invocations++;
        }
    }
}
