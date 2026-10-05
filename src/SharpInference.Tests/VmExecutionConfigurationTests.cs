using System.Runtime.InteropServices;
using System.Xml.Linq;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class VmExecutionConfigurationTests
{
    private static readonly PrecisionRequirement Precision = new(GraphElementType.Float32, GraphElementType.Float32);

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void ConfigurationSurvivesBindingStrippingAndXml(VmTarget target)
    {
        var provider = new ConfiguredProvider(target);
        var program = Program(target, Configuration(target));
        var bound = VmInstructionContracts.Bind(program, new([provider]));
        var xml = VmProgramXml.Serialize(bound);
        Assert.Contains("<ExecutionConfiguration", xml);
        Assert.DoesNotContain("name=\"implementation\"", xml);
        var restored = VmProgramXml.Deserialize(xml);
        Assert.Equal(Operation(program).ExecutionConfiguration, Operation(restored).ExecutionConfiguration);
        Assert.Equal(xml, VmProgramXml.Serialize(restored));
        Assert.Equal(VmProgramXml.Serialize(program),
            VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(restored)));
        VmInstructionContracts.ValidateResolved(restored);
    }

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void InvalidSelectionsFailBeforeGenerationOrRecording(VmTarget target)
    {
        var provider = new ConfiguredProvider(target);
        var program = Program(target, Configuration(target));
        var parameters = VmInstructionParameters.Create(Operation(program), program.Definitions[0].Parameters,
            parameter => (parameter.Name, "0"));
        var recorder = new SourceInstructionRecorder(provider.Value.Target);
        foreach (var configuration in new InstructionExecutionConfiguration?[]
                 { null, Configuration(target, "missing"), Configuration(target == VmTarget.Cpu ? VmTarget.Direct3D12 : VmTarget.Cpu) })
            Assert.Throws<InstructionAdaptationException>(() =>
                provider.Value.Invoke(recorder, parameters, Precision, configuration));
        Assert.Empty(recorder.Recordings);
        Assert.Equal(0, provider.Value.Generated);
        provider.Value.Invoke(recorder, parameters, Precision, Configuration(target));
        Assert.Single(recorder.Recordings);
        Assert.Equal(1, provider.Value.Generated);
    }

    [Fact]
    public void TierZeroKeepsItsUnconfiguredContract()
    {
        var instruction = new CpuFloat32InstructionCollection().QueryInstruction(Guid.Empty, "core.copy").Single();
        var program = Program(VmTarget.Cpu, null);
        var parameters = VmInstructionParameters.Create(Operation(program), program.Definitions[0].Parameters,
            parameter => (parameter.Name, "0"));
        var recorder = new SourceInstructionRecorder(InstructionTarget.Cpu);
        Assert.Throws<InstructionAdaptationException>(() =>
            instruction.Invoke(recorder, parameters, Precision, Configuration(VmTarget.Cpu)));
        Assert.Empty(recorder.Recordings);
        instruction.Invoke(recorder, parameters, Precision);
        Assert.Single(recorder.Recordings);
    }

    [Fact]
    public void ConfigurationValuesAreImmutableAndIdentifiersAreValidated()
    {
        var configuration = Configuration(VmTarget.Cpu);
        Assert.Equal(configuration, Configuration(VmTarget.Cpu));
        Assert.NotEqual(configuration, Configuration(VmTarget.Direct3D12));
        Assert.Null(typeof(InstructionExecutionConfiguration).GetProperty("Implementation")!.SetMethod);
        Assert.Null(typeof(InstructionImplementationId).GetProperty("Value")!.SetMethod);
        Assert.Throws<ArgumentNullException>(() => new CpuInstructionExecutionConfiguration(null!));
        foreach (var value in new[] { "", " ", "bad value", "bad/name", new string('a', 129) })
            Assert.Throws<ArgumentException>(() => new InstructionImplementationId(value));
    }

    [Fact]
    public void XmlRejectsUnknownDuplicateAndCrossTargetConfiguration()
    {
        var xml = VmProgramXml.Serialize(Program(VmTarget.Cpu, Configuration(VmTarget.Cpu)));
        foreach (var mutation in new Action<XElement>[]
                 {
                     element => element.SetAttributeValue("target", "unknown"),
                     element => element.SetAttributeValue("width", "256"),
                     element => element.SetAttributeValue("implementation", "bad value"),
                     element => element.Add(new XElement("Unknown")),
                     element => element.Parent!.Add(new XElement(element)),
                 })
        {
            var document = XElement.Parse(xml);
            mutation(document.Descendants("ExecutionConfiguration").Single());
            Assert.Throws<InvalidDataException>(() => VmProgramXml.Deserialize(document.ToString()));
        }
        Assert.Throws<InvalidDataException>(() => Program(VmTarget.Cpu, Configuration(VmTarget.Direct3D12)));
    }

    [Fact]
    public void CpuSourceAndPublishedArtifactConsumeTheSelectedConfiguration()
    {
        var provider = new ConfiguredProvider(VmTarget.Cpu);
        var artifact = new CpuVmCompiler([provider]).Compile(Program(VmTarget.Cpu, Configuration(VmTarget.Cpu)));
        var directory = Path.Combine(Path.GetTempPath(), "t1-config-" + Guid.NewGuid().ToString("N"));
        try
        {
            artifact.Export(directory);
            var restored = CpuVmCompiledArtifact.Load(directory);
            Assert.Equal(Configuration(VmTarget.Cpu), Operation(restored.Program).ExecutionConfiguration);
            using var executor = restored.LoadExecutable();
            var input = MemoryMarshal.AsBytes(new float[] { 1, -2, 3 }.AsSpan()).ToArray();
            var output = new byte[input.Length];
            executor.Execute("run", [input, output]);
            Assert.Equal(input, output);
            Assert.Equal(1, provider.Value.Generated);
        }
        finally { Directory.Delete(directory, recursive: true); }
    }

    [Fact]
    public void GpuSourceAndArtifactReloadKeepTheSelectedConfiguration()
    {
        var provider = new ConfiguredProvider(VmTarget.Direct3D12);
        var artifact = new D3D12VmCompiler([provider]).Compile(Program(VmTarget.Direct3D12,
            Configuration(VmTarget.Direct3D12)));
        using var stream = new MemoryStream();
        artifact.Export(stream);
        stream.Position = 0;
        var restored = D3D12VmArtifact.Import(stream);
        Assert.Equal(Configuration(VmTarget.Direct3D12), Operation(restored.Program).ExecutionConfiguration);
        using var executor = restored.CreateExecutor();
        var input = MemoryMarshal.AsBytes(new float[] { 1, -2, 3 }.AsSpan()).ToArray();
        executor.Upload("input", input);
        executor.Execute("run");
        Assert.Equal(input, executor.Readback("output"));
    }

    private static InstructionExecutionConfiguration Configuration(VmTarget target, string implementation = "copy") =>
        target == VmTarget.Cpu ? new CpuInstructionExecutionConfiguration(new(implementation)) :
            new D3D12InstructionExecutionConfiguration(new(implementation));

    private static VmOperator Operation(VmProgram program) =>
        (VmOperator)program.Definitions[0].Nodes[0].Instruction;

    private static VmProgram Program(VmTarget target, InstructionExecutionConfiguration? configuration)
    {
        var tensor = new VmTensor(VmElementType.Float32, [3]);
        VmParameter[] parameters = [new("input", VmAccess.ReadOnly, tensor), new("output", VmAccess.ReadWrite, tensor)];
        VmArgument[] arguments = [new("input", "input"), new("output", "output")];
        var body = new VmDefinition("body", target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Kernel,
            parameters, [new("copy", new VmOperator(Precision, ConfiguredProvider.Id, "test.configured-copy",
                arguments, executionConfiguration: configuration))], target == VmTarget.Cpu ? null : new(64));
        var root = new VmDefinition("root", VmDefinitionKind.Orchestration, parameters,
            [new("call", target == VmTarget.Cpu ? new VmCall("body", arguments) :
                new VmDispatch("body", arguments, new(1)))]);
        return new("configured", "test", target,
            [new("input", VmSlotScope.Local, VmAccess.ReadOnly, tensor),
                new("output", VmSlotScope.Local, VmAccess.ReadWrite, tensor)],
            [body, root], [new("run", "root", arguments)], new("empty", 1, []));
    }

    private sealed class ConfiguredProvider : IInstructionCollectionProvider
    {
        public static readonly Guid Id = new("a148a3e1-c60d-4862-b3f4-6124be452d70");
        public ConfiguredProvider(VmTarget target) => Value = new(target);
        public ConfiguredInstruction Value { get; }
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(Id, "Configured test", 1, Value.Target)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            collectionId == Id && instructionName == Value.Name ? [Value] : [];
    }

    private sealed class ConfiguredInstruction(VmTarget target) : Instruction
    {
        public int Generated { get; private set; }
        public override Guid CollectionId => ConfiguredProvider.Id;
        public override string Name => "test.configured-copy";
        public override InstructionTarget Target =>
            target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12;
        public override bool RequiresDispatchIsolation => false;
        public override IReadOnlyList<InstructionSignature> Signatures { get; } =
            [new([new("input", GraphElementType.Float32, GraphResourceAccess.Read),
                new("output", GraphElementType.Float32, GraphResourceAccess.Write)], [],
                new(GraphElementType.Float32, GraphElementType.Float32))];

        protected override void ValidateExecutionConfiguration(InstructionExecutionConfiguration? configuration)
        {
            if (configuration is null || configuration.Implementation.Value != "copy")
                throw new InstructionAdaptationException(CollectionId, Name, "Unsupported execution configuration.");
        }

        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            Generated++;
            var input = parameters.OfType<InstructionTensorParameter>().Single(parameter => parameter.Name == "input");
            var output = parameters.OfType<InstructionTensorParameter>().Single(parameter => parameter.Name == "output");
            return new(target == VmTarget.Cpu ? $"{input.Expression}.CopyTo({output.Expression});" :
                $"if(i<3u) {output.Expression}.Store({output.OffsetExpression}+i*4u,{input.Expression}.Load({input.OffsetExpression}+i*4u));");
        }
    }
}
