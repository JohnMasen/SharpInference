using System.Runtime.InteropServices;
using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using System.Xml.Linq;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class VmPrecisionTests
{
    private static readonly PrecisionRequirement F32 = new(GraphElementType.Float32, GraphElementType.Float32);
    private static readonly PrecisionRequirement F16 = new(GraphElementType.Float16, GraphElementType.Float16);
    private static readonly KernelPrecisionProfile Provided32 = new(GraphElementType.Float32, GraphElementType.Float32);

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void HalfStorageAndMixedWeightsAdvertiseIndependentFp32InternalPrecision(VmTarget target)
    {
        var registry = new InstructionRegistry(DefaultInstructionCollections.Create());
        foreach (var mixed in new[] { false, true })
        {
            var program = Program(target, F32, mixed);
            var bound = VmInstructionContracts.Bind(program, registry);
            var operation = Operator(bound);
            Assert.Equal(F32, operation.Precision);
            Assert.Equal(Provided32, operation.ResolvedPrecision);
            Assert.Contains(program.Definitions[0].Parameters, parameter => parameter.Tensor.ElementType == VmElementType.Float16);
            Assert.Equal(mixed ? VmElementType.Float32 : VmElementType.Float16,
                program.Definitions[0].Parameters.Single(parameter => parameter.Name == "output").Tensor.ElementType);
            var restored = VmProgramXml.Deserialize(VmProgramXml.Serialize(bound));
            Assert.Equal(Provided32, Operator(restored).ResolvedPrecision);
            Assert.Equal(F32, Operator(restored).Precision);
            Assert.Equal(VmProgramXml.Serialize(program),
                VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(restored)));
            VmInstructionContracts.ValidateResolved(restored);
        }
    }

    [Theory]
    [InlineData(GraphElementType.Float16, GraphElementType.Float32)]
    [InlineData(GraphElementType.Float32, GraphElementType.Float16)]
    public void InsufficientArithmeticOrAccumulatorFailsBeforeRecordingOnBothTargets(
        GraphElementType arithmetic, GraphElementType accumulator)
    {
        foreach (var target in new[] { VmTarget.Cpu, VmTarget.Direct3D12 })
        {
            var provider = new PrecisionProvider(target, new(arithmetic, accumulator));
            var program = Program(target, F32, collection: PrecisionProvider.Id);
            var recorder = new SourceInstructionRecorder(provider.Value.Target);
            var parameters = VmInstructionParameters.Create(Operator(program), program.Definitions[0].Parameters,
                parameter => (parameter.Name, "0"));
            var error = Assert.Throws<InstructionAdaptationException>(() => provider.Value.Invoke(recorder, parameters, F32));
            Assert.Contains("required arithmetic=Float32, accumulator=Float32", error.Message);
            Assert.Empty(recorder.Recordings);
            Assert.Equal(0, provider.Value.Generated);
            error = Assert.Throws<InstructionAdaptationException>(() =>
                VmInstructionContracts.Bind(program, new([provider])));
            Assert.Contains("op/body", error.Message);
            Assert.Throws<InstructionAdaptationException>(() =>
            {
                if (target == VmTarget.Cpu) new CpuVmCompiler([provider]).GenerateSource(program);
                else new D3D12VmCompiler([provider]).GenerateSources(program);
            });
            Assert.Equal(0, provider.Value.Generated);
        }
    }

    [Fact]
    public void HigherProvidedPrecisionMeetsLowerRequirementButMissingTypesAndAmbiguityDoNot()
    {
        var program = Program(VmTarget.Cpu, F16);
        var parameters = VmInstructionParameters.Create(Operator(program), program.Definitions[0].Parameters,
            parameter => (parameter.Name, "0"));
        var provider = new PrecisionProvider(VmTarget.Cpu, Provided32);
        Assert.Equal(Provided32, provider.Value.GetSignature(parameters, F16).Precision);
        var recorder = new SourceInstructionRecorder(InstructionTarget.Cpu);
        provider.Value.Invoke(recorder, parameters, F16);
        Assert.Single(recorder.Recordings);
        var unsupported = parameters.Select(parameter => parameter is InstructionTensorParameter tensor && tensor.Name == "input"
            ? tensor with { Tensor = new(GraphElementType.Float32, tensor.Tensor.Dimensions) } : parameter).ToArray();
        Assert.Throws<InstructionAdaptationException>(() => provider.Value.Invoke(recorder, unsupported, F16));
        Assert.Single(recorder.Recordings);
        var ambiguous = new PrecisionProvider(VmTarget.Cpu, Provided32, duplicate: true);
        Assert.Throws<InstructionAdaptationException>(() => ambiguous.Value.Invoke(recorder, parameters, F16));
        Assert.Single(recorder.Recordings);
        Assert.Equal(0, ambiguous.Value.Generated);
    }

    [Fact]
    public void SignaturesSnapshotCollectionsAndGenerationFailureIsAtomic()
    {
        var ports = new List<InstructionPort> { new("input", GraphElementType.Float16, GraphResourceAccess.Read) };
        var attributes = new List<string> { "value" };
        var signature = new InstructionSignature(ports, attributes, Provided32);
        ports.Clear();
        attributes.Clear();
        Assert.Single(signature.Ports);
        Assert.Single(signature.Attributes);

        var provider = new PrecisionProvider(VmTarget.Cpu, Provided32, failGeneration: true);
        var program = Program(VmTarget.Cpu, F32, collection: PrecisionProvider.Id);
        var parameters = VmInstructionParameters.Create(Operator(program), program.Definitions[0].Parameters,
            parameter => (parameter.Name, "0"));
        var recorder = new SourceInstructionRecorder(InstructionTarget.Cpu);
        Assert.Throws<InvalidDataException>(() => provider.Value.Invoke(recorder, parameters, F32));
        Assert.Empty(recorder.Recordings);
        Assert.Empty(recorder.Helpers);
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("duplicate")]
    [InlineData("missing-attribute")]
    [InlineData("unknown-attribute")]
    [InlineData("unknown-type")]
    [InlineData("integer-type")]
    [InlineData("child")]
    [InlineData("unexpected-version")]
    public void XmlRequiresExplicitStrictPrecision(string mutation)
    {
        var xml = XElement.Parse(VmProgramXml.Serialize(Program(VmTarget.Cpu, F32)));
        var precision = xml.Descendants("Precision").Single();
        switch (mutation)
        {
            case "missing": precision.Remove(); break;
            case "duplicate": precision.AddAfterSelf(new XElement(precision)); break;
            case "missing-attribute": precision.Attribute("minimumAccumulatorType")!.Remove(); break;
            case "unknown-attribute": precision.SetAttributeValue("fast", "true"); break;
            case "unknown-type": precision.SetAttributeValue("minimumArithmeticType", "Unknown"); break;
            case "integer-type": precision.SetAttributeValue("minimumArithmeticType", "Int32"); break;
            case "child": precision.Add(new XElement("Extra")); break;
            case "unexpected-version": xml.SetAttributeValue("version", "3"); break;
        }
        Assert.Throws<InvalidDataException>(() => VmProgramXml.Deserialize(xml.ToString()));
    }

    [Fact]
    public void BoundPrecisionConflictsAndInsufficientSerializedCapabilitiesAreRejected()
    {
        var registry = new InstructionRegistry(DefaultInstructionCollections.Create());
        var bound = VmInstructionContracts.Bind(Program(VmTarget.Cpu, F16), registry);
        Assert.Same(bound, VmInstructionContracts.Bind(bound, registry));
        var xml = XElement.Parse(VmProgramXml.Serialize(bound));
        xml.Descendants("ResolvedPrecision").Single().SetAttributeValue("arithmeticType", "Float16");
        var conflicting = VmProgramXml.Deserialize(xml.ToString());
        Assert.Throws<InvalidDataException>(() => VmInstructionContracts.Bind(conflicting, registry));
        xml.Descendants("Precision").Single().SetAttributeValue("minimumArithmeticType", "Float32");
        Assert.Throws<InvalidDataException>(() => VmProgramXml.Deserialize(xml.ToString()));
        Assert.Throws<ArgumentNullException>(() =>
            new VmOperator(null!, Guid.Empty, "core.copy", []));
        Assert.Throws<ArgumentException>(() => new PrecisionRequirement(GraphElementType.Int32, GraphElementType.Float32));
        Assert.Throws<ArgumentNullException>(() => new InstructionSignature([], [], null!));
    }

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void LoweringPreservesRequirementsAndSeparatesDefinitionKeys(VmTarget target)
    {
        var original = TierZeroOperationTests.Example("core.add", 0).Graph();
        var output = original.Resources.Single(resource => resource.Id == new ResourceId("output"));
        var second = original.Nodes[0] with
        {
            Id = new("second"),
            Requirements = F16,
            Resources = original.Nodes[0].Resources.Select(binding => binding.Port == "output"
                ? binding with { Resource = new("second-output") } : binding).ToArray(),
        };
        var logical = new LogicalGraph(original.Identity, original.Model,
            original.Resources.Append(output with { Id = new("second-output"), Name = "second-output" }),
            original.Regions, original.Nodes.Append(second), original.Inputs,
            original.Outputs.Append(new("second-output")));
        var program = new VmExecutionGraphGenerator(
            target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12,
            DefaultInstructionCollections.Create()).Generate(logical, new(PrefillCapacity: 1, WeightViews: false));
        var operations = program.Definitions.SelectMany(definition => definition.Nodes)
            .Select(node => node.Instruction).OfType<VmOperator>().ToArray();
        Assert.Equal(2, operations.Length);
        Assert.Equal(F32, operations[0].Precision);
        Assert.Equal(F16, operations[1].Precision);
        Assert.All(operations, operation => Assert.Equal(Provided32, operation.ResolvedPrecision));
    }

    [Fact]
    public void PrecisionChangesProgramHashAndBoundCapabilityChangesContractXml()
    {
        var compiler = new CpuVmCompiler(DefaultInstructionCollections.Create());
        Assert.NotEqual(compiler.GenerateSource(Program(VmTarget.Cpu, F32)).ProgramHash,
            compiler.GenerateSource(Program(VmTarget.Cpu, F16)).ProgramHash);
        var program = Program(VmTarget.Cpu, F16, collection: PrecisionProvider.Id);
        var provided16 = new KernelPrecisionProfile(GraphElementType.Float16, GraphElementType.Float16);
        var low = VmInstructionContracts.Bind(program, new([new PrecisionProvider(VmTarget.Cpu, provided16)]));
        var high = VmInstructionContracts.Bind(program, new([new PrecisionProvider(VmTarget.Cpu, Provided32)]));
        Assert.NotEqual(VmProgramXml.Serialize(low), VmProgramXml.Serialize(high));
        Assert.Equal(VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(low)),
            VmProgramXml.Serialize(VmInstructionContracts.WithoutContracts(high)));
    }

    [Theory]
    [InlineData(VmTarget.Cpu, false)]
    [InlineData(VmTarget.Cpu, true)]
    [InlineData(VmTarget.Direct3D12, false)]
    [InlineData(VmTarget.Direct3D12, true)]
    public void ArtifactsPreservePrecisionAndExecuteHalfOrMixedSignatures(VmTarget target, bool mixed)
    {
        var program = Program(target, F32, mixed);
        byte[][] buffers = mixed
            ? [HalfBytes([1, 2, 3, 4, 5, 6]), FloatBytes([1, 2, 3]), new byte[8]]
            : [HalfBytes([1, -2, 3]), new byte[6]];
        if (target == VmTarget.Cpu)
        {
            var directory = Path.Combine(Path.GetTempPath(), "vm-precision-" + Guid.NewGuid().ToString("N"));
            try
            {
                new CpuVmCompiler(DefaultInstructionCollections.Create()).Compile(program).Export(directory);
                var artifact = CpuVmCompiledArtifact.Load(directory);
                Assert.Equal(F32, Operator(artifact.Program).Precision);
                var contract = VmProgramXml.Deserialize(File.ReadAllText(Path.Combine(directory, "contracts.xml")));
                Assert.Equal(Provided32, Operator(contract).ResolvedPrecision);
                using var executable = artifact.LoadExecutable();
                executable.Execute("run", buffers);
            }
            finally { if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true); }
        }
        else
        {
            var artifact = new D3D12VmCompiler(DefaultInstructionCollections.Create()).Compile(program);
            using var stream = new MemoryStream();
            artifact.Export(stream);
            using (var archive = new ZipArchive(stream, ZipArchiveMode.Read, leaveOpen: true))
            {
                var contract = VmProgramXml.Deserialize(Encoding.UTF8.GetString(ReadEntry(archive, "contracts.xml")));
                Assert.Equal(Provided32, Operator(contract).ResolvedPrecision);
            }
            stream.Position = 0;
            var restored = D3D12VmArtifact.Import(stream);
            Assert.Equal(F32, Operator(restored.Program).Precision);
            using var executable = restored.CreateExecutor();
            executable.Execute("run", buffers);
        }
        if (mixed) Assert.Equal(new[] { 14f, 32f }, MemoryMarshal.Cast<byte, float>(buffers[^1]).ToArray());
        else Assert.Equal(new Half[] { (Half)1, (Half)(-2), (Half)3 }, MemoryMarshal.Cast<byte, Half>(buffers[^1]).ToArray());
    }

    [Fact]
    public void CpuPackageRejectsMissingResolvedPrecisionEvenWithRecomputedIntegrityHashes()
    {
        var directory = Path.Combine(Path.GetTempPath(), "vm-precision-missing-" + Guid.NewGuid().ToString("N"));
        try
        {
            new CpuVmCompiler(DefaultInstructionCollections.Create()).GenerateSource(Program(VmTarget.Cpu, F32))
                .Export(directory, includeBinary: false);
            var files = Directory.GetFiles(directory).ToDictionary(path =>
                Path.GetFileName(path) ?? throw new InvalidDataException("Missing package filename."),
                File.ReadAllBytes, StringComparer.Ordinal);
            var contracts = XElement.Parse(Encoding.UTF8.GetString(files["contracts.xml"]));
            contracts.Descendants("ResolvedPrecision").Single().Remove();
            files["contracts.xml"] = Encoding.UTF8.GetBytes(contracts.ToString());
            var manifest = JsonNode.Parse(files["manifest.json"])!;
            manifest["Files"]!["contracts.xml"] = Convert.ToHexString(SHA256.HashData(files["contracts.xml"]));
            files["manifest.json"] = Encoding.UTF8.GetBytes(manifest.ToJsonString());
            Assert.Throws<InvalidDataException>(() => CpuVmCompiledArtifact.LoadFromResources(files));
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GpuPackageRejectsMissingOrInsufficientResolvedPrecisionWithRecomputedHashes(bool insufficient)
    {
        var artifact = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(Program(VmTarget.Direct3D12, F32));
        using var stream = new MemoryStream();
        artifact.Export(stream);
        stream.Position = 0;
        using (var archive = new ZipArchive(stream, ZipArchiveMode.Update, leaveOpen: true))
        {
            var contracts = XElement.Parse(Encoding.UTF8.GetString(ReadEntry(archive, "contracts.xml")));
            if (insufficient)
                contracts.Descendants("ResolvedPrecision").Single().SetAttributeValue("accumulatorType", "Float16");
            else contracts.Descendants("ResolvedPrecision").Single().Remove();
            var contractBytes = Encoding.UTF8.GetBytes(contracts.ToString());
            ReplaceEntry(archive, "contracts.xml", contractBytes);
            var envelope = JsonNode.Parse(ReadEntry(archive, "manifest.json"))!;
            var manifest = JsonNode.Parse(envelope["Manifest"]!.GetValue<string>())!;
            manifest["ContractsHash"] = Convert.ToHexString(SHA256.HashData(contractBytes));
            var manifestText = manifest.ToJsonString();
            envelope["Manifest"] = manifestText;
            envelope["Sha256"] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(manifestText)));
            ReplaceEntry(archive, "manifest.json", Encoding.UTF8.GetBytes(envelope.ToJsonString()));
        }
        stream.Position = 0;
        Assert.Throws<InvalidDataException>(() => D3D12VmArtifact.Import(stream));
    }

    private static byte[] ReadEntry(ZipArchive archive, string name)
    {
        using var entry = (archive.GetEntry(name) ?? throw new InvalidDataException($"Missing {name}.")).Open();
        using var bytes = new MemoryStream();
        entry.CopyTo(bytes);
        return bytes.ToArray();
    }

    private static void ReplaceEntry(ZipArchive archive, string name, byte[] bytes)
    {
        (archive.GetEntry(name) ?? throw new InvalidDataException($"Missing {name}.")).Delete();
        using var entry = archive.CreateEntry(name).Open();
        entry.Write(bytes);
    }

    private static byte[] HalfBytes(float[] values) =>
        MemoryMarshal.AsBytes(values.Select(value => (Half)value).ToArray().AsSpan()).ToArray();
    private static byte[] FloatBytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    private static VmOperator Operator(VmProgram program) =>
        (VmOperator)program.Definitions[0].Nodes[0].Instruction;

    private static VmProgram Program(VmTarget target, PrecisionRequirement precision, bool mixed = false, Guid? collection = null)
    {
        VmParameter[] parameters = mixed
            ? [new("matrix", VmAccess.ReadOnly, new(VmElementType.Float16, [2, 3])),
                new("input", VmAccess.ReadOnly, new(VmElementType.Float32, [3])),
                new("output", VmAccess.ReadWrite, new(VmElementType.Float32, [2]))]
            : [new("input", VmAccess.ReadOnly, new(VmElementType.Float16, [3])),
                new("output", VmAccess.ReadWrite, new(VmElementType.Float16, [3]))];
        var arguments = parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var operation = new VmOperator(precision, collection ?? (mixed
            ? InstructionCollectionIds.TierZeroFloat32 : InstructionCollectionIds.TierZeroFloat16),
            collection.HasValue ? "test.precision" : mixed ? "core.mat-vec" : "core.copy", arguments);
        var definitions = new List<VmDefinition>
        {
            new("op", target == VmTarget.Cpu ? VmDefinitionKind.Function : VmDefinitionKind.Kernel,
                parameters, [new("body", operation)], target == VmTarget.Cpu ? null : new(64)),
        };
        if (target == VmTarget.Direct3D12)
            definitions.Add(new("host", VmDefinitionKind.Orchestration, parameters,
                [new("dispatch", new VmDispatch("op", arguments, new(1)))]));
        return new("precision-test", "precision-test", target, parameters.Select(parameter =>
            new VmSlot(parameter.Name, VmSlotScope.Local, parameter.Access, parameter.Tensor)),
            definitions, [new("run", target == VmTarget.Cpu ? "op" : "host", arguments)], new("none", 1, []));
    }

    private sealed class PrecisionProvider : IInstructionCollectionProvider
    {
        internal static readonly Guid Id = new("19c6067b-73a0-4510-9bf2-e64a6974f0dc");
        internal PrecisionInstruction Value { get; }
        internal PrecisionProvider(VmTarget target, KernelPrecisionProfile precision, bool duplicate = false, bool failGeneration = false) =>
            Value = new(target == VmTarget.Cpu ? InstructionTarget.Cpu : InstructionTarget.Direct3D12,
                precision, duplicate, failGeneration);
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() => [new(Id, "precision-test", 0, Value.Target)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            collectionId == Id && instructionName == Value.Name ? [Value] : [];
    }

    private sealed class PrecisionInstruction : Instruction
    {
        private readonly bool failGeneration;
        internal PrecisionInstruction(InstructionTarget target, KernelPrecisionProfile precision, bool duplicate, bool failGeneration)
        {
            Target = target;
            this.failGeneration = failGeneration;
            var signature = new InstructionSignature(
                [new("input", GraphElementType.Float16, GraphResourceAccess.Read),
                    new("output", GraphElementType.Float16, GraphResourceAccess.Write)], [], precision);
            Signatures = duplicate ? [signature, signature] : [signature];
        }
        public override Guid CollectionId => PrecisionProvider.Id;
        public override string Name => "test.precision";
        public override InstructionTarget Target { get; }
        public override IReadOnlyList<InstructionSignature> Signatures { get; }
        internal int Generated { get; private set; }
        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            if (failGeneration) throw new InvalidDataException("Test generation failure.");
            Generated++;
            return new("{}");
        }
    }
}
