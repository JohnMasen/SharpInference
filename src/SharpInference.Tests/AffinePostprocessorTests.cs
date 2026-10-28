using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using SharpInference.Instructions.Cpu;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;
using SharpInference.Vm.Optimization.GraphHelper;

namespace SharpInference.Tests;

public sealed class AffinePostprocessorTests
{
    private static readonly ExecutionPassContext Context = new(new("test", "cpu-vm"), 0, "test");

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FusesRegisteredAffinePreservingBindingsTransposeOriginsAndScope(bool transposeLeft, bool transposeRight)
    {
        var package = Create(transposeLeft, transposeRight);
        var before = ExecutionGraphPackageXml.Serialize(package);
        var processor = new AffinePostprocessor([new TestInstruction()]);
        var result = processor.Process(package, Context);
        var graph = result.MainGraph!;
        var operation = Assert.IsType<ExecutionOperation>(Assert.Single(graph.Actions));
        Assert.Equal(TierOneAffineOperations.Affine, operation.Operation);
        Assert.Equal("bias-add", operation.Id);
        Assert.Equal(transposeLeft.ToString(), operation.Attributes["transpose_left"]);
        Assert.Equal(transposeRight.ToString(), operation.Attributes["transpose_right"]);
        Assert.Equal("root/projector", operation.Metadata["architecture.path"]);
        Assert.Equal("projector", operation.Category);
        Assert.Equal(2, operation.Sources.Count);
        Assert.Equal(new PrecisionRequirement(ElementType.Float32, ElementType.Float32), operation.Precision);
        Assert.Equal(TierOneAffineOperations.CollectionFloat32, operation.SelectedImplementation!.CollectionId);
        Assert.Equal(["left", "right", "bias", "output"], operation.Bindings.Select(binding => binding.SourceProperty));
        Assert.DoesNotContain(operation.Bindings, binding => binding.Target.TargetProperty == "temporary");
        Assert.Equal(before, ExecutionGraphPackageXml.Serialize(package));
        Assert.Equal(ExecutionGraphPackageXml.Serialize(result),
            ExecutionGraphPackageXml.Serialize(processor.Process(result, Context)));
    }

    [Theory]
    [InlineData("fp16")]
    [InlineData("alias")]
    [InlineData("storage-alias")]
    [InlineData("binding-alias")]
    [InlineData("third-party-binding-alias")]
    [InlineData("public-temporary")]
    [InlineData("state-temporary")]
    [InlineData("global-temporary")]
    [InlineData("external-consumer")]
    [InlineData("cross-graph-consumer")]
    [InlineData("cross-graph-writer")]
    [InlineData("unordered")]
    [InlineData("intervening")]
    [InlineData("shape")]
    [InlineData("view")]
    [InlineData("selected-fp16")]
    [InlineData("version")]
    public void UnsafeOrUnsupportedPairsRemainUnchanged(string scenario)
    {
        var package = Create();
        var graph = package.MainGraph!;
        var multiply = (ExecutionOperation)graph.Actions[0];
        var add = (ExecutionOperation)graph.Actions[1];
        var temporary = graph.Resources.Single(resource => resource.Id == "temporary");
        var output = graph.Resources.Single(resource => resource.Id == "output");
        switch (scenario)
        {
            case "fp16": output.Tensor = new(ElementType.Float16, [2, 4]); break;
            case "alias":
                add.Bindings[2] = add.Bindings[2] with { Target = add.Bindings[0].Target };
                Assert.DoesNotContain(package.ValidateStructure(),
                    diagnostic => diagnostic.Severity == GraphDiagnosticSeverity.Error);
                break;
            case "storage-alias": graph.Storage["temporary"] = new(ResourceStorageKind.LocalBuffer, "output"); break;
            case "binding-alias":
                graph.Resources.Single(resource => resource.Id == "left").BindingKey = "same";
                graph.Resources.Single(resource => resource.Id == "right").BindingKey = "same";
                break;
            case "third-party-binding-alias":
                var left = graph.Resources.Single(resource => resource.Id == "left");
                left.BindingKey = "same";
                graph.Resource("alias", left.Tensor, bindingKey: "same");
                break;
            case "public-temporary": temporary.GraphIO = GraphIO.Out; break;
            case "state-temporary": temporary.IsState = true; break;
            case "global-temporary": temporary.Scope = ResourceScope.Global; break;
            case "external-consumer":
                var consumer = graph.Operation("consumer", PrimitiveOperations.Copy);
                consumer.Bind("input", temporary, ResourceAccess.Read);
                consumer.Bind("output", graph.Resource("extra", temporary.Tensor), ResourceAccess.Write);
                multiply.Then(consumer);
                break;
            case "cross-graph-consumer":
                var other = new ExecutionGraph { Id = "other" };
                var external = other.Operation("consumer", PrimitiveOperations.Copy);
                external.Bindings.Add(new("input", new ResourceRef(temporary), ResourceAccess.Read));
                external.Bind("output", other.Resource("extra", temporary.Tensor), ResourceAccess.Write);
                package.Add(other);
                break;
            case "cross-graph-writer":
                var writerGraph = new ExecutionGraph { Id = "writer" };
                var writer = writerGraph.Operation("write", PrimitiveOperations.Copy);
                writer.Bind("input", writerGraph.Resource("input", output.Tensor, GraphIO.In), ResourceAccess.Read);
                writer.Bindings.Add(new("output", new ResourceRef(output), ResourceAccess.Write));
                package.Add(writerGraph);
                break;
            case "unordered": graph.GraphFlow.Clear(); break;
            case "intervening":
                var middle = graph.Operation("middle", PrimitiveOperations.Copy);
                middle.Bind("input", temporary, ResourceAccess.Read);
                middle.Bind("output", graph.Resource("extra", temporary.Tensor), ResourceAccess.Write);
                multiply.Then(middle);
                middle.Then(add);
                break;
            case "shape": output.Tensor = new(ElementType.Float32, [4, 2]); break;
            case "view": multiply.Attributes["view.left.offset"] = "0"; break;
            case "selected-fp16":
                multiply.SelectedImplementation = new(Guid.Empty, "test", ExecutionTarget.Cpu,
                    new(ElementType.Float16, ElementType.Float16));
                break;
            case "version": multiply.Operation = new(PrimitiveOperations.MatrixMultiply.Name, 2); break;
        }
        var before = ExecutionGraphPackageXml.Serialize(package);
        Assert.Equal(before, ExecutionGraphPackageXml.Serialize(new AffinePostprocessor([new TestInstruction()])
            .Process(package, Context)));
    }

    [Theory]
    [InlineData("absent")]
    [InlineData("unsupported")]
    [InlineData("signature")]
    [InlineData("access")]
    [InlineData("precision")]
    [InlineData("other-target")]
    public void NoLegalAlternativeLeavesGraphUnchanged(string scenario)
    {
        var package = Create();
        IInstructionCollectionProvider[] providers = scenario == "absent" ? [] : [new TestInstruction(scenario)];
        Assert.Equal(ExecutionGraphPackageXml.Serialize(package),
            ExecutionGraphPackageXml.Serialize(new AffinePostprocessor(providers).Process(package, Context)));
    }

    [Theory]
    [InlineData("duplicate-provider")]
    [InlineData("invalid-signature")]
    [InlineData("invalid-transpose")]
    public void InvalidDeclarationsAreNotSilentlyIgnored(string scenario)
    {
        var package = Create();
        if (scenario == "invalid-transpose")
        {
            ((ExecutionOperation)package.MainGraph!.Actions[0]).Attributes["transpose_right"] = "not-a-bool";
        }
        IInstructionCollectionProvider[] providers = scenario == "duplicate-provider"
            ? [new TestInstruction(), new TestInstruction()] : [new TestInstruction(scenario)];
        Assert.Throws<InvalidDataException>(() => new AffinePostprocessor(providers).Process(package, Context));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void RegisteredCpuAffinePreservesObservableResults(bool transposeLeft, bool transposeRight)
    {
        IInstructionCollectionProvider[] providers =
            [new CpuFloat32InstructionCollection(), new CpuTierOneInstructionCollection()];
        var source = Create(transposeLeft, transposeRight);
        var optimized = new AffinePostprocessor(providers).Process(source, Context);
        Assert.Single(optimized.MainGraph!.Actions);
        Assert.Equal(Execute(source), Execute(optimized));

        float[] Execute(ExecutionGraphPackage package)
        {
            var program = package.Compile(VmTarget.Cpu, providers);
            var artifact = new CpuVmCompiler(providers).Compile(program);
            using var executor = artifact.CreateExecutor();
            var slots = program.Slots.ToList();
            var buffers = slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
            foreach (var name in new[] { "left", "right", "bias" })
            {
                var index = slots.FindIndex(slot => slot.Id == name);
                var values = MemoryMarshal.Cast<byte, float>(buffers[index].AsSpan());
                for (var element = 0; element < values.Length; element++)
                {
                    values[element] = name == "bias" ? element + 0.5f : element + 1;
                }
            }
            executor.Execute("forward", buffers);
            return MemoryMarshal.Cast<byte, float>(buffers[slots.FindIndex(slot => slot.Id == "output")]).ToArray();
        }
    }

    private static ExecutionGraphPackage Create(bool transposeLeft = false, bool transposeRight = true)
    {
        var graph = new ExecutionGraph { Id = "main" };
        var left = graph.Resource("left", new(ElementType.Float32, transposeLeft ? [3, 2] : [2, 3]), GraphIO.In);
        var right = graph.Resource("right", new(ElementType.Float32, transposeRight ? [4, 3] : [3, 4]), GraphIO.In);
        var bias = graph.Resource("bias", new(ElementType.Float32, [4]), GraphIO.In);
        var temporary = graph.Resource("temporary", new(ElementType.Float32, [2, 4]));
        var output = graph.Resource("output", temporary.Tensor, GraphIO.Out);
        var multiply = graph.Operation("multiply", PrimitiveOperations.MatrixMultiply);
        multiply.Bind("left", left, ResourceAccess.Read);
        multiply.Bind("right", right, ResourceAccess.Read);
        multiply.Bind("output", temporary, ResourceAccess.Write);
        multiply.Attributes.Add("transpose_left", transposeLeft.ToString());
        multiply.Attributes.Add("transpose_right", transposeRight.ToString());
        multiply.Source("model", "main", "multiply");
        var add = graph.Operation("bias-add", PrimitiveOperations.BiasAdd);
        add.Bind("input", temporary, ResourceAccess.Read);
        add.Bind("bias", bias, ResourceAccess.Read);
        add.Bind("output", output, ResourceAccess.Write);
        add.Source("model", "main", "bias-add");
        add.Category = "projector";
        add.Metadata["architecture.path"] = "root/projector";
        multiply.Then(add);
        return new(graph);
    }

    private sealed class TestInstruction(string scenario = "") : Instruction, IInstructionCollectionProvider
    {
        public override Guid CollectionId => TierOneAffineOperations.CollectionFloat32;
        public override string Name => TierOneAffineOperations.Affine.Name;
        public override ExecutionTarget Target => scenario == "other-target" ? ExecutionTarget.Direct3D12 : ExecutionTarget.Cpu;
        public override IReadOnlyList<InstructionSignature> Signatures => [
            new([
                new(scenario == "signature" ? "unknown" : "left", ElementType.Float32, ResourceAccess.Read),
                new(scenario == "invalid-signature" ? "left" : "right", ElementType.Float32, ResourceAccess.Read),
                new("bias", ElementType.Float32, ResourceAccess.Read),
                new("output", ElementType.Float32, scenario == "access" ? ResourceAccess.Read : ResourceAccess.Write),
            ], ["transpose_left", "transpose_right"],
                scenario == "precision" ? new(ElementType.Float16, ElementType.Float16) :
                    new(ElementType.Float32, ElementType.Float32))];
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(CollectionId, "test-affine", 1, Target)];
        public IReadOnlyList<Instruction> EnumerateInstructions() => [this];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            collectionId == CollectionId && instructionName == Name ? [this] : [];
        protected override void Validate(InstructionParameter[] parameters)
        {
            if (scenario == "unsupported")
            {
                throw new InstructionAdaptationException(CollectionId, Name, "Unsupported test shape.");
            }
        }
        protected override InstructionRecording Generate(InstructionParameter[] parameters) =>
            throw new NotSupportedException("Used only for adaptation tests.");
    }
}
