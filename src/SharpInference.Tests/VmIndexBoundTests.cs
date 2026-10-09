using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Vm;

namespace SharpInference.Tests;

[Collection(VmGraphGpuCollection.Name)]
public sealed class VmIndexBoundTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ArrayBoundsRespectViewsAndRejectInvalidLastElementBeforeUpload(int minimum)
    {
        var provider = new LookupProvider(minimum);
        var artifact = new D3D12VmCompiler([provider]).Compile(Program());
        using var executor = artifact.CreateExecutor();
        executor.Upload("table", Bytes([1f, 2f, 3f, 4f]));
        executor.Upload("workspace", new byte[32]);
        int[] indices = [777, 0, minimum, 1, 888];
        executor.Upload("indices", MemoryMarshal.AsBytes(indices.AsSpan()).ToArray());
        executor.Execute("forward");
        var actual = MemoryMarshal.Cast<byte, float>(executor.Readback("workspace")).ToArray();
        Assert.Equal(minimum == 0 ? [0f, 1f, 2f, 1f, 2f, 3f, 4f, 0f] : [0f, 1f, 2f, 0f, 0f, 3f, 4f, 0f], actual);
        foreach (var invalid in new[] { minimum - 1, 2 })
        {
            var broken = indices.ToArray();
            broken[3] = invalid;
            Assert.Throws<ArgumentOutOfRangeException>(() => executor.Upload("indices", MemoryMarshal.AsBytes(broken.AsSpan()).ToArray()));
        }
        var bound = VmInstructionContracts.Bind(Program(), new InstructionRegistry([provider]));
        var xml = VmProgramXml.Serialize(bound);
        if (minimum == 0) Assert.DoesNotContain("minimumIndex", xml);
        else Assert.Contains("minimumIndex=\"-1\"", xml);
        var restored = VmProgramXml.Deserialize(xml);
        Assert.Equal(xml, VmProgramXml.Serialize(restored));
        var operation = Assert.IsType<VmOperator>(restored.Definitions.Single(definition => definition.Id == "lookup").Nodes[0].Instruction);
        Assert.Equal(minimum, Assert.Single(operation.IndexBounds).MinimumIndex);
        Assert.Throws<InvalidDataException>(() => new D3D12VmCompiler([new LookupProvider(minimum - 1)]).Compile(restored));
    }

    [Fact]
    public void PositiveMinimumAndMalformedSerializedBoundsAreRejected()
    {
        var registry = new InstructionRegistry([new LookupProvider(1)]);
        Assert.Throws<InvalidDataException>(() => VmInstructionContracts.Bind(Program(), registry));
        var valid = VmInstructionContracts.Bind(Program(), new InstructionRegistry([new LookupProvider(-1)]));
        var xml = VmProgramXml.Serialize(valid);
        Assert.Throws<InvalidDataException>(() => VmProgramXml.Deserialize(xml.Replace("minimumIndex=\"-1\"", "minimumIndex=\"1\"", StringComparison.Ordinal)));
        Assert.Throws<InvalidDataException>(() => VmProgramXml.Deserialize(xml.Replace("minimumIndex=\"-1\"", "minimumIndex=\"NaN\"", StringComparison.Ordinal)));
    }

    private static VmProgram Program()
    {
        var parameters = new VmParameter[]
        {
            new("table", VmAccess.ReadOnly, new(VmElementType.Float32, [2, 2])),
            new("indices", VmAccess.ReadOnly, new(VmElementType.Int32, [3])),
            new("output", VmAccess.ReadWrite, new(VmElementType.Float32, [3, 2])),
        };
        var slots = new VmSlot[]
        {
            new("table", VmSlotScope.Global, VmAccess.ReadOnly, new(VmElementType.Float32, [2, 2])),
            new("indices", VmSlotScope.Local, VmAccess.ReadOnly, new(VmElementType.Int32, [5])),
            new("workspace", VmSlotScope.Local, VmAccess.ReadWrite, new(VmElementType.Float32, [8])),
        };
        var operation = new VmOperator(new(GraphElementType.Float32, GraphElementType.Float32), LookupProvider.Id, "test.lookup",
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
        var definitions = new VmDefinition[]
        {
            new("lookup", VmDefinitionKind.Kernel, parameters, [new("body", operation)], new(64)),
            new("forward", VmDefinitionKind.Orchestration, slots.Select(slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)),
                [new("dispatch", new VmDispatch("lookup", [new("table", "table"), new("indices", "indices", 4), new("output", "workspace", 4)], new(1))),
                 new("barrier", new VmBarrier(["workspace"]), ["dispatch"])]),
        };
        return new("bounded-lookup", "bounded-lookup.empty@1", VmTarget.Direct3D12, slots, definitions,
            [new("forward", "forward", slots.Select(slot => new VmArgument(slot.Id, slot.Id)))], new("none", 1, []));
    }

    private static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private sealed class LookupProvider(int minimum) : IInstructionCollectionProvider
    {
        public static readonly Guid Id = new("0c41cb9b-1d54-4e94-b8a9-86a357218df8");
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            [new(Id, "Bounded lookup test", 2, InstructionTarget.Direct3D12)];
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string name) =>
            collectionId == Id && name == "test.lookup" ? [new LookupInstruction(minimum)] : [];
    }

    private sealed class LookupInstruction(int minimum) : Instruction
    {
        public override Guid CollectionId => LookupProvider.Id;
        public override string Name => "test.lookup";
        public override InstructionTarget Target => InstructionTarget.Direct3D12;
        public override IReadOnlyList<InstructionSignature> Signatures =>
            [new([new("table", GraphElementType.Float32, GraphResourceAccess.Read),
                new("indices", GraphElementType.Int32, GraphResourceAccess.Read), new("output", GraphElementType.Float32, GraphResourceAccess.Write)],
                [], new(GraphElementType.Float32, GraphElementType.Float32))];
        public override IReadOnlyList<InstructionIndexBound> IndexBounds => [new("indices", "table", 0, minimum)];
        protected override InstructionRecording Generate(InstructionParameter[] parameters)
        {
            var tensors = parameters.OfType<InstructionTensorParameter>().ToDictionary(parameter => parameter.Name);
            var table = tensors["table"];
            var indices = tensors["indices"];
            var output = tensors["output"];
            return new($$"""
                {
                    uint element=gpu.groupIndex;
                    if(element<6u) {
                        int row=asint({{indices.Expression}}.Load({{indices.OffsetExpression}}+(element/2u)*4u));
                        float value=0.0f;
                        [branch] if(row>=0)
                            value=asfloat({{table.Expression}}.Load({{table.OffsetExpression}}+(uint(row)*2u+element%2u)*4u));
                        {{output.Expression}}.Store({{output.OffsetExpression}}+element*4u,asuint(value));
                    }
                }
                """);
        }
    }
}
