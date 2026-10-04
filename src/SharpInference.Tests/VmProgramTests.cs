using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class VmProgramTests
{
    private static readonly VmTensor Vector = new(VmElementType.Float32, [4]);
    private static readonly VmParameter[] Parameters =
    [
        new("input", VmAccess.ReadOnly, Vector),
        new("output", VmAccess.ReadWrite, Vector),
    ];
    private static readonly VmArgument[] FunctionArguments =
    [
        new("input", "input"),
        new("output", "output"),
    ];

    [Fact]
    public void CpuDefinitionsAndStateRoundTripWithoutLayerOrRegion()
    {
        var original = CpuProgram();
        var xml = VmProgramXml.Serialize(original);
        var restored = VmProgramXml.Deserialize(xml);
        Assert.Equal(xml, VmProgramXml.Serialize(restored));
        Assert.Equal("VmProgram", System.Xml.Linq.XElement.Parse(xml).Name.LocalName);
        Assert.DoesNotContain("Region", xml, StringComparison.Ordinal);
        Assert.Equal("state", Assert.Single(restored.State.Entries).Slot);
        Assert.Equal(3, restored.Slots.Count);
        var calls = restored.Definitions.Single(definition => definition.Id == "forward").Nodes;
        Assert.Equal(2, calls.Count);
        Assert.All(calls, node => Assert.Equal("copy", Assert.IsType<VmCall>(node.Instruction).Definition));
        Assert.Equal(["first"], calls[1].Dependencies);
        Assert.Equal(16UL, restored.Slots[0].Tensor.ByteLength);
    }

    [Fact]
    public void GpuFunctionCallsAreSeparateFromDispatchAndBarrier()
    {
        var gpu = GpuProgram();
        var restored = VmProgramXml.Deserialize(VmProgramXml.Serialize(gpu));
        var kernel = restored.Definitions.Single(definition => definition.Kind == VmDefinitionKind.Kernel);
        Assert.IsType<VmCall>(Assert.Single(kernel.Nodes).Instruction);
        Assert.Equal(new VmThreadGroup(64), kernel.Threads);
        var nodes = restored.Definitions.Single(definition => definition.Id == "forward").Nodes;
        Assert.IsType<VmDispatch>(nodes[0].Instruction);
        Assert.IsType<VmBarrier>(nodes[1].Instruction);
        Assert.IsType<VmDispatch>(nodes[2].Instruction);
    }

    [Fact]
    public void ProgramSnapshotsCallerCollections()
    {
        var slots = CpuProgram().Slots.ToList();
        var nodes = new List<VmNode>();
        var parameters = new List<VmParameter>(Parameters);
        var attributes = new Dictionary<string, string> { ["test"] = "original" };
        var operation = new VmOperator("copy", 1, FunctionArguments, attributes);
        nodes.Add(new VmNode("copy", operation));
        var definition = new VmDefinition("copy", VmDefinitionKind.Function, parameters, nodes);
        var program = Create(slots, [definition],
            [new VmEntry("forward", "copy", [new("input", "weight"), new("output", "state")])]);
        attributes["test"] = "modified";
        nodes.Clear();
        slots.Clear();
        parameters.Clear();
        Assert.Equal("original", operation.Attributes["test"]);
        Assert.Single(definition.Nodes);
        Assert.Equal(2, definition.Parameters.Count);
        Assert.Equal(3, program.Slots.Count);
    }

    [Theory]
    [InlineData(VmSlotScope.Global)]
    [InlineData(VmSlotScope.Local)]
    public void StateOnlyAllowsSessionSlots(VmSlotScope scope)
    {
        var original = CpuProgram();
        Assert.Throws<InvalidDataException>(() => Create(
            original.Slots.Select(slot => slot.Id == "state" ? slot with { Scope = scope } : slot),
            original.Definitions, original.Entries));
    }

    [Fact]
    public void SessionSlotsNeedNotAllBelongToState()
    {
        var original = CpuProgram();
        var program = Create(original.Slots.Append(new VmSlot("private", VmSlotScope.Session,
            VmAccess.ReadWrite, Vector)), original.Definitions, original.Entries);
        Assert.Single(program.State.Entries);
        Assert.Equal(2, program.Slots.Count(slot => slot.Scope == VmSlotScope.Session));
    }

    [Theory]
    [InlineData("weight")]
    [InlineData("missing")]
    public void EntryRejectsReadOnlyOrUnknownOutput(string output)
    {
        var program = CpuProgram();
        Assert.Throws<InvalidDataException>(() => Create(program.Slots, program.Definitions,
            [new VmEntry("forward", "forward", [new("input", "weight"), new("output", output)])]));
    }

    [Theory]
    [InlineData(1UL)]
    [InlineData(4UL)]
    [InlineData(ulong.MaxValue)]
    public void EntryRejectsMisalignedOrOutOfBoundsView(ulong offset)
    {
        var program = CpuProgram();
        Assert.Throws<InvalidDataException>(() => Create(program.Slots, program.Definitions,
            [new VmEntry("forward", "forward",
                [new("input", "weight"), new("output", "state", offset)])]));
    }

    [Fact]
    public void ByteWorkspaceCanBindAnAlignedTypedView()
    {
        var program = CpuProgram();
        var slots = program.Slots.Append(new VmSlot("workspace", VmSlotScope.Local,
            VmAccess.ReadWrite, new VmTensor(VmElementType.Byte, [64])));
        var updated = Create(slots, program.Definitions,
            [new VmEntry("forward", "forward",
                [new("input", "weight"), new("output", "workspace", 16)])]);
        Assert.Equal(16UL, updated.Entries[0].Arguments[1].ByteOffset);
    }

    [Fact]
    public void RejectsDuplicateDefinitionsAndRecursiveCalls()
    {
        var program = CpuProgram();
        Assert.Throws<InvalidDataException>(() => Create(program.Slots,
            program.Definitions.Concat([program.Definitions[0]]), program.Entries));
        var recursive = new VmDefinition("forward", VmDefinitionKind.Function, Parameters,
            [new VmNode("call", new VmCall("forward", FunctionArguments))]);
        Assert.Throws<InvalidDataException>(() => Create(program.Slots, [recursive], program.Entries));
    }

    [Fact]
    public void RejectsExcessiveNestingRegardlessOfDefinitionOrder()
    {
        var definitions = new List<VmDefinition>
        {
            new("d0", VmDefinitionKind.Function, [], []),
        };
        for (var i = 1; i <= 65; i++)
            definitions.Add(new($"d{i}", VmDefinitionKind.Function, [],
                [new VmNode("call", new VmCall($"d{i - 1}", []))]));
        Assert.Throws<InvalidDataException>(() => Create(CpuProgram().Slots, definitions,
            [new VmEntry("entry", "d65", [])]));
    }

    [Fact]
    public void RejectsForwardDependencies()
    {
        var program = CpuProgram();
        var definition = new VmDefinition("forward", VmDefinitionKind.Function, Parameters,
        [
            new VmNode("first", new VmCall("copy", FunctionArguments), ["second"]),
            new VmNode("second", new VmCall("copy", FunctionArguments)),
        ]);
        Assert.Throws<InvalidDataException>(() => Create(program.Slots,
            [program.Definitions[0], definition], program.Entries));
    }

    [Fact]
    public void FunctionCannotHideAKernelLaunch()
    {
        var gpu = GpuProgram();
        var definition = new VmDefinition("forward", VmDefinitionKind.Function, Parameters,
            [new VmNode("launch", new VmDispatch("kernel", FunctionArguments, new(1)))]);
        Assert.Throws<InvalidDataException>(() => Create(gpu.Slots,
            gpu.Definitions.Where(item => item.Id != "forward").Append(definition),
            gpu.Entries, VmTarget.Direct3D12));
    }

    [Fact]
    public void KernelCannotCallAnotherKernelAsADeviceFunction()
    {
        var gpu = GpuProgram();
        var kernel = new VmDefinition("kernel", VmDefinitionKind.Kernel, Parameters,
            [new VmNode("call", new VmCall("kernel", FunctionArguments))], new(64));
        Assert.Throws<InvalidDataException>(() => Create(gpu.Slots,
            gpu.Definitions.Where(item => item.Id != "kernel").Append(kernel),
            gpu.Entries, VmTarget.Direct3D12));
    }

    [Theory]
    [InlineData(0U, 1U, 1U)]
    [InlineData(1025U, 1U, 1U)]
    [InlineData(1024U, 2U, 1U)]
    [InlineData(1U, 1U, 65U)]
    public void RejectsIllegalThreadGroups(uint x, uint y, uint z)
    {
        var gpu = GpuProgram();
        Assert.Throws<InvalidDataException>(() => Create(gpu.Slots,
            gpu.Definitions.Select(definition => definition.Id == "kernel" ?
                new VmDefinition("kernel", VmDefinitionKind.Kernel, Parameters, definition.Nodes,
                    new(x, y, z)) : definition), gpu.Entries, VmTarget.Direct3D12));
    }

    [Theory]
    [InlineData("<Graph kind=\"ExecutionGraph\" version=\"1\" />")]
    [InlineData("<VmProgram version=\"99\" />")]
    [InlineData("<!DOCTYPE VmProgram [<!ENTITY x 'bad'>]><VmProgram>&x;</VmProgram>")]
    [InlineData("<VmProgram xmlns=\"urn:unexpected\" />")]
    public void RejectsOldOrUnsafeXml(string xml) =>
        Assert.Throws<InvalidDataException>(() => VmProgramXml.Deserialize(xml));

    [Fact]
    public void XmlRejectsUnknownFieldsDuplicateContainersAndNumericEnums()
    {
        var xml = VmProgramXml.Serialize(CpuProgram());
        Assert.Throws<InvalidDataException>(() =>
            VmProgramXml.Deserialize(xml.Replace("target=\"Cpu\"", "target=\"Cpu\" extra=\"x\"")));
        Assert.Throws<InvalidDataException>(() =>
            VmProgramXml.Deserialize(xml.Replace("<Slots>", "<Slots /><Slots>")));
        Assert.Throws<InvalidDataException>(() =>
            VmProgramXml.Deserialize(xml.Replace("target=\"Cpu\"", "target=\"0\"")));
    }

    private static VmProgram CpuProgram()
    {
        var copy = new VmDefinition("copy", VmDefinitionKind.Function, Parameters,
            [new VmNode("copy", new VmOperator("core.copy", 1, FunctionArguments))]);
        var forward = new VmDefinition("forward", VmDefinitionKind.Function, Parameters,
        [
            new VmNode("first", new VmCall("copy", FunctionArguments)),
            new VmNode("second", new VmCall("copy", FunctionArguments), ["first"]),
        ]);
        return Create(Slots(), [copy, forward],
            [new VmEntry("forward", "forward", [new("input", "weight"), new("output", "state")])]);
    }

    private static VmProgram GpuProgram()
    {
        var copy = CpuProgram().Definitions[0];
        var kernel = new VmDefinition("kernel", VmDefinitionKind.Kernel, Parameters,
            [new VmNode("call", new VmCall("copy", FunctionArguments))], new(64));
        var forward = new VmDefinition("forward", VmDefinitionKind.Orchestration, Parameters,
        [
            new VmNode("first", new VmDispatch("kernel", FunctionArguments, new(1))),
            new VmNode("barrier", new VmBarrier(["output"]), ["first"]),
            new VmNode("second", new VmDispatch("kernel", FunctionArguments, new(1)), ["barrier"]),
        ]);
        return Create(Slots(), [copy, kernel, forward],
            [new VmEntry("forward", "forward", [new("input", "weight"), new("output", "state")])],
            VmTarget.Direct3D12);
    }

    private static VmSlot[] Slots() =>
    [
        new("weight", VmSlotScope.Global, VmAccess.ReadOnly, Vector, "model.weight"),
        new("state", VmSlotScope.Session, VmAccess.ReadWrite, Vector),
        new("scratch", VmSlotScope.Local, VmAccess.ReadWrite, new VmTensor(VmElementType.Byte, [64])),
    ];

    private static VmProgram Create(IEnumerable<VmSlot> slots, IEnumerable<VmDefinition> definitions,
        IEnumerable<VmEntry> entries, VmTarget target = VmTarget.Cpu) =>
        new("test", "test@1", target, slots, definitions, entries,
            new VmState("state", 1, [new("recurrence", "state")]));
}
