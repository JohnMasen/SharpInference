using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class VmGraphBindingValidatorTests
{
    [Theory]
    [InlineData(GraphElementType.Byte, VmElementType.Byte)]
    [InlineData(GraphElementType.Int32, VmElementType.Int32)]
    [InlineData(GraphElementType.UInt32, VmElementType.UInt32)]
    [InlineData(GraphElementType.Float16, VmElementType.Float16)]
    [InlineData(GraphElementType.Float32, VmElementType.Float32)]
    public void TypedStateAndNonTokenIoFollowGraphDescriptors(GraphElementType graphType, VmElementType vmType)
    {
        var graph = new LogicalGraphBuilder(new("external", 1, "typed-io"),
                new GraphModelSignature("external", "external.typed@2", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .SetStateSchema(new("External_State"))
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(graphType, [2]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(graphType, [2]), graphOutput: true)
            .AddResource("state", "State", GraphResourceKind.SessionState, GraphResourceLifetime.Session,
                new TensorDescriptor(graphType, [3]))
            .AddStateSlot("position", "state").Build();
        var execution = new GraphOptimizer().Optimize(graph, new(OptimizationBoundary.Off));
        var slots = new VmSlot[]
        {
            new("input", VmSlotScope.Local, VmAccess.ReadOnly, new(vmType, [2])),
            new("output", VmSlotScope.Local, VmAccess.ReadWrite, new(vmType, [2])),
            new("state", VmSlotScope.Session, VmAccess.ReadWrite, new(vmType, [3])),
        };
        var parameters = slots.Select(slot => new VmParameter(slot.Id, slot.Access, slot.Tensor)).ToArray();
        var program = new VmProgram("typed", "vm:external.typed@2", VmTarget.Cpu, slots,
            [new VmDefinition("forward", VmDefinitionKind.Orchestration, parameters, [])],
            [new VmEntry("forward", "forward", slots.Select(slot => new VmArgument(slot.Id, slot.Id)))],
            new VmState("External_State", 1, [new VmStateEntry("position", "state")]));
        VmGraphBindingValidator.Validate(program, execution);
        var wrong = new VmProgram(program.Name, program.Abi, program.Target,
            slots.Select(slot => slot.Id == "state" ? slot with { Tensor = new VmTensor(vmType, [4]) } : slot),
            program.Definitions, program.Entries, program.State);
        var error = Assert.Throws<InvalidDataException>(() => VmGraphBindingValidator.Validate(wrong, execution));
        Assert.Contains("descriptor", error.Message, StringComparison.Ordinal);
    }
}
