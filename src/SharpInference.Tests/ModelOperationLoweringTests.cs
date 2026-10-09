using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm.Optimization;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class ModelOperationLoweringTests
{
    [Fact]
    public void RegisteredNativeOperationCompilesWithoutGenericExpansion()
    {
        var original = CreateGraph();
        var graph = new LogicalGraph(original.Identity, original.Model, original.Resources, original.Regions,
            original.Nodes.Select(node => node with
            {
                Resources = [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "input"),
                    GraphBindings.Write("output", "output")],
            }), original.Inputs, original.Outputs, original.GraphState);
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Cpu, [new NativeProvider()]);
        var program = generator.Generate(graph, new(ReuseLocalStorage: false, PrefillCapacity: 1));
        using var executable = new CpuVmCompiler([new NativeProvider()]).Compile(program).LoadExecutable();
        var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
        var input = program.Slots.ToList().FindIndex(slot => slot.Id == "input");
        var output = program.Slots.ToList().FindIndex(slot => slot.Id == "output");
        MemoryMarshal.AsBytes(new float[] { 1, -2, 3 }.AsSpan()).CopyTo(buffers[input]);
        executable.Invoke("forward", new CpuVmContext(program, buffers, ["input"]));
        Assert.Equal([2f, -4f, 6f], MemoryMarshal.Cast<byte, float>(buffers[output]).ToArray());
    }

    [Fact]
    public void AmbiguousNativeMappingsAreRejected()
    {
        var registry = new InstructionRegistry([new NativeProvider(), new NativeProvider()]);
        Assert.Throws<InvalidDataException>(() => registry.QueryGraphInstructionBindings());
    }

    [Fact]
    public void ModelOwnedOperationLowersAndExecutesThroughRegisteredInstructions()
    {
        var graph = CreateGraph();
        var generator = new VmExecutionGraphGenerator(InstructionTarget.Cpu,
            SharpInference.Runtime.Cpu.CpuInstructionCollections.Create(), new DoubleLowerer());
        var program = generator.Generate(graph, new(ReuseLocalStorage: false, PrefillCapacity: 1));
        using var executable = new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()).Compile(program).LoadExecutable();
        var buffers = program.Slots.Select(slot => new byte[checked((int)slot.Tensor.ByteLength)]).ToArray();
        var input = program.Slots.ToList().FindIndex(slot => slot.Id == "input");
        var output = program.Slots.ToList().FindIndex(slot => slot.Id == "output");
        MemoryMarshal.AsBytes(new float[] { 1, -2, 3 }.AsSpan()).CopyTo(buffers[input]);
        executable.Invoke("forward", new CpuVmContext(program, buffers, ["input"]));
        Assert.Equal([2f, -4f, 6f], MemoryMarshal.Cast<byte, float>(buffers[output]).ToArray());
    }

    [Fact]
    public void LoweringCannotChangePublicBindings()
    {
        Assert.Throws<InvalidDataException>(() => GraphOperationLowering.Apply(CreateGraph(), new InvalidLowerer()));
    }

    [Fact]
    public void UnregisteredOperatorVersionIsRejected()
    {
        var graph = CreateGraph();
        var invalid = new LogicalGraph(graph.Identity, graph.Model, graph.Resources, graph.Regions,
            graph.Nodes.Select(node => node with { Operation = new GraphOperationId("external.double", 3) }),
            graph.Inputs, graph.Outputs, graph.GraphState);
        Assert.Throws<NotSupportedException>(() => GraphOperationLowering.Apply(invalid, new DoubleLowerer()));
    }

    private static LogicalGraph CreateGraph() =>
        new LogicalGraphBuilder(new("external", 1, "double"),
                new GraphModelSignature("external", "external.empty@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [3]), graphOutput: true)
            .AddNode("double", new GraphOperationId("external.double", 2), "root",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")])
            .Build();

    private sealed class DoubleLowerer : IGraphOperationLowerer
    {
        public LogicalGraph Lower(LogicalGraph graph)
        {
            var nodes = graph.Nodes.Select(node =>
            {
                if (node.Operation != new GraphOperationId("external.double", 2))
                    throw new NotSupportedException($"Unsupported external operation '{node.Operation}'.");
                return node with
                {
                    Operation = PrimitiveGraphOperations.Add,
                    Resources = [GraphBindings.Read("left", "input"), GraphBindings.Read("right", "input"),
                        GraphBindings.Write("output", "output")],
                };
            }).ToArray();
            return new(graph.Identity, graph.Model, graph.Resources, graph.Regions, nodes,
                graph.Inputs, graph.Outputs, graph.GraphState);
        }
    }

    private sealed class NativeProvider : IInstructionCollectionProvider, IGraphInstructionProvider
    {
        private readonly InstructionRegistry instructions = new(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create());
        public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
            instructions.QueryInstructionCollection();
        public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
            instructions.QueryInstruction(collectionId, instructionName);
        public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
            [new(new GraphOperationId("external.double", 2), InstructionCollectionIds.TierZeroFloat32,
                "core.add", InstructionTarget.Cpu)];
    }

    private sealed class InvalidLowerer : IGraphOperationLowerer
    {
        public LogicalGraph Lower(LogicalGraph graph) => new(graph.Identity, graph.Model,
            graph.Resources.Select(resource => resource.Id.Value == "input"
                ? resource with { Tensor = new TensorDescriptor(GraphElementType.Float32, [4]) } : resource),
            graph.Regions, graph.Nodes, graph.Inputs, graph.Outputs, graph.GraphState);
    }
}
