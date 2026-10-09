using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphOperationValidationTests
{
    [Fact]
    public void RegisteredOperationOwnsShapeAndAttributeValidation()
    {
        var validator = new ExternalOperationValidator();
        var graph = CreateGraph(3);
        GraphValidator.Validate(graph, validator);
        Assert.Equal(1, validator.Calls);
        Assert.Throws<InvalidDataException>(() => GraphValidator.Validate(CreateGraph(2), validator));
        GraphValidator.Validate(CreateGraph(2));
    }

    [Fact]
    public void InvalidDependenciesAreRejectedBeforeModuleCallbacks()
    {
        var validator = new ExternalOperationValidator();
        Assert.Throws<InvalidDataException>(() => GraphValidator.Validate(CreateGraph(3, true), validator));
        Assert.Equal(0, validator.Calls);
    }

    private static LogicalGraph CreateGraph(int width, bool invalidDependency = false)
    {
        var graph = new LogicalGraphBuilder(new GraphIdentity("external-model", 1, "custom-op"),
                new GraphModelSignature("external", "external.empty@1", new Dictionary<string, int>()))
            .AddRegion("graph", GraphRegionTypes.Graph, "Graph")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [width]), graphOutput: true)
            .AddNode("op", new GraphOperationId("external.scale", 2), "graph",
                [GraphBindings.Read("input", "input"), GraphBindings.Write("output", "output")],
                attributes: new Dictionary<string, string> { ["scale"] = "2" })
            .Build();
        return invalidDependency
            ? new LogicalGraph(graph.Identity, graph.Model, graph.Resources, graph.Regions,
                    graph.Nodes.Select(node => node with { Dependencies = [new LogicalNodeId("missing")] }).ToArray(),
                    graph.Inputs, graph.Outputs, graphState: graph.GraphState)
            : graph;
    }

    private sealed class ExternalOperationValidator : IGraphOperationValidator
    {
        public int Calls { get; private set; }

        public void Validate(GraphOperationValidationContext context)
        {
            Calls++;
            if (context.Operation != new GraphOperationId("external.scale", 2) ||
                !context.Attributes.TryGetValue("scale", out var scale) || scale != "2")
                throw new InvalidDataException("The external scale operation requires version 2 and scale=2.");
            var input = context.Resources[context.Bindings.Single(binding => binding.Port == "input").Resource];
            if (!input.Tensor.Dimensions.SequenceEqual([3]))
                throw new InvalidDataException("The external model requires a three-element scale input.");
        }
    }
}
