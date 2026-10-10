using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class GraphBuilderExtensionsTests
{
    [Theory]
    [InlineData("Copy")]
    [InlineData("Square")]
    [InlineData("Relu")]
    [InlineData("Sigmoid")]
    [InlineData("Tanh")]
    [InlineData("Exp")]
    [InlineData("ReciprocalSquareRoot")]
    [InlineData("ReduceSum")]
    [InlineData("ReduceMean")]
    [InlineData("Add")]
    [InlineData("Subtract")]
    [InlineData("Multiply")]
    [InlineData("Divide")]
    [InlineData("Maximum")]
    [InlineData("MatVec")]
    [InlineData("GatherRow")]
    [InlineData("MatrixMultiply")]
    [InlineData("Affine")]
    public void TypedOperationsMatchExplicitBindingsAndValidateAgainstExistingContracts(string operation)
    {
        LogicalGraphBuilder Builder()
        {
            var builder = new LogicalGraphBuilder(new("test", 1, "typed"),
                new("test", "test.empty@1", new Dictionary<string, int>()));
            var matrix = operation is "MatVec" or "GatherRow" ? new[] { 2, 2 } : new[] { 3, 2 };
            var input = operation is "MatrixMultiply" or "Affine" ? new[] { 1, 2 } : new[] { 2 };
            var output = operation is "ReduceSum" or "ReduceMean" ? new[] { 1 } :
                operation is "MatrixMultiply" or "Affine" ? new[] { 1, 3 } : new[] { 2 };
            return builder
                .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External, new(GraphElementType.Float32, input), graphInput: true)
                .AddResource("right", "Right", GraphResourceKind.Input, GraphResourceLifetime.External, new(GraphElementType.Float32, [2]), graphInput: true)
                .AddResource("matrix", "Matrix", GraphResourceKind.Weight, GraphResourceLifetime.Model, new(GraphElementType.Float32, matrix), "matrix")
                .AddResource("bias", "Bias", GraphResourceKind.Weight, GraphResourceLifetime.Model, new(GraphElementType.Float32, [3]), "bias")
                .AddResource("index", "Index", GraphResourceKind.Input, GraphResourceLifetime.External, new(GraphElementType.Int32, [1]), graphInput: true)
                .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External, new(GraphElementType.Float32, output), graphOutput: true);
        }
        var typed = Builder().WithRegion("root", GraphRegionTypes.Graph, "Root", scoped =>
        {
            switch (operation)
            {
                case "Copy": scoped.Copy("node", "input", "output"); break;
                case "Square": scoped.Square("node", "input", "output"); break;
                case "Relu": scoped.Relu("node", "input", "output"); break;
                case "Sigmoid": scoped.Sigmoid("node", "input", "output"); break;
                case "Tanh": scoped.Tanh("node", "input", "output"); break;
                case "Exp": scoped.Exp("node", "input", "output"); break;
                case "ReciprocalSquareRoot": scoped.ReciprocalSquareRoot("node", "input", "output"); break;
                case "ReduceSum": scoped.ReduceSum("node", "input", "output"); break;
                case "ReduceMean": scoped.ReduceMean("node", "input", "output"); break;
                case "Add": scoped.Add("node", "input", "right", "output"); break;
                case "Subtract": scoped.Subtract("node", "input", "right", "output"); break;
                case "Multiply": scoped.Multiply("node", "input", "right", "output"); break;
                case "Divide": scoped.Divide("node", "input", "right", "output"); break;
                case "Maximum": scoped.Maximum("node", "input", "right", "output"); break;
                case "MatVec": scoped.MatVec("node", "matrix", "input", "output"); break;
                case "GatherRow": scoped.GatherRow("node", "matrix", "index", "output"); break;
                case "MatrixMultiply": scoped.MatrixMultiply("node", "input", "matrix", "output", transposeRight: true); break;
                case "Affine": scoped.Affine("node", "input", "matrix", "bias", "output", transposeRight: true); break;
            }
        }).Build();
        TierZeroOperationContracts.ValidateGraph(typed);
        var node = Assert.Single(typed.Nodes);
        var explicitGraph = Builder().AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddNode("node", node.Operation, "root", node.Resources, attributes: node.Attributes).Build();
        Assert.Equal(GraphXml.Serialize(explicitGraph), GraphXml.Serialize(typed));
        Assert.All(node.Resources, binding => Assert.Equal(binding.Port == "output" ?
            GraphResourceAccess.Write : GraphResourceAccess.Read, binding.Access));
    }

    [Fact]
    public void TypedOperationsSupportExplicitRegionsAndRejectMissingScope()
    {
        var builder = new LogicalGraphBuilder(new("test", 1, "explicit"),
            new("test", "test.empty@1", new Dictionary<string, int>()))
            .AddRegion("root", GraphRegionTypes.Graph, "Root")
            .AddResource("input", "Input", GraphResourceKind.Input, GraphResourceLifetime.External, new(GraphElementType.Float32, [2]), graphInput: true)
            .AddResource("output", "Output", GraphResourceKind.Output, GraphResourceLifetime.External, new(GraphElementType.Float32, [2]), graphOutput: true);
        Assert.Throws<InvalidOperationException>(() => builder.Copy("invalid", "input", "output"));
        builder.Copy("copy", "input", "output", regionId: "root");
        Assert.Equal("root", Assert.Single(builder.Build().Nodes).Region.Value);
        Assert.Throws<ArgumentNullException>(() => GraphBuilderExtensions.Copy(null!, "copy", "input", "output"));
        Assert.Throws<ArgumentException>(() => builder.Copy("bad", "", "output", regionId: "root"));
    }
}
