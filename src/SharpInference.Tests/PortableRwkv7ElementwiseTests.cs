using SharpInference.Architectures.Rwkv7;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class PortableRwkv7ElementwiseTests
{
    [Fact]
    public void DecompositionsUsePortablePrimitivesAndMatchScalarEquations()
    {
        var builder = new LogicalGraphBuilder(
            new GraphIdentity("rwkv-7", 1, "elementwise-test"),
            TestGraphSignatures.Create(32, 4, 1, 1, 4, "rwkv-7.state.fp32@1"));
        builder.AddRegion("graph", GraphRegionTypes.Graph, "Elementwise test");
        var inputs = new Dictionary<string, float[]>
        {
            ["x"] = [0.4f, -0.5f, 1.2f, 2f],
            ["previous"] = [1f, 0.5f, -0.2f, -1f],
            ["coefficient"] = [0.2f, 0.7f, -0.5f, 1.5f],
            ["value"] = [-1f, 3f, 2f, 0.25f],
            ["first"] = [2f, -1f, 0f, 1f],
            ["gate"] = [0.1f, 0.9f, 0f, 1f],
            ["key"] = [0.3f, -0.7f, 2f, -1f],
            ["adaptation"] = [0.2f, 1f, 0.5f, 0f],
            ["key-adapt"] = [0.9f, -0.1f, 0.5f, 2f],
        };
        foreach (var input in inputs.Keys)
            builder.AddResource(input, input, GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, [4]), graphInput: true);

        var portable = new Rwkv7PortableElementwiseBuilder(builder, "graph", 4);
        var mix = portable.Mix("time-mix", "x", "previous", "coefficient");
        var value = portable.InterpolateValue("value-mix", "value", "first", "gate");
        var key = portable.AdaptKey("key", "key", "adaptation", "key-adapt");
        var activation = portable.ReluSquare("ffn", "value");
        var graph = builder.Build();
        GraphValidator.Validate(graph);
        ModelGraphNamingAssertions.Validate(graph);
        Assert.Equal("time-mix.output_output", mix);
        Assert.Equal("value-mix.output_output", value);
        Assert.Equal("key.output_output", key);
        Assert.Equal("ffn.output_output", activation);
        var standard = PrimitiveGraphOperations.CreateStandardDescriptions(false)
            .Select(description => description.Operation).ToHashSet();
        Assert.All(graph.Nodes, node => Assert.Contains(node.Operation, standard));
        Assert.All(graph.Resources.Where(resource => resource.Kind == GraphResourceKind.Temporary),
            resource => Assert.Equal([4], resource.Tensor.Dimensions));

        var values = inputs.ToDictionary(item => item.Key, item => item.Value, StringComparer.Ordinal);
        foreach (var node in graph.Nodes)
        {
            float[] Input(string port) => values[node.Resources.Single(binding => binding.Port == port).Resource.Value];
            var output = new float[4];
            for (var index = 0; index < output.Length; index++)
            {
                output[index] = node.Operation == PrimitiveGraphOperations.Add
                    ? Input("left")[index] + Input("right")[index]
                    : node.Operation == PrimitiveGraphOperations.Subtract
                        ? Input("left")[index] - Input("right")[index]
                        : node.Operation == PrimitiveGraphOperations.Multiply
                            ? Input("left")[index] * Input("right")[index]
                            : node.Operation == PrimitiveGraphOperations.Relu
                                ? MathF.Max(0, Input("input")[index])
                                : node.Operation == PrimitiveGraphOperations.Square
                                    ? Input("input")[index] * Input("input")[index]
                                    : throw new InvalidOperationException($"Unexpected operation: {node.Operation}");
            }
            values.Add(node.Resources.Single(binding => binding.Port == "output").Resource.Value, output);
        }

        for (var i = 0; i < 4; i++)
        {
            Assert.Equal(inputs["x"][i] + (inputs["previous"][i] - inputs["x"][i]) *
                inputs["coefficient"][i], values[mix][i], 6);
            Assert.Equal(inputs["value"][i] + (inputs["first"][i] - inputs["value"][i]) *
                inputs["gate"][i], values[value][i], 6);
            var ka = inputs["key"][i] * inputs["key-adapt"][i];
            Assert.Equal(inputs["key"][i] + inputs["adaptation"][i] * ka - ka, values[key][i], 6);
            Assert.Equal(MathF.Pow(MathF.Max(0, inputs["value"][i]), 2), values[activation][i], 6);
        }
    }

    [Fact]
    public void RejectsNonPositiveVectorWidth()
    {
        var builder = new LogicalGraphBuilder(new GraphIdentity("rwkv-7", 1, "test"),
            TestGraphSignatures.Create(32, 4, 1, 1, 4, "state"));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Rwkv7PortableElementwiseBuilder(builder, "graph", 0));
    }
}
