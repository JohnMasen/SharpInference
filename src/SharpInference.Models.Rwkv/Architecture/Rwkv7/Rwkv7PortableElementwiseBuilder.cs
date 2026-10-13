using SharpInference.Graphs;

namespace SharpInference.Architectures.Rwkv7;

/// <summary>Composes RWKV-7 FP32 elementwise expressions in the definition IR.</summary>
internal sealed class Rwkv7PortableElementwiseBuilder
{
    private readonly GraphDefinitionBuilder<RwkvGraphPorts> builder;
    private readonly int width;
    private readonly HashSet<string> names = new(StringComparer.Ordinal);

    internal Rwkv7PortableElementwiseBuilder(
        GraphDefinitionBuilder<RwkvGraphPorts> builder,
        int width)
    {
        this.builder = builder ?? throw new ArgumentNullException(nameof(builder));
        this.width = width > 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));
    }

    internal GraphResource<RwkvTensor> Mix(
        string id,
        GraphResource<RwkvTensor> x,
        GraphResource<RwkvTensor> previous,
        GraphResource<RwkvTensor> coefficient) =>
        Binary($"{id}.output", PrimitiveGraphOperations.Add, x,
            Binary($"{id}.scale", PrimitiveGraphOperations.Multiply,
                Binary($"{id}.difference", PrimitiveGraphOperations.Subtract, previous, x),
                coefficient));

    internal GraphResource<RwkvTensor> InterpolateValue(
        string id,
        GraphResource<RwkvTensor> value,
        GraphResource<RwkvTensor> firstValue,
        GraphResource<RwkvTensor> gate) =>
        Binary($"{id}.output", PrimitiveGraphOperations.Add, value,
            Binary($"{id}.scale", PrimitiveGraphOperations.Multiply,
                Binary($"{id}.difference", PrimitiveGraphOperations.Subtract, firstValue, value),
                gate));

    internal GraphResource<RwkvTensor> AdaptKey(
        string id,
        GraphResource<RwkvTensor> key,
        GraphResource<RwkvTensor> adaptation,
        GraphResource<RwkvTensor> keyAdaptWeight)
    {
        var scaledKey = Binary($"{id}.scaled-key", PrimitiveGraphOperations.Multiply,
            key, keyAdaptWeight);
        return Binary($"{id}.output", PrimitiveGraphOperations.Subtract,
            Binary($"{id}.adapted", PrimitiveGraphOperations.Add, key,
                Binary($"{id}.scale", PrimitiveGraphOperations.Multiply, adaptation, scaledKey)),
            scaledKey);
    }

    internal GraphResource<RwkvTensor> ReluSquare(
        string id,
        GraphResource<RwkvTensor> input) =>
        Unary($"{id}.output", PrimitiveGraphOperations.Square,
            Unary($"{id}.relu", PrimitiveGraphOperations.Relu, input));

    private GraphResource<RwkvTensor> Unary(
        string id,
        GraphOperationId operation,
        GraphResource<RwkvTensor> input)
    {
        var output = AddOutput(id);
        builder.AddNode(id, operation, node =>
        {
            node.AddInput("input", input);
            return node.AddOutput("output", output);
        });
        return output;
    }

    private GraphResource<RwkvTensor> Binary(
        string id,
        GraphOperationId operation,
        GraphResource<RwkvTensor> left,
        GraphResource<RwkvTensor> right)
    {
        var output = AddOutput(id);
        builder.AddNode(id, operation, node =>
        {
            node.AddInput("left", left);
            node.AddInput("right", right);
            return node.AddOutput("output", output);
        });
        return output;
    }

    private GraphResource<RwkvTensor> AddOutput(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !names.Add(id))
        {
            throw new ArgumentException($"Duplicate or invalid elementwise resource '{id}'.", nameof(id));
        }

        var output = $"{id}_output";
        return builder.AddResource<RwkvTensor>(
            output,
            output,
            new TensorDescriptor(GraphElementType.Float32, [width]));
    }
}
