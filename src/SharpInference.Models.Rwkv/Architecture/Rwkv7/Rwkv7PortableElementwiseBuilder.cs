using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Rwkv7;

/// <summary>Composes RWKV-7 FP32 elementwise expressions in the definition IR.</summary>
internal sealed class Rwkv7PortableElementwiseBuilder
{
    private readonly GraphDefinitionBuilder builder;
    private readonly int width;
    private readonly HashSet<string> names = new(StringComparer.Ordinal);

    internal Rwkv7PortableElementwiseBuilder(
        GraphDefinitionBuilder builder,
        int width)
    {
        this.builder = builder ?? throw new ArgumentNullException(nameof(builder));
        this.width = width > 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));
    }

    internal ResourceId Mix(
        string id,
        ResourceId x,
        ResourceId previous,
        ResourceId coefficient) =>
        Binary($"{id}.output", PrimitiveGraphOperations.Add, x,
            Binary($"{id}.scale", PrimitiveGraphOperations.Multiply,
                Binary($"{id}.difference", PrimitiveGraphOperations.Subtract, previous, x),
                coefficient));

    internal ResourceId InterpolateValue(
        string id,
        ResourceId value,
        ResourceId firstValue,
        ResourceId gate) =>
        Binary($"{id}.output", PrimitiveGraphOperations.Add, value,
            Binary($"{id}.scale", PrimitiveGraphOperations.Multiply,
                Binary($"{id}.difference", PrimitiveGraphOperations.Subtract, firstValue, value),
                gate));

    internal ResourceId AdaptKey(
        string id,
        ResourceId key,
        ResourceId adaptation,
        ResourceId keyAdaptWeight)
    {
        var scaledKey = Binary($"{id}.scaled-key", PrimitiveGraphOperations.Multiply,
            key, keyAdaptWeight);
        return Binary($"{id}.output", PrimitiveGraphOperations.Subtract,
            Binary($"{id}.adapted", PrimitiveGraphOperations.Add, key,
                Binary($"{id}.scale", PrimitiveGraphOperations.Multiply, adaptation, scaledKey)),
            scaledKey);
    }

    internal ResourceId ReluSquare(
        string id,
        ResourceId input) =>
        Unary($"{id}.output", PrimitiveGraphOperations.Square,
            Unary($"{id}.relu", PrimitiveGraphOperations.Relu, input));

    private ResourceId Unary(
        string id,
        GraphOperationId operation,
        ResourceId input)
    {
        var output = AddOutput(id);
        builder.AddNode(id, operation, node =>
        {
            node.AddInput("input", input);
            return node.AddOutput("output", output);
        });
        return output;
    }

    private ResourceId Binary(
        string id,
        GraphOperationId operation,
        ResourceId left,
        ResourceId right)
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

    private ResourceId AddOutput(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !names.Add(id))
        {
            throw new ArgumentException($"Duplicate or invalid elementwise resource '{id}'.", nameof(id));
        }

        var output = $"{id}_output";
        return builder.AddResource(
            output,
            output,
            new TensorDescriptor(GraphElementType.Float32, [width]));
    }
}
