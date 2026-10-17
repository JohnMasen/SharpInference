using SharpInference.Instructions;
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
        Binary($"{id}.output", PrimitiveOperations.Add, x,
            Binary($"{id}.scale", PrimitiveOperations.Multiply,
                Binary($"{id}.difference", PrimitiveOperations.Subtract, previous, x),
                coefficient));

    internal ResourceId InterpolateValue(
        string id,
        ResourceId value,
        ResourceId firstValue,
        ResourceId gate) =>
        Binary($"{id}.output", PrimitiveOperations.Add, value,
            Binary($"{id}.scale", PrimitiveOperations.Multiply,
                Binary($"{id}.difference", PrimitiveOperations.Subtract, firstValue, value),
                gate));

    internal ResourceId AdaptKey(
        string id,
        ResourceId key,
        ResourceId adaptation,
        ResourceId keyAdaptWeight)
    {
        var scaledKey = Binary($"{id}.scaled-key", PrimitiveOperations.Multiply,
            key, keyAdaptWeight);
        return Binary($"{id}.output", PrimitiveOperations.Subtract,
            Binary($"{id}.adapted", PrimitiveOperations.Add, key,
                Binary($"{id}.scale", PrimitiveOperations.Multiply, adaptation, scaledKey)),
            scaledKey);
    }

    internal ResourceId ReluSquare(
        string id,
        ResourceId input) =>
        Unary($"{id}.output", PrimitiveOperations.Square,
            Unary($"{id}.relu", PrimitiveOperations.Relu, input));

    private ResourceId Unary(
        string id,
        OperationId operation,
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
        OperationId operation,
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
            new TensorDescriptor(ElementType.Float32, [width]));
    }
}
