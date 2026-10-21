using SharpInference.Instructions;
using SharpInference.Graphs;
using SharpInference.Graphs.Logical;

namespace SharpInference.Architectures.Rwkv7;

/// <summary>Composes RWKV-7 FP32 elementwise expressions in a logical graph.</summary>
internal sealed class Rwkv7PortableElementwiseBuilder
{
    private readonly LogicalGraph graph;
    private readonly int width;
    private readonly HashSet<string> names = new(StringComparer.Ordinal);

    internal Rwkv7PortableElementwiseBuilder(
        LogicalGraph graph,
        int width)
    {
        this.graph = graph ?? throw new ArgumentNullException(nameof(graph));
        this.width = width > 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));
    }

    internal string Mix(
        string id,
        string x,
        string previous,
        string coefficient) =>
        Binary($"{id}.output", PrimitiveOperations.Add, x,
            Binary($"{id}.scale", PrimitiveOperations.Multiply,
                Binary($"{id}.difference", PrimitiveOperations.Subtract, previous, x),
                coefficient));

    internal string InterpolateValue(
        string id,
        string value,
        string firstValue,
        string gate) =>
        Binary($"{id}.output", PrimitiveOperations.Add, value,
            Binary($"{id}.scale", PrimitiveOperations.Multiply,
                Binary($"{id}.difference", PrimitiveOperations.Subtract, firstValue, value),
                gate));

    internal string AdaptKey(
        string id,
        string key,
        string adaptation,
        string keyAdaptWeight)
    {
        var scaledKey = Binary($"{id}.scaled-key", PrimitiveOperations.Multiply,
            key, keyAdaptWeight);
        return Binary($"{id}.output", PrimitiveOperations.Subtract,
            Binary($"{id}.adapted", PrimitiveOperations.Add, key,
                Binary($"{id}.scale", PrimitiveOperations.Multiply, adaptation, scaledKey)),
            scaledKey);
    }

    internal string ReluSquare(
        string id,
        string input) =>
        Unary($"{id}.output", PrimitiveOperations.Square,
            Unary($"{id}.relu", PrimitiveOperations.Relu, input));

    private string Unary(
        string id,
        OperationId operation,
        string input)
    {
        var output = AddOutput(id);
        var node = graph.Operation(id, operation);
        node.Bind("input", input, ResourceAccess.Read);
        node.Bind("output", output, ResourceAccess.Write);
        return output;
    }

    private string Binary(
        string id,
        OperationId operation,
        string left,
        string right)
    {
        var output = AddOutput(id);
        var node = graph.Operation(id, operation);
        node.Bind("left", left, ResourceAccess.Read);
        node.Bind("right", right, ResourceAccess.Read);
        node.Bind("output", output, ResourceAccess.Write);
        return output;
    }

    private string AddOutput(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !names.Add(id))
        {
            throw new ArgumentException($"Duplicate or invalid elementwise resource '{id}'.", nameof(id));
        }

        var output = $"{id}_output";
        graph.Resource(output, new TensorDescriptor(ElementType.Float32, [width]));
        return output;
    }
}
