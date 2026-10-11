using SharpInference.Graphs;

namespace SharpInference.Architectures.Rwkv7;

/// <summary>
/// Composes only existing FP32 elementwise primitives on equal-length vectors.
/// Input resources and the region must already exist in the supplied graph builder.
/// No WKV state update or complete RWKV-7 forward graph is implied.
/// </summary>
public sealed class Rwkv7PortableElementwiseBuilder
{
    private readonly LogicalGraphBuilder builder;
    private readonly string region;
    private readonly int width;
    private readonly HashSet<string> names = new(StringComparer.Ordinal);

    /// <summary>Creates an elementwise composer for a graph region and fixed vector width.</summary>
    /// <param name="builder">The graph builder containing the input resources.</param>
    /// <param name="region">The existing region in which to place nodes.</param>
    /// <param name="width">The positive number of elements in each vector.</param>
    public Rwkv7PortableElementwiseBuilder(LogicalGraphBuilder builder, string region, int width)
    {
        this.builder = builder ?? throw new ArgumentNullException(nameof(builder));
        this.region = string.IsNullOrWhiteSpace(region)
            ? throw new ArgumentException("A graph region is required.", nameof(region)) : region;
        this.width = width > 0 ? width : throw new ArgumentOutOfRangeException(nameof(width));
    }

    // x + (previous - x) * coefficient
    /// <summary>Builds linear mixing of the current and previous vectors.</summary>
    /// <param name="id">The unique node and resource prefix.</param>
    /// <param name="x">The current vector resource.</param>
    /// <param name="previous">The previous vector resource.</param>
    /// <param name="coefficient">The per-element mixing coefficient resource.</param>
    /// <returns>The mixed output resource identifier.</returns>
    public string Mix(string id, string x, string previous, string coefficient) =>
        Binary($"{id}.output", PrimitiveGraphOperations.Add, x,
            Binary($"{id}.scale", PrimitiveGraphOperations.Multiply,
                Binary($"{id}.difference", PrimitiveGraphOperations.Subtract, previous, x),
                coefficient));

    // value + (firstValue - value) * gate
    /// <summary>Builds gated interpolation toward the first layer's value vector.</summary>
    /// <param name="id">The unique node and resource prefix.</param>
    /// <param name="value">The current value resource.</param>
    /// <param name="firstValue">The first layer's value resource.</param>
    /// <param name="gate">The interpolation gate resource.</param>
    /// <returns>The interpolated output resource identifier.</returns>
    public string InterpolateValue(string id, string value, string firstValue, string gate) =>
        Binary($"{id}.output", PrimitiveGraphOperations.Add, value,
            Binary($"{id}.scale", PrimitiveGraphOperations.Multiply,
                Binary($"{id}.difference", PrimitiveGraphOperations.Subtract, firstValue, value),
                gate));

    // key + adaptation * (key * keyAdaptWeight) - key * keyAdaptWeight
    /// <summary>Builds the elementwise RWKV-7 key adaptation expression.</summary>
    /// <param name="id">The unique node and resource prefix.</param>
    /// <param name="key">The key vector resource.</param>
    /// <param name="adaptation">The adaptation coefficient resource.</param>
    /// <param name="keyAdaptWeight">The key adaptation weight resource.</param>
    /// <returns>The adapted key resource identifier.</returns>
    public string AdaptKey(string id, string key, string adaptation, string keyAdaptWeight)
    {
        var scaledKey = Binary($"{id}.scaled-key", PrimitiveGraphOperations.Multiply,
            key, keyAdaptWeight);
        return Binary($"{id}.output", PrimitiveGraphOperations.Subtract,
            Binary($"{id}.adapted", PrimitiveGraphOperations.Add, key,
                Binary($"{id}.scale", PrimitiveGraphOperations.Multiply, adaptation, scaledKey)),
            scaledKey);
    }

    // max(0, input)^2
    /// <summary>Builds rectification followed by elementwise squaring.</summary>
    /// <param name="id">The unique node and resource prefix.</param>
    /// <param name="input">The input vector resource.</param>
    /// <returns>The squared rectified output resource identifier.</returns>
    public string ReluSquare(string id, string input) =>
        Unary($"{id}.output", PrimitiveGraphOperations.Square,
            Unary($"{id}.relu", PrimitiveGraphOperations.Relu, input));

    private string Unary(string id, GraphOperationId operation, string input)
    {
        var output = AddOutput(id);
        builder.AddNode(id, operation, region,
            [GraphBindings.Read("input", RequireInput(input)), GraphBindings.Write("output", output)]);
        return output;
    }

    private string Binary(string id, GraphOperationId operation, string left, string right)
    {
        var output = AddOutput(id);
        builder.AddNode(id, operation, region,
            [
                GraphBindings.Read("left", RequireInput(left)),
                GraphBindings.Read("right", RequireInput(right)),
                GraphBindings.Write("output", output),
            ]);
        return output;
    }

    private string AddOutput(string id)
    {
        if (string.IsNullOrWhiteSpace(id) || !names.Add(id))
            throw new ArgumentException($"Duplicate or invalid elementwise resource '{id}'.", nameof(id));
        var output = $"{id}_output";
        builder.AddResource(output, output, GraphResourceKind.Temporary, GraphResourceLifetime.Invocation,
            new TensorDescriptor(GraphElementType.Float32, [width]));
        return output;
    }

    private static string RequireInput(string input) =>
        string.IsNullOrWhiteSpace(input)
            ? throw new ArgumentException("An input resource is required.", nameof(input)) : input;
}
