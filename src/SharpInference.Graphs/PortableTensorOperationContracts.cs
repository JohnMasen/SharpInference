using System.Globalization;

namespace SharpInference.Graphs;

public sealed record PortableTensorOpContract(
    GraphOperationId Operation,
    OperatorSignature Signature,
    IReadOnlyList<string> InputPorts);

/// <summary>
/// Typed attribute value for a Float32 tensor fill. Non-finite values are not supported.
/// </summary>
public readonly record struct TensorFillValue(float Value)
{
    public IReadOnlyDictionary<string, string> ToAttributes()
    {
        if (!float.IsFinite(Value))
            throw new ArgumentOutOfRangeException(nameof(Value));
        return new Dictionary<string, string> { ["value"] = Value.ToString("R", CultureInfo.InvariantCulture) };
    }
}

/// <summary>
/// Zero-based slice axis and start, with a strictly positive number of elements.
/// A slice preserves rank and copies the selected elements in row-major order.
/// </summary>
public readonly record struct TensorSlice(int Axis, int Start, int Length)
{
    public IReadOnlyDictionary<string, string> ToAttributes()
    {
        if (Axis < 0 || Start < 0 || Length <= 0)
            throw new ArgumentOutOfRangeException(nameof(Length), "Slice axis/start must be non-negative and length positive.");
        return new Dictionary<string, string>
        {
            ["axis"] = Axis.ToString(CultureInfo.InvariantCulture),
            ["start"] = Start.ToString(CultureInfo.InvariantCulture),
            ["length"] = Length.ToString(CultureInfo.InvariantCulture),
        };
    }
}

/// <summary>
/// Versioned tensor-operation contracts shared by logical graphs and T0 VM backends.
/// Call ValidateGraph explicitly when validating portable tensor operations alone.
/// Inputs and outputs are dense; dimensions are row-major in their declared order.
/// All arithmetic and reductions use FP32.
/// </summary>
public static class PortableTensorOperationContracts
{
    public static readonly GraphOperationId Fill = new("core.tensor.fill", 1);
    public static readonly GraphOperationId CastFp16ToFp32 = new("core.tensor.cast-f16-f32", 1);
    public static readonly GraphOperationId Reshape = new("core.tensor.reshape", 1);
    public static readonly GraphOperationId Slice = new("core.tensor.slice", 1);
    public static readonly GraphOperationId Broadcast = new("core.tensor.broadcast", 1);
    public static readonly GraphOperationId BatchedMatVec = new("core.tensor.batched-mat-vec", 1);
    public static readonly GraphOperationId ReduceLastSum = new("core.tensor.reduce-last-sum", 1);
    public static readonly GraphOperationId ReduceLastMean = new("core.tensor.reduce-last-mean", 1);
    public static readonly GraphOperationId HeadOuter = new("core.tensor.head-outer", 1);

    // Reshape preserves flat element order. Broadcast aligns trailing dimensions
    // and repeats input axes of length one. BatchedMatVec computes
    // y[b,i] = sum_j matrix[b,i,j] * vector[b,j].
    // ReduceLast{Sum,Mean} removes the final axis (e.g. [heads,width] -> [heads]
    // or [heads,width,width] -> [heads,width]).
    // HeadOuter computes output[h,i,j] = left[h,i] * right[h,j].

    public static IReadOnlyList<PortableTensorOpContract> Contracts { get; } =
    [
        Description(Fill, [], [GraphElementType.Float32], []),
        Description(CastFp16ToFp32, [GraphElementType.Float16], [GraphElementType.Float32], ["input"]),
        Description(Reshape, [GraphElementType.Float32], [GraphElementType.Float32], ["input"]),
        Description(Slice, [GraphElementType.Float32], [GraphElementType.Float32], ["input"]),
        Description(Broadcast, [GraphElementType.Float32], [GraphElementType.Float32], ["input"]),
        Description(BatchedMatVec, [GraphElementType.Float32, GraphElementType.Float32],
            [GraphElementType.Float32], ["matrix", "vector"]),
        Description(ReduceLastSum, [GraphElementType.Float32], [GraphElementType.Float32], ["input"]),
        Description(ReduceLastMean, [GraphElementType.Float32], [GraphElementType.Float32], ["input"]),
        Description(HeadOuter, [GraphElementType.Float32, GraphElementType.Float32],
            [GraphElementType.Float32], ["left", "right"]),
    ];

    public static void ValidateGraph(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        foreach (var node in graph.Nodes.Where(node =>
                     node.Operation.Name.StartsWith("core.tensor.", StringComparison.Ordinal)))
            ValidateNode(node, resources);
    }

    public static void ValidateNode(
        LogicalNode node, IReadOnlyDictionary<ResourceId, GraphResource> resources)
    {
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(resources);
        var contract = Contracts.SingleOrDefault(candidate => candidate.Operation == node.Operation)
            ?? throw new InvalidDataException($"Unsupported tensor operation or version '{node.Operation}'.");
        if (!NumericTypeCompatibility.Satisfies(
            new KernelPrecisionProfile(GraphElementType.Float32, GraphElementType.Float32), node.Requirements))
            throw new InvalidDataException($"Tensor operation '{node.Id}' requires FP32 arithmetic and accumulation.");

        var ports = contract.InputPorts;
        var expected = ports.Append("output").ToArray();
        if (node.Resources.Count != expected.Length ||
            !expected.ToHashSet(StringComparer.Ordinal).SetEquals(node.Resources.Select(binding => binding.Port)) ||
            node.Resources.Any(binding => binding.Access !=
                (binding.Port == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read) ||
                binding.InitializedBeforeRead))
            throw new InvalidDataException($"Tensor operation '{node.Id}' has invalid ports or access.");

        TensorDescriptor Get(string port)
        {
            var id = node.Resources.Single(binding => binding.Port == port).Resource;
            if (!resources.TryGetValue(id, out var resource))
                throw new InvalidDataException($"Tensor operation '{node.Id}' references unknown resource '{id}'.");
            if (resource.Tensor.Layout != "dense" || resource.Tensor.Dimensions.Count == 0)
                throw new InvalidDataException($"Tensor operation '{node.Id}' requires dense, non-scalar tensors.");
            return resource.Tensor;
        }
        var outputId = node.Resources.Single(binding => binding.Port == "output").Resource;
        if (node.Resources.Any(binding => binding.Port != "output" && binding.Resource == outputId))
            throw new InvalidDataException($"Tensor operation '{node.Id}' cannot overwrite an input.");
        if (resources.TryGetValue(outputId, out var outputResource) &&
            outputResource.Kind is GraphResourceKind.Input or GraphResourceKind.Weight or GraphResourceKind.Constant)
            throw new InvalidDataException($"Tensor operation '{node.Id}' cannot write a read-only resource.");
        var output = Get("output");
        var inputs = ports.Select(Get).ToArray();
        var signature = new OperatorSignature(inputs.Select(tensor => tensor.ElementType),
            [output.ElementType]);
        var mixedMatrix = node.Operation == BatchedMatVec &&
            inputs[0].ElementType == GraphElementType.Float16 &&
            inputs[1].ElementType == GraphElementType.Float32 &&
            output.ElementType == GraphElementType.Float32;
        if (!contract.Signature.Matches(signature) && !mixedMatrix)
            throw new InvalidDataException($"Tensor operation '{node.Id}' has unsupported element types.");

        var dims = output.Dimensions;
        if (node.Operation == Fill)
        {
            RequireAttributes(node, "value");
            var text = node.Attributes["value"];
            if (text != text.Trim() ||
                !float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ||
                !float.IsFinite(value))
                throw new InvalidDataException($"Tensor fill '{node.Id}' requires a finite invariant-culture FP32 value.");
        }
        else if (node.Operation == Slice)
        {
            RequireAttributes(node, "axis", "start", "length");
            var axis = ParseNonNegative(node, "axis");
            var start = ParseNonNegative(node, "start");
            var length = ParseNonNegative(node, "length");
            var source = inputs[0].Dimensions;
            if (axis >= source.Count || length == 0 || start > source[axis] ||
                (long)start + length > source[axis] || dims.Count != source.Count ||
                dims.Where((value, index) => value != (index == axis ? length : source[index])).Any())
                throw new InvalidDataException($"Tensor slice '{node.Id}' has incompatible dimensions.");
        }
        else
        {
            RequireAttributes(node);
            if (node.Operation == CastFp16ToFp32 && !dims.SequenceEqual(inputs[0].Dimensions))
                throw ShapeError(node);
            if (node.Operation == Reshape &&
                ElementCount(dims) != ElementCount(inputs[0].Dimensions))
                throw ShapeError(node);
            if (node.Operation == Broadcast)
            {
                var source = inputs[0].Dimensions;
                if (source.Count > dims.Count || source.Where((value, index) =>
                    value != 1 && value != dims[dims.Count - source.Count + index]).Any())
                    throw ShapeError(node);
            }
            if (node.Operation == BatchedMatVec)
            {
                var matrix = inputs[0].Dimensions;
                var vector = inputs[1].Dimensions;
                if (matrix.Count != 3 || vector.Count != 2 || dims.Count != 2 ||
                    matrix[0] != vector[0] || matrix[2] != vector[1] ||
                    dims[0] != matrix[0] || dims[1] != matrix[1])
                    throw ShapeError(node);
            }
            if (node.Operation == ReduceLastSum || node.Operation == ReduceLastMean)
            {
                var source = inputs[0].Dimensions;
                if (source.Count is not (2 or 3) || dims.Count != source.Count - 1 ||
                    !dims.SequenceEqual(source.Take(source.Count - 1)))
                    throw ShapeError(node);
            }
            if (node.Operation == HeadOuter)
            {
                var left = inputs[0].Dimensions;
                var right = inputs[1].Dimensions;
                if (left.Count != 2 || right.Count != 2 || dims.Count != 3 ||
                    left[0] != right[0] || dims[0] != left[0] ||
                    dims[1] != left[1] || dims[2] != right[1])
                    throw ShapeError(node);
            }
        }
    }

    private static PortableTensorOpContract Description(
        GraphOperationId operation, GraphElementType[] inputs, GraphElementType[] outputs,
        string[] inputPorts) =>
        new(operation, new OperatorSignature(inputs, outputs), Array.AsReadOnly(inputPorts));

    private static void RequireAttributes(LogicalNode node, params string[] keys)
    {
        if (node.Attributes.Count != keys.Length ||
            !keys.ToHashSet(StringComparer.Ordinal).SetEquals(node.Attributes.Keys))
            throw new InvalidDataException($"Tensor operation '{node.Id}' has invalid attributes.");
    }

    private static int ParseNonNegative(LogicalNode node, string key)
    {
        var text = node.Attributes[key];
        if (text.Length == 0 || text.Any(character => character is < '0' or > '9') ||
            !int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var result))
            throw new InvalidDataException($"Tensor operation '{node.Id}' requires a non-negative integer '{key}'.");
        return result;
    }

    private static long ElementCount(IReadOnlyList<int> dimensions)
    {
        try
        {
            return dimensions.Aggregate(1L, (size, dimension) => checked(size * dimension));
        }
        catch (OverflowException exception)
        {
            throw new InvalidDataException("Tensor element count exceeds Int64.", exception);
        }
    }

    private static InvalidDataException ShapeError(LogicalNode node) =>
        new($"Tensor operation '{node.Id}' has incompatible dimensions.");
}
