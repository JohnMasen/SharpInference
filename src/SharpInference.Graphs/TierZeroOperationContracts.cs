using System.Collections.Frozen;

namespace SharpInference.Graphs;

public sealed class TierZeroOperationContract
{
    internal TierZeroOperationContract(GraphOperationId operation, IEnumerable<string> inputPorts,
        IEnumerable<OperatorSignature> signatures, bool required)
    {
        Operation = operation;
        InputPorts = Array.AsReadOnly(inputPorts.ToArray());
        Signatures = Array.AsReadOnly(signatures.ToArray());
        RequiredByRwkv = required;
    }

    public GraphOperationId Operation { get; }
    public IReadOnlyList<string> InputPorts { get; }
    public IReadOnlyList<OperatorSignature> Signatures { get; }
    public bool RequiredByRwkv { get; }
    public KernelPrecisionProfile Precision { get; } =
        new(GraphElementType.Float32, GraphElementType.Float32);

    public IReadOnlyList<string> ResolveInputPorts(IEnumerable<string> ports)
    {
        ArgumentNullException.ThrowIfNull(ports);
        // Existing logical graphs use "weight"; "matrix" is the canonical T0 port.
        return Operation == PrimitiveGraphOperations.MatVec && ports.Contains("weight", StringComparer.Ordinal)
            ? Array.AsReadOnly(new[] { "weight", "input" }) : InputPorts;
    }
}

public static class TierZeroOperationContracts
{
    public const string Profile = "core.t0.fp32@1";

    public static IReadOnlyList<TierZeroOperationContract> Contracts { get; } = CreateContracts();
    public static IReadOnlyList<TierZeroOperationContract> RequiredContracts { get; } =
        Array.AsReadOnly(Contracts.Where(contract => contract.RequiredByRwkv).ToArray());

    private static readonly FrozenDictionary<GraphOperationId, TierZeroOperationContract> ByOperation =
        Contracts.ToFrozenDictionary(contract => contract.Operation);

    public static TierZeroOperationContract Get(GraphOperationId operation) =>
        ByOperation.TryGetValue(operation, out var contract) ? contract :
            throw new NotSupportedException($"Unsupported T0 operation or version '{operation}'.");

    public static void ValidateGraph(LogicalGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        var resources = graph.Resources.ToDictionary(resource => resource.Id);
        foreach (var node in graph.Nodes)
            ValidateNode(node, resources);
    }

    public static void ValidateNode(LogicalNode node,
        IReadOnlyDictionary<ResourceId, GraphResource> resources, bool allowLegacyFloat16 = false)
    {
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(resources);
        var contract = Get(node.Operation);
        if (node.Operation.Name.StartsWith("core.tensor.", StringComparison.Ordinal))
        {
            PortableTensorOperationContracts.ValidateNode(node, resources);
            return;
        }
        InvalidDataException Error(string reason) => new($"T0 operation '{node.Id}' ({node.Operation}): {reason}");
        if (!NumericTypeCompatibility.Satisfies(contract.Precision, node.Requirements))
            throw Error("requires FP32 arithmetic and accumulation.");
        var ports = contract.ResolveInputPorts(node.Resources.Select(binding => binding.Port));
        if (node.Attributes.Count != 0 || node.Resources.Count != ports.Count + 1 ||
            !ports.Append("output").ToHashSet(StringComparer.Ordinal)
                .SetEquals(node.Resources.Select(binding => binding.Port)) ||
            node.Resources.Any(binding => binding.Access !=
                (binding.Port == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read)))
            throw Error("invalid ports, attributes or access.");
        GraphResource GetResource(string port)
        {
            var binding = node.Resources.Single(binding => binding.Port == port);
            if (!resources.TryGetValue(binding.Resource, out var resource))
                throw Error($"unknown resource '{binding.Resource}'.");
            if (resource.Tensor.Layout != "dense" || resource.Tensor.Dimensions.Count == 0)
                throw Error("requires dense, non-scalar tensors.");
            return resource;
        }
        var output = GetResource("output");
        var inputs = ports.Select(GetResource).ToArray();
        if (output.Kind is GraphResourceKind.Input or GraphResourceKind.Weight or GraphResourceKind.Constant)
            throw Error("cannot write a read-only resource.");
        if (inputs.Any(input => input.Id == output.Id))
            throw Error("cannot overwrite an input.");
        var signature = new OperatorSignature(inputs.Select(input => input.Tensor.ElementType),
            [output.Tensor.ElementType]);
        var legacySignature = allowLegacyFloat16 &&
            output.Tensor.ElementType == GraphElementType.Float16 &&
            inputs.All(input => input.Tensor.ElementType ==
                (node.Operation == PrimitiveGraphOperations.GatherRow && input == inputs[1]
                    ? GraphElementType.Int32 : GraphElementType.Float16));
        if (!contract.Signatures.Any(candidate => candidate.Matches(signature)) && !legacySignature)
            throw new NotSupportedException($"T0 operation '{node.Operation}' has unsupported element types.");
        var shape = inputs[0].Tensor.Dimensions;
        var dimensions = output.Tensor.Dimensions;
        var valid = node.Operation.Name switch
        {
            "core.mat-vec" => shape.Count == 2 &&
                inputs[1].Tensor.Dimensions.SequenceEqual([shape[1]]) &&
                dimensions.SequenceEqual([shape[0]]),
            "core.gather-row" => shape.Count == 2 &&
                inputs[1].Tensor.Dimensions.SequenceEqual([1]) &&
                dimensions.SequenceEqual([shape[1]]),
            "core.reduce-sum" or "core.reduce-mean" => dimensions.SequenceEqual([1]),
            _ => inputs.All(input => input.Tensor.Dimensions.SequenceEqual(dimensions)),
        };
        if (!valid) throw Error("incompatible dimensions.");
    }

    private static IReadOnlyList<TierZeroOperationContract> CreateContracts()
    {
        const GraphElementType f32 = GraphElementType.Float32;
        const GraphElementType f16 = GraphElementType.Float16;
        var contracts = PrimitiveGraphOperations.CreateStandardDescriptions(includeFp16: false)
            .Select(description =>
            {
                var name = description.Operation.Name;
                string[] ports = name switch
                {
                    "core.gather-row" => ["table", "index"],
                    "core.mat-vec" => ["matrix", "input"],
                    "core.add" or "core.subtract" or "core.multiply" or "core.divide" or "core.maximum" =>
                        ["left", "right"],
                    _ => ["input"],
                };
                OperatorSignature[] signatures = name switch
                {
                    "core.gather-row" => [new([f32, GraphElementType.Int32], [f32]),
                        new([f16, GraphElementType.Int32], [f32])],
                    "core.mat-vec" => [new([f32, f32], [f32]), new([f16, f32], [f32])],
                    _ => description.Signatures.ToArray(),
                };
                return new TierZeroOperationContract(description.Operation, ports, signatures,
                    name is not ("core.divide" or "core.reduce-sum"));
            }).ToList();
        contracts.AddRange(PortableTensorOperationContracts.Contracts.Select(contract =>
            new TierZeroOperationContract(contract.Operation, contract.InputPorts,
                contract.Operation == PortableTensorOperationContracts.BatchedMatVec
                    ? [contract.Signature, new OperatorSignature([f16, f32], [f32])]
                    : [contract.Signature], required: true)));
        return Array.AsReadOnly(contracts.ToArray());
    }
}
