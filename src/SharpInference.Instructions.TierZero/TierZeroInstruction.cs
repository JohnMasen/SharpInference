using SharpInference.Graphs;

namespace SharpInference.Instructions;

public abstract class TierZeroInstruction(Guid collectionId, string name, InstructionTarget target) : Instruction
{
    public override Guid CollectionId { get; } = collectionId;
    public override string Name { get; } = name;
    public override InstructionTarget Target { get; } = target;
    public override bool RequiresDispatchIsolation => Name is not (
        "core.copy" or "core.add" or "core.subtract" or "core.multiply" or "core.divide" or "core.maximum" or
        "core.exp" or "core.tanh" or "core.sigmoid" or "core.rsqrt" or "core.square" or "core.relu" or
        "core.tensor.fill" or "core.tensor.cast-f16-f32" or "core.tensor.reshape");
    public override IReadOnlyList<InstructionSignature> Signatures { get; } = CreateSignatures(collectionId, name);
    public override IReadOnlyList<InstructionIndexBound> IndexBounds { get; } =
        name == "core.gather-row" ? Array.AsReadOnly<InstructionIndexBound>([new("index", "table", 0)]) : [];

    protected sealed override void Validate(InstructionParameter[] parameters)
    {
        try
        {
            if (parameters.Select(parameter => parameter.Name).Distinct(StringComparer.Ordinal).Count() != parameters.Length)
                throw new InvalidDataException("Duplicate parameters.");
            var tensors = parameters.OfType<InstructionTensorParameter>().ToArray();
            var attributes = parameters.OfType<InstructionAttributeParameter>().ToDictionary(parameter => parameter.Name,
                parameter => parameter.Value, StringComparer.Ordinal);
            if (tensors.Length + attributes.Count != parameters.Length)
                throw new InvalidDataException("Unknown parameter kind.");
            if (tensors.Any(parameter => parameter.ByteOffset != 0))
                throw new InvalidDataException("Put typed offsets on calls, not instruction parameters.");
            var resources = tensors.GroupBy(parameter => parameter.Source).ToDictionary(group => new ResourceId(group.Key),
                group =>
                {
                    var parameter = group.First();
                    return new GraphResource(new(group.Key), group.Key,
                        parameter.Access == GraphResourceAccess.Read ? GraphResourceKind.Input : GraphResourceKind.Temporary,
                        GraphResourceLifetime.Invocation, parameter.Tensor);
                });
            var contract = TierZeroOperationContracts.Get(new(Name));
            var node = new LogicalNode(new("instruction"), contract.Operation, new("instruction"),
                tensors.Select(parameter => new NodeResourceBinding(parameter.Name, new(parameter.Source),
                    parameter.Name == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read)).ToArray(),
                [], attributes, new(GraphElementType.Float32, GraphElementType.Float32));
            if (tensors.Any(parameter => parameter.Name == "output" && parameter.Access == GraphResourceAccess.Read))
                throw new InvalidDataException("Output is read-only.");
            contract.ResolveInputPorts(tensors.Select(parameter => parameter.Name));
            if (tensors.GroupBy(parameter => parameter.Source).Any(group =>
                group.Select(parameter => (parameter.Tensor.ElementType,
                    Shape: string.Join(",", parameter.Tensor.Dimensions))).Distinct().Count() != 1))
                throw new InvalidDataException("Aliased source parameters have conflicting tensor descriptors.");
            var signature = tensors.ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
            if (!Signatures.Any(candidate =>
                candidate.Ports.Count == tensors.Length &&
                candidate.Ports.All(port =>
                    signature.TryGetValue(port.Name == "matrix" && signature.ContainsKey("weight") ? "weight" : port.Name,
                        out var parameter) && parameter.Tensor.ElementType == port.ElementType)))
                throw new InvalidDataException("Unsupported parameter types or ports.");
            if (CollectionId == InstructionCollectionIds.TierZeroFloat16 &&
                Name.StartsWith("core.tensor.", StringComparison.Ordinal))
            {
                resources = resources.ToDictionary(pair => pair.Key, pair => pair.Value with
                {
                    Tensor = new TensorDescriptor(GraphElementType.Float32, pair.Value.Tensor.Dimensions, pair.Value.Tensor.Layout),
                });
            }
            TierZeroOperationContracts.ValidateNode(node, resources,
                allowLegacyFloat16: CollectionId == InstructionCollectionIds.TierZeroFloat16);
        }
        catch (Exception exception) when (exception is InvalidDataException or NotSupportedException or ArgumentException)
        {
            throw new InstructionAdaptationException(CollectionId, Name,
                exception.Message + " Actual: " + string.Join(", ", parameters.Select(parameter =>
                    parameter is InstructionTensorParameter tensor
                        ? $"{tensor.Name}:{tensor.Tensor.ElementType}[{string.Join(",", tensor.Tensor.Dimensions)}]/{tensor.Access}"
                        : parameter.Name)), exception);
        }
    }

    private static IReadOnlyList<InstructionSignature> CreateSignatures(Guid collectionId, string name)
    {
        if (collectionId != InstructionCollectionIds.TierZeroFloat32 &&
            collectionId != InstructionCollectionIds.TierZeroFloat16)
            throw new ArgumentException("Unknown T0 collection.", nameof(collectionId));
        var contract = TierZeroOperationContracts.Get(new(name));
        if (collectionId == InstructionCollectionIds.TierZeroFloat16 && name == "core.tensor.cast-f16-f32")
            return [];
        IReadOnlyList<OperatorSignature> signatures = collectionId == InstructionCollectionIds.TierZeroFloat32 ? contract.Signatures :
            [new OperatorSignature(contract.Signatures[0].InputTypes.Select(type =>
                type == GraphElementType.Float32 ? GraphElementType.Float16 : type), [GraphElementType.Float16])];
        string[] attributes = name switch
        {
            "core.tensor.fill" => ["value"], "core.tensor.slice" => ["axis", "start", "length"], _ => [],
        };
        var result = signatures.Select(signature => new InstructionSignature(
            Array.AsReadOnly(contract.InputPorts.Select((port, index) =>
                new InstructionPort(port, signature.InputTypes[index], GraphResourceAccess.Read))
                .Append(new("output", signature.OutputTypes[0], GraphResourceAccess.Write)).ToArray()),
            Array.AsReadOnly(attributes), contract.Precision)).ToList();
        if (name == "core.mat-vec")
            result.AddRange(result.ToArray().Select(signature => new InstructionSignature(
                Array.AsReadOnly(signature.Ports.Select(port => port.Name == "matrix" ? port with { Name = "weight" } : port).ToArray()),
                signature.Attributes, signature.Precision)));
        return result.AsReadOnly();
    }
}
