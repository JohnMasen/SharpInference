using SharpInference.Graphs;

namespace SharpInference.Vm;

public static class VmTierZeroOperations
{
    public static IReadOnlyList<string> Validate(VmOperator operation,
        IReadOnlyList<VmParameter> parameters, bool allowLegacyFloat16 = false)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(parameters);
        var contract = TierZeroOperationContracts.Get(new(operation.Operation, operation.Version));
        var byName = parameters.ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var resources = new Dictionary<ResourceId, GraphResource>();
        foreach (var argument in operation.Arguments)
        {
            if (argument.ByteOffset != 0)
                throw new NotSupportedException("T0 operator views require full parameters; put offsets on calls.");
            if (!byName.TryGetValue(argument.Source, out var parameter))
                throw new InvalidDataException($"T0 operator references unknown parameter '{argument.Source}'.");
            var type = parameter.Tensor.ElementType switch
            {
                VmElementType.Float32 => GraphElementType.Float32,
                VmElementType.Float16 => GraphElementType.Float16,
                VmElementType.Int32 => GraphElementType.Int32,
                VmElementType.UInt32 => GraphElementType.UInt32,
                VmElementType.Byte => GraphElementType.Byte,
                _ => throw new NotSupportedException($"Unsupported T0 element type '{parameter.Tensor.ElementType}'."),
            };
            var id = new ResourceId(argument.Source);
            resources.TryAdd(id, new(id, argument.Source,
                parameter.Access == VmAccess.ReadOnly ? GraphResourceKind.Input : GraphResourceKind.Temporary,
                GraphResourceLifetime.Invocation, new(type, parameter.Tensor.Dimensions)));
        }
        var node = new LogicalNode(new("vm"), contract.Operation, new("vm"),
            operation.Arguments.Select(argument => new NodeResourceBinding(argument.Parameter, new(argument.Source),
                argument.Parameter == "output" ? GraphResourceAccess.Write : GraphResourceAccess.Read)).ToArray(),
            [], operation.Attributes, new(GraphElementType.Float32, GraphElementType.Float32));
        TierZeroOperationContracts.ValidateNode(node, resources, allowLegacyFloat16);
        return contract.ResolveInputPorts(operation.Arguments.Select(argument => argument.Parameter));
    }
}
