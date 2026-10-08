using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Vm;

/// <summary>Converts VM operator arguments and attributes into instruction parameters.</summary>
public static class VmInstructionParameters
{
    /// <summary>Creates instruction parameters using VM tensor metadata and an expression resolver.</summary>
    /// <param name="operation">The VM operator to convert.</param>
    /// <param name="parameters">The definition parameters referenced by the operator.</param>
    /// <param name="expression">Resolves each VM parameter to an instruction expression and offset.</param>
    /// <returns>Instruction tensor and attribute parameters in the operator's argument order.</returns>
    public static InstructionParameter[] Create(VmOperator operation, IReadOnlyList<VmParameter> parameters,
        Func<VmParameter, (string Expression, string Offset)> expression)
    {
        ArgumentNullException.ThrowIfNull(operation);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(expression);
        if (operation.Version != 1)
            throw new NotSupportedException("Versioned legacy operator identities are not supported by IC assembly.");
        var byName = parameters.ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var result = new List<InstructionParameter>();
        foreach (var argument in operation.Arguments)
        {
            if (!byName.TryGetValue(argument.Source, out var parameter))
                throw new InvalidDataException($"Unknown instruction parameter '{argument.Source}'.");
            var (value, offset) = expression(parameter);
            var type = parameter.Tensor.ElementType switch
            {
                VmElementType.Float16 => GraphElementType.Float16,
                VmElementType.Float32 => GraphElementType.Float32,
                VmElementType.Int32 => GraphElementType.Int32,
                VmElementType.UInt32 => GraphElementType.UInt32,
                VmElementType.Byte => GraphElementType.Byte,
                _ => throw new NotSupportedException($"Unsupported instruction type '{parameter.Tensor.ElementType}'."),
            };
            result.Add(new InstructionTensorParameter(argument.Parameter, new(type, parameter.Tensor.Dimensions),
                parameter.Access == VmAccess.ReadOnly ? GraphResourceAccess.Read : GraphResourceAccess.ReadWrite,
                parameter.Name, value, offset, argument.ByteOffset));
        }
        result.AddRange(operation.Attributes.OrderBy(pair => pair.Key, StringComparer.Ordinal)
            .Select(pair => new InstructionAttributeParameter(pair.Key, pair.Value)));
        return result.ToArray();
    }
}
