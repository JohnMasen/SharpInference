using System.Text.Json;

namespace SharpInference.Graphs;

/// <summary>Provides implementations for fused elementwise expressions with a specified signature and tensor shape.</summary>
public interface IFusedElementwiseExpressionProvider
{
    /// <summary>Returns backend implementations matching the fused expression signature and tensor descriptor.</summary>
    /// <param name="signature">The ordered input and output element-type signature.</param>
    /// <param name="tensor">The tensor element type, dimensions, and layout.</param>
    IReadOnlyList<OperatorImplementationDescription> GetExpressionImplementations(
        OperatorSignature signature, TensorDescriptor tensor);
}

/// <summary>References either an expression input or an earlier instruction result.</summary>
/// <param name="InputIndex">The zero-based expression input index, or null when referencing a step.</param>
/// <param name="StepIndex">The zero-based earlier instruction index, or null when referencing an input.</param>
public sealed record ElementwiseOperand(int? InputIndex = null, int? StepIndex = null);

/// <summary>Describes one primitive operation and its operands in a fused expression.</summary>
/// <param name="Operation">The operation contract identifier.</param>
/// <param name="Operands">The references to inputs or preceding step results.</param>
public sealed record ElementwiseInstruction(
    GraphOperationId Operation, IReadOnlyList<ElementwiseOperand> Operands);

/// <summary>Describes an ordered expression whose instruction results are materialized in FP32.</summary>
/// <param name="InputCount">The number of external expression inputs.</param>
/// <param name="Steps">The expression instructions in execution order.</param>
public sealed record FusedElementwiseExpression(
    int InputCount, IReadOnlyList<ElementwiseInstruction> Steps);

/// <summary>Steps execute in order, materializing each result as FP32 without reassociation.</summary>
public static class FusedElementwiseExpressionContract
{
    /// <summary>The versioned operation identifier for an ordered fused elementwise expression.</summary>
    public static readonly GraphOperationId Operation = new("core.fused-elementwise-expression", 1);
    /// <summary>The attribute key storing the serialized fused elementwise expression.</summary>
    public const string Attribute = "expression";

    /// <summary>Validates the typed attribute value and encodes it as portable operation attributes.</summary>
    /// <param name="expression">The ordered fused elementwise expression.</param>
    public static IReadOnlyDictionary<string, string> ToAttributes(FusedElementwiseExpression expression)
    {
        Validate(expression);
        return new Dictionary<string, string> { [Attribute] = JsonSerializer.Serialize(expression) };
    }

    /// <summary>Reads and validates the fused elementwise expression and its node bindings.</summary>
    /// <param name="node">The operation node to inspect.</param>
    public static FusedElementwiseExpression Read(ExecutionNode node)
    {
        ArgumentNullException.ThrowIfNull(node);
        if (node.Operation != Operation || node.Attributes.Count != 1 ||
            !node.Attributes.TryGetValue(Attribute, out var json))
            throw new InvalidDataException("Invalid fused elementwise expression node.");
        FusedElementwiseExpression expression;
        try
        {
            expression = JsonSerializer.Deserialize<FusedElementwiseExpression>(json)
                ?? throw new InvalidDataException("Missing fused elementwise expression.");
        }
        catch (JsonException error)
        {
            throw new InvalidDataException("Malformed fused elementwise expression.", error);
        }

        Validate(expression);
        if (node.Resources.Count != expression.InputCount + 1 ||
            node.Resources.Take(expression.InputCount)
                .Select((binding, index) => (binding, index))
                .Any(item => item.binding.Port != $"input{item.index}" ||
                    item.binding.Access != GraphResourceAccess.Read) ||
            node.Resources[^1].Port != "output" ||
            node.Resources[^1].Access != GraphResourceAccess.Write ||
            node.Resources.Take(expression.InputCount).Any(binding =>
                binding.Resource == node.Resources[^1].Resource) ||
            node.InternalResources.Count != 0 || node.FirstWriteResources.Count != 0)
            throw new InvalidDataException("Invalid fused elementwise expression bindings.");
        return expression;
    }

    /// <summary>Reads and validates the fused elementwise expression and its node bindings.</summary>
    /// <param name="node">The operation node to inspect.</param>
    /// <param name="resources">The graph resource descriptors.</param>
    public static FusedElementwiseExpression Read(
        ExecutionNode node, IReadOnlyDictionary<ResourceId, GraphResource> resources)
    {
        ArgumentNullException.ThrowIfNull(resources);
        var expression = Read(node);
        var target = resources[node.Resources[^1].Resource];
        var output = target.Tensor;
        if (node.Requirements.MinimumArithmeticType != GraphElementType.Float32 ||
            node.Requirements.MinimumAccumulatorType != GraphElementType.Float32 ||
            target.Kind is GraphResourceKind.Input or GraphResourceKind.Weight or
                GraphResourceKind.Constant or GraphResourceKind.SessionState ||
            target.Scope != GraphResourceScope.Local || target.BindingKey is not null ||
            output.ElementType != GraphElementType.Float32 || output.Layout != "dense" ||
            node.Resources.Any(binding =>
            {
                var tensor = resources[binding.Resource].Tensor;
                return tensor.ElementType != GraphElementType.Float32 || tensor.Layout != "dense" ||
                    !tensor.Dimensions.SequenceEqual(output.Dimensions);
            }))
            throw new InvalidDataException("Fused elementwise expression requires identical dense FP32 tensors.");
        return expression;
    }

    /// <summary>Returns the supported elementwise operation&apos;s operand count, or zero for an unsupported operation.</summary>
    /// <param name="operation">The operation contract identifier.</param>
    public static int Arity(GraphOperationId operation)
    {
        if (operation == PrimitiveGraphOperations.Add ||
            operation == PrimitiveGraphOperations.Subtract ||
            operation == PrimitiveGraphOperations.Multiply ||
            operation == PrimitiveGraphOperations.Divide ||
            operation == PrimitiveGraphOperations.Maximum)
            return 2;
        if (operation == PrimitiveGraphOperations.Copy ||
            operation == PrimitiveGraphOperations.Exp ||
            operation == PrimitiveGraphOperations.Tanh ||
            operation == PrimitiveGraphOperations.Sigmoid ||
            operation == PrimitiveGraphOperations.ReciprocalSquareRoot ||
            operation == PrimitiveGraphOperations.Square ||
            operation == PrimitiveGraphOperations.Relu)
            return 1;
        return 0;
    }

    private static void Validate(FusedElementwiseExpression expression)
    {
        if (expression.InputCount <= 0 || expression.Steps is null || expression.Steps.Count < 2)
            throw new InvalidDataException("A fused expression requires inputs and at least two steps.");
        for (var stepIndex = 0; stepIndex < expression.Steps.Count; stepIndex++)
        {
            var step = expression.Steps[stepIndex];
            if (step is null || step.Operands is null ||
                Arity(step.Operation) == 0 || Arity(step.Operation) != step.Operands.Count ||
                step.Operands.Any(operand => operand is null ||
                    (operand.InputIndex.HasValue == operand.StepIndex.HasValue) ||
                    operand.InputIndex is int input && (input < 0 || input >= expression.InputCount) ||
                    operand.StepIndex is int prior && (prior < 0 || prior >= stepIndex)) ||
                stepIndex > 0 && !step.Operands.Any(operand => operand.StepIndex == stepIndex - 1))
                throw new InvalidDataException("Invalid fused elementwise expression step.");
        }
    }
}
