using System.Text;
using System.Text.Json;
using SharpInference.Graphs;

namespace SharpInference.Backends.Vortice;

internal static class VorticeFusedExpressionKernel
{
    public const int MaximumInputs = 8;
    public const int MaximumSteps = 32;

    public static VorticePrimitiveGraphStep Compile(
        ExecutionNode node,
        IReadOnlyDictionary<ResourceId, GraphResource> resources,
        IReadOnlySet<ResourceId> initialized)
    {
        NotSupportedException Unsupported(string reason) =>
            new($"Vortice primitive graph node '{node.Id}' ({node.Operation}): {reason}");
        FusedElementwiseExpression expression;
        try
        {
            expression = FusedElementwiseExpressionContract.Read(node, resources);
        }
        catch (InvalidDataException error)
        {
            throw Unsupported(error.Message);
        }
        if (expression.InputCount > MaximumInputs || expression.Steps.Count > MaximumSteps)
            throw Unsupported($"GPU fused expression supports at most {MaximumInputs} inputs and {MaximumSteps} ordered steps.");
        var inputs = node.Resources.Take(expression.InputCount).Select(binding => binding.Resource).ToArray();
        foreach (var input in inputs)
            if (!initialized.Contains(input))
                throw Unsupported($"resource '{input}' is read before it is produced.");
        var output = node.Resources[^1].Resource;
        var count = checked((uint)resources[output].Tensor.Dimensions.Aggregate(
            1L, (size, dimension) => checked(size * dimension)));
        var groups = checked((uint)(((ulong)count + 63) / 64));
        if (groups > 65535UL * 65535UL)
            throw Unsupported($"dispatch requires {groups} thread groups, exceeding the D3D12 XY grid limit.");
        return new VorticePrimitiveGraphStep(node.Id, "FusedExpression", inputs[0],
            null, output, null, count, 0, 0, groups,
            FusedInputs: inputs, Expression: expression);
    }

    public static string Key(FusedElementwiseExpression expression) =>
        "FusedExpression:" + JsonSerializer.Serialize(expression);

    public static string Source(FusedElementwiseExpression expression)
    {
        var source = new StringBuilder();
        for (var input = 0; input < expression.InputCount; input++)
            source.AppendLine($"StructuredBuffer<float> input{input} : register(t{input});");
        source.AppendLine("""
            RWStructuredBuffer<float> output : register(u0);
            cbuffer Constants : register(b0) { uint elementCount; };
            [numthreads(64, 1, 1)]
            void FusedExpression(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID)
            {
                uint index = id.x + group.y * 65535u * 64u;
                if (index >= elementCount) return;
            """);
        for (var index = 0; index < expression.Steps.Count; index++)
        {
            var instruction = expression.Steps[index];
            string Operand(ElementwiseOperand operand) => operand.InputIndex is int input
                ? $"input{input}[index]" : $"step{operand.StepIndex!.Value}";
            var left = Operand(instruction.Operands[0]);
            var right = instruction.Operands.Count == 2 ? Operand(instruction.Operands[1]) : "";
            var value = instruction.Operation switch
            {
                var op when op == PrimitiveGraphOperations.Copy => left,
                var op when op == PrimitiveGraphOperations.Add => $"({left} + {right})",
                var op when op == PrimitiveGraphOperations.Subtract => $"({left} - {right})",
                var op when op == PrimitiveGraphOperations.Multiply => $"({left} * {right})",
                var op when op == PrimitiveGraphOperations.Divide => $"({left} / {right})",
                var op when op == PrimitiveGraphOperations.Maximum => $"max({left}, {right})",
                var op when op == PrimitiveGraphOperations.Exp => $"exp({left})",
                var op when op == PrimitiveGraphOperations.Tanh => $"tanh({left})",
                var op when op == PrimitiveGraphOperations.Sigmoid =>
                    $"(1.0f / (1.0f + exp(-({left}))))",
                var op when op == PrimitiveGraphOperations.ReciprocalSquareRoot => $"rsqrt({left})",
                var op when op == PrimitiveGraphOperations.Square => $"({left} * {left})",
                var op when op == PrimitiveGraphOperations.Relu => $"max(0.0f, {left})",
                _ => throw new NotSupportedException($"GPU fused expression step '{instruction.Operation}' is unsupported."),
            };
            source.AppendLine($"    precise float step{index} = {value};");
        }
        source.AppendLine($"    output[index] = step{expression.Steps.Count - 1};");
        source.AppendLine("}");
        return source.ToString();
    }
}
