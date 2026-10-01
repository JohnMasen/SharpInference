using System.Runtime.InteropServices;
using SharpInference.Gguf;
using SharpInference.Runtime;

internal static class StateComparison
{
    public static float MaximumDifference(ProcessorSession cpu, ProcessorSession gpu, string label)
    {
        var expected = Capture(cpu);
        var actual = Capture(gpu);
        if (expected.SchemaName != actual.SchemaName || expected.Tensors.Count != actual.Tensors.Count)
            throw new InvalidOperationException($"{label} state schema differs.");
        var byName = actual.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        var maximum = 0f;
        foreach (var tensor in expected.Tensors)
        {
            if (!byName.TryGetValue(tensor.Name, out var other) ||
                tensor.Type != GgufTensorType.Float32 || other.Type != tensor.Type ||
                !tensor.Dimensions.SequenceEqual(other.Dimensions) ||
                tensor.Data.Length != other.Data.Length ||
                (ulong)tensor.Data.Length != checked(tensor.Dimensions.Aggregate(
                    1UL, (count, dimension) => checked(count * dimension)) * sizeof(float)))
                throw new InvalidOperationException($"{label} state tensor {tensor.Name} differs.");
            var cpuValues = MemoryMarshal.Cast<byte, float>(tensor.Data.Span);
            var gpuValues = MemoryMarshal.Cast<byte, float>(other.Data.Span);
            for (var index = 0; index < cpuValues.Length; index++)
            {
                if (!float.IsFinite(cpuValues[index]) || !float.IsFinite(gpuValues[index]))
                    throw new InvalidOperationException($"{label} contains non-finite state at {tensor.Name}[{index}].");
                maximum = MathF.Max(maximum, MathF.Abs(cpuValues[index] - gpuValues[index]));
            }
        }
        return maximum;
    }

    private static GgufState Capture(ProcessorSession session)
    {
        using var stream = new MemoryStream();
        session.SaveState(stream);
        stream.Position = 0;
        return GgufStateFile.Read(stream);
    }
}
