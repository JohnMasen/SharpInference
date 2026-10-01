using System.Runtime.InteropServices;
using SharpInference.Gguf;
using SharpInference.Runtime;

namespace SharpInference.PrefillExperiment;

internal static class ExperimentStateSnapshots
{
    public static GgufState Capture(ProcessorSession session)
    {
        using var stream = new MemoryStream();
        session.SaveState(stream);
        stream.Position = 0;
        return GgufStateFile.Read(stream);
    }

    public static void Restore(ProcessorSession session, GgufState state)
    {
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, state);
        stream.Position = 0;
        session.LoadState(stream);
    }

    // The experimental RWKV-6 kernels use layer-major [ffn-previous, att-previous, WKV] FP32.
    public static float[] LayerMajor(GgufState state, int layers, int width, int headSize)
    {
        if (state.SchemaName != "RWKV6_State" || width <= 0 || headSize <= 0 ||
            width % headSize != 0 || state.Tensors.Count != checked(layers * 3) ||
            !BitConverter.IsLittleEndian)
            throw new InvalidDataException("Expected RWKV-6 layer-major FP32 state.");
        var byName = state.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        var result = new List<float>();
        for (var layer = 0; layer < layers; layer++)
        {
            Add($"state.{layer}.ffn-previous", [(ulong)width]);
            Add($"state.{layer}.att-previous", [(ulong)width]);
            Add($"state.{layer}.wkv", [(ulong)(width / headSize), (ulong)headSize, (ulong)headSize]);
        }
        return result.ToArray();

        void Add(string name, ulong[] dimensions)
        {
            if (!byName.TryGetValue(name, out var tensor) || tensor.Type != GgufTensorType.Float32 ||
                !tensor.Dimensions.SequenceEqual(dimensions) ||
                (ulong)tensor.Data.Length != checked(dimensions.Aggregate(1UL, (n, d) => checked(n * d)) * 4))
                throw new InvalidDataException($"Unexpected tensor {name} in experimental state.");
            result.AddRange(MemoryMarshal.Cast<byte, float>(tensor.Data.Span).ToArray());
        }
    }

    public static float[] LayerMajor(ProcessorSession session, int layers, int width, int headSize) =>
        LayerMajor(Capture(session), layers, width, headSize);

    public static float Compare(GgufState expected, GgufState actual,
        Func<ReadOnlyMemory<float>, ReadOnlyMemory<float>, string, float> compare)
    {
        if (expected.SchemaName != actual.SchemaName || expected.Tensors.Count != actual.Tensors.Count)
            throw new InvalidDataException("State schemas differ.");
        var byName = actual.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        var maximum = 0f;
        foreach (var tensor in expected.Tensors)
        {
            if (!byName.TryGetValue(tensor.Name, out var other) || tensor.Type != GgufTensorType.Float32 ||
                other.Type != tensor.Type || !other.Dimensions.SequenceEqual(tensor.Dimensions) ||
                other.Data.Length != tensor.Data.Length ||
                (ulong)tensor.Data.Length != checked(tensor.Dimensions.Aggregate(
                    1UL, (count, dimension) => checked(count * dimension)) * sizeof(float)))
                throw new InvalidDataException($"State tensor {tensor.Name} differs.");
            maximum = MathF.Max(maximum, compare(
                MemoryMarshal.Cast<byte, float>(tensor.Data.Span).ToArray(),
                MemoryMarshal.Cast<byte, float>(other.Data.Span).ToArray(), tensor.Name));
        }
        return maximum;
    }
}
