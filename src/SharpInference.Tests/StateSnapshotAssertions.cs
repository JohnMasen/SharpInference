using System.Runtime.InteropServices;
using SharpInference.Gguf;
using SharpInference.Runtime;

namespace SharpInference.Tests;

internal static class StateSnapshotAssertions
{
    public static GgufState Capture(ProcessorSession session)
    {
        using var stream = new MemoryStream();
        session.SaveState(stream);
        stream.Position = 0;
        return GgufStateFile.Read(stream);
    }

    public static void Restore(ProcessorSession session, GgufState snapshot)
    {
        using var stream = new MemoryStream();
        GgufStateFile.Write(stream, snapshot);
        stream.Position = 0;
        session.LoadState(stream);
    }

    public static float[] Values(GgufState state)
    {
        return state.Tensors.SelectMany(tensor => Decode(tensor)).ToArray();
    }

    public static void Equal(GgufState expected, GgufState actual) => Near(expected, actual, 0, 0, "state");

    public static void Near(GgufState expected, GgufState actual, float absolute, float relative, string label,
        bool scaleAtLeastOne = false)
    {
        Assert.Equal(expected.SchemaName, actual.SchemaName);
        Assert.Equal(expected.Tensors.Count, actual.Tensors.Count);
        var actualByName = actual.Tensors.ToDictionary(tensor => tensor.Name, StringComparer.Ordinal);
        foreach (var tensor in expected.Tensors)
        {
            Assert.True(actualByName.TryGetValue(tensor.Name, out var other), $"Missing tensor {tensor.Name}");
            Assert.Equal(tensor.Type, other.Type);
            Assert.Equal(tensor.Dimensions, other.Dimensions);
            var values = Decode(tensor);
            var otherValues = Decode(other);
            Assert.Equal(values.Length, otherValues.Length);
            for (var index = 0; index < values.Length; index++)
            {
                var tolerance = scaleAtLeastOne
                    ? relative * MathF.Max(1, MathF.Abs(values[index]))
                    : absolute + relative * MathF.Abs(values[index]);
                Assert.True(float.IsFinite(otherValues[index]) &&
                    MathF.Abs(values[index] - otherValues[index]) <= tolerance,
                    $"{label} {tensor.Name}[{index}]: expected {values[index]}, actual {otherValues[index]}");
            }
        }
    }

    private static float[] Decode(GgufStateTensor tensor)
    {
        Assert.Equal(GgufTensorType.Float32, tensor.Type);
        Assert.Equal(checked(tensor.Dimensions.Aggregate(1UL, (size, dimension) => checked(size * dimension)) * 4),
            (ulong)tensor.Data.Length);
        Assert.True(BitConverter.IsLittleEndian);
        return MemoryMarshal.Cast<byte, float>(tensor.Data.Span).ToArray();
    }
}
