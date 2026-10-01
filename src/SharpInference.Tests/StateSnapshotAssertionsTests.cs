using SharpInference.Gguf;

namespace SharpInference.Tests;

public sealed class StateSnapshotAssertionsTests
{
    [Fact]
    public void Comparison_MatchesByTensorNameAndRejectsDescriptorChanges()
    {
        var expected = new GgufState("test", [
            Tensor("a", [2], [1f, 2f]),
            Tensor("b", [1], [3f]),
        ]);
        var reordered = new GgufState("test", [
            Tensor("b", [1], [3f]),
            Tensor("a", [2], [1f, 2f]),
        ]);
        StateSnapshotAssertions.Equal(expected, reordered);
        Assert.ThrowsAny<Exception>(() => StateSnapshotAssertions.Equal(expected,
            new GgufState("test", [Tensor("b", [1], [3f]), Tensor("a", [1, 2], [1f, 2f])])));
        Assert.ThrowsAny<Exception>(() => StateSnapshotAssertions.Equal(expected,
            new GgufState("test", [Tensor("b", [1], [3f]), Tensor("c", [2], [1f, 2f])])));
    }

    private static GgufStateTensor Tensor(string name, ulong[] dimensions, float[] values) =>
        new(name, GgufTensorType.Float32, dimensions,
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
}
