using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.Tests;

public sealed class PrimitiveOperatorTests
{
    [Fact]
    public void CpuFp32_PrimitivesProduceExpectedValues()
    {
        var backend = CpuPrimitiveOperatorBackend.Instance;
        var left = new[] { 1f, -2f, 4f, 9f };
        var right = new[] { 2f, 4f, -1f, 3f };
        var output = new float[4];

        backend.Add(left, right, output);
        Assert.Equal([3f, 2f, 3f, 12f], output);
        backend.Subtract(left, right, output);
        Assert.Equal([-1f, -6f, 5f, 6f], output);
        backend.Multiply(left, right, output);
        Assert.Equal([2f, -8f, -4f, 27f], output);
        backend.Divide(left, right, output);
        Assert.Equal([0.5f, -0.5f, -4f, 3f], output);
        backend.Maximum(left, right, output);
        Assert.Equal([2f, 4f, 4f, 9f], output);

        backend.Square(left, output);
        Assert.Equal([1f, 4f, 16f, 81f], output);
        backend.Relu(left, output);
        Assert.Equal([1f, 0f, 4f, 9f], output);
        backend.ReciprocalSquareRoot(new[] { 1f, 4f, 16f, 25f }, output);
        Assert.Equal([1f, 0.5f, 0.25f, 0.2f], output);
        Assert.Equal(12f, backend.ReduceSum(left));
        Assert.Equal(3f, backend.ReduceMean(left));
    }

    [Fact]
    public void CpuFp16_PrimitivesUseSameTypeInputsAndFp32Accumulation()
    {
        var backend = CpuPrimitiveOperatorBackend.Instance;
        Half[] left = [(Half)1f, (Half)(-2f), (Half)4f, (Half)9f];
        Half[] right = [(Half)2f, (Half)4f, (Half)(-1f), (Half)3f];
        var output = new Half[4];

        backend.Add(left, right, output);
        Assert.Equal([3f, 2f, 3f, 12f], output.Select(value => (float)value));
        backend.Sigmoid(left, output);
        Assert.Equal(
            left.Select(value => 1f / (1f + MathF.Exp(-(float)value))),
            output.Select(value => (float)value),
            new FloatToleranceComparer(0.001f));
        Assert.InRange((float)backend.ReduceSum(left), 11.999f, 12.001f);
        Assert.InRange((float)backend.ReduceMean(left), 2.999f, 3.001f);
    }

    [Fact]
    public void Cpu_MatVecAndGatherSupportFp32AndFp16()
    {
        var backend = CpuPrimitiveOperatorBackend.Instance;
        var matrix = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var input = new[] { 2f, -1f, 0.5f };
        var output = new float[2];
        backend.MatVec(matrix, input, output, 2, 3);
        Assert.Equal([1.5f, 6f], output);

        var gathered = new float[3];
        backend.GatherRow(matrix, 1, 3, gathered);
        Assert.Equal([4f, 5f, 6f], gathered);

        var halfMatrix = matrix.Select(value => (Half)value).ToArray();
        var halfInput = input.Select(value => (Half)value).ToArray();
        var halfOutput = new Half[2];
        backend.MatVec(halfMatrix, halfInput, halfOutput, 2, 3);
        Assert.Equal([1.5f, 6f], halfOutput.Select(value => (float)value));
    }

    [Fact]
    public void PrimitiveDescriptionsDoNotAdvertiseMixedFloatingPointInputs()
    {
        var descriptions = CpuPrimitiveOperatorBackend.Instance.PrimitiveOperators;
        Assert.NotEmpty(descriptions);
        foreach (var signature in descriptions.SelectMany(description => description.Signatures))
        {
            var floatingInputs = signature.InputTypes
                .Where(type => type is GraphElementType.Float16 or GraphElementType.Float32)
                .Distinct()
                .ToArray();
            Assert.True(floatingInputs.Length <= 1);
        }
    }

    [Fact]
    public void CpuFp16ImplementationsAdvertisePromotedArithmetic()
    {
        var implementations = CpuPrimitiveOperatorBackend.Instance.OperatorImplementations;
        var add = Assert.Single(implementations, implementation =>
            implementation.Operation == PrimitiveGraphOperations.Add &&
            implementation.Signature.InputTypes.SequenceEqual(
                [GraphElementType.Float16, GraphElementType.Float16]));
        var matVec = Assert.Single(implementations, implementation =>
            implementation.Operation == PrimitiveGraphOperations.MatVec &&
            implementation.Signature.InputTypes.SequenceEqual(
                [GraphElementType.Float16, GraphElementType.Float16]));

        Assert.Equal(GraphElementType.Float32, add.Precision.ArithmeticType);
        Assert.Equal(GraphElementType.Float32, add.Precision.AccumulatorType);
        Assert.Equal(GraphElementType.Float32, matVec.Precision.ArithmeticType);
        Assert.Equal(GraphElementType.Float32, matVec.Precision.AccumulatorType);
    }

    [Fact]
    public void PrimitiveDescriptionsRejectMixedTypesButCustomPrimitivesAllowThem()
    {
        var mixed = new OperatorSignature(
            [GraphElementType.Float32, GraphElementType.Float16],
            [GraphElementType.Float32]);

        Assert.Throws<ArgumentException>(() =>
            new PrimitiveOperatorDescription(new GraphOperationId("test.mixed"), [mixed]));
        var custom = new CustomPrimitiveOperatorDescription(new GraphOperationId("test.custom"), [mixed]);
        Assert.Same(mixed, Assert.Single(custom.Signatures));
    }

    private sealed class FloatToleranceComparer(float tolerance) : IEqualityComparer<float>
    {
        public bool Equals(float x, float y) => MathF.Abs(x - y) <= tolerance;
        public int GetHashCode(float obj) => obj.GetHashCode();
    }
}
