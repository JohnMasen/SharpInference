using SharpInference.Instructions;

namespace SharpInference.Tests;

public sealed class InstructionContractTests
{
    [Fact]
    public void ContractsAreOwnedByInstructions()
    {
        Type[] types = [typeof(ElementType), typeof(TensorDescriptor), typeof(ResourceAccess),
            typeof(PrecisionRequirement), typeof(NumericTypeCompatibility), typeof(OperationId),
            typeof(PrimitiveOperations)];

        Assert.All(types, type =>
        {
            Assert.Equal("SharpInference.Instructions", type.Namespace);
            Assert.Equal(typeof(Instruction).Assembly, type.Assembly);
        });
    }

    [Fact]
    public void EnumNamesAndValuesAreUnchanged()
    {
        Assert.Equal(["Byte", "Int32", "UInt32", "Float16", "Float32"], Enum.GetNames<ElementType>());
        Assert.Equal([0, 1, 2, 3, 4], Enum.GetValues<ElementType>().Select(value => (int)value));
        Assert.Equal(["Read", "Write", "ReadWrite"], Enum.GetNames<ResourceAccess>());
        Assert.Equal([0, 1, 2], Enum.GetValues<ResourceAccess>().Select(value => (int)value));
    }

    [Fact]
    public void TensorDescriptorSnapshotsDimensionsAndPreservesLayout()
    {
        int[] dimensions = [2, 4];
        var tensor = new TensorDescriptor(ElementType.Float32, dimensions);
        dimensions[0] = 9;

        Assert.Equal(ElementType.Float32, tensor.ElementType);
        Assert.Equal([2, 4], tensor.Dimensions);
        Assert.Equal("dense", tensor.Layout);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)tensor.Dimensions)[0] = 8);
        Assert.Equal("custom", new TensorDescriptor(ElementType.Float16, [1], "custom").Layout);
        Assert.Equal((ElementType)99, new TensorDescriptor((ElementType)99, [1]).ElementType);
    }

    [Fact]
    public void TensorDescriptorKeepsInvalidInputBehavior()
    {
        Assert.Throws<ArgumentNullException>(() => new TensorDescriptor(ElementType.Float32, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorDescriptor(ElementType.Float32, []));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorDescriptor(ElementType.Float32, [0]));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TensorDescriptor(ElementType.Float32, [-1]));
        Assert.Throws<ArgumentException>(() => new TensorDescriptor(ElementType.Float32, [1], " "));
    }

    [Theory]
    [InlineData(ElementType.Byte, false)]
    [InlineData(ElementType.Int32, false)]
    [InlineData(ElementType.UInt32, false)]
    [InlineData(ElementType.Float16, true)]
    [InlineData(ElementType.Float32, true)]
    [InlineData((ElementType)99, false)]
    public void FloatingPointClassificationIsUnchanged(ElementType type, bool expected)
    {
        Assert.Equal(expected, NumericTypeCompatibility.IsFloatingPoint(type));
    }

    [Fact]
    public void MaximumPrecisionKeepsSelectionAndValidation()
    {
        Assert.Equal(ElementType.Float16, NumericTypeCompatibility.Maximum([ElementType.Float16]));
        Assert.Equal(ElementType.Float32,
            NumericTypeCompatibility.Maximum([ElementType.Float16, ElementType.Float32]));
        Assert.Throws<ArgumentNullException>(() => NumericTypeCompatibility.Maximum(null!));
        Assert.Throws<ArgumentException>(() => NumericTypeCompatibility.Maximum([]));
        Assert.Throws<ArgumentOutOfRangeException>(() => NumericTypeCompatibility.Maximum([ElementType.Int32]));
        Assert.Throws<ArgumentOutOfRangeException>(() => NumericTypeCompatibility.Maximum([(ElementType)99]));
    }

    [Fact]
    public void PrecisionRequirementsValidateEachComponentIndependently()
    {
        var requirement = new PrecisionRequirement(ElementType.Float32, ElementType.Float16);
        Assert.Equal(ElementType.Float32, requirement.MinimumArithmeticType);
        Assert.Equal(ElementType.Float16, requirement.MinimumAccumulatorType);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PrecisionRequirement(ElementType.Int32, ElementType.Float32));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PrecisionRequirement(ElementType.Float32, ElementType.Byte));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PrecisionRequirement((ElementType)99, ElementType.Float32));

        Assert.True(InstructionPrecisionCompatibility.Satisfies(
            new KernelPrecisionProfile(ElementType.Float32, ElementType.Float32), requirement));
        Assert.False(InstructionPrecisionCompatibility.Satisfies(
            new KernelPrecisionProfile(ElementType.Float16, ElementType.Float32), requirement));
    }

    [Fact]
    public void OperationIdentityKeepsValueSemanticsAndText()
    {
        var operation = new OperationId("custom.operation");
        Assert.Equal("custom.operation", operation.Name);
        Assert.Equal(1, operation.Version);
        Assert.Equal(new OperationId("custom.operation", 1), operation);
        Assert.NotEqual(new OperationId("custom.operation", 2), operation);
        Assert.Equal("custom.operation@1", operation.ToString());
    }

    [Fact]
    public void AllPrimitiveIdentifiersAreUnchanged()
    {
        (OperationId Operation, string Name)[] expected =
        [
            (PrimitiveOperations.Copy, "core.copy"),
            (PrimitiveOperations.Add, "core.add"),
            (PrimitiveOperations.Subtract, "core.subtract"),
            (PrimitiveOperations.Multiply, "core.multiply"),
            (PrimitiveOperations.Divide, "core.divide"),
            (PrimitiveOperations.Maximum, "core.maximum"),
            (PrimitiveOperations.Exp, "core.exp"),
            (PrimitiveOperations.Tanh, "core.tanh"),
            (PrimitiveOperations.Sigmoid, "core.sigmoid"),
            (PrimitiveOperations.ReciprocalSquareRoot, "core.rsqrt"),
            (PrimitiveOperations.Square, "core.square"),
            (PrimitiveOperations.Relu, "core.relu"),
            (PrimitiveOperations.ReduceSum, "core.reduce-sum"),
            (PrimitiveOperations.ReduceMean, "core.reduce-mean"),
            (PrimitiveOperations.MatVec, "core.mat-vec"),
            (PrimitiveOperations.MatrixMultiply, "core.matrix-multiply"),
            (PrimitiveOperations.Affine, "core.affine"),
            (PrimitiveOperations.BiasAdd, "core.bias-add"),
            (PrimitiveOperations.GatherRow, "core.gather-row"),
        ];

        Assert.Equal(19, typeof(PrimitiveOperations).GetFields().Length);
        Assert.All(expected, entry => Assert.Equal(new OperationId(entry.Name, 1), entry.Operation));
    }
}
