using System.Runtime.InteropServices;
using SharpInference.Backends.CpuVm;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Vm;
using SharpInference.Vm.Optimization;

namespace SharpInference.Tests;

public sealed class TierZeroOperationTests
{
    public static IEnumerable<object[]> Signatures() =>
        TierZeroOperationContracts.Contracts.SelectMany(contract =>
            Enumerable.Range(0, contract.Signatures.Count).Select(index =>
                new object[] { contract.Operation.Name, index }));

    [Fact]
    public void CatalogHasOneCommonVersionedProfileAndImmutableSignatures()
    {
        Assert.Equal("core.t0.fp32@1", TierZeroOperationContracts.Profile);
        Assert.Equal(28, TierZeroOperationContracts.Contracts.Count);
        Assert.Equal(23, TierZeroOperationContracts.RequiredContracts.Count);
        Assert.Equal(35, Signatures().Count());
        Assert.Equal(3, new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create())
            .InstructionCollections.QueryInstructionCollection().Count);
        Assert.Equal(4, new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create())
            .InstructionCollections.QueryInstructionCollection().Count);
        Assert.Equal(
            ["core.divide", "core.reduce-sum", "core.matrix-multiply", "core.bias-add", "core.affine"],
            TierZeroOperationContracts.Contracts.Where(contract => !contract.RequiredByBaseProfile)
                .Select(contract => contract.Operation.Name));
        Assert.All(TierZeroOperationContracts.Contracts, contract =>
        {
            Assert.Equal(1, contract.Operation.Version);
            Assert.Same(contract, TierZeroOperationContracts.Get(contract.Operation));
            Assert.Equal(GraphElementType.Float32, contract.Precision.ArithmeticType);
            Assert.Equal(GraphElementType.Float32, contract.Precision.AccumulatorType);
            Assert.All(contract.Signatures, signature =>
                Assert.Equal([GraphElementType.Float32], signature.OutputTypes));
        });
        var signature = TierZeroOperationContracts.Contracts[0].Signatures[0];
        Assert.Throws<NotSupportedException>(() =>
            ((IList<GraphElementType>)signature.InputTypes)[0] = GraphElementType.Byte);
    }

    [Theory]
    [MemberData(nameof(Signatures))]
    public void EverySignatureRunsOnCpuThroughLogicalLowering(string operation, int signature) =>
        Execute(Example(operation, signature), VmTarget.Cpu);

    [Theory]
    [MemberData(nameof(Signatures))]
    public void EverySignatureRunsOnActualGpuThroughLogicalLowering(string operation, int signature) =>
        Execute(Example(operation, signature), VmTarget.Direct3D12);

    public static IEnumerable<object[]> NumericalOperations()
    {
        foreach (var operation in new[] { "copy", "exp", "tanh", "sigmoid", "rsqrt", "square", "relu",
                     "maximum", "add", "subtract", "multiply", "divide" })
            yield return ["core." + operation];
    }

    [Theory]
    [MemberData(nameof(NumericalOperations))]
    public void CpuSpecialValuesFollowReferenceSemantics(string operation) =>
        Execute(SpecialValues(operation), VmTarget.Cpu, exactZero: true);

    [Theory]
    [MemberData(nameof(NumericalOperations))]
    public void GpuSpecialValuesFollowReferenceSemantics(string operation) =>
        Execute(SpecialValues(operation), VmTarget.Direct3D12, exactZero: true);

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void CastPreservesHalfSpecialValuesAndOddElementCounts(VmTarget target)
    {
        float[] values = [float.NaN, float.NegativeInfinity, float.PositiveInfinity, -0f, 0f,
            (float)Half.Epsilon, (float)Half.MaxValue];
        var example = Example("core.tensor.cast-f16-f32", 0);
        Execute(example with
        {
            Inputs = [new("input", new(GraphElementType.Float16, [values.Length]), values)],
            Output = new(GraphElementType.Float32, [values.Length]),
            Expected = values,
        }, target, exactZero: true);
    }

    [Theory]
    [InlineData(VmTarget.Cpu)]
    [InlineData(VmTarget.Direct3D12)]
    public void FillPreservesNegativeZero(VmTarget target)
    {
        var example = Example("core.tensor.fill", 0);
        Execute(example with
        {
            Attributes = new TensorFillValue(-0f).ToAttributes(),
            Expected = Enumerable.Repeat(-0f, example.Expected.Length).ToArray(),
        }, target, exactZero: true);
    }

    [Fact]
    public void BothCompilersRejectTheSameMalformedContractsBeforeCodeCompilation()
    {
        var example = Example("core.mat-vec", 0);
        var valid = VmGraphOptimizer.Optimize(example.Graph(), VmTarget.Cpu,
            new(PrefillCapacity: 1, WeightViews: false));
        var parameters = valid.Definitions[0].Parameters;
        var arguments = parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var precision = new PrecisionRequirement(GraphElementType.Float32, GraphElementType.Float32);
        var failures = new (VmOperator Operator, IReadOnlyList<VmParameter> Parameters, Type Error)[]
        {
            (new(precision, "core.unknown", 1, arguments), parameters, typeof(NotSupportedException)),
            (new(precision, "core.mat-vec", 2, arguments), parameters, typeof(NotSupportedException)),
            (new(precision, "core.mat-vec", 1, arguments, new Dictionary<string, string> { ["extra"] = "1" }),
                parameters, typeof(InvalidDataException)),
            (new(precision, "core.mat-vec", 1, arguments),
                parameters.Select(p => p.Name == "input" ? p with { Tensor = new(VmElementType.Float32, [1, 5]) } : p).ToArray(),
                typeof(InvalidDataException)),
            (new(precision, "core.mat-vec", 1, arguments),
                parameters.Select(p => p.Name == "input" ? p with { Tensor = new(VmElementType.Float16, [5]) } : p).ToArray(),
                typeof(NotSupportedException)),
            (new(precision, "core.mat-vec", 1, arguments.Select(a => a.Parameter == "input"
                ? a with { ByteOffset = 4 } : a)), parameters, typeof(NotSupportedException)),
            (new(precision, "core.mat-vec", 1, arguments.Select(a => a.Parameter == "output"
                ? a with { Source = "matrix" } : a)), parameters, typeof(InvalidDataException)),
            (new(precision, "core.mat-vec", 1, arguments.Append(new("weight", "matrix"))),
                parameters, typeof(InvalidDataException)),
        };
        foreach (var failure in failures)
        {
            Assert.IsType(failure.Error, Record.Exception(() =>
                VmTierZeroOperations.Validate(failure.Operator, failure.Parameters)));
            foreach (var target in new[] { VmTarget.Cpu, VmTarget.Direct3D12 })
            {
                var leaf = new VmDefinition("op", target == VmTarget.Cpu
                    ? VmDefinitionKind.Function : VmDefinitionKind.Kernel, failure.Parameters,
                    [new("body", failure.Operator)], target == VmTarget.Cpu ? null : new(64));
                var program = new VmProgram("invalid", "t0", target,
                    failure.Parameters.Select(p => new VmSlot(p.Name, VmSlotScope.Local, p.Access, p.Tensor)),
                    [leaf], [new("run", "op", failure.Parameters.Select(p => new VmArgument(p.Name, p.Name)))],
                    new("none", 1, []));
                var compilerError = Record.Exception(() =>
                {
                    if (target == VmTarget.Cpu) new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()).GenerateSource(program);
                    else new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()).GenerateSources(program);
                });
                if (compilerError is SharpInference.Instructions.InstructionAdaptationException adaptation)
                    Assert.True(adaptation.InnerException is null or InvalidDataException or NotSupportedException);
                else Assert.IsType(failure.Error, compilerError);
            }
        }
        var malformed = Example("core.tensor.fill", 0) with
        {
            Attributes = new Dictionary<string, string> { ["value"] = "NaN" },
        };
        Assert.Throws<InvalidDataException>(() => TierZeroOperationContracts.ValidateGraph(malformed.Graph()));
    }

    [Theory]
    [InlineData("core.reduce-sum")]
    [InlineData("core.reduce-mean")]
    [InlineData("core.tensor.reduce-last-sum")]
    [InlineData("core.tensor.reduce-last-mean")]
    [InlineData("core.mat-vec")]
    [InlineData("core.tensor.batched-mat-vec")]
    public void GpuModelWidthAccumulationRetainsSmallContributions(string operation)
    {
        var values = Enumerable.Repeat(1f, 4096).ToArray();
        values[0] = 100_000_000;
        values[^1] = -100_000_000;
        var example = Example(operation, 0);
        var expected = operation.EndsWith("mean", StringComparison.Ordinal) ? 4094f / 4096 : 4094;
        if (operation is "core.mat-vec" or "core.tensor.batched-mat-vec")
        {
            var batched = operation.StartsWith("core.tensor.", StringComparison.Ordinal);
            example = example with
            {
                Inputs =
                [
                    new("matrix", new(GraphElementType.Float32, batched ? [2, 2, 4096] : [2, 4096]),
                        Enumerable.Repeat(1f, batched ? 16384 : 8192).ToArray()),
                    new(batched ? "vector" : "input", new(GraphElementType.Float32, batched ? [2, 4096] : [4096]),
                        batched ? values.Concat(values).ToArray() : values),
                ],
                Output = new(GraphElementType.Float32, batched ? [2, 2] : [2]),
                Expected = Enumerable.Repeat(expected, batched ? 4 : 2).ToArray(),
            };
        }
        else
        {
            var last = operation.StartsWith("core.tensor.", StringComparison.Ordinal);
            example = example with
            {
                Inputs = [new("input", new(GraphElementType.Float32, last ? [2, 4096] : [4096]),
                    last ? values.Concat(values).ToArray() : values)],
                Output = new(GraphElementType.Float32, last ? [2] : [1]),
                Expected = Enumerable.Repeat(expected, last ? 2 : 1).ToArray(),
            };
        }
        Execute(example, VmTarget.Direct3D12);
    }

    [Theory]
    [InlineData("core.reduce-sum")]
    [InlineData("core.reduce-mean")]
    public void GpuCompensationDoesNotTurnInfinityIntoNaN(string operation)
    {
        var example = Example(operation, 0);
        foreach (var values in new[] { new[] { float.PositiveInfinity, 1f },
                     new[] { float.NegativeInfinity, 1f },
                     new[] { float.PositiveInfinity, float.NegativeInfinity },
                     new[] { float.NaN, 1f } })
            Execute(example with
            {
                Inputs = [new("input", new(GraphElementType.Float32, [2]), values)],
                Expected = [values.Sum() / (operation.EndsWith("mean", StringComparison.Ordinal) ? 2 : 1)],
            }, VmTarget.Direct3D12);
    }

    private static void Execute(OperationExample example, VmTarget target, bool exactZero = false)
    {
        var graph = example.Graph();
        TierZeroOperationContracts.ValidateGraph(graph);
        var program = VmGraphOptimizer.Optimize(graph, target,
            new(PrefillCapacity: 1, WeightViews: false, ReuseLocalStorage: false));
        Assert.Single(program.Definitions.SelectMany(definition => definition.Nodes)
            .Select(node => node.Instruction).OfType<VmOperator>());
        var buffers = program.Slots.Select(slot => slot.Id == "output"
            ? new byte[checked((int)slot.Tensor.ByteLength)]
            : Bytes(example.Inputs.Single(input => input.Port == slot.Id))).ToArray();
        var originalBuffers = buffers.ToArray();
        var originalInputs = buffers.Take(buffers.Length - 1).Select(buffer => buffer.ToArray()).ToArray();
        var output = program.Slots.Select((slot, index) => (slot, index)).Single(pair => pair.slot.Id == "output").index;
        if (target == VmTarget.Cpu)
        {
            using var executable = new CpuVmCompiler(SharpInference.Runtime.Cpu.CpuInstructionCollections.Create()).Compile(program).LoadExecutable();
            var context = executable.Prepare(buffers, program.Slots
                .Where(slot => slot.Scope == VmSlotScope.Local && slot.Access == VmAccess.ReadOnly)
                .Select(slot => slot.Id));
            for (var repeat = 0; repeat < 2; repeat++)
            {
                executable.Invoke("forward", context);
                Check();
            }
        }
        else
        {
            using var executable = new D3D12VmCompiler(SharpInference.Runtime.D3D12.D3D12InstructionCollections.Create()).Compile(program).CreateExecutor();
            for (var repeat = 0; repeat < 2; repeat++)
            {
                executable.Execute("forward", buffers);
                Check();
            }
        }
        void Check()
        {
            var actual = MemoryMarshal.Cast<byte, float>(buffers[output]).ToArray();
            Assert.Equal(example.Expected.Length, actual.Length);
            for (var i = 0; i < actual.Length; i++)
            {
                var expected = example.Expected[i];
                if (float.IsNaN(expected)) Assert.True(float.IsNaN(actual[i]), $"{example.Operation}[{i}] expected NaN, got {actual[i]}.");
                else if (float.IsInfinity(expected)) Assert.Equal(expected, actual[i]);
                else if (exactZero && expected == 0)
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(actual[i]));
                else Assert.InRange(Math.Abs(actual[i] - expected), 0, 0.00003f + Math.Abs(expected) * 0.00003f);
            }
            for (var i = 0; i < buffers.Length; i++) Assert.Same(originalBuffers[i], buffers[i]);
            for (var i = 0; i < originalInputs.Length; i++) Assert.Equal(originalInputs[i], buffers[i]);
        }
    }

    internal sealed record InputExample(string Port, TensorDescriptor Tensor, float[] Values);
    internal sealed record OperationExample(string Operation, InputExample[] Inputs, TensorDescriptor Output,
        float[] Expected, IReadOnlyDictionary<string, string> Attributes)
    {
        internal LogicalGraph Graph() => new(new("t0-test", 1, Operation), TestGraphSignatures.Create(7, 5, 1, 1, 5, "none"),
            Inputs.Select(input => new GraphResource(new(input.Port), input.Port,
                input.Port is "matrix" or "table" ? GraphResourceKind.Weight : GraphResourceKind.Input,
                input.Port is "matrix" or "table" ? GraphResourceLifetime.Model : GraphResourceLifetime.External,
                input.Tensor, input.Port is "matrix" or "table" ? input.Port : null))
                .Append(new(new("output"), "output", GraphResourceKind.Output, GraphResourceLifetime.External, Output)),
            [new(new("root"), null, GraphRegionTypes.Graph, null, "root", new Dictionary<string, string>())],
            [new(new("op"), new(Operation), new("root"),
                Inputs.Select(input => new NodeResourceBinding(input.Port, new(input.Port), GraphResourceAccess.Read))
                    .Append(new("output", new("output"), GraphResourceAccess.Write)).ToArray(),
                [], Attributes, new(GraphElementType.Float32, GraphElementType.Float32))],
            Inputs.Where(input => input.Port is not ("matrix" or "table")).Select(input => new ResourceId(input.Port)),
            [new("output")]);
    }

    internal static OperationExample Example(string operation, int signatureIndex)
    {
        var signature = TierZeroOperationContracts.Get(new(operation)).Signatures[signatureIndex];
        var inputs = new List<InputExample>();
        IReadOnlyDictionary<string, string> attributes = new Dictionary<string, string>();
        void Input(string port, int[] dimensions, float[] values) =>
            inputs.Add(new(port, new(signature.InputTypes[inputs.Count], dimensions), values));
        static float[] Values(int count) => Enumerable.Range(0, count).Select(i => ((i % 17) - 8) * .125f).ToArray();
        int[] outputShape = [67];
        float[] expected;
        switch (operation)
        {
            case "core.tensor.fill":
                attributes = new TensorFillValue(-1.25f).ToAttributes();
                expected = Enumerable.Repeat(-1.25f, 67).ToArray(); break;
            case "core.tensor.cast-f16-f32":
                Input("input", [67], Values(67)); expected = Values(67); break;
            case "core.tensor.reshape":
                Input("input", [7, 11], Values(77)); outputShape = [11, 7]; expected = Values(77); break;
            case "core.tensor.slice":
                Input("input", [2, 3, 5], Values(30)); outputShape = [2, 2, 5];
                attributes = new TensorSlice(1, 1, 2).ToAttributes();
                expected = inputs[0].Values.Skip(5).Take(10).Concat(inputs[0].Values.Skip(20).Take(10)).ToArray(); break;
            case "core.tensor.broadcast":
                Input("input", [2, 1, 5], Values(10)); outputShape = [2, 3, 5];
                expected = Enumerable.Range(0, 30).Select(i => inputs[0].Values[(i / 15) * 5 + i % 5]).ToArray(); break;
            case "core.tensor.head-outer":
                Input("left", [2, 3], Values(6)); Input("right", [2, 5], Values(10)); outputShape = [2, 3, 5];
                expected = Enumerable.Range(0, 30).Select(i =>
                    inputs[0].Values[i / 15 * 3 + i / 5 % 3] * inputs[1].Values[i / 15 * 5 + i % 5]).ToArray(); break;
            case "core.mat-vec":
            case "core.tensor.batched-mat-vec":
                var batched = operation.StartsWith("core.tensor.", StringComparison.Ordinal);
                Input("matrix", batched ? [2, 3, 5] : [3, 5], Values(batched ? 30 : 15));
                Input(batched ? "vector" : "input", batched ? [2, 5] : [5], Values(batched ? 10 : 5));
                outputShape = batched ? [2, 3] : [3];
                expected = Enumerable.Range(0, batched ? 6 : 3).Select(row =>
                    Enumerable.Range(0, 5).Sum(column =>
                        inputs[0].Values[row * 5 + column] * inputs[1].Values[(batched ? row / 3 * 5 : 0) + column])).ToArray(); break;
            case "core.matrix-multiply":
            case "core.affine":
                Input("left", [2, 3], Values(6));
                Input("right", [4, 3], Values(12));
                if (operation == "core.affine") Input("bias", [4], Values(4));
                outputShape = [2, 4];
                attributes = new Dictionary<string, string>
                {
                    ["transpose_left"] = "false",
                    ["transpose_right"] = "true",
                };
                expected = Enumerable.Range(0, 8).Select(element =>
                {
                    var row = element / 4;
                    var column = element % 4;
                    var value = Enumerable.Range(0, 3).Sum(index =>
                        inputs[0].Values[row * 3 + index] *
                        inputs[1].Values[column * 3 + index]);
                    return operation == "core.affine"
                        ? value + inputs[2].Values[column]
                        : value;
                }).ToArray();
                break;
            case "core.bias-add":
                Input("input", [2, 3], Values(6));
                Input("bias", [3], Values(3));
                outputShape = [2, 3];
                expected = inputs[0].Values.Select((value, index) =>
                    value + inputs[1].Values[index % 3]).ToArray();
                break;
            case "core.gather-row":
                Input("table", [7, 5], Values(35)); Input("index", [1], [6]);
                outputShape = [5]; expected = inputs[0].Values.Skip(30).ToArray(); break;
            case "core.tensor.reduce-last-sum":
            case "core.tensor.reduce-last-mean":
                Input("input", [2, 3, 5], Values(30)); outputShape = [2, 3];
                expected = Enumerable.Range(0, 6).Select(row => inputs[0].Values.Skip(row * 5).Take(5).Sum() /
                    (operation.EndsWith("mean", StringComparison.Ordinal) ? 5 : 1)).ToArray(); break;
            case "core.reduce-sum":
            case "core.reduce-mean":
                Input("input", [7, 11], Values(77)); outputShape = [1];
                expected = [inputs[0].Values.Sum() / (operation.EndsWith("mean", StringComparison.Ordinal) ? 77 : 1)]; break;
            default:
                if (signature.InputTypes.Count == 2)
                {
                    Input("left", [67], Values(67));
                    Input("right", [67], Values(67).Select(value => value + 2).ToArray());
                    expected = inputs[0].Values.Zip(inputs[1].Values, (left, right) => Binary(operation, left, right)).ToArray();
                }
                else
                {
                    Input("input", [67], operation == "core.rsqrt" ? Values(67).Select(value => value + 2).ToArray() : Values(67));
                    expected = inputs[0].Values.Select(value => Unary(operation, value)).ToArray();
                }
                break;
        }
        return new(operation, inputs.ToArray(), new(GraphElementType.Float32, outputShape), expected, attributes);
    }

    private static OperationExample SpecialValues(string operation)
    {
        float[] left = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, -0f, 0f, -100f, 100f, -1f, 1f];
        float[] right = [1f, float.NegativeInfinity, float.PositiveInfinity, 0f, -0f, float.NaN, 0f, -1f, -1f];
        var example = Example(operation, 0);
        var binary = example.Inputs.Length == 2;
        var count = 67;
        left = Enumerable.Range(0, count).Select(i => left[i % left.Length]).ToArray();
        right = Enumerable.Range(0, count).Select(i => right[i % right.Length]).ToArray();
        return example with
        {
            Inputs = example.Inputs.Select((input, index) => input with { Values = index == 0 ? left : right }).ToArray(),
            Expected = binary ? left.Zip(right, (a, b) => Binary(operation, a, b)).ToArray()
                : left.Select(value => Unary(operation, value)).ToArray(),
        };
    }

    private static float Unary(string operation, float value) => operation switch
    {
        "core.copy" => value, "core.exp" => MathF.Exp(value), "core.tanh" => MathF.Tanh(value),
        "core.sigmoid" => 1 / (1 + MathF.Exp(-value)), "core.rsqrt" => 1 / MathF.Sqrt(value),
        "core.square" => value * value, "core.relu" => MathF.Max(value, 0), _ => throw new ArgumentException(operation),
    };

    private static float Binary(string operation, float left, float right) => operation switch
    {
        "core.add" => left + right, "core.subtract" => left - right, "core.multiply" => left * right,
        "core.divide" => left / right, "core.maximum" => MathF.Max(left, right), _ => throw new ArgumentException(operation),
    };

    private static byte[] Bytes(InputExample input) => input.Tensor.ElementType switch
    {
        GraphElementType.Float16 => MemoryMarshal.AsBytes(input.Values.Select(value => (Half)value).ToArray().AsSpan()).ToArray(),
        GraphElementType.Int32 => MemoryMarshal.AsBytes(input.Values.Select(value => (int)value).ToArray().AsSpan()).ToArray(),
        _ => MemoryMarshal.AsBytes(input.Values.AsSpan()).ToArray(),
    };
}
