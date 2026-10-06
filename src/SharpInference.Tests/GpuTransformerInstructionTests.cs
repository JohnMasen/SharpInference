using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Tests;

public sealed class GpuTransformerInstructionTests
{
    [Fact]
    public void ResidualRmsNormUpdatesHiddenAndProducesNormalizedOutput()
    {
        const int count = 8;
        var tensor = new VmTensor(VmElementType.Float32, [count]);
        VmParameter[] parameters =
        [
            new("hidden", VmAccess.ReadOnly, tensor),
            new("residual", VmAccess.ReadOnly, tensor),
            new("weight", VmAccess.ReadOnly, tensor),
            new("updated", VmAccess.ReadWrite, tensor),
            new("output", VmAccess.ReadWrite, tensor),
        ];
        var operation = new VmOperator(
            new(GraphElementType.Float32, GraphElementType.Float32),
            InstructionCollectionIds.TransformerFloat32,
            "transformer.residual-rms-norm",
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
            new Dictionary<string, string> { ["epsilon"] = "1e-5" });
        var kernel = new VmDefinition(
            "residual_norm",
            VmDefinitionKind.Kernel,
            parameters,
            [new("body", operation)],
            new(64));
        var dispatchArguments = parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var forward = new VmDefinition(
            "forward",
            VmDefinitionKind.Orchestration,
            parameters,
            [
                new("compute", new VmDispatch(kernel.Id, dispatchArguments, new(1))),
                new("barrier", new VmBarrier(["updated", "output"]), ["compute"]),
            ]);
        var program = new VmProgram(
            "transformer-residual-rms-norm",
            "transformer.fp32",
            VmTarget.Direct3D12,
            parameters.Select(parameter => new VmSlot(
                parameter.Name,
                VmSlotScope.Local,
                parameter.Access,
                parameter.Tensor)),
            [kernel, forward],
            [new("forward", forward.Id, dispatchArguments)],
            new("none", 1, []));

        float[] hidden = [1, -2, 3, -4, 5, -6, 7, -8];
        float[] residual = [0.5f, 1, -1.5f, 2, -2.5f, 3, -3.5f, 4];
        float[] weight = [1, 0.5f, 1.5f, 2, 0.75f, 1.25f, 0.25f, 1.75f];
        var expectedHidden = hidden.Zip(residual, static (left, right) => left + right).ToArray();
        var scale = 1f / MathF.Sqrt(expectedHidden.Select(value => value * value).Average() + 1e-5f);
        var expectedOutput = expectedHidden.Zip(
            weight, (value, multiplier) => value * scale * multiplier).ToArray();

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("hidden", Bytes(hidden));
        executor.Upload("residual", Bytes(residual));
        executor.Upload("weight", Bytes(weight));
        executor.Execute("forward");

        AssertClose(expectedHidden, Values(executor.Readback("updated")));
        AssertClose(expectedOutput, Values(executor.Readback("output")));
    }

    [Fact]
    public void SwiGluProducesPhi4GateActivation()
    {
        const int count = 4;
        var gateUpTensor = new VmTensor(VmElementType.Float32, [count * 2]);
        var outputTensor = new VmTensor(VmElementType.Float32, [count]);
        VmParameter[] parameters =
        [
            new("gate_up", VmAccess.ReadOnly, gateUpTensor),
            new("output", VmAccess.ReadWrite, outputTensor),
        ];
        var operation = new VmOperator(
            new(GraphElementType.Float32, GraphElementType.Float32),
            InstructionCollectionIds.TransformerFloat32,
            "transformer.swiglu",
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)));
        var kernel = new VmDefinition(
            "swiglu",
            VmDefinitionKind.Kernel,
            parameters,
            [new("body", operation)],
            new(64));
        var arguments = parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var forward = new VmDefinition(
            "forward",
            VmDefinitionKind.Orchestration,
            parameters,
            [
                new("compute", new VmDispatch(kernel.Id, arguments, new(1))),
                new("barrier", new VmBarrier(["output"]), ["compute"]),
            ]);
        var program = new VmProgram(
            "transformer-swiglu",
            "transformer.fp32",
            VmTarget.Direct3D12,
            [
                new("gate_up", VmSlotScope.Local, VmAccess.ReadOnly, gateUpTensor),
                new("output", VmSlotScope.Local, VmAccess.ReadWrite, outputTensor),
            ],
            [kernel, forward],
            [new("forward", forward.Id, arguments)],
            new("none", 1, []));
        float[] gateUp = [-2, -1, 1, 2, 0.5f, 1.5f, -2, 3];
        var expected = Enumerable.Range(0, count).Select(index =>
        {
            var gate = gateUp[index];
            return gate / (1 + MathF.Exp(-gate)) * gateUp[count + index];
        }).ToArray();

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("gate_up", Bytes(gateUp));
        executor.Execute("forward");

        AssertClose(expected, Values(executor.Readback("output")));
    }

    [Fact]
    public void ScaledAddCombinesBaseAndLoraProjection()
    {
        const int count = 4;
        var tensor = new VmTensor(VmElementType.Float32, [count]);
        VmParameter[] parameters =
        [
            new("input", VmAccess.ReadOnly, tensor),
            new("update", VmAccess.ReadOnly, tensor),
            new("output", VmAccess.ReadWrite, tensor),
        ];
        var operation = new VmOperator(
            new(GraphElementType.Float32, GraphElementType.Float32),
            InstructionCollectionIds.TransformerFloat32,
            "transformer.scaled-add",
            parameters.Select(parameter => new VmArgument(parameter.Name, parameter.Name)),
            new Dictionary<string, string> { ["scale"] = "2" });
        var kernel = new VmDefinition(
            "scaled_add", VmDefinitionKind.Kernel, parameters,
            [new("body", operation)], new(64));
        var arguments = parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var forward = new VmDefinition(
            "forward", VmDefinitionKind.Orchestration, parameters,
            [
                new("compute", new VmDispatch(kernel.Id, arguments, new(1))),
                new("barrier", new VmBarrier(["output"]), ["compute"]),
            ]);
        var program = new VmProgram(
            "transformer-scaled-add", "transformer.fp32", VmTarget.Direct3D12,
            parameters.Select(parameter => new VmSlot(
                parameter.Name, VmSlotScope.Local, parameter.Access, parameter.Tensor)),
            [kernel, forward],
            [new("forward", forward.Id, arguments)],
            new("none", 1, []));
        float[] input = [1, -2, 3, -4];
        float[] update = [0.5f, 1, -1.5f, 2];

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("input", Bytes(input));
        executor.Upload("update", Bytes(update));
        executor.Execute("forward");

        AssertClose([2, 0, 0, 0], Values(executor.Readback("output")));
    }

    [Fact]
    public void BatchRmsNormNormalizesValidRowsAndSkipsPadding()
    {
        const int batch = 3;
        const int width = 5;
        var matrix = new VmTensor(VmElementType.Float32, [batch, width]);
        var weightTensor = new VmTensor(VmElementType.Float32, [width]);
        var controlTensor = new VmTensor(VmElementType.Int32, [2]);
        VmParameter[] parameters =
        [
            new("input", VmAccess.ReadOnly, matrix),
            new("weight", VmAccess.ReadOnly, weightTensor),
            new("control", VmAccess.ReadOnly, controlTensor),
            new("output", VmAccess.ReadWrite, matrix),
        ];
        var program = SingleOperationProgram(
            "batch-rms-norm",
            parameters,
            "transformer.batch-rms-norm",
            new(batch),
            new Dictionary<string, string> { ["epsilon"] = "1e-5" });
        float[] input =
        [
            1, -2, 3, -4, 5,
            2, 4, -1, -3, 0.5f,
            9, 9, 9, 9, 9,
        ];
        float[] weight = [1, 0.5f, 1.5f, 2, 0.25f];
        var expected = Enumerable.Repeat(-77f, batch * width).ToArray();
        for (var row = 0; row < 2; row++)
        {
            var rowValues = input.AsSpan(row * width, width);
            var scale = 1f / MathF.Sqrt(
                rowValues.ToArray().Select(value => value * value).Average() + 1e-5f);
            for (var column = 0; column < width; column++)
                expected[row * width + column] = rowValues[column] * scale * weight[column];
        }

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("input", Bytes(input));
        executor.Upload("weight", Bytes(weight));
        executor.Upload("control", Bytes([7, 2]));
        executor.Upload("output", Bytes(Enumerable.Repeat(-77f, batch * width).ToArray()));
        executor.Execute("forward");

        AssertClose(expected, Values(executor.Readback("output")));
    }

    [Fact]
    public void BatchGatherRowsCopiesChunkAndSkipsPadding()
    {
        const int rows = 5;
        const int batch = 4;
        const int width = 3;
        var tableTensor = new VmTensor(VmElementType.Float32, [rows, width]);
        var controlTensor = new VmTensor(VmElementType.Int32, [2]);
        var outputTensor = new VmTensor(VmElementType.Float32, [batch, width]);
        VmParameter[] parameters =
        [
            new("table", VmAccess.ReadOnly, tableTensor),
            new("control", VmAccess.ReadOnly, controlTensor),
            new("output", VmAccess.ReadWrite, outputTensor),
        ];
        var program = SingleOperationProgram(
            "batch-gather-rows",
            parameters,
            "transformer.batch-gather-rows",
            new(1));
        float[] table =
        [
            0, 1, 2,
            10, 11, 12,
            20, 21, 22,
            30, 31, 32,
            40, 41, 42,
        ];

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("table", Bytes(table));
        executor.Upload("control", Bytes([2, 2]));
        executor.Upload("output", Bytes(Enumerable.Repeat(-17f, batch * width).ToArray()));
        executor.Execute("forward");

        AssertClose(
            [20, 21, 22, 30, 31, 32, -17, -17, -17, -17, -17, -17],
            Values(executor.Readback("output")));
    }

    [Fact]
    public void BatchSharedMatVecProjectsValidRowsWithSharedHalfWeights()
    {
        const int batch = 3;
        const int inputSize = 4;
        const int outputSize = 2;
        var weightTensor = new VmTensor(VmElementType.Float16, [outputSize, inputSize]);
        var inputTensor = new VmTensor(VmElementType.Float32, [batch, inputSize]);
        var controlTensor = new VmTensor(VmElementType.Int32, [2]);
        var outputTensor = new VmTensor(VmElementType.Float32, [batch, outputSize]);
        VmParameter[] parameters =
        [
            new("weight", VmAccess.ReadOnly, weightTensor),
            new("input", VmAccess.ReadOnly, inputTensor),
            new("control", VmAccess.ReadOnly, controlTensor),
            new("output", VmAccess.ReadWrite, outputTensor),
        ];
        var program = SingleOperationProgram(
            "batch-shared-matvec",
            parameters,
            "transformer.batch-shared-matvec",
            new(3, 2));
        float[] weight =
        [
            1, -0.5f, 2, 0.25f,
            -1, 3, 0.5f, -2,
        ];
        float[] input =
        [
            2, 4, -1, 8,
            -3, 1, 2, -2,
            9, 9, 9, 9,
        ];

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("weight", HalfBytes(weight));
        executor.Upload("input", Bytes(input));
        executor.Upload("control", Bytes([0, 2]));
        executor.Upload("output", Bytes(Enumerable.Repeat(-31f, batch * outputSize).ToArray()));
        executor.Execute("forward");

        AssertClose(
            [0, -6.5f, 0f, 11f, -31f, -31f],
            Values(executor.Readback("output")));
    }

    [Fact]
    public void BatchAttentionWritesContinuousKvCacheAndAttendsThroughEachPosition()
    {
        const int batch = 2;
        const int queryHeads = 2;
        const int keyValueHeads = 1;
        const int headSize = 2;
        const int context = 4;
        const int querySize = queryHeads * headSize;
        const int keyValueSize = keyValueHeads * headSize;
        const int qkvWidth = querySize + 2 * keyValueSize;
        VmParameter[] parameters =
        [
            new("qkv", VmAccess.ReadOnly,
                new(VmElementType.Float32, [batch, qkvWidth])),
            new("control", VmAccess.ReadOnly, new(VmElementType.Int32, [2])),
            new("frequencies", VmAccess.ReadOnly, new(VmElementType.Float32, [1])),
            new("query", VmAccess.ReadWrite,
                new(VmElementType.Float32, [batch, querySize])),
            new("key_cache", VmAccess.ReadWrite,
                new(VmElementType.Float32, [context, keyValueHeads, headSize])),
            new("value_cache", VmAccess.ReadWrite,
                new(VmElementType.Float32, [context, keyValueHeads, headSize])),
            new("scores", VmAccess.ReadWrite,
                new(VmElementType.Float32, [batch, queryHeads, context])),
            new("probabilities", VmAccess.ReadWrite,
                new(VmElementType.Float32, [batch, queryHeads, context])),
            new("output", VmAccess.ReadWrite,
                new(VmElementType.Float32, [batch, querySize])),
        ];
        var byName = parameters.ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var rope = Kernel(
            "rope",
            ["qkv", "control", "frequencies", "query", "key_cache", "value_cache"],
            byName,
            "transformer.batch-rope-kv-write",
            new Dictionary<string, string>
            {
                ["head_size"] = headSize.ToString(),
                ["rotary_size"] = headSize.ToString(),
                ["rope_scale"] = "1",
            });
        var scores = Kernel(
            "scores",
            ["query", "key_cache", "control", "scores"],
            byName,
            "transformer.batch-gqa-scores");
        var softmax = Kernel(
            "softmax",
            ["scores", "control", "probabilities"],
            byName,
            "transformer.batch-causal-softmax");
        var values = Kernel(
            "values",
            ["probabilities", "value_cache", "control", "output"],
            byName,
            "transformer.batch-gqa-values");
        var arguments = parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var forward = new VmDefinition(
            "forward",
            VmDefinitionKind.Orchestration,
            parameters,
            [
                new("rope", new VmDispatch(rope.Id, Arguments(rope), new(1))),
                new("rope_barrier",
                    new VmBarrier(["query", "key_cache", "value_cache"]), ["rope"]),
                new("scores", new VmDispatch(scores.Id, Arguments(scores), new(1)),
                    ["rope_barrier"]),
                new("scores_barrier", new VmBarrier(["scores"]), ["scores"]),
                new("softmax",
                    new VmDispatch(softmax.Id, Arguments(softmax), new(batch * queryHeads)),
                    ["scores_barrier"]),
                new("probability_barrier",
                    new VmBarrier(["probabilities"]), ["softmax"]),
                new("values", new VmDispatch(values.Id, Arguments(values), new(1)),
                    ["probability_barrier"]),
                new("output_barrier", new VmBarrier(["output"]), ["values"]),
            ]);
        var program = new VmProgram(
            "batch-attention-continuity",
            "transformer.fp32",
            VmTarget.Direct3D12,
            parameters.Select(parameter => new VmSlot(
                parameter.Name, VmSlotScope.Local, parameter.Access, parameter.Tensor)),
            [rope, scores, softmax, values, forward],
            [new("forward", forward.Id, arguments)],
            new("none", 1, []));
        float[] qkv =
        [
            1, 0, 0, 1, 1, 1, 10, 20,
            1, 1, 2, 0, 0, 2, 30, 40,
        ];
        float[] keyCache = [1, 0, -9, -9, -9, -9, -9, -9];
        float[] valueCache = [2, 4, -8, -8, -8, -8, -8, -8];
        float[] expectedKeyCache = [1, 0, 1, 1, 0, 2, -9, -9];
        float[] expectedValueCache = [2, 4, 10, 20, 30, 40, -8, -8];
        var expectedOutput = AttentionReference(
            qkv, expectedKeyCache, expectedValueCache,
            batch, queryHeads, keyValueHeads, headSize, context, 1);

        using var executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutor();
        executor.Upload("qkv", Bytes(qkv));
        executor.Upload("control", Bytes([1, batch]));
        executor.Upload("frequencies", Bytes([0f]));
        executor.Upload("key_cache", Bytes(keyCache));
        executor.Upload("value_cache", Bytes(valueCache));
        executor.Execute("forward");

        AssertClose(expectedKeyCache, Values(executor.Readback("key_cache")));
        AssertClose(expectedValueCache, Values(executor.Readback("value_cache")));
        AssertClose(expectedOutput, Values(executor.Readback("output")), 2e-5f);
    }

    private static VmProgram SingleOperationProgram(
        string name,
        VmParameter[] parameters,
        string operationName,
        VmThreadGroup groups,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        var arguments = parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        var operation = new VmOperator(
            new(GraphElementType.Float32, GraphElementType.Float32),
            InstructionCollectionIds.TransformerFloat32,
            operationName,
            arguments,
            attributes);
        var kernel = new VmDefinition(
            name,
            VmDefinitionKind.Kernel,
            parameters,
            [new("body", operation)],
            new(64));
        var outputNames = parameters
            .Where(parameter => parameter.Access == VmAccess.ReadWrite)
            .Select(parameter => parameter.Name)
            .ToArray();
        var forward = new VmDefinition(
            "forward",
            VmDefinitionKind.Orchestration,
            parameters,
            [
                new("compute", new VmDispatch(kernel.Id, arguments, groups)),
                new("barrier", new VmBarrier(outputNames), ["compute"]),
            ]);
        return new(
            name,
            "transformer.fp32",
            VmTarget.Direct3D12,
            parameters.Select(parameter => new VmSlot(
                parameter.Name, VmSlotScope.Local, parameter.Access, parameter.Tensor)),
            [kernel, forward],
            [new("forward", forward.Id, arguments)],
            new("none", 1, []));
    }

    private static VmDefinition Kernel(
        string id,
        string[] names,
        IReadOnlyDictionary<string, VmParameter> parameters,
        string operation,
        IReadOnlyDictionary<string, string>? attributes = null)
    {
        var selected = names.Select(name => parameters[name]).ToArray();
        var arguments = selected.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
        return new(
            id,
            VmDefinitionKind.Kernel,
            selected,
            [new("body", new VmOperator(
                new(GraphElementType.Float32, GraphElementType.Float32),
                InstructionCollectionIds.TransformerFloat32,
                operation,
                arguments,
                attributes))],
            new(64));
    }

    private static VmArgument[] Arguments(VmDefinition definition) =>
        definition.Parameters.Select(
            parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();

    private static float[] AttentionReference(
        float[] qkv,
        float[] keyCache,
        float[] valueCache,
        int batch,
        int queryHeads,
        int keyValueHeads,
        int headSize,
        int context,
        int start)
    {
        var querySize = queryHeads * headSize;
        var keyValueSize = keyValueHeads * headSize;
        var qkvWidth = querySize + 2 * keyValueSize;
        var output = new float[batch * querySize];
        var scale = 1f / MathF.Sqrt(headSize);
        for (var row = 0; row < batch; row++)
        {
            var active = start + row;
            for (var head = 0; head < queryHeads; head++)
            {
                var keyValueHead = head / (queryHeads / keyValueHeads);
                var scores = new float[active + 1];
                for (var token = 0; token <= active; token++)
                {
                    for (var d = 0; d < headSize; d++)
                    {
                        scores[token] +=
                            qkv[row * qkvWidth + head * headSize + d] *
                            keyCache[(token * keyValueHeads + keyValueHead) * headSize + d];
                    }
                    scores[token] *= scale;
                }
                var maximum = scores.Max();
                var exponentials = scores.Select(score => MathF.Exp(score - maximum)).ToArray();
                var denominator = exponentials.Sum();
                for (var d = 0; d < headSize; d++)
                {
                    for (var token = 0; token <= active; token++)
                    {
                        output[row * querySize + head * headSize + d] +=
                            exponentials[token] / denominator *
                            valueCache[(token * keyValueHeads + keyValueHead) * headSize + d];
                    }
                }
            }
        }
        return output;
    }

    private static byte[] Bytes(float[] values) =>
        MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static byte[] Bytes(int[] values) =>
        MemoryMarshal.AsBytes(values.AsSpan()).ToArray();

    private static byte[] HalfBytes(float[] values) =>
        MemoryMarshal.AsBytes(values.Select(value => (Half)value).ToArray().AsSpan()).ToArray();

    private static float[] Values(byte[] bytes) =>
        MemoryMarshal.Cast<byte, float>(bytes).ToArray();

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var index = 0; index < expected.Length; index++)
            Assert.InRange(MathF.Abs(expected[index] - actual[index]), 0, tolerance);
    }
}
