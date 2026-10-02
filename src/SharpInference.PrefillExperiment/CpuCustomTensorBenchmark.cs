using System.Diagnostics;
using SharpInference.Backends.Cpu;
using SharpInference.Graphs;

namespace SharpInference.PrefillExperiment;

internal static class CpuCustomTensorBenchmark
{
    public static void Run(int width, int repeats)
    {
        var input = Enumerable.Range(0, width).Select(i => (i % 19 - 9) / 16f).ToArray();
        var halves = input.Select(value => (Half)value).ToArray();
        var cast = Create(PortableTensorOperationContracts.CastFp16ToFp32, [width],
            ("input", [width], halves));
        var castReference = new float[width];
        Compare("CastFp16ToFp32", width, repeats, cast.Executor, cast.Inputs, castReference, () =>
        {
            for (var i = 0; i < width; i++) castReference[i] = (float)halves[i];
        });

        int[] sourceShape = [1, width, 1];
        int[] outputShape = [4, width, 8];
        var broadcast = Create(PortableTensorOperationContracts.Broadcast, outputShape,
            ("input", sourceShape, input));
        var broadcastReference = new float[4 * width * 8];
        Compare("Broadcast [1,width,1] -> [4,width,8]", broadcastReference.Length, repeats,
            broadcast.Executor, broadcast.Inputs, broadcastReference, () =>
                ScalarBroadcast(input, broadcastReference, sourceShape, outputShape));

        var outer = Create(PortableTensorOperationContracts.HeadOuter, [1, 8, width],
            ("left", [1, 8], input.Take(8).Concat(Enumerable.Repeat(1f, 8)).Take(8).ToArray()),
            ("right", [1, width], input));
        var left = (float[])outer.Inputs[new("left")];
        var outerReference = new float[8 * width];
        Compare("HeadOuter [1,8,width]", outerReference.Length, repeats,
            outer.Executor, outer.Inputs, outerReference, () =>
        {
            for (var row = 0; row < 8; row++)
                for (var column = 0; column < width; column++)
                    outerReference[row * width + column] = left[row] * input[column];
        });

        var expression = new FusedElementwiseExpression(2,
        [
            new(PrimitiveGraphOperations.Add, [new(InputIndex: 0), new(InputIndex: 1)]),
            new(PrimitiveGraphOperations.Square, [new(StepIndex: 0)]),
            new(PrimitiveGraphOperations.Tanh, [new(StepIndex: 1)]),
            new(PrimitiveGraphOperations.Multiply, [new(StepIndex: 1), new(StepIndex: 2)]),
            new(PrimitiveGraphOperations.Add, [new(StepIndex: 3), new(StepIndex: 0)]),
        ]);
        var fused = Create(FusedElementwiseExpressionContract.Operation, [width],
            FusedElementwiseExpressionContract.ToAttributes(expression),
            ("input0", [width], input), ("input1", [width], input));
        var fusedReference = new float[width];
        Compare("Branched fused expression (5 steps)", width, repeats,
            fused.Executor, fused.Inputs, fusedReference, () =>
        {
            for (var i = 0; i < width; i++)
            {
                var sum = input[i] + input[i];
                var square = sum * sum;
                var tanh = MathF.Tanh(square);
                var product = square * tanh;
                fusedReference[i] = product + sum;
            }
        });
    }

    private static void Compare(string name, int elements, int repeats,
        CpuPrimitiveGraphExecutor executor, Dictionary<ResourceId, Array> inputs,
        float[] reference, Action scalar)
    {
        scalar();
        var actual = (float[])executor.Execute(inputs).Outputs[new("output")];
        CpuWkv6Benchmark.AssertClose(reference, actual, name);
        var iterations = checked(Math.Max(16, 1048576 / elements) * repeats);
        Action tensor = () => executor.Execute(inputs);
        for (var warm = 0; warm < 16; warm++)
        {
            scalar();
            tensor();
        }
        var scalarTime = Measure(iterations, scalar);
        var tensorTime = Measure(iterations, tensor);
        Console.WriteLine($"  {name}: scalar-kernel={scalarTime.TotalMilliseconds:F3}ms " +
            $"tensor-graph={tensorTime.TotalMilliseconds:F3}ms " +
            $"ratio={scalarTime.TotalMilliseconds / tensorTime.TotalMilliseconds:F2}x " +
            $"iterations={iterations} (graph includes dispatch and output allocation)");
    }

    private static TimeSpan Measure(int iterations, Action action)
    {
        var start = Stopwatch.GetTimestamp();
        for (var i = 0; i < iterations; i++) action();
        return Stopwatch.GetElapsedTime(start);
    }

    private static void ScalarBroadcast(ReadOnlySpan<float> source, Span<float> output,
        int[] sourceShape, int[] outputShape)
    {
        for (var flat = 0; flat < output.Length; flat++)
        {
            var remainder = flat;
            var sourceOffset = 0;
            var sourceStride = 1;
            for (var axis = outputShape.Length - 1; axis >= 0; axis--)
            {
                var position = remainder % outputShape[axis];
                remainder /= outputShape[axis];
                var sourceAxis = axis - (outputShape.Length - sourceShape.Length);
                if (sourceAxis < 0) continue;
                if (sourceShape[sourceAxis] != 1) sourceOffset += position * sourceStride;
                sourceStride *= sourceShape[sourceAxis];
            }
            output[flat] = source[sourceOffset];
        }
    }

    private static (CpuPrimitiveGraphExecutor Executor, Dictionary<ResourceId, Array> Inputs) Create(
        GraphOperationId operation, int[] shape,
        params (string Port, int[] Shape, Array Data)[] sources) =>
        Create(operation, shape, null, sources);

    private static (CpuPrimitiveGraphExecutor Executor, Dictionary<ResourceId, Array> Inputs) Create(
        GraphOperationId operation, int[] shape, IReadOnlyDictionary<string, string>? attributes,
        params (string Port, int[] Shape, Array Data)[] sources)
    {
        var builder = new LogicalGraphBuilder(new GraphIdentity("synthetic", 1, "cpu-tensor-benchmark"),
                new GraphModelSignature(2, 2, 1, 1, 2, "synthetic.state"))
            .AddRegion("root", GraphRegionTypes.Graph, "root")
            .AddResource("output", "output", GraphResourceKind.Output, GraphResourceLifetime.External,
                new TensorDescriptor(GraphElementType.Float32, shape), graphOutput: true);
        foreach (var source in sources)
            builder.AddResource(source.Port, source.Port, GraphResourceKind.Input, GraphResourceLifetime.External,
                new TensorDescriptor(source.Data is Half[] ? GraphElementType.Float16 : GraphElementType.Float32,
                    source.Shape), graphInput: true);
        builder.AddNode("operation", operation, "root",
            sources.Select(source => GraphBindings.Read(source.Port, source.Port))
                .Append(GraphBindings.Write("output", "output")), attributes: attributes);
        var graph = new GraphOptimizer().Optimize(builder.Build(),
            new GraphOptimizationOptions(OptimizationBoundary.Off,
                DefinitionPolicy: GraphDefinitionPolicy.PreserveExpanded),
            CpuPrimitiveGraphBackend.Instance.KernelCatalog);
        return (new CpuPrimitiveGraphExecutor(graph, new EmptyCatalog()),
            sources.ToDictionary(source => new ResourceId(source.Port), source => source.Data));
    }

    private sealed class EmptyCatalog : IModelTensorCatalog
    {
        public int VocabularySize => 2;
        public int EmbeddingSize => 2;
        public int LayerCount => 1;
        public IReadOnlyCollection<string> Names => [];
        public bool TryGet(string name, out IModelTensor tensor)
        {
            tensor = null!;
            return false;
        }
        public IModelTensor GetRequired(string name) => throw new InvalidDataException(name);
        public void Dispose() { }
    }
}
