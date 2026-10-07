using System.Runtime.InteropServices;
using SharpInference.Backends.D3D12Vm;
using SharpInference.Gguf;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Runtime;
using SharpInference.Vm;

namespace SharpInference.Architectures.Phi4.D3D12;

public sealed class Phi4D3D12MatrixProjector : IPhi4MatrixProjector, IDisposable
{
    private readonly D3D12VmExecutor executor;
    private readonly IReadOnlyDictionary<string, Projection> projections;
    private readonly IReadOnlyDictionary<Shape, Buffers> buffers;
    private readonly object gate = new();
    private bool disposed;

    public Phi4D3D12MatrixProjector(Phi4ModelPackage package, int adapterIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(package);
        var weights = Weights(package).ToArray();
        var projectionEntries = weights.Select((weight, index) =>
        {
            ValidateWeight(weight);
            var shape = new Shape(
                checked((int)weight.Dimensions[0]),
                checked((int)weight.Dimensions[1]));
            return new Projection(weight, $"project_{index}", $"weight_{index}", shape);
        }).ToArray();
        var program = CreateProgram(projectionEntries);
        executor = new D3D12VmCompiler(DefaultInstructionCollections.Create())
            .Compile(program)
            .CreateExecutorForExplicitGlobalUploads(adapterIndex);
        try
        {
            foreach (var projection in projectionEntries)
                executor.Upload(projection.WeightSlot, projection.Weight.Bytes);
        }
        catch
        {
            executor.Dispose();
            throw;
        }
        projections = projectionEntries.ToDictionary(
            projection => projection.Weight.Name,
            StringComparer.Ordinal);
        buffers = projectionEntries.Select(projection => projection.Shape)
            .Distinct()
            .ToDictionary(
                shape => shape,
                shape => new Buffers(
                    new byte[shape.Input * sizeof(float)],
                    new byte[shape.Output * sizeof(float)]));
    }

    public void Project(GgufModelTensor weight, ReadOnlySpan<float> input, Span<float> output)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (!projections.TryGetValue(weight.Name, out var projection) ||
            !ReferenceEquals(projection.Weight, weight))
            throw new ArgumentException("The matrix is not resident in this Phi-4 projector.", nameof(weight));
        if (input.Length != projection.Shape.Input)
            throw new ArgumentException($"Expected {projection.Shape.Input} input values.", nameof(input));
        if (output.Length != projection.Shape.Output)
            throw new ArgumentException($"Expected {projection.Shape.Output} output values.", nameof(output));

        lock (gate)
        {
            var io = buffers[projection.Shape];
            MemoryMarshal.AsBytes(input).CopyTo(io.Input);
            executor.Upload(InputSlot(projection.Shape), io.Input);
            executor.Execute(projection.Entry);
            executor.Readback(OutputSlot(projection.Shape), io.Output);
            MemoryMarshal.Cast<byte, float>(io.Output).CopyTo(output);
        }
    }

    public void Dispose()
    {
        if (disposed)
            return;
        executor.Dispose();
        disposed = true;
    }

    private static IEnumerable<GgufModelTensor> Weights(Phi4ModelPackage package)
    {
        for (var layer = 0; layer < 32; layer++)
        {
            yield return package.Text.GetTensor($"blk.{layer}.attn_qkv.weight");
            yield return package.Text.GetTensor($"blk.{layer}.attn_output.weight");
            yield return package.Text.GetTensor($"blk.{layer}.ffn_up.weight");
            yield return package.Text.GetTensor($"blk.{layer}.ffn_down.weight");
        }
        yield return package.Text.GetTensor("token_embd.weight");
    }

    private static void ValidateWeight(GgufModelTensor weight)
    {
        if (weight.Type != GgufModelTensorType.Float16 || weight.Dimensions.Count != 2)
            throw new InvalidDataException(
                $"Phi-4 projection '{weight.Name}' must be a rank-two Float16 tensor.");
    }

    private static VmProgram CreateProgram(IReadOnlyList<Projection> projections)
    {
        var shapes = projections.Select(projection => projection.Shape).Distinct().ToArray();
        var slots = new List<VmSlot>();
        foreach (var shape in shapes)
        {
            slots.Add(new(InputSlot(shape), VmSlotScope.Local, VmAccess.ReadOnly,
                new(VmElementType.Float32, [shape.Input])));
            slots.Add(new(OutputSlot(shape), VmSlotScope.Local, VmAccess.ReadWrite,
                new(VmElementType.Float32, [shape.Output])));
        }
        slots.AddRange(projections.Select(projection => new VmSlot(
            projection.WeightSlot,
            VmSlotScope.Global,
            VmAccess.ReadOnly,
            new(VmElementType.Float16, [projection.Shape.Output, projection.Shape.Input]))));

        var definitions = new List<VmDefinition>();
        var kernelIds = new Dictionary<Shape, string>();
        foreach (var shape in shapes)
        {
            var kernelId = $"matvec_{shape.Input}_{shape.Output}";
            kernelIds.Add(shape, kernelId);
            VmParameter[] parameters =
            [
                new("weight", VmAccess.ReadOnly,
                    new(VmElementType.Float16, [shape.Output, shape.Input])),
                new("input", VmAccess.ReadOnly, new(VmElementType.Float32, [shape.Input])),
                new("output", VmAccess.ReadWrite, new(VmElementType.Float32, [shape.Output])),
            ];
            var arguments = parameters.Select(
                parameter => new VmArgument(parameter.Name, parameter.Name)).ToArray();
            var operation = new VmOperator(
                new(GraphElementType.Float32, GraphElementType.Float32),
                InstructionCollectionIds.TierZeroFloat32,
                "core.mat-vec",
                arguments,
                executionConfiguration: GpuMatVecExecution.Cooperative);
            definitions.Add(new(
                kernelId,
                VmDefinitionKind.Kernel,
                parameters,
                [new("body", operation)],
                new(GpuMatVecExecution.Threads)));
        }

        var entries = new List<VmEntry>();
        foreach (var projection in projections)
        {
            VmParameter[] parameters =
            [
                new("weight", VmAccess.ReadOnly,
                    new(VmElementType.Float16,
                        [projection.Shape.Output, projection.Shape.Input])),
                new("input", VmAccess.ReadOnly,
                    new(VmElementType.Float32, [projection.Shape.Input])),
                new("output", VmAccess.ReadWrite,
                    new(VmElementType.Float32, [projection.Shape.Output])),
            ];
            VmArgument[] arguments =
            [
                new("weight", projection.WeightSlot),
                new("input", InputSlot(projection.Shape)),
                new("output", OutputSlot(projection.Shape)),
            ];
            VmArgument[] dispatchArguments =
            [
                new("weight", "weight"),
                new("input", "input"),
                new("output", "output"),
            ];
            var definitionId = $"entry_{projection.Entry}";
            var groups = GpuMatVecExecution.Groups((ulong)projection.Shape.Output);
            definitions.Add(new(
                definitionId,
                VmDefinitionKind.Orchestration,
                parameters,
                [
                    new("compute", new VmDispatch(
                        kernelIds[projection.Shape], dispatchArguments, new(groups.X, groups.Y))),
                    new("barrier", new VmBarrier(["output"]), ["compute"]),
                ]));
            entries.Add(new(
                projection.Entry,
                definitionId,
                arguments));
        }
        return new(
            "phi4-resident-projections",
            "phi4.projections.fp16",
            VmTarget.Direct3D12,
            slots,
            definitions,
            entries,
            new("none", 1, []));
    }

    private static string InputSlot(Shape shape) =>
        $"input_{shape.Input}_{shape.Output}";

    private static string OutputSlot(Shape shape) =>
        $"output_{shape.Input}_{shape.Output}";

    private sealed record Projection(
        GgufModelTensor Weight,
        string Entry,
        string WeightSlot,
        Shape Shape);

    private sealed record Buffers(byte[] Input, byte[] Output);
    private readonly record struct Shape(int Input, int Output);
}
