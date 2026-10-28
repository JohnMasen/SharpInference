using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Instructions.D3D12;

/// <summary>Shares FP32 source-generation utilities without owning an instruction set.</summary>
public abstract class GpuTransformerInstruction(
    Guid collectionId,
    string name,
    IReadOnlyList<InstructionPort> ports,
    IReadOnlyList<string>? attributes = null) : Instruction
{
    private static readonly ExecutionPrecision Precision =
        new(ElementType.Float32, ElementType.Float32);

    public override Guid CollectionId => collectionId;
    public override string Name => name;
    public override ExecutionTarget Target => ExecutionTarget.Direct3D12;
    public override string? Comment => collectionId == TierZeroTensorInstructionSet.Id
        ? TierZeroTensorInstructionSet.Comment(Name) : base.Comment;
    public override IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions =>
        collectionId == TierZeroTensorInstructionSet.Id ? TierZeroTensorInstructionSet.Attributes(Name) : [];
    public override IReadOnlyList<InstructionSignature> Signatures { get; } =
        collectionId == TierZeroTensorInstructionSet.Id
            ? [TierZeroTensorInstructionSet.Signatures[name]]
            : [new(ports, attributes ?? [], Precision)];

    protected override void Validate(InstructionParameter[] parameters)
    {
        if (CollectionId == TierZeroTensorInstructionSet.Id)
        {
            TierZeroTensorInstructionSet.Validate(Name, parameters);
        }
    }

    protected static InstructionHelper ReductionHelper { get; } =
        new("transformer.reduction64", "groupshared float transformerReduction[64];");

    protected static IReadOnlyDictionary<string, InstructionTensorParameter> Tensors(
        InstructionParameter[] parameters) =>
        parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);

    protected float PositiveFloat(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>()
            .Single(parameter => parameter.Name == name).Value;
        var value = float.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (!float.IsFinite(value) || value <= 0)
        {
            throw new InstructionAdaptationException(
                collectionId, name,
                $"Attribute '{name}' must be finite and positive.");
        }

        return value;
    }

    protected static string Load(InstructionTensorParameter tensor, string index) =>
        $"load32({tensor.Expression},{tensor.OffsetExpression},{index})";

    protected static string Store(
        InstructionTensorParameter tensor,
        string index,
        string value) =>
        $"{tensor.Expression}.Store({tensor.OffsetExpression}+({index})*4u,asuint({value}));";
}

/// <summary>Shares shape checks and dispatch helpers for batched GPU instructions.</summary>
public abstract class GpuBatchTransformerInstruction(
    Guid collectionId,
    string name,
    IReadOnlyList<InstructionPort> ports,
    IReadOnlyList<string>? attributes = null)
    : GpuTransformerInstruction(collectionId, name, ports, attributes)
{
    protected (int Batch, int Width) BatchMatrix(
        InstructionTensorParameter tensor,
        string name)
    {
        RequireRank(tensor, name, 2);
        var batch = tensor.Tensor.Dimensions[0];
        var width = tensor.Tensor.Dimensions[1];
        if (batch <= 0 || width <= 0)
        {
            throw Error($"'{name}' dimensions must be positive.");
        }

        return (batch, width);
    }

    protected int Vector(InstructionTensorParameter tensor, string name)
    {
        RequireRank(tensor, name, 1);
        var width = tensor.Tensor.Dimensions[0];
        if (width <= 0)
        {
            throw Error($"'{name}' dimensions must be positive.");
        }

        return width;
    }

    protected (int Context, int Heads, int HeadSize) Cache(
        InstructionTensorParameter tensor,
        string name)
    {
        RequireRank(tensor, name, 3);
        var dimensions = tensor.Tensor.Dimensions;
        if (dimensions.Any(dimension => dimension <= 0))
        {
            throw Error($"'{name}' dimensions must be positive.");
        }

        return (dimensions[0], dimensions[1], dimensions[2]);
    }

    protected void RequireControl(InstructionTensorParameter control)
    {
        if (control.Tensor.Dimensions.Count != 1 ||
            control.Tensor.Dimensions[0] != 2)
        {
            throw Error("'control' must have shape [2].");
        }
    }

    protected void RequireSameShape(
        InstructionTensorParameter left,
        string leftName,
        InstructionTensorParameter right,
        string rightName)
    {
        if (!left.Tensor.Dimensions.SequenceEqual(right.Tensor.Dimensions))
        {
            throw Error($"'{leftName}' and '{rightName}' must have matching shapes.");
        }
    }

    protected int PositiveInteger(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>()
            .Single(parameter => parameter.Name == name).Value;
        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value) ||
            value <= 0)
        {
            throw Error($"Attribute '{name}' must be a positive integer.");
        }

        return value;
    }

    protected static string FloatLiteral(float value)
    {
        var text = value.ToString("R", CultureInfo.InvariantCulture);
        return text.Contains('.') || text.Contains('E', StringComparison.OrdinalIgnoreCase)
            ? text
            : text + ".0";
    }

    protected static string ValidCount(InstructionTensorParameter control) =>
        $"asint({control.Expression}.Load({control.OffsetExpression}+4u))";

    protected static string StartPosition(InstructionTensorParameter control) =>
        $"asint({control.Expression}.Load({control.OffsetExpression}))";

    protected static InstructionHelper BatchReduction { get; } =
        new("transformer.batch.reduction64", "groupshared float transformerBatchReduction[64];");

    protected const string LinearGroup =
        "gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x";

    protected InstructionAdaptationException Error(string message) =>
        new(CollectionId, Name, message);

    private void RequireRank(
        InstructionTensorParameter tensor,
        string name,
        int rank)
    {
        if (tensor.Tensor.Dimensions.Count != rank)
        {
            throw Error($"'{name}' must have rank {rank}.");
        }
    }
}

internal sealed class GpuEmbeddingGatherBatchInstruction() : GpuTransformerInstruction(
    TierZeroTensorInstructionSet.Id,
    "transformer.embedding-gather-batch",
    [
        new("table", ElementType.Float16, ResourceAccess.Read),
        new("indices", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ])
{
    public override IReadOnlyList<InstructionIndexBound> IndexBounds => [new("indices", "table", 0)];

    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var table = tensors["table"];
        var indices = tensors["indices"];
        var output = tensors["output"];
        var width = output.Tensor.Dimensions[^1];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var source = $$"""
            {
                uint element=i;
                if(element<{{count}}u) {
                    uint row=element/{{width}}u;
                    uint column=element-row*{{width}}u;
                    uint token={{indices.Expression}}.Load({{indices.OffsetExpression}}+row*4u);
                    float value=load16({{table.Expression}},{{table.OffsetExpression}},token*{{width}}u+column);
                    {{Store(output, "element", "value")}}
                }
            }
            """;
        return new(source,
            [new InstructionHelper(
                "transformer.load16",
                "float load16(RWByteAddressBuffer b, uint o, uint i) { uint a=o+i*2; return f16tof32((b.Load(a & ~3u) >> ((a & 2u)*8u)) & 65535u); }")],
            InstructionSynchronization.None);
    }
}

internal sealed class GpuCopyEmbeddingRowsInstruction() : GpuTransformerInstruction(
    TierZeroTensorInstructionSet.Id,
    "transformer.copy-embedding-rows",
    [
        new("input", ElementType.Float32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ],
    ["rows", "output_start"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var output = tensors["output"];
        var rows = PositiveInteger(parameters, "rows");
        var outputStart = NonNegativeInteger(parameters, "output_start");
        var width = input.Tensor.Dimensions[^1];
        var count = checked((ulong)rows * (ulong)width);
        var destinationOffset = checked((ulong)outputStart * (ulong)width);
        var source = $$"""
            {
                uint element=i;
                if(element<{{count}}u)
                    {{Store(output, $"element+{destinationOffset}u", Load(input, "element"))}}
            }
            """;
        return new(source, [GpuInstructionHelpers.Load32],
            InstructionSynchronization.None);
    }

    private static int PositiveInteger(InstructionParameter[] parameters, string name)
    {
        var value = Integer(parameters, name);
        if (value <= 0)
        {
            throw new InstructionAdaptationException(
                TierZeroTensorInstructionSet.Id, name,
                $"Attribute '{name}' must be positive.");
        }

        return value;
    }

    private static int NonNegativeInteger(InstructionParameter[] parameters, string name)
    {
        var value = Integer(parameters, name);
        if (value < 0)
        {
            throw new InstructionAdaptationException(
                TierZeroTensorInstructionSet.Id, name,
                $"Attribute '{name}' must be non-negative.");
        }

        return value;
    }

    private static int Integer(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>()
            .Single(parameter => parameter.Name == name).Value;
        return int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new InstructionAdaptationException(
                TierZeroTensorInstructionSet.Id, name,
                $"Attribute '{name}' must be an integer.");
    }
}

internal sealed class GpuArgMaxInstruction() : GpuTransformerInstruction(
    TierZeroTensorInstructionSet.Id,
    "transformer.argmax",
    [
        new("input", ElementType.Float32, ResourceAccess.Read),
                    new("output", ElementType.Int32, ResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var output = tensors["output"];
        var count = input.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        if (output.Tensor.Dimensions.Aggregate(
                1UL, (product, dimension) => checked(product * (ulong)dimension)) != 1)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Argmax output must be scalar.");
        }

        return new($$"""
                        {
                            precise float best=asfloat(0xff800000u);
                            uint bestIndex=0xffffffffu;
                            for(uint j=gpu.groupIndex;j<{{count}}u;j+=64u) {
                                precise float value={{Load(input, "j")}};
                                if(value>best || (value==best && j<bestIndex)) {
                                    best=value;
                                    bestIndex=j;
                                }
                            }
                            transformerReduction[gpu.groupIndex]=best;
                            transformerIndices[gpu.groupIndex]=bestIndex;
                            GroupMemoryBarrierWithGroupSync();
                            for(uint step=32u;step>0u;step>>=1u) {
                                if(gpu.groupIndex<step) {
                                    precise float other=transformerReduction[gpu.groupIndex+step];
                                    uint otherIndex=transformerIndices[gpu.groupIndex+step];
                                    if(other>transformerReduction[gpu.groupIndex] ||
                                        (other==transformerReduction[gpu.groupIndex] &&
                                         otherIndex<transformerIndices[gpu.groupIndex])) {
                                        transformerReduction[gpu.groupIndex]=other;
                                        transformerIndices[gpu.groupIndex]=otherIndex;
                                    }
                                }
                                GroupMemoryBarrierWithGroupSync();
                            }
                            if(gpu.groupIndex==0u)
                                {{output.Expression}}.Store({{output.OffsetExpression}},
                                    transformerIndices[0]==0xffffffffu ? 0u : transformerIndices[0]);
                        }
                        """, [GpuInstructionHelpers.Load32, ReductionHelper,
                            new("transformer.indices64", "groupshared uint transformerIndices[64];")],
                        InstructionSynchronization.GroupMemoryBarrier);
    }
}
internal sealed class GpuBatchSharedMatVecInstruction() : GpuBatchTransformerInstruction(
    TierZeroTensorInstructionSet.Id,
    "transformer.batch-shared-matvec",
    [
        new("weight", ElementType.Float16, ResourceAccess.Read),
        new("input", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var weight = tensors["weight"];
        var input = tensors["input"];
        var control = tensors["control"];
        var output = tensors["output"];
        var (outputSize, inputSize) = BatchMatrix(weight, "weight");
        var (batch, inputWidth) = BatchMatrix(input, "input");
        var (outputBatch, outputWidth) = BatchMatrix(output, "output");
        if (inputWidth != inputSize || outputBatch != batch || outputWidth != outputSize)
        {
            throw Error("Expected weight [O,I], input [B,I], and output [B,O].");
        }

        RequireControl(control);
        return new($$"""
            {
                uint work={{LinearGroup}};
                uint row=work/{{outputSize}}u;
                uint outputColumn=work-row*{{outputSize}}u;
                int validCount={{ValidCount(control)}};
                if(row<{{batch}}u && int(row)<validCount) {
                    precise float sum=0.0f;
                    for(uint column=gpu.groupIndex;column<{{inputSize}}u;column+=64u)
                        sum+=load16({{weight.Expression}},{{weight.OffsetExpression}},
                                outputColumn*{{inputSize}}u+column)*
                            {{Load(input, $"row*{inputSize}u+column")}};
                    transformerBatchReduction[gpu.groupIndex]=sum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(gpu.groupIndex<step)
                            transformerBatchReduction[gpu.groupIndex]+=
                                transformerBatchReduction[gpu.groupIndex+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    if(gpu.groupIndex==0u)
                        {{Store(output, $"row*{outputSize}u+outputColumn",
                            "transformerBatchReduction[0]")}}
                }
            }
            """,
            [Load16, GpuInstructionHelpers.Load32, BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }

    private static InstructionHelper Load16 { get; } = new(
        "transformer.batch.load16",
        "float load16(RWByteAddressBuffer b,uint o,uint i) { uint a=o+i*2u; return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u); }");
}
