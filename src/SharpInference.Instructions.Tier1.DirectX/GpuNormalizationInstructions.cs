using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Instructions.D3D12;

internal abstract class GpuNormalizationInstruction(
    string name, IReadOnlyList<InstructionPort> ports, IReadOnlyList<string>? attributes = null)
    : GpuTransformerInstruction(TierOneNormalizationInstructionSet.Id, name, ports, attributes)
{
    public override string Comment => TierOneNormalizationInstructionSet.Comment(Name);
    public override IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions =>
        TierOneNormalizationInstructionSet.Attributes(Name);
    public override IReadOnlyList<InstructionSignature> Signatures
    {
        get;
    } =
        [TierOneNormalizationInstructionSet.Signatures[name]];

    protected override void Validate(InstructionParameter[] parameters) =>
        TierOneNormalizationInstructionSet.Validate(Name, parameters);
}

internal sealed class GpuScaledAddInstruction() : GpuNormalizationInstruction(
    "transformer.scaled-add",
    [
        new("input", ElementType.Float32, ResourceAccess.Read),
        new("update", ElementType.Float32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ],
    ["scale"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var update = tensors["update"];
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var scale = PositiveFloat(parameters, "scale")
            .ToString("R", CultureInfo.InvariantCulture);
        if (!scale.Contains('.') && !scale.Contains('E', StringComparison.OrdinalIgnoreCase))
        {
            scale += ".0";
        }

        var source = $$"""
            {
                uint j=i;
                if(j<{{count}}u)
                    {{Store(output, "j", $"{Load(input, "j")}+{scale}f*{Load(update, "j")}")}}
            }
            """;
        return new(source, [GpuInstructionHelpers.Load32],
            InstructionSynchronization.None);
    }
}

internal sealed class GpuRmsNormInstruction() : GpuNormalizationInstruction(
    "transformer.rms-norm",
    [
        new("input", ElementType.Float32, ResourceAccess.Read),
        new("weight", ElementType.Float32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ],
    ["epsilon"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var weight = tensors["weight"];
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var epsilon = PositiveFloat(parameters, "epsilon")
            .ToString("R", CultureInfo.InvariantCulture);
        var source = $$"""
            {
                precise float sum=0.0f;
                for(uint j=gpu.groupIndex;j<{{count}}u;j+=64u) {
                    precise float value={{Load(input, "j")}};
                    sum+=value*value;
                }
                transformerReduction[gpu.groupIndex]=sum;
                GroupMemoryBarrierWithGroupSync();
                for(uint step=32u;step>0u;step>>=1u) {
                    if(gpu.groupIndex<step)
                        transformerReduction[gpu.groupIndex]+=transformerReduction[gpu.groupIndex+step];
                    GroupMemoryBarrierWithGroupSync();
                }
                precise float scale=rsqrt(transformerReduction[0]/{{count}}.0f+{{epsilon}}f);
                for(uint j=gpu.groupIndex;j<{{count}}u;j+=64u)
                    {{Store(output, "j", $"{Load(input, "j")}*scale*{Load(weight, "j")}")}}
            }
            """;
        return new(source,
            [GpuInstructionHelpers.Load32, ReductionHelper],
            InstructionSynchronization.GroupMemoryBarrier);
    }

}

internal sealed class GpuResidualRmsNormInstruction() : GpuNormalizationInstruction(
    "transformer.residual-rms-norm",
    [
        new("hidden", ElementType.Float32, ResourceAccess.Read),
        new("residual", ElementType.Float32, ResourceAccess.Read),
        new("weight", ElementType.Float32, ResourceAccess.Read),
        new("updated", ElementType.Float32, ResourceAccess.Write),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ],
    ["epsilon"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var hidden = tensors["hidden"];
        var residual = tensors["residual"];
        var weight = tensors["weight"];
        var updated = tensors["updated"];
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var epsilon = PositiveFloat(parameters, "epsilon")
            .ToString("R", CultureInfo.InvariantCulture);
        var source = $$"""
            {
                precise float sum=0.0f;
                for(uint j=gpu.groupIndex;j<{{count}}u;j+=64u) {
                    precise float value={{Load(hidden, "j")}}+{{Load(residual, "j")}};
                    {{Store(updated, "j", "value")}}
                    sum+=value*value;
                }
                transformerReduction[gpu.groupIndex]=sum;
                GroupMemoryBarrierWithGroupSync();
                for(uint step=32u;step>0u;step>>=1u) {
                    if(gpu.groupIndex<step)
                        transformerReduction[gpu.groupIndex]+=transformerReduction[gpu.groupIndex+step];
                    GroupMemoryBarrierWithGroupSync();
                }
                precise float scale=rsqrt(transformerReduction[0]/{{count}}.0f+{{epsilon}}f);
                for(uint j=gpu.groupIndex;j<{{count}}u;j+=64u)
                    {{Store(output, "j", $"{Load(updated, "j")}*scale*{Load(weight, "j")}")}}
            }
            """;
        return new(source,
            [GpuInstructionHelpers.Load32, ReductionHelper],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuSwiGluInstruction() : GpuNormalizationInstruction(
    "transformer.swiglu",
    [
        new("gate_up", ElementType.Float32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var gateUp = tensors["gate_up"];
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        return new($$"""
            {
                if(i<{{count}}u) {
                    precise float gate={{Load(gateUp, "i")}};
                    precise float up={{Load(gateUp, $"i+{count}u")}};
                    {{Store(output, "i", "gate/(1.0f+exp(-gate))*up")}}
                }
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}

internal abstract class GpuBatchNormalizationInstruction(
    string name, IReadOnlyList<InstructionPort> ports, IReadOnlyList<string>? attributes = null)
    : GpuBatchTransformerInstruction(TierOneNormalizationInstructionSet.Id, name, ports, attributes)
{
    public override string Comment => TierOneNormalizationInstructionSet.Comment(Name);
    public override IReadOnlyList<InstructionAttributeDescription> AttributeDescriptions =>
        TierOneNormalizationInstructionSet.Attributes(Name);
    public override IReadOnlyList<InstructionSignature> Signatures
    {
        get;
    } =
        [TierOneNormalizationInstructionSet.Signatures[name]];

    protected override void Validate(InstructionParameter[] parameters) =>
        TierOneNormalizationInstructionSet.Validate(Name, parameters);
}

internal sealed class GpuBatchRmsNormInstruction() : GpuBatchNormalizationInstruction(
    "transformer.batch-rms-norm",
    [
        new("input", ElementType.Float32, ResourceAccess.Read),
        new("weight", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ],
    ["epsilon"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var weight = tensors["weight"];
        var control = tensors["control"];
        var output = tensors["output"];
        var (batch, width) = BatchMatrix(input, "input");
        RequireSameShape(input, "input", output, "output");
        if (Vector(weight, "weight") != width)
        {
            throw Error("'weight' must have shape [H].");
        }

        RequireControl(control);
        var epsilon = FloatLiteral(PositiveFloat(parameters, "epsilon"));
        return new($$"""
            {
                uint row={{LinearGroup}};
                int validCount={{ValidCount(control)}};
                if(row<{{batch}}u && int(row)<validCount) {
                    precise float sum=0.0f;
                    for(uint column=gpu.groupIndex;column<{{width}}u;column+=64u) {
                        precise float value={{Load(input, $"row*{width}u+column")}};
                        sum+=value*value;
                    }
                    transformerBatchReduction[gpu.groupIndex]=sum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(gpu.groupIndex<step)
                            transformerBatchReduction[gpu.groupIndex]+=
                                transformerBatchReduction[gpu.groupIndex+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float scale=rsqrt(
                        transformerBatchReduction[0]/{{width}}.0f+{{epsilon}}f);
                    for(uint column=gpu.groupIndex;column<{{width}}u;column+=64u)
                        {{Store(output, $"row*{width}u+column",
                            $"{Load(input, $"row*{width}u+column")}*scale*{Load(weight, "column")}")}}
                }
            }
            """,
            [GpuInstructionHelpers.Load32, BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }

}

internal sealed class GpuBatchResidualRmsNormInstruction() : GpuBatchNormalizationInstruction(
    "transformer.batch-residual-rms-norm",
    [
        new("hidden", ElementType.Float32, ResourceAccess.Read),
        new("residual", ElementType.Float32, ResourceAccess.Read),
        new("weight", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("updated", ElementType.Float32, ResourceAccess.Write),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ],
    ["epsilon"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var hidden = tensors["hidden"];
        var residual = tensors["residual"];
        var weight = tensors["weight"];
        var control = tensors["control"];
        var updated = tensors["updated"];
        var output = tensors["output"];
        var (batch, width) = BatchMatrix(hidden, "hidden");
        RequireSameShape(hidden, "hidden", residual, "residual");
        RequireSameShape(hidden, "hidden", updated, "updated");
        RequireSameShape(hidden, "hidden", output, "output");
        if (Vector(weight, "weight") != width)
        {
            throw Error("'weight' must have shape [H].");
        }

        RequireControl(control);
        var epsilon = FloatLiteral(PositiveFloat(parameters, "epsilon"));
        return new($$"""
            {
                uint row={{LinearGroup}};
                int validCount={{ValidCount(control)}};
                if(row<{{batch}}u && int(row)<validCount) {
                    precise float sum=0.0f;
                    for(uint column=gpu.groupIndex;column<{{width}}u;column+=64u) {
                        uint element=row*{{width}}u+column;
                        precise float value={{Load(hidden, "element")}}+
                            {{Load(residual, "element")}};
                        {{Store(updated, "element", "value")}}
                        sum+=value*value;
                    }
                    transformerBatchReduction[gpu.groupIndex]=sum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(gpu.groupIndex<step)
                            transformerBatchReduction[gpu.groupIndex]+=
                                transformerBatchReduction[gpu.groupIndex+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float scale=rsqrt(
                        transformerBatchReduction[0]/{{width}}.0f+{{epsilon}}f);
                    for(uint column=gpu.groupIndex;column<{{width}}u;column+=64u) {
                        uint element=row*{{width}}u+column;
                        {{Store(output, "element",
                            $"{Load(updated, "element")}*scale*{Load(weight, "column")}")}}
                    }
                }
            }
            """,
            [GpuInstructionHelpers.Load32, BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuBatchSwiGluInstruction() : GpuBatchNormalizationInstruction(
    "transformer.batch-swiglu",
    [
        new("gate_up", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var gateUp = tensors["gate_up"];
        var control = tensors["control"];
        var output = tensors["output"];
        var (batch, combinedWidth) = BatchMatrix(gateUp, "gate_up");
        var (outputBatch, width) = BatchMatrix(output, "output");
        if (combinedWidth != checked(width * 2) || outputBatch != batch)
        {
            throw Error("Expected gate_up [B,2I] and output [B,I].");
        }

        RequireControl(control);
        var count = checked((uint)((long)batch * width));
        return new($$"""
            {
                uint element=i;
                if(element<{{count}}u) {
                    uint row=element/{{width}}u;
                    int validCount={{ValidCount(control)}};
                    if(int(row)<validCount) {
                        uint column=element-row*{{width}}u;
                        uint source=row*{{combinedWidth}}u+column;
                        precise float gate={{Load(gateUp, "source")}};
                        precise float up={{Load(gateUp, $"source+{width}u")}};
                        {{Store(output, "element", "gate/(1.0f+exp(-gate))*up")}}
                    }
                }
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}
