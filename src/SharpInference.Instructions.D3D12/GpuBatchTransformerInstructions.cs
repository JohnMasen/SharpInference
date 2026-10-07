using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Instructions.D3D12;

internal abstract class GpuBatchTransformerInstruction(
    string name,
    IReadOnlyList<InstructionPort> ports,
    IReadOnlyList<string>? attributes = null)
    : GpuTransformerInstruction(name, ports, attributes)
{
    protected (int Batch, int Width) BatchMatrix(
        InstructionTensorParameter tensor,
        string name)
    {
        RequireRank(tensor, name, 2);
        var batch = tensor.Tensor.Dimensions[0];
        var width = tensor.Tensor.Dimensions[1];
        if (batch <= 0 || width <= 0)
            throw Error($"'{name}' dimensions must be positive.");
        return (batch, width);
    }

    protected int Vector(InstructionTensorParameter tensor, string name)
    {
        RequireRank(tensor, name, 1);
        var width = tensor.Tensor.Dimensions[0];
        if (width <= 0)
            throw Error($"'{name}' dimensions must be positive.");
        return width;
    }

    protected (int Context, int Heads, int HeadSize) Cache(
        InstructionTensorParameter tensor,
        string name)
    {
        RequireRank(tensor, name, 3);
        var dimensions = tensor.Tensor.Dimensions;
        if (dimensions.Any(dimension => dimension <= 0))
            throw Error($"'{name}' dimensions must be positive.");
        return (dimensions[0], dimensions[1], dimensions[2]);
    }

    protected void RequireControl(InstructionTensorParameter control)
    {
        if (control.Tensor.Dimensions.Count != 1 ||
            control.Tensor.Dimensions[0] != 2)
            throw Error("'control' must have shape [2].");
    }

    protected void RequireSameShape(
        InstructionTensorParameter left,
        string leftName,
        InstructionTensorParameter right,
        string rightName)
    {
        if (!left.Tensor.Dimensions.SequenceEqual(right.Tensor.Dimensions))
            throw Error($"'{leftName}' and '{rightName}' must have matching shapes.");
    }

    protected int PositiveInteger(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>()
            .Single(parameter => parameter.Name == name).Value;
        if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value) ||
            value <= 0)
            throw Error($"Attribute '{name}' must be a positive integer.");
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
            throw Error($"'{name}' must have rank {rank}.");
    }
}

internal sealed class GpuBatchRmsNormInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-rms-norm",
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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
            throw Error("'weight' must have shape [H].");
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

    internal static InstructionHelper BatchReduction { get; } =
        new("transformer.batch.reduction64",
            "groupshared float transformerBatchReduction[64];");
}

internal sealed class GpuBatchResidualRmsNormInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-residual-rms-norm",
    [
        new("hidden", GraphElementType.Float32, GraphResourceAccess.Read),
        new("residual", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("updated", GraphElementType.Float32, GraphResourceAccess.Write),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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
            throw Error("'weight' must have shape [H].");
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
            [GpuInstructionHelpers.Load32, GpuBatchRmsNormInstruction.BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuBatchSharedMatVecInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-shared-matvec",
    [
        new("weight", GraphElementType.Float16, GraphResourceAccess.Read),
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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
            throw Error("Expected weight [O,I], input [B,I], and output [B,O].");
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
            [Load16, GpuInstructionHelpers.Load32, GpuBatchRmsNormInstruction.BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }

    private static InstructionHelper Load16 { get; } = new(
        "transformer.batch.load16",
        "float load16(RWByteAddressBuffer b,uint o,uint i) { uint a=o+i*2u; return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u); }");
}

internal sealed class GpuBatchSwiGluInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-swiglu",
    [
        new("gate_up", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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
            throw Error("Expected gate_up [B,2I] and output [B,I].");
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

internal sealed class GpuBatchRopeKeyValueWriteInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-rope-kv-write",
    [
        new("qkv", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("frequencies", GraphElementType.Float32, GraphResourceAccess.Read),
        new("query", GraphElementType.Float32, GraphResourceAccess.Write),
        new("key_cache", GraphElementType.Float32, GraphResourceAccess.Write),
        new("value_cache", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["head_size", "rotary_size", "rope_scale"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var qkv = tensors["qkv"];
        var control = tensors["control"];
        var frequencies = tensors["frequencies"];
        var query = tensors["query"];
        var keyCache = tensors["key_cache"];
        var valueCache = tensors["value_cache"];
        var (batch, qkvWidth) = BatchMatrix(qkv, "qkv");
        var (queryBatch, querySize) = BatchMatrix(query, "query");
        var (context, keyValueHeads, cacheHeadSize) = Cache(keyCache, "key_cache");
        var valueShape = Cache(valueCache, "value_cache");
        var headSize = PositiveInteger(parameters, "head_size");
        var rotarySize = PositiveInteger(parameters, "rotary_size");
        var ropeScale = FloatLiteral(PositiveFloat(parameters, "rope_scale"));
        var keyValueSize = checked(keyValueHeads * headSize);
        if (queryBatch != batch ||
            valueShape != (context, keyValueHeads, cacheHeadSize) ||
            cacheHeadSize != headSize ||
            querySize % headSize != 0 ||
            rotarySize > headSize ||
            rotarySize % 2 != 0 ||
            qkvWidth != checked(querySize + 2 * keyValueSize))
            throw Error("Invalid qkv, query, cache, head, or rotary dimensions.");
        if (Vector(frequencies, "frequencies") != rotarySize / 2)
            throw Error("'frequencies' must have shape [rotary_size/2].");
        RequireControl(control);
        var total = checked((uint)((long)batch * qkvWidth));
        return new($$"""
            {
                uint element=i;
                if(element<{{total}}u) {
                    uint row=element/{{qkvWidth}}u;
                    int validCount={{ValidCount(control)}};
                    int position={{StartPosition(control)}}+int(row);
                    if(int(row)<validCount && position>=0 && position<{{context}}) {
                        uint local=element-row*{{qkvWidth}}u;
                        if(local<{{querySize}}u) {
                            uint d=local%{{headSize}}u;
                            precise float value={{Load(qkv, "element")}};
                            if(d<{{rotarySize}}u) {
                                uint half={{rotarySize / 2}}u;
                                uint rotaryIndex=d<half ? d : d-half;
                                uint pair=d<half ? element+half : element-half;
                                precise float angle=float(position)*
                                    {{Load(frequencies, "rotaryIndex")}};
                                precise float c=cos(angle)*{{ropeScale}}f;
                                precise float s=sin(angle)*{{ropeScale}}f;
                                precise float other={{Load(qkv, "pair")}};
                                value=d<half ? value*c-other*s : value*c+other*s;
                            }
                            {{Store(query, $"row*{querySize}u+local", "value")}}
                        } else if(local<{{querySize + keyValueSize}}u) {
                            uint cacheLocal=local-{{querySize}}u;
                            uint d=cacheLocal%{{headSize}}u;
                            precise float value={{Load(qkv, "element")}};
                            if(d<{{rotarySize}}u) {
                                uint half={{rotarySize / 2}}u;
                                uint rotaryIndex=d<half ? d : d-half;
                                uint pair=d<half ? element+half : element-half;
                                precise float angle=float(position)*
                                    {{Load(frequencies, "rotaryIndex")}};
                                precise float c=cos(angle)*{{ropeScale}}f;
                                precise float s=sin(angle)*{{ropeScale}}f;
                                precise float other={{Load(qkv, "pair")}};
                                value=d<half ? value*c-other*s : value*c+other*s;
                            }
                            {{Store(keyCache,
                                $"uint(position)*{keyValueSize}u+cacheLocal", "value")}}
                        } else {
                            uint cacheLocal=local-{{querySize + keyValueSize}}u;
                            {{Store(valueCache,
                                $"uint(position)*{keyValueSize}u+cacheLocal",
                                Load(qkv, "element"))}}
                        }
                    }
                }
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}

internal sealed class GpuBatchGroupedQueryScoresInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-gqa-scores",
    [
        new("query", GraphElementType.Float32, GraphResourceAccess.Read),
        new("key_cache", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("scores", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var query = tensors["query"];
        var keyCache = tensors["key_cache"];
        var control = tensors["control"];
        var scores = tensors["scores"];
        var (batch, querySize) = BatchMatrix(query, "query");
        var (context, keyValueHeads, headSize) = Cache(keyCache, "key_cache");
        if (scores.Tensor.Dimensions.Count != 3 ||
            scores.Tensor.Dimensions[0] != batch ||
            scores.Tensor.Dimensions[2] != context)
            throw Error("'scores' must have shape [B,qheads,context].");
        var queryHeads = scores.Tensor.Dimensions[1];
        if (queryHeads <= 0 ||
            queryHeads % keyValueHeads != 0 ||
            querySize != checked(queryHeads * headSize))
            throw Error("Incompatible query, cache, and GQA score shapes.");
        RequireControl(control);
        var scale = FloatLiteral(1f / MathF.Sqrt(headSize));
        var count = checked((uint)((long)batch * queryHeads * context));
        return new($$"""
            {
                uint element=i;
                if(element<{{count}}u) {
                    uint row=element/({{queryHeads}}u*{{context}}u);
                    int validCount={{ValidCount(control)}};
                    if(int(row)<validCount) {
                        uint local=element-row*{{queryHeads}}u*{{context}}u;
                        uint head=local/{{context}}u;
                        uint token=local-head*{{context}}u;
                        int active={{StartPosition(control)}}+int(row);
                        precise float sum=-3.402823466e+38f;
                        if(active>=0 && token<=uint(active)) {
                            uint kvHead=head/{{queryHeads / keyValueHeads}}u;
                            sum=0.0f;
                            for(uint d=0u;d<{{headSize}}u;d++)
                                sum+={{Load(query,
                                    $"(row*{queryHeads}u+head)*{headSize}u+d")}}*
                                    {{Load(keyCache,
                                        $"(token*{keyValueHeads}u+kvHead)*{headSize}u+d")}};
                            sum*={{scale}}f;
                        }
                        {{Store(scores, "element", "sum")}}
                    }
                }
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}

internal sealed class GpuBatchCausalSoftmaxInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-causal-softmax",
    [
        new("scores", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("probabilities", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var scores = tensors["scores"];
        var control = tensors["control"];
        var probabilities = tensors["probabilities"];
        if (scores.Tensor.Dimensions.Count != 3)
            throw Error("'scores' must have shape [B,heads,context].");
        RequireSameShape(scores, "scores", probabilities, "probabilities");
        var batch = scores.Tensor.Dimensions[0];
        var heads = scores.Tensor.Dimensions[1];
        var context = scores.Tensor.Dimensions[2];
        if (batch <= 0 || heads <= 0 || context <= 0)
            throw Error("'scores' dimensions must be positive.");
        RequireControl(control);
        return new($$"""
            {
                uint work={{LinearGroup}};
                uint row=work/{{heads}}u;
                uint head=work-row*{{heads}}u;
                int validCount={{ValidCount(control)}};
                if(row<{{batch}}u && int(row)<validCount) {
                    int active={{StartPosition(control)}}+int(row);
                    uint base=(row*{{heads}}u+head)*{{context}}u;
                    precise float maximum=-3.402823466e+38f;
                    if(active>=0) {
                        for(uint token=gpu.groupIndex;
                            token<=uint(active) && token<{{context}}u;token+=64u)
                            maximum=max(maximum,{{Load(scores, "base+token")}});
                    }
                    transformerBatchReduction[gpu.groupIndex]=maximum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(gpu.groupIndex<step)
                            transformerBatchReduction[gpu.groupIndex]=max(
                                transformerBatchReduction[gpu.groupIndex],
                                transformerBatchReduction[gpu.groupIndex+step]);
                        GroupMemoryBarrierWithGroupSync();
                    }
                    maximum=transformerBatchReduction[0];
                    GroupMemoryBarrierWithGroupSync();
                    precise float sum=0.0f;
                    if(active>=0) {
                        for(uint token=gpu.groupIndex;
                            token<=uint(active) && token<{{context}}u;token+=64u)
                            sum+=exp({{Load(scores, "base+token")}}-maximum);
                    }
                    transformerBatchReduction[gpu.groupIndex]=sum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(gpu.groupIndex<step)
                            transformerBatchReduction[gpu.groupIndex]+=
                                transformerBatchReduction[gpu.groupIndex+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float denominator=transformerBatchReduction[0];
                    for(uint token=gpu.groupIndex;token<{{context}}u;token+=64u) {
                        precise float probability=active>=0 && token<=uint(active)
                            ? exp({{Load(scores, "base+token")}}-maximum)/denominator
                            : 0.0f;
                        {{Store(probabilities, "base+token", "probability")}}
                    }
                }
            }
            """,
            [GpuInstructionHelpers.Load32, GpuBatchRmsNormInstruction.BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuBatchGroupedQueryValuesInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-gqa-values",
    [
        new("probabilities", GraphElementType.Float32, GraphResourceAccess.Read),
        new("value_cache", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var probabilities = tensors["probabilities"];
        var valueCache = tensors["value_cache"];
        var control = tensors["control"];
        var output = tensors["output"];
        if (probabilities.Tensor.Dimensions.Count != 3)
            throw Error("'probabilities' must have shape [B,qheads,context].");
        var batch = probabilities.Tensor.Dimensions[0];
        var queryHeads = probabilities.Tensor.Dimensions[1];
        var probabilityContext = probabilities.Tensor.Dimensions[2];
        var (context, keyValueHeads, headSize) = Cache(valueCache, "value_cache");
        var (outputBatch, outputWidth) = BatchMatrix(output, "output");
        if (batch <= 0 || queryHeads <= 0 ||
            probabilityContext != context ||
            queryHeads % keyValueHeads != 0 ||
            outputBatch != batch ||
            outputWidth != checked(queryHeads * headSize))
            throw Error("Incompatible probability, value cache, and output shapes.");
        RequireControl(control);
        var count = checked((uint)((long)batch * outputWidth));
        return new($$"""
            {
                uint element=i;
                if(element<{{count}}u) {
                    uint row=element/{{outputWidth}}u;
                    int validCount={{ValidCount(control)}};
                    if(int(row)<validCount) {
                        uint local=element-row*{{outputWidth}}u;
                        uint head=local/{{headSize}}u;
                        uint d=local-head*{{headSize}}u;
                        uint kvHead=head/{{queryHeads / keyValueHeads}}u;
                        int active={{StartPosition(control)}}+int(row);
                        precise float sum=0.0f;
                        if(active>=0) {
                            for(uint token=0u;
                                token<=uint(active) && token<{{context}}u;token++)
                                sum+={{Load(probabilities,
                                    $"(row*{queryHeads}u+head)*{context}u+token")}}*
                                    {{Load(valueCache,
                                        $"(token*{keyValueHeads}u+kvHead)*{headSize}u+d")}};
                        }
                        {{Store(output, "element", "sum")}}
                    }
                }
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}

internal sealed class GpuBatchSelectLastInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-select-last",
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var control = tensors["control"];
        var output = tensors["output"];
        var (batch, width) = BatchMatrix(input, "input");
        if (Vector(output, "output") != width)
            throw Error("Expected input [B,H] and output [H].");
        RequireControl(control);
        return new($$"""
            {
                uint column=i;
                int validCount={{ValidCount(control)}};
                if(column<{{width}}u && validCount>0 && validCount<={{batch}})
                    {{Store(output, "column",
                        Load(input, $"uint(validCount-1)*{width}u+column"))}}
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}

internal sealed class GpuBatchGatherRowsInstruction() : GpuBatchTransformerInstruction(
    "transformer.batch-gather-rows",
    [
        new("table", GraphElementType.Float32, GraphResourceAccess.Read),
        new("control", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var table = tensors["table"];
        var control = tensors["control"];
        var output = tensors["output"];
        var (rowCount, width) = BatchMatrix(table, "table");
        var (batch, outputWidth) = BatchMatrix(output, "output");
        if (outputWidth != width)
            throw Error("Expected table [N,H] and output [B,H].");
        RequireControl(control);
        var count = checked((uint)((long)batch * width));
        return new($$"""
            {
                uint element=i;
                if(element<{{count}}u) {
                    uint row=element/{{width}}u;
                    int validCount={{ValidCount(control)}};
                    int sourceRow={{StartPosition(control)}}+int(row);
                    if(int(row)<validCount && sourceRow>=0 && sourceRow<{{rowCount}}) {
                        uint column=element-row*{{width}}u;
                        {{Store(output, "element",
                            Load(table, $"uint(sourceRow)*{width}u+column"))}}
                    }
                }
            }
            """, [GpuInstructionHelpers.Load32]);
    }
}
