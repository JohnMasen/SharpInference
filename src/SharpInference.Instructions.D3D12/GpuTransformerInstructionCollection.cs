using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Instructions.D3D12;

public sealed class GpuTransformerInstructionCollection : IInstructionCollectionProvider
{
    private readonly IReadOnlyDictionary<string, Instruction> instructions =
        new Instruction[]
        {
            new GpuRmsNormInstruction(),
            new GpuResidualRmsNormInstruction(),
            new GpuSwiGluInstruction(),
            new GpuRopeKeyValueWriteInstruction(),
            new GpuGroupedQueryScoresInstruction(),
            new GpuCausalSoftmaxInstruction(),
            new GpuGroupedQueryValuesInstruction(),
            new GpuGroupedQueryAttentionInstruction(),
            new GpuArgMaxInstruction(),
            new GpuScaledAddInstruction(),
            new GpuEmbeddingGatherBatchInstruction(),
            new GpuCopyEmbeddingRowsInstruction(),
            new GpuBatchRmsNormInstruction(),
            new GpuBatchResidualRmsNormInstruction(),
            new GpuBatchSharedMatVecInstruction(),
            new GpuBatchSwiGluInstruction(),
            new GpuBatchRopeKeyValueWriteInstruction(),
            new GpuBatchGroupedQueryScoresInstruction(),
            new GpuBatchCausalSoftmaxInstruction(),
            new GpuBatchGroupedQueryValuesInstruction(),
            new GpuBatchSelectLastInstruction(),
            new GpuBatchGatherRowsInstruction(),
        }.ToDictionary(instruction => instruction.Name, StringComparer.Ordinal);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(InstructionCollectionIds.TransformerFloat32,
            "FP32 transformer operations", 1, InstructionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == InstructionCollectionIds.TransformerFloat32 &&
        instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction]
            : [];
}

internal sealed class GpuGroupedQueryAttentionInstruction() : GpuTransformerInstruction(
    "transformer.grouped-query-attention",
    [
        new("query", GraphElementType.Float32, GraphResourceAccess.Read),
        new("key_cache", GraphElementType.Float32, GraphResourceAccess.Read),
        new("value_cache", GraphElementType.Float32, GraphResourceAccess.Read),
        new("position", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var query = tensors["query"];
        var keyCache = tensors["key_cache"];
        var valueCache = tensors["value_cache"];
        var position = tensors["position"];
        var output = tensors["output"];
        if (keyCache.Tensor.Dimensions.Count != 3 ||
            !keyCache.Tensor.Dimensions.SequenceEqual(valueCache.Tensor.Dimensions))
            throw new InstructionAdaptationException(
                CollectionId, Name, "Expected matching rank-three key and value caches.");
        var context = keyCache.Tensor.Dimensions[0];
        var keyValueHeads = keyCache.Tensor.Dimensions[1];
        var headSize = keyCache.Tensor.Dimensions[2];
        var queryValues = query.Tensor.Dimensions.Aggregate(
            1, (product, dimension) => checked(product * dimension));
        var outputValues = output.Tensor.Dimensions.Aggregate(
            1, (product, dimension) => checked(product * dimension));
        if (queryValues != outputValues || queryValues % headSize != 0)
            throw new InstructionAdaptationException(
                CollectionId, Name, "Query and output must contain complete matching heads.");
        var queryHeads = queryValues / headSize;
        if (queryHeads % keyValueHeads != 0)
            throw new InstructionAdaptationException(
                CollectionId, Name, "Query heads must be divisible by key/value heads.");
        var scale = (1f / MathF.Sqrt(headSize)).ToString(
            "R", CultureInfo.InvariantCulture);
        return new($$"""
            {
                uint head=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint lane=gpu.groupIndex;
                if(head<{{queryHeads}}u) {
                    uint kvHead=head/{{queryHeads / keyValueHeads}}u;
                    uint active=min(
                        (uint)max(0,asint({{position.Expression}}.Load(
                            {{position.OffsetExpression}}))),
                        {{context - 1}}u);
                    precise float localMaximum=-3.402823466e+38f;
                    for(uint token=lane;token<=active;token+=64u) {
                        precise float score=0.0f;
                        for(uint channel=0u;channel<{{headSize}}u;channel++)
                            score+={{Load(query, $"head*{headSize}u+channel")}}*
                                {{Load(keyCache,
                                    $"(token*{keyValueHeads}u+kvHead)*{headSize}u+channel")}};
                        score*={{scale}}f;
                        transformerAttentionScores[token]=score;
                        localMaximum=max(localMaximum,score);
                    }
                    transformerReduction[lane]=localMaximum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            transformerReduction[lane]=max(
                                transformerReduction[lane],
                                transformerReduction[lane+step]);
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float maximum=transformerReduction[0];
                    GroupMemoryBarrierWithGroupSync();
                    precise float localDenominator=0.0f;
                    for(uint token=lane;token<=active;token+=64u) {
                        precise float probability=exp(
                            transformerAttentionScores[token]-maximum);
                        transformerAttentionScores[token]=probability;
                        localDenominator+=probability;
                    }
                    transformerReduction[lane]=localDenominator;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            transformerReduction[lane]+=transformerReduction[lane+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float denominator=transformerReduction[0];
                    for(uint channel=lane;channel<{{headSize}}u;channel+=64u) {
                        precise float value=0.0f;
                        for(uint token=0u;token<=active;token++)
                            value+=transformerAttentionScores[token]*
                                {{Load(valueCache,
                                    $"(token*{keyValueHeads}u+kvHead)*{headSize}u+channel")}};
                        {{Store(output, $"head*{headSize}u+channel", "value/denominator")}}
                    }
                }
            }
            """,
        [
            GpuInstructionHelpers.Load32,
            GpuRmsNormInstruction.ReductionHelper,
            new("transformer.attention-scores",
                $"groupshared float transformerAttentionScores[{context}];")
        ], InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuScaledAddInstruction() : GpuTransformerInstruction(
    "transformer.scaled-add",
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("update", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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
            scale += ".0";
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

internal sealed class GpuEmbeddingGatherBatchInstruction() : GpuTransformerInstruction(
    "transformer.embedding-gather-batch",
    [
        new("table", GraphElementType.Float16, GraphResourceAccess.Read),
        new("indices", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
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
    "transformer.copy-embedding-rows",
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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
            throw new InstructionAdaptationException(
                InstructionCollectionIds.TransformerFloat32, name,
                $"Attribute '{name}' must be positive.");
        return value;
    }

    private static int NonNegativeInteger(InstructionParameter[] parameters, string name)
    {
        var value = Integer(parameters, name);
        if (value < 0)
            throw new InstructionAdaptationException(
                InstructionCollectionIds.TransformerFloat32, name,
                $"Attribute '{name}' must be non-negative.");
        return value;
    }

    private static int Integer(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>()
            .Single(parameter => parameter.Name == name).Value;
        return int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new InstructionAdaptationException(
                InstructionCollectionIds.TransformerFloat32, name,
                $"Attribute '{name}' must be an integer.");
    }
}

internal abstract class GpuTransformerInstruction(
    string name,
    IReadOnlyList<InstructionPort> ports,
    IReadOnlyList<string>? attributes = null) : Instruction
{
    private static readonly KernelPrecisionProfile Precision =
        new(GraphElementType.Float32, GraphElementType.Float32);

    public override Guid CollectionId => InstructionCollectionIds.TransformerFloat32;
    public override string Name => name;
    public override InstructionTarget Target => InstructionTarget.Direct3D12;
    public override IReadOnlyList<InstructionSignature> Signatures { get; } =
        [new(ports, attributes ?? [], Precision)];

    protected static IReadOnlyDictionary<string, InstructionTensorParameter> Tensors(
        InstructionParameter[] parameters) =>
        parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);

    protected static float PositiveFloat(InstructionParameter[] parameters, string name)
    {
        var text = parameters.OfType<InstructionAttributeParameter>()
            .Single(parameter => parameter.Name == name).Value;
        var value = float.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (!float.IsFinite(value) || value <= 0)
            throw new InstructionAdaptationException(
                InstructionCollectionIds.TransformerFloat32, name,
                $"Attribute '{name}' must be finite and positive.");
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

internal sealed class GpuRmsNormInstruction() : GpuTransformerInstruction(
    "transformer.rms-norm",
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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

    internal static InstructionHelper ReductionHelper { get; } =
        new("transformer.reduction64", "groupshared float transformerReduction[64];");
}

internal sealed class GpuResidualRmsNormInstruction() : GpuTransformerInstruction(
    "transformer.residual-rms-norm",
    [
        new("hidden", GraphElementType.Float32, GraphResourceAccess.Read),
        new("residual", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float32, GraphResourceAccess.Read),
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
            [GpuInstructionHelpers.Load32, GpuRmsNormInstruction.ReductionHelper],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuSwiGluInstruction() : GpuTransformerInstruction(
    "transformer.swiglu",
    [
        new("gate_up", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
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

internal sealed class GpuRopeKeyValueWriteInstruction() : GpuTransformerInstruction(
                "transformer.rope-kv-write",
                [
                    new("qkv", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("position", GraphElementType.Int32, GraphResourceAccess.Read),
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
                    var position = tensors["position"];
                    var frequencies = tensors["frequencies"];
                    var query = tensors["query"];
                    var keyCache = tensors["key_cache"];
                    var valueCache = tensors["value_cache"];
                    var headSize = PositiveInteger(parameters, "head_size");
                    var rotarySize = PositiveInteger(parameters, "rotary_size");
                    var ropeScale = PositiveFloat(parameters, "rope_scale").ToString("R", CultureInfo.InvariantCulture);
                    var querySize = checked((uint)query.Tensor.Dimensions.Aggregate(
                        1UL, (product, dimension) => checked(product * (ulong)dimension)));
                    if (keyCache.Tensor.Dimensions.Count != 3 ||
                        valueCache.Tensor.Dimensions.Count != 3 ||
                        !keyCache.Tensor.Dimensions.SequenceEqual(valueCache.Tensor.Dimensions) ||
                        keyCache.Tensor.Dimensions[2] != headSize ||
                        rotarySize > headSize || rotarySize % 2 != 0 ||
                        querySize % headSize != 0)
                        throw new InstructionAdaptationException(
                            CollectionId, Name, "Invalid query, cache, head, or rotary dimensions.");
        if (frequencies.Tensor.Dimensions.Aggregate(
                            1UL, (product, dimension) => checked(product * (ulong)dimension)) !=
            (ulong)(rotarySize / 2))
            throw new InstructionAdaptationException(CollectionId, Name, "RoPE frequency shape is invalid.");
                    var context = keyCache.Tensor.Dimensions[0];
                    var keyValueSize = checked((uint)(
                        keyCache.Tensor.Dimensions[1] * keyCache.Tensor.Dimensions[2]));
                    if (qkv.Tensor.Dimensions.Aggregate(
                            1UL, (product, dimension) => checked(product * (ulong)dimension)) !=
                        querySize + 2UL * keyValueSize)
                        throw new InstructionAdaptationException(CollectionId, Name, "QKV shape does not match query and cache shapes.");
                    var total = querySize + 2u * keyValueSize;
                    return new($$"""
                        {
                            int token=asint({{position.Expression}}.Load({{position.OffsetExpression}}));
                            if(token>=0 && token<{{context}} && i<{{total}}u) {
                                if(i<{{querySize}}u) {
                                    uint d=i%{{headSize}}u;
                                    precise float value={{Load(qkv, "i")}};
                                    if(d<{{rotarySize}}u) {
                                        uint half={{rotarySize / 2}}u;
                                        uint rotaryIndex=d<half ? d : d-half;
                                        uint pair=d<half ? i+half : i-half;
                                        precise float angle=float(token)*{{Load(frequencies, "rotaryIndex")}};
                                        precise float c=cos(angle)*{{ropeScale}}f;
                                        precise float s=sin(angle)*{{ropeScale}}f;
                                        precise float other={{Load(qkv, "pair")}};
                                        value=d<half ? value*c-other*s : value*c+other*s;
                                    }
                                    {{Store(query, "i", "value")}}
                                } else if(i<{{querySize + keyValueSize}}u) {
                                    uint local=i-{{querySize}}u;
                                    uint d=local%{{headSize}}u;
                                    precise float value={{Load(qkv, "i")}};
                                    if(d<{{rotarySize}}u) {
                                        uint half={{rotarySize / 2}}u;
                                        uint rotaryIndex=d<half ? d : d-half;
                                        uint pair=d<half ? i+half : i-half;
                                        precise float angle=float(token)*{{Load(frequencies, "rotaryIndex")}};
                                        precise float c=cos(angle)*{{ropeScale}}f;
                                        precise float s=sin(angle)*{{ropeScale}}f;
                                        precise float other={{Load(qkv, "pair")}};
                                        value=d<half ? value*c-other*s : value*c+other*s;
                                    }
                                    {{Store(keyCache, $"uint(token)*{keyValueSize}u+local", "value")}}
                                } else {
                                    uint local=i-{{querySize + keyValueSize}}u;
                                    {{Store(valueCache, $"uint(token)*{keyValueSize}u+local", Load(qkv, "i"))}}
                                }
                            }
                        }
                        """, [GpuInstructionHelpers.Load32]);
                }

                private static int PositiveInteger(InstructionParameter[] parameters, string name)
                {
                    var text = parameters.OfType<InstructionAttributeParameter>()
                        .Single(parameter => parameter.Name == name).Value;
                    if (!int.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out var value) || value <= 0)
                        throw new InstructionAdaptationException(
                            InstructionCollectionIds.TransformerFloat32, name,
                            $"Attribute '{name}' must be a positive integer.");
                    return value;
                }
            }

            internal sealed class GpuGroupedQueryScoresInstruction() : GpuTransformerInstruction(
                "transformer.gqa-scores",
                [
                    new("query", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("key_cache", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("position", GraphElementType.Int32, GraphResourceAccess.Read),
                    new("scores", GraphElementType.Float32, GraphResourceAccess.Write),
                ])
            {
                protected override InstructionRecording Generate(InstructionParameter[] parameters)
                {
                    var tensors = Tensors(parameters);
                    var query = tensors["query"];
                    var keyCache = tensors["key_cache"];
                    var position = tensors["position"];
                    var scores = tensors["scores"];
                    if (keyCache.Tensor.Dimensions.Count != 3 || scores.Tensor.Dimensions.Count != 2)
                        throw new InstructionAdaptationException(CollectionId, Name, "Expected rank-three cache and rank-two scores.");
                    var context = scores.Tensor.Dimensions[1];
                    var queryHeads = scores.Tensor.Dimensions[0];
                    var keyValueHeads = keyCache.Tensor.Dimensions[1];
                    var headSize = keyCache.Tensor.Dimensions[2];
                    var scale = (1f / MathF.Sqrt(headSize)).ToString(
                        "R", CultureInfo.InvariantCulture);
                    if (queryHeads % keyValueHeads != 0 ||
                        keyCache.Tensor.Dimensions[0] != context ||
                        query.Tensor.Dimensions.Aggregate(1L, (value, dimension) => value * dimension) != queryHeads * headSize)
                        throw new InstructionAdaptationException(CollectionId, Name, "Incompatible GQA shapes.");
                    var count = checked((uint)(queryHeads * context));
                    return new($$"""
                        {
                            if(i<{{count}}u) {
                                uint head=i/{{context}}u;
                                uint token=i%{{context}}u;
                                int active=asint({{position.Expression}}.Load({{position.OffsetExpression}}));
                                precise float sum=-3.402823466e+38f;
                                if(token<=uint(active)) {
                                    uint kvHead=head/{{queryHeads / keyValueHeads}}u;
                                    sum=0.0f;
                                    for(uint d=0u;d<{{headSize}}u;d++)
                                        sum+={{Load(query, $"head*{headSize}u+d")}}*
                                            {{Load(keyCache, $"(token*{keyValueHeads}u+kvHead)*{headSize}u+d")}};
                                    sum*={{scale}}f;
                                }
                                {{Store(scores, "i", "sum")}}
                            }
                        }
                        """, [GpuInstructionHelpers.Load32]);
                }
            }

internal sealed class GpuCausalSoftmaxInstruction() : GpuTransformerInstruction(
                "transformer.causal-softmax",
                [
                    new("scores", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("position", GraphElementType.Int32, GraphResourceAccess.Read),
                    new("probabilities", GraphElementType.Float32, GraphResourceAccess.Write),
                ])
            {
                protected override InstructionRecording Generate(InstructionParameter[] parameters)
                {
                    var tensors = Tensors(parameters);
                    var scores = tensors["scores"];
                    var position = tensors["position"];
                    var probabilities = tensors["probabilities"];
                    if (scores.Tensor.Dimensions.Count != 2 ||
                        !scores.Tensor.Dimensions.SequenceEqual(probabilities.Tensor.Dimensions))
                        throw new InstructionAdaptationException(CollectionId, Name, "Softmax requires matching rank-two tensors.");
                    var heads = scores.Tensor.Dimensions[0];
                    var context = scores.Tensor.Dimensions[1];
                    return new($$"""
                        {
                            uint head=gpu.groupId.x;
                            int active=asint({{position.Expression}}.Load({{position.OffsetExpression}}));
                            precise float maximum=-3.402823466e+38f;
                            for(uint token=gpu.groupIndex;token<=uint(active) && token<{{context}}u;token+=64u)
                                maximum=max(maximum,{{Load(scores, $"head*{context}u+token")}});
                            transformerReduction[gpu.groupIndex]=maximum;
                            GroupMemoryBarrierWithGroupSync();
                            for(uint step=32u;step>0u;step>>=1u) {
                                if(gpu.groupIndex<step)
                                    transformerReduction[gpu.groupIndex]=max(
                                        transformerReduction[gpu.groupIndex],
                                        transformerReduction[gpu.groupIndex+step]);
                                GroupMemoryBarrierWithGroupSync();
                            }
                            maximum=transformerReduction[0];
                            GroupMemoryBarrierWithGroupSync();
                            precise float sum=0.0f;
                            for(uint token=gpu.groupIndex;token<=uint(active) && token<{{context}}u;token+=64u)
                                sum+=exp({{Load(scores, $"head*{context}u+token")}}-maximum);
                            transformerReduction[gpu.groupIndex]=sum;
                            GroupMemoryBarrierWithGroupSync();
                            for(uint step=32u;step>0u;step>>=1u) {
                                if(gpu.groupIndex<step)
                                    transformerReduction[gpu.groupIndex]+=transformerReduction[gpu.groupIndex+step];
                                GroupMemoryBarrierWithGroupSync();
                            }
                            precise float denominator=transformerReduction[0];
                            for(uint token=gpu.groupIndex;token<{{context}}u;token+=64u) {
                                precise float probability=token<=uint(active)
                                    ? exp({{Load(scores, $"head*{context}u+token")}}-maximum)/denominator
                                    : 0.0f;
                                {{Store(probabilities, $"head*{context}u+token", "probability")}}
                            }
                        }
                        """, [GpuInstructionHelpers.Load32, GpuRmsNormInstruction.ReductionHelper],
                        InstructionSynchronization.GroupMemoryBarrier);
                }
            }

            internal sealed class GpuGroupedQueryValuesInstruction() : GpuTransformerInstruction(
                "transformer.gqa-values",
                [
                    new("probabilities", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("value_cache", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("position", GraphElementType.Int32, GraphResourceAccess.Read),
                    new("output", GraphElementType.Float32, GraphResourceAccess.Write),
                ])
            {
                protected override InstructionRecording Generate(InstructionParameter[] parameters)
                {
                    var tensors = Tensors(parameters);
                    var probabilities = tensors["probabilities"];
                    var valueCache = tensors["value_cache"];
                    var position = tensors["position"];
                    var output = tensors["output"];
                    if (probabilities.Tensor.Dimensions.Count != 2 || valueCache.Tensor.Dimensions.Count != 3)
                        throw new InstructionAdaptationException(CollectionId, Name, "Invalid probability or value-cache rank.");
                    var queryHeads = probabilities.Tensor.Dimensions[0];
                    var context = probabilities.Tensor.Dimensions[1];
                    var keyValueHeads = valueCache.Tensor.Dimensions[1];
                    var headSize = valueCache.Tensor.Dimensions[2];
                    var count = checked((uint)(queryHeads * headSize));
                    return new($$"""
                        {
                            if(i<{{count}}u) {
                                uint head=i/{{headSize}}u;
                                uint d=i%{{headSize}}u;
                                uint kvHead=head/{{queryHeads / keyValueHeads}}u;
                                int active=asint({{position.Expression}}.Load({{position.OffsetExpression}}));
                                precise float sum=0.0f;
                                for(uint token=0u;token<=uint(active) && token<{{context}}u;token++)
                                    sum+={{Load(probabilities, $"head*{context}u+token")}}*
                                        {{Load(valueCache, $"(token*{keyValueHeads}u+kvHead)*{headSize}u+d")}};
                                {{Store(output, "i", "sum")}}
                            }
                        }
                        """, [GpuInstructionHelpers.Load32]);
                }
            }

            internal sealed class GpuArgMaxInstruction() : GpuTransformerInstruction(
                "transformer.argmax",
                [
                    new("input", GraphElementType.Float32, GraphResourceAccess.Read),
                    new("output", GraphElementType.Int32, GraphResourceAccess.Write),
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
                        throw new InstructionAdaptationException(CollectionId, Name, "Argmax output must be scalar.");
                    return new($$"""
                        {
                            precise float best=-3.402823466e+38f;
                            uint bestIndex=0u;
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
                                {{output.Expression}}.Store({{output.OffsetExpression}},transformerIndices[0]);
                        }
                        """, [GpuInstructionHelpers.Load32, GpuRmsNormInstruction.ReductionHelper,
                            new("transformer.indices64", "groupshared uint transformerIndices[64];")],
                        InstructionSynchronization.GroupMemoryBarrier);
    }
}
