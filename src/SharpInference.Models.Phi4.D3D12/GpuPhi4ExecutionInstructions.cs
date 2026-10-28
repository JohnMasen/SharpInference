using SharpInference.Graphs.Execution;
using SharpInference.Instructions;
using System.Globalization;
using SharpInference.Graphs;

namespace SharpInference.Instructions.D3D12;

internal abstract class GpuPhi4Instruction(
    string name, IReadOnlyList<InstructionPort> ports, IReadOnlyList<string>? attributes = null)
    : GpuTransformerInstruction(Phi4ExecutionInstructionCollectionIds.Float32, name, ports, attributes);

internal abstract class GpuPhi4BatchInstruction(
    string name, IReadOnlyList<InstructionPort> ports, IReadOnlyList<string>? attributes = null)
    : GpuBatchTransformerInstruction(Phi4ExecutionInstructionCollectionIds.Float32, name, ports, attributes);

internal sealed class GpuGroupedQueryAttentionInstruction() : GpuPhi4Instruction(
    "transformer.grouped-query-attention",
    [
        new("query", ElementType.Float32, ResourceAccess.Read),
        new("key_cache", ElementType.Float32, ResourceAccess.Read),
        new("value_cache", ElementType.Float32, ResourceAccess.Read),
        new("position", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw new InstructionAdaptationException(
                CollectionId, Name, "Expected matching rank-three key and value caches.");
        }

        var context = keyCache.Tensor.Dimensions[0];
        var keyValueHeads = keyCache.Tensor.Dimensions[1];
        var headSize = keyCache.Tensor.Dimensions[2];
        var queryValues = query.Tensor.Dimensions.Aggregate(
            1, (product, dimension) => checked(product * dimension));
        var outputValues = output.Tensor.Dimensions.Aggregate(
            1, (product, dimension) => checked(product * dimension));
        if (queryValues != outputValues || queryValues % headSize != 0)
        {
            throw new InstructionAdaptationException(
                CollectionId, Name, "Query and output must contain complete matching heads.");
        }

        var queryHeads = queryValues / headSize;
        if (queryHeads % keyValueHeads != 0)
        {
            throw new InstructionAdaptationException(
                CollectionId, Name, "Query heads must be divisible by key/value heads.");
        }

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
            ReductionHelper,
            new("transformer.attention-scores",
                $"groupshared float transformerAttentionScores[{context}];")
        ], InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuRopeKeyValueWriteInstruction() : GpuPhi4Instruction(
                "transformer.rope-kv-write",
                [
                    new("qkv", ElementType.Float32, ResourceAccess.Read),
                    new("position", ElementType.Int32, ResourceAccess.Read),
                    new("frequencies", ElementType.Float32, ResourceAccess.Read),
                    new("query", ElementType.Float32, ResourceAccess.Write),
                    new("key_cache", ElementType.Float32, ResourceAccess.Write),
                    new("value_cache", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw new InstructionAdaptationException(
                            CollectionId, Name, "Invalid query, cache, head, or rotary dimensions.");
        }

        if (frequencies.Tensor.Dimensions.Aggregate(
                            1UL, (product, dimension) => checked(product * (ulong)dimension)) !=
            (ulong)(rotarySize / 2))
        {
            throw new InstructionAdaptationException(CollectionId, Name, "RoPE frequency shape is invalid.");
        }

        var context = keyCache.Tensor.Dimensions[0];
        var keyValueSize = checked((uint)(
            keyCache.Tensor.Dimensions[1] * keyCache.Tensor.Dimensions[2]));
        if (qkv.Tensor.Dimensions.Aggregate(
                1UL, (product, dimension) => checked(product * (ulong)dimension)) !=
            querySize + 2UL * keyValueSize)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "QKV shape does not match query and cache shapes.");
        }

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
        {
            throw new InstructionAdaptationException(
                            Phi4ExecutionInstructionCollectionIds.Float32, name,
                            $"Attribute '{name}' must be a positive integer.");
        }

        return value;
    }
}

internal sealed class GpuGroupedQueryScoresInstruction() : GpuPhi4Instruction(
    "transformer.gqa-scores",
    [
        new("query", ElementType.Float32, ResourceAccess.Read),
                    new("key_cache", ElementType.Float32, ResourceAccess.Read),
                    new("position", ElementType.Int32, ResourceAccess.Read),
                    new("scores", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Expected rank-three cache and rank-two scores.");
        }

        var context = scores.Tensor.Dimensions[1];
        var queryHeads = scores.Tensor.Dimensions[0];
        var keyValueHeads = keyCache.Tensor.Dimensions[1];
        var headSize = keyCache.Tensor.Dimensions[2];
        var scale = (1f / MathF.Sqrt(headSize)).ToString(
            "R", CultureInfo.InvariantCulture);
        if (queryHeads % keyValueHeads != 0 ||
            keyCache.Tensor.Dimensions[0] != context ||
            query.Tensor.Dimensions.Aggregate(1L, (value, dimension) => value * dimension) != queryHeads * headSize)
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Incompatible GQA shapes.");
        }

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

internal sealed class GpuCausalSoftmaxInstruction() : GpuPhi4Instruction(
                "transformer.causal-softmax",
                [
                    new("scores", ElementType.Float32, ResourceAccess.Read),
                    new("position", ElementType.Int32, ResourceAccess.Read),
                    new("probabilities", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Softmax requires matching rank-two tensors.");
        }

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
                        """, [GpuInstructionHelpers.Load32, ReductionHelper],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuGroupedQueryValuesInstruction() : GpuPhi4Instruction(
    "transformer.gqa-values",
    [
        new("probabilities", ElementType.Float32, ResourceAccess.Read),
                    new("value_cache", ElementType.Float32, ResourceAccess.Read),
                    new("position", ElementType.Int32, ResourceAccess.Read),
                    new("output", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw new InstructionAdaptationException(CollectionId, Name, "Invalid probability or value-cache rank.");
        }

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

internal sealed class GpuBatchRopeKeyValueWriteInstruction() : GpuPhi4BatchInstruction(
    "transformer.batch-rope-kv-write",
    [
        new("qkv", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("frequencies", ElementType.Float32, ResourceAccess.Read),
        new("query", ElementType.Float32, ResourceAccess.Write),
        new("key_cache", ElementType.Float32, ResourceAccess.Write),
        new("value_cache", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw Error("Invalid qkv, query, cache, head, or rotary dimensions.");
        }

        if (Vector(frequencies, "frequencies") != rotarySize / 2)
        {
            throw Error("'frequencies' must have shape [rotary_size/2].");
        }

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

internal sealed class GpuBatchGroupedQueryScoresInstruction() : GpuPhi4BatchInstruction(
    "transformer.batch-gqa-scores",
    [
        new("query", ElementType.Float32, ResourceAccess.Read),
        new("key_cache", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("scores", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw Error("'scores' must have shape [B,qheads,context].");
        }

        var queryHeads = scores.Tensor.Dimensions[1];
        if (queryHeads <= 0 ||
            queryHeads % keyValueHeads != 0 ||
            querySize != checked(queryHeads * headSize))
        {
            throw Error("Incompatible query, cache, and GQA score shapes.");
        }

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

internal sealed class GpuBatchCausalSoftmaxInstruction() : GpuPhi4BatchInstruction(
    "transformer.batch-causal-softmax",
    [
        new("scores", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("probabilities", ElementType.Float32, ResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var scores = tensors["scores"];
        var control = tensors["control"];
        var probabilities = tensors["probabilities"];
        if (scores.Tensor.Dimensions.Count != 3)
        {
            throw Error("'scores' must have shape [B,heads,context].");
        }

        RequireSameShape(scores, "scores", probabilities, "probabilities");
        var batch = scores.Tensor.Dimensions[0];
        var heads = scores.Tensor.Dimensions[1];
        var context = scores.Tensor.Dimensions[2];
        if (batch <= 0 || heads <= 0 || context <= 0)
        {
            throw Error("'scores' dimensions must be positive.");
        }

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
            [GpuInstructionHelpers.Load32, BatchReduction],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class GpuBatchGroupedQueryValuesInstruction() : GpuPhi4BatchInstruction(
    "transformer.batch-gqa-values",
    [
        new("probabilities", ElementType.Float32, ResourceAccess.Read),
        new("value_cache", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw Error("'probabilities' must have shape [B,qheads,context].");
        }

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
        {
            throw Error("Incompatible probability, value cache, and output shapes.");
        }

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

internal sealed class GpuBatchSelectLastInstruction() : GpuPhi4BatchInstruction(
    "transformer.batch-select-last",
    [
        new("input", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw Error("Expected input [B,H] and output [H].");
        }

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

internal sealed class GpuBatchGatherRowsInstruction() : GpuPhi4BatchInstruction(
    "transformer.batch-gather-rows",
    [
        new("table", ElementType.Float32, ResourceAccess.Read),
        new("control", ElementType.Int32, ResourceAccess.Read),
        new("output", ElementType.Float32, ResourceAccess.Write),
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
        {
            throw Error("Expected table [N,H] and output [B,H].");
        }

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
