using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Instructions.Phi4.D3D12;

public sealed class Phi4D3D12VisionInstructionCollection : IInstructionCollectionProvider
{
    private readonly IReadOnlyDictionary<string, Instruction> instructions =
        new Instruction[]
        {
            new VisionPatchEmbeddingInstruction(),
            new VisionLayerNormInstruction(),
            new VisionAttentionInstruction(),
            new VisionGeluInstruction(),
            new VisionPool2x2Instruction(),
            new VisionHdGatherInstruction(),
        }.ToDictionary(instruction => instruction.Name, StringComparer.Ordinal);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(Phi4InstructionCollectionIds.VisionFloat32,
            "Phi-4 FP32 Vision operations", 1, InstructionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == Phi4InstructionCollectionIds.VisionFloat32 &&
        instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction]
            : [];
}

internal abstract class VisionInstruction(
    string name,
    IReadOnlyList<InstructionPort> ports,
    IReadOnlyList<string>? attributes = null) : Instruction
{
    private static readonly KernelPrecisionProfile Precision =
        new(GraphElementType.Float32, GraphElementType.Float32);

    protected static readonly InstructionHelper LoadStoreHelper = new(
        "vision.load-store",
        """
        float visionLoad16(RWByteAddressBuffer b,uint o,uint i) {
            uint a=o+i*2u;
            return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u);
        }
        float visionLoad32(RWByteAddressBuffer b,uint o,uint i) {
            return asfloat(b.Load(o+i*4u));
        }
        int visionLoadI32(RWByteAddressBuffer b,uint o,uint i) {
            return asint(b.Load(o+i*4u));
        }
        void visionStore32(RWByteAddressBuffer b,uint o,uint i,float value) {
            b.Store(o+i*4u,asuint(value));
        }
        """);

    public override Guid CollectionId => Phi4InstructionCollectionIds.VisionFloat32;
    public override string Name => name;
    public override InstructionTarget Target => InstructionTarget.Direct3D12;
    public override IReadOnlyList<InstructionSignature> Signatures { get; } =
        [new(ports, attributes ?? [], Precision)];

    protected static IReadOnlyDictionary<string, InstructionTensorParameter> Tensors(
        InstructionParameter[] parameters) =>
        parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);

    protected static IReadOnlyDictionary<string, string> Attributes(
        InstructionParameter[] parameters) =>
        parameters.OfType<InstructionAttributeParameter>()
            .ToDictionary(parameter => parameter.Name, parameter => parameter.Value, StringComparer.Ordinal);

    protected static int PositiveInt(IReadOnlyDictionary<string, string> attributes, string name)
    {
        if (!int.TryParse(attributes[name], NumberStyles.None, CultureInfo.InvariantCulture, out var value) ||
            value <= 0)
            throw new InstructionAdaptationException(
                Phi4InstructionCollectionIds.VisionFloat32, name,
                $"Attribute '{name}' must be a positive integer.");
        return value;
    }

    protected static string Buffer(InstructionTensorParameter tensor) => tensor.Expression;
    protected static string Offset(InstructionTensorParameter tensor) => tensor.OffsetExpression;
}

internal sealed class VisionPatchEmbeddingInstruction() : VisionInstruction(
    Phi4VisionInstructionNames.PatchEmbedding,
    [
        new("pixels", GraphElementType.Float32, GraphResourceAccess.Read),
        new("mask", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float16, GraphResourceAccess.Read),
        new("bias", GraphElementType.Float16, GraphResourceAccess.Read),
        new("position", GraphElementType.Float16, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["crop_size", "patch_size", "width"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var cropSize = PositiveInt(attributes, "crop_size");
        var patchSize = PositiveInt(attributes, "patch_size");
        var width = PositiveInt(attributes, "width");
        var grid = cropSize / patchSize;
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var pixels = tensors["pixels"];
        var mask = tensors["mask"];
        var weight = tensors["weight"];
        var bias = tensors["bias"];
        var position = tensors["position"];
        var source = $$"""
            {
                uint group=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint i=group*64u+gpu.groupIndex;
                if(i<{{count}}u) {
                    uint channel=i%{{width}}u;
                    uint token=(i/{{width}}u)%{{grid * grid}}u;
                    uint crop=i/({{width * grid * grid}}u);
                    uint patchY=token/{{grid}}u;
                    uint patchX=token%{{grid}}u;
                    precise float sum=visionLoad16({{Buffer(bias)}},{{Offset(bias)}},channel);
                    for(uint inputChannel=0u;inputChannel<3u;inputChannel++)
                    for(uint y=0u;y<{{patchSize}}u;y++)
                    for(uint x=0u;x<{{patchSize}}u;x++) {
                        uint pixel=((crop*3u+inputChannel)*{{cropSize}}u+
                            patchY*{{patchSize}}u+y)*{{cropSize}}u+
                            patchX*{{patchSize}}u+x;
                        uint coefficient=((channel*3u+inputChannel)*{{patchSize}}u+y)*
                            {{patchSize}}u+x;
                        sum+=visionLoad32({{Buffer(pixels)}},{{Offset(pixels)}},pixel)*
                            visionLoad16({{Buffer(weight)}},{{Offset(weight)}},coefficient);
                    }
                    uint validRows=0u;
                    while(validRows<{{grid}}u &&
                        visionLoad32({{Buffer(mask)}},{{Offset(mask)}},
                            (crop*{{grid}}u+validRows)*{{grid}}u)!=0.0f)
                        validRows++;
                    uint validColumns=0u;
                    while(validColumns<{{grid}}u &&
                        visionLoad32({{Buffer(mask)}},{{Offset(mask)}},
                            crop*{{grid * grid}}u+validColumns)!=0.0f)
                        validColumns++;
                    if(patchY<validRows && patchX<validColumns) {
                        uint positionId=(patchY*{{grid}}u/validRows)*{{grid}}u+
                            patchX*{{grid}}u/validColumns;
                        sum+=visionLoad16({{Buffer(position)}},{{Offset(position)}},
                            positionId*{{width}}u+channel);
                    }
                    visionStore32({{Buffer(output)}},{{Offset(output)}},i,sum);
                }
            }
            """;
        return new(source, [LoadStoreHelper]);
    }
}

internal sealed class VisionLayerNormInstruction() : VisionInstruction(
    Phi4VisionInstructionNames.LayerNorm,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float16, GraphResourceAccess.Read),
        new("bias", GraphElementType.Float16, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["epsilon"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        if (!float.TryParse(attributes["epsilon"], NumberStyles.Float, CultureInfo.InvariantCulture,
                out var epsilon) || !float.IsFinite(epsilon) || epsilon <= 0)
            throw new InstructionAdaptationException(
                CollectionId, Name, "Layer normalization epsilon must be finite and positive.");
        var input = tensors["input"];
        var weight = tensors["weight"];
        var bias = tensors["bias"];
        var output = tensors["output"];
        var width = weight.Tensor.Dimensions.Single();
        var rows = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension)) / (ulong)width;
        var epsilonText = epsilon.ToString("R", CultureInfo.InvariantCulture);
        var source = $$"""
            {
                uint row=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint lane=gpu.groupIndex;
                if(row<{{rows}}u) {
                    precise float sum=0.0f;
                    for(uint channel=lane;channel<{{width}}u;channel+=64u)
                        sum+=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                            row*{{width}}u+channel);
                    visionReduction[lane]=sum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            visionReduction[lane]+=visionReduction[lane+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float mean=visionReduction[0]/{{width}}.0f;
                    precise float squared=0.0f;
                    for(uint channel=lane;channel<{{width}}u;channel+=64u) {
                        precise float centered=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                            row*{{width}}u+channel)-mean;
                        squared+=centered*centered;
                    }
                    visionReduction[lane]=squared;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            visionReduction[lane]+=visionReduction[lane+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float scale=rsqrt(visionReduction[0]/{{width}}.0f+{{epsilonText}}f);
                    for(uint channel=lane;channel<{{width}}u;channel+=64u) {
                        precise float normalized=(visionLoad32({{Buffer(input)}},{{Offset(input)}},
                            row*{{width}}u+channel)-mean)*scale;
                        precise float value=normalized*
                            visionLoad16({{Buffer(weight)}},{{Offset(weight)}},channel)+
                            visionLoad16({{Buffer(bias)}},{{Offset(bias)}},channel);
                        visionStore32({{Buffer(output)}},{{Offset(output)}},
                            row*{{width}}u+channel,value);
                    }
                }
            }
            """;
        return new(source,
            [LoadStoreHelper, new("vision.reduction64", "groupshared float visionReduction[64];")],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class VisionAttentionInstruction() : VisionInstruction(
    Phi4VisionInstructionNames.Attention,
    [
        new("query", GraphElementType.Float32, GraphResourceAccess.Read),
        new("key", GraphElementType.Float32, GraphResourceAccess.Read),
        new("value", GraphElementType.Float32, GraphResourceAccess.Read),
        new("mask", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["heads"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var heads = PositiveInt(Attributes(parameters), "heads");
        var query = tensors["query"];
        var key = tensors["key"];
        var value = tensors["value"];
        var mask = tensors["mask"];
        var output = tensors["output"];
        var dimensions = output.Tensor.Dimensions;
        var crops = dimensions[0];
        var tokens = dimensions[1];
        var width = dimensions[2];
        var headWidth = width / heads;
        var work = checked((ulong)crops * (ulong)tokens * (ulong)heads);
        var scale = (1f / MathF.Sqrt(headWidth)).ToString("R", CultureInfo.InvariantCulture);
        var source = $$"""
            {
                uint work=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint lane=gpu.groupIndex;
                if(work<{{work}}u) {
                    uint head=work%{{heads}}u;
                    uint token=(work/{{heads}}u)%{{tokens}}u;
                    uint crop=work/({{heads * tokens}}u);
                    uint queryBase=(crop*{{tokens}}u+token)*{{width}}u+head*{{headWidth}}u;
                    precise float localMaximum=-3.402823466e+38f;
                    for(uint sourceToken=lane;sourceToken<{{tokens}}u;sourceToken+=64u) {
                        if(visionLoad32({{Buffer(mask)}},{{Offset(mask)}},
                            crop*{{tokens}}u+sourceToken)==0.0f) {
                            visionAttentionScores[sourceToken]=-3.402823466e+38f;
                            continue;
                        }
                        uint keyBase=(crop*{{tokens}}u+sourceToken)*{{width}}u+
                            head*{{headWidth}}u;
                        precise float score=0.0f;
                        for(uint channel=0u;channel<{{headWidth}}u;channel++)
                            score+=visionLoad32({{Buffer(query)}},{{Offset(query)}},
                                queryBase+channel)*
                                visionLoad32({{Buffer(key)}},{{Offset(key)}},keyBase+channel);
                        score*={{scale}}f;
                        visionAttentionScores[sourceToken]=score;
                        localMaximum=max(localMaximum,score);
                    }
                    visionReduction[lane]=localMaximum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            visionReduction[lane]=max(
                                visionReduction[lane],visionReduction[lane+step]);
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float maximum=visionReduction[0];
                    GroupMemoryBarrierWithGroupSync();
                    precise float localDenominator=0.0f;
                    for(uint sourceToken=lane;sourceToken<{{tokens}}u;sourceToken+=64u) {
                        precise float probability=
                            visionAttentionScores[sourceToken]==-3.402823466e+38f
                                ? 0.0f
                                : exp(visionAttentionScores[sourceToken]-maximum);
                        visionAttentionScores[sourceToken]=probability;
                        localDenominator+=probability;
                    }
                    visionReduction[lane]=localDenominator;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            visionReduction[lane]+=visionReduction[lane+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float denominator=visionReduction[0];
                    for(uint channel=lane;channel<{{headWidth}}u;channel+=64u) {
                        precise float sum=0.0f;
                        for(uint sourceToken=0u;sourceToken<{{tokens}}u;sourceToken++) {
                            uint valueIndex=(crop*{{tokens}}u+sourceToken)*{{width}}u+
                                head*{{headWidth}}u+channel;
                            sum+=visionAttentionScores[sourceToken]*
                                visionLoad32({{Buffer(value)}},{{Offset(value)}},valueIndex);
                        }
                        visionStore32({{Buffer(output)}},{{Offset(output)}},
                            queryBase+channel,sum/denominator);
                    }
                }
            }
            """;
        return new(source,
            [
                LoadStoreHelper,
                new("vision.reduction64", "groupshared float visionReduction[64];"),
                new("vision.attention-scores",
                    $"groupshared float visionAttentionScores[{tokens}];"),
            ],
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class VisionGeluInstruction() : VisionInstruction(
    Phi4VisionInstructionNames.Gelu,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var output = tensors["output"];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var source = $$"""
            {
                uint group=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint i=group*64u+gpu.groupIndex;
                if(i<{{count}}u) {
                    precise float x=visionLoad32({{Buffer(input)}},{{Offset(input)}},i);
                    precise float value=0.5f*x*(1.0f+tanh(
                        0.7978845608028654f*(x+0.044715f*x*x*x)));
                    visionStore32({{Buffer(output)}},{{Offset(output)}},i,value);
                }
            }
            """;
        return new(source, [LoadStoreHelper]);
    }
}

internal sealed class VisionPool2x2Instruction() : VisionInstruction(
    Phi4VisionInstructionNames.Pool2x2,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var output = tensors["output"];
        var dimensions = output.Tensor.Dimensions;
        var grid = dimensions[1];
        var width = dimensions[3];
        var inputGrid = grid * 2;
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var source = $$"""
            {
                uint group=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint i=group*64u+gpu.groupIndex;
                if(i<{{count}}u) {
                    uint channel=i%{{width}}u;
                    uint x=(i/{{width}}u)%{{grid}}u;
                    uint y=(i/({{width * grid}}u))%{{grid}}u;
                    uint crop=i/({{width * grid * grid}}u);
                    uint topLeft=((crop*{{inputGrid}}u+y*2u)*{{inputGrid}}u+x*2u)*
                        {{width}}u+channel;
                    precise float value=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                        topLeft);
                    value+=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                        topLeft+{{width}}u);
                    value+=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                        topLeft+{{inputGrid * width}}u);
                    value+=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                        topLeft+{{(inputGrid + 1) * width}}u);
                    visionStore32({{Buffer(output)}},{{Offset(output)}},i,value*0.25f);
                }
            }
            """;
        return new(source, [LoadStoreHelper]);
    }
}

internal sealed class VisionHdGatherInstruction() : VisionInstruction(
    Phi4VisionInstructionNames.HdGather,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("mapping", GraphElementType.Int32, GraphResourceAccess.Read),
        new("sub_separator", GraphElementType.Float16, GraphResourceAccess.Read),
        new("global_separator", GraphElementType.Float16, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var input = tensors["input"];
        var mapping = tensors["mapping"];
        var subSeparator = tensors["sub_separator"];
        var globalSeparator = tensors["global_separator"];
        var output = tensors["output"];
        var width = output.Tensor.Dimensions[^1];
        var count = output.Tensor.Dimensions.Aggregate(
            1UL, (product, dimension) => checked(product * (ulong)dimension));
        var source = $$"""
            {
                uint group=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint i=group*64u+gpu.groupIndex;
                if(i<{{count}}u) {
                    uint channel=i%{{width}}u;
                    uint token=i/{{width}}u;
                    int sourceIndex=visionLoadI32({{Buffer(mapping)}},{{Offset(mapping)}},token);
                    precise float value;
                    if(sourceIndex>=0)
                        value=visionLoad32({{Buffer(input)}},{{Offset(input)}},
                            uint(sourceIndex)*{{width}}u+channel);
                    else if(sourceIndex==-1)
                        value=visionLoad16({{Buffer(subSeparator)}},{{Offset(subSeparator)}},channel);
                    else
                        value=visionLoad16({{Buffer(globalSeparator)}},{{Offset(globalSeparator)}},channel);
                    visionStore32({{Buffer(output)}},{{Offset(output)}},i,value);
                }
            }
            """;
        return new(source, [LoadStoreHelper]);
    }
}
