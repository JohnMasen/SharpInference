using System.Globalization;
using SharpInference.Graphs;
using SharpInference.Instructions;
using SharpInference.Instructions.Phi4;

namespace SharpInference.Instructions.Phi4.D3D12;

public sealed class Phi4D3D12AudioInstructionCollection : IInstructionCollectionProvider
{
    private readonly IReadOnlyDictionary<string, Instruction> instructions =
        new Instruction[]
        {
            new NormalizeFeaturesInstruction(),
            new ConvolutionInstruction(Phi4AudioInstructionNames.Conv1D, dimensions: 1),
            new ConvolutionInstruction(Phi4AudioInstructionNames.Conv2D, dimensions: 2),
            new FlattenSubsamplingInstruction(),
            new LayerNormInstruction(),
            new BiasActivationInstruction(),
            new SwiGluInstruction(),
            new ResidualInstruction(),
            new RelativeAttentionInstruction(),
        }.ToDictionary(value => value.Name, StringComparer.Ordinal);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(Phi4InstructionCollectionIds.AudioFloat32,
            "Phi-4 FP32 Audio operations", 2, InstructionTarget.Direct3D12)];

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == Phi4InstructionCollectionIds.AudioFloat32 &&
        instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction]
            : [];
}

internal abstract class ConformerInstruction(
    string name,
    IReadOnlyList<InstructionPort> ports,
    IReadOnlyList<string> attributes) : Instruction
{
    private static readonly KernelPrecisionProfile Precision =
        new(GraphElementType.Float32, GraphElementType.Float32);

    public override Guid CollectionId => Phi4InstructionCollectionIds.AudioFloat32;
    public override string Name => name;
    public override InstructionTarget Target => InstructionTarget.Direct3D12;
    public override IReadOnlyList<InstructionSignature> Signatures { get; } =
        [new(ports, attributes, Precision)];

    protected static IReadOnlyDictionary<string, InstructionTensorParameter> Tensors(
        InstructionParameter[] parameters) =>
        parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(value => value.Name, StringComparer.Ordinal);

    protected static IReadOnlyDictionary<string, string> Attributes(
        InstructionParameter[] parameters) =>
        parameters.OfType<InstructionAttributeParameter>()
            .ToDictionary(value => value.Name, value => value.Value, StringComparer.Ordinal);

    protected static int PositiveInt(IReadOnlyDictionary<string, string> attributes, string name)
    {
        if (!int.TryParse(attributes[name], NumberStyles.None, CultureInfo.InvariantCulture, out var value) ||
            value <= 0)
            throw new InstructionAdaptationException(
                Phi4InstructionCollectionIds.AudioFloat32, name,
                $"Attribute '{name}' must be a positive integer.");
        return value;
    }

    protected static int NonNegativeInt(
        IReadOnlyDictionary<string, string> attributes,
        string name)
    {
        if (!int.TryParse(attributes[name], NumberStyles.None, CultureInfo.InvariantCulture, out var value) ||
            value < 0)
            throw new InstructionAdaptationException(
                Phi4InstructionCollectionIds.AudioFloat32, name,
                $"Attribute '{name}' must be a non-negative integer.");
        return value;
    }

    protected static float PositiveFloat(
        IReadOnlyDictionary<string, string> attributes,
        string name)
    {
        if (!float.TryParse(
                attributes[name], NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ||
            !float.IsFinite(value) ||
            value <= 0)
            throw new InstructionAdaptationException(
                Phi4InstructionCollectionIds.AudioFloat32, name,
                $"Attribute '{name}' must be finite and positive.");
        return value;
    }

    protected static string Load(InstructionTensorParameter tensor, string index) =>
        tensor.Tensor.ElementType switch
        {
            GraphElementType.Float16 =>
                $"conformerLoad16({tensor.Expression},{tensor.OffsetExpression},{index})",
            GraphElementType.Float32 =>
                $"conformerLoad32({tensor.Expression},{tensor.OffsetExpression},{index})",
            GraphElementType.Int32 =>
                $"asint({tensor.Expression}.Load({tensor.OffsetExpression}+({index})*4u))",
            _ => throw new NotSupportedException(),
        };

    protected static string Store(
        InstructionTensorParameter tensor,
        string index,
        string value) =>
        $"{tensor.Expression}.Store({tensor.OffsetExpression}+({index})*4u,asuint({value}));";

    protected static IReadOnlyList<InstructionHelper> LoadHelpers { get; } =
    [
        new("conformer.load32",
            "float conformerLoad32(RWByteAddressBuffer b,uint o,uint i){return asfloat(b.Load(o+i*4u));}"),
        new("conformer.load16",
            "float conformerLoad16(RWByteAddressBuffer b,uint o,uint i){uint a=o+i*2u;return f16tof32((b.Load(a&~3u)>>((a&2u)*8u))&65535u);}"),
    ];
}

internal sealed class NormalizeFeaturesInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.NormalizeFeatures,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("mean", GraphElementType.Float32, GraphResourceAccess.Read),
        new("inverse_standard_deviation", GraphElementType.Float32, GraphResourceAccess.Read),
        new("frame_count", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["frames", "features"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var frames = PositiveInt(attributes, "frames");
        var features = PositiveInt(attributes, "features");
        var input = tensors["input"];
        var mean = tensors["mean"];
        var inverse = tensors["inverse_standard_deviation"];
        var frameCount = tensors["frame_count"];
        var output = tensors["output"];
        var source = $$"""
            {
                if(i<{{frames * features}}u) {
                    uint feature=i%{{features}}u;
                    int validFrames={{Load(frameCount, "0u")}};
                    float value=i<(uint)validFrames*{{features}}u
                        ? ({{Load(input, "i")}}-{{Load(mean, "feature")}})*{{Load(inverse, "feature")}}
                        : 0.0f;
                    {{Store(output, "i", "value")}}
                }
            }
            """;
        return new(source, LoadHelpers);
    }
}

internal sealed class ConvolutionInstruction(string name, int dimensions) : ConformerInstruction(
    name,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float32, GraphResourceAccess.Read),
        new("bias", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    dimensions == 1
        ? ["input_length", "input_channels", "output_length", "output_channels",
            "kernel", "padding", "stride", "groups", "activation"]
        : ["input_height", "input_width", "input_channels", "output_height", "output_width",
            "output_channels", "kernel_height", "kernel_width", "padding", "stride", "groups",
            "activation"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var input = tensors["input"];
        var weight = tensors["weight"];
        var bias = tensors["bias"];
        var output = tensors["output"];
        var inputChannels = PositiveInt(attributes, "input_channels");
        var outputChannels = PositiveInt(attributes, "output_channels");
        var groups = PositiveInt(attributes, "groups");
        var padding = NonNegativeInt(attributes, "padding");
        var stride = PositiveInt(attributes, "stride");
        var activation = attributes["activation"] switch
        {
            "none" => "sum",
            "relu" => "max(0.0f,sum)",
            _ => throw new InstructionAdaptationException(
                CollectionId, Name, "Unsupported convolution activation."),
        };
        if (inputChannels % groups != 0 || outputChannels % groups != 0)
            throw new InstructionAdaptationException(
                CollectionId, Name, "Convolution channel counts must be divisible by groups.");
        var inputPerGroup = inputChannels / groups;
        var outputPerGroup = outputChannels / groups;

        string source;
        if (dimensions == 1)
        {
            var inputLength = PositiveInt(attributes, "input_length");
            var outputLength = PositiveInt(attributes, "output_length");
            var kernel = PositiveInt(attributes, "kernel");
            source = $$"""
                {
                    if(i<{{outputLength * outputChannels}}u) {
                        uint position=i/{{outputChannels}}u;
                        uint outputChannel=i%{{outputChannels}}u;
                        uint group=outputChannel/{{outputPerGroup}}u;
                        precise float sum={{Load(bias, "outputChannel")}};
                        uint weightBase=outputChannel*{{inputPerGroup * kernel}}u;
                        for(uint inputChannel=0u;inputChannel<{{inputPerGroup}}u;inputChannel++)
                        for(uint k=0u;k<{{kernel}}u;k++) {
                            int sourcePosition=(int)(position*{{stride}}u+k)-{{padding}};
                            if(sourcePosition>=0&&sourcePosition<{{inputLength}})
                                sum+={{Load(input, $"(uint)sourcePosition*{inputChannels}u+group*{inputPerGroup}u+inputChannel")}}*
                                    {{Load(weight, $"weightBase+inputChannel*{kernel}u+k")}};
                        }
                        {{Store(output, "i", activation)}}
                    }
                }
                """;
        }
        else
        {
            var inputHeight = PositiveInt(attributes, "input_height");
            var inputWidth = PositiveInt(attributes, "input_width");
            var outputHeight = PositiveInt(attributes, "output_height");
            var outputWidth = PositiveInt(attributes, "output_width");
            var kernelHeight = PositiveInt(attributes, "kernel_height");
            var kernelWidth = PositiveInt(attributes, "kernel_width");
            source = $$"""
                {
                    if(i<{{outputHeight * outputWidth * outputChannels}}u) {
                        uint outputChannel=i%{{outputChannels}}u;
                        uint position=i/{{outputChannels}}u;
                        uint outputY=position/{{outputWidth}}u;
                        uint outputX=position%{{outputWidth}}u;
                        uint group=outputChannel/{{outputPerGroup}}u;
                        precise float sum={{Load(bias, "outputChannel")}};
                        uint weightBase=outputChannel*{{inputPerGroup * kernelHeight * kernelWidth}}u;
                        for(uint inputChannel=0u;inputChannel<{{inputPerGroup}}u;inputChannel++)
                        for(uint kernelY=0u;kernelY<{{kernelHeight}}u;kernelY++)
                        for(uint kernelX=0u;kernelX<{{kernelWidth}}u;kernelX++) {
                            int sourceY=(int)(outputY*{{stride}}u+kernelY)-{{padding}};
                            int sourceX=(int)(outputX*{{stride}}u+kernelX)-{{padding}};
                            if(sourceY>=0&&sourceY<{{inputHeight}}&&sourceX>=0&&sourceX<{{inputWidth}}) {
                                uint source=((uint)sourceY*{{inputWidth}}u+(uint)sourceX)*{{inputChannels}}u+
                                    group*{{inputPerGroup}}u+inputChannel;
                                uint coefficient=weightBase+
                                    (inputChannel*{{kernelHeight}}u+kernelY)*{{kernelWidth}}u+kernelX;
                                sum+={{Load(input, "source")}}*{{Load(weight, "coefficient")}};
                            }
                        }
                        {{Store(output, "i", activation)}}
                    }
                }
                """;
        }
        return new(source, LoadHelpers);
    }
}

internal sealed class FlattenSubsamplingInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.FlattenSubsampling,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["tokens", "frequency", "channels"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var tokens = PositiveInt(attributes, "tokens");
        var frequency = PositiveInt(attributes, "frequency");
        var channels = PositiveInt(attributes, "channels");
        var input = tensors["input"];
        var output = tensors["output"];
        var source = $$"""
            {
                if(i<{{tokens * frequency * channels}}u) {
                    uint token=i/{{frequency * channels}}u;
                    uint within=i%{{frequency * channels}}u;
                    uint channel=within/{{frequency}}u;
                    uint frequency=within%{{frequency}}u;
                    {{Store(output, "i", Load(input, $"(token*{frequency}u+frequency)*{channels}u+channel"))}}
                }
            }
            """;
        return new(source, LoadHelpers);
    }
}

internal sealed class LayerNormInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.LayerNorm,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("weight", GraphElementType.Float32, GraphResourceAccess.Read),
        new("bias", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["rows", "width", "epsilon"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var rows = PositiveInt(attributes, "rows");
        var width = PositiveInt(attributes, "width");
        var epsilon = PositiveFloat(attributes, "epsilon")
            .ToString("R", CultureInfo.InvariantCulture);
        var input = tensors["input"];
        var weight = tensors["weight"];
        var bias = tensors["bias"];
        var output = tensors["output"];
        var source = $$"""
            {
                uint row=gpu.groupId.x;
                if(row<{{rows}}u) {
                    if(gpu.groupIndex==0u) {
                        precise float sum=0.0f;
                        for(uint channel=0u;channel<{{width}}u;channel++)
                            sum+={{Load(input, $"row*{width}u+channel")}};
                        precise float mean=sum/{{width}}.0f;
                        precise float variance=0.0f;
                        for(uint channel=0u;channel<{{width}}u;channel++) {
                            precise float centered={{Load(input, $"row*{width}u+channel")}}-mean;
                            variance+=centered*centered;
                        }
                        conformerReduction[0]=mean;
                        conformerReduction[1]=rsqrt(variance/{{width}}.0f+{{epsilon}}f);
                    }
                    GroupMemoryBarrierWithGroupSync();
                    precise float mean=conformerReduction[0];
                    precise float scale=conformerReduction[1];
                    for(uint channel=gpu.groupIndex;channel<{{width}}u;channel+=64u) {
                        uint index=row*{{width}}u+channel;
                        {{Store(output, "index",
                            $"({Load(input, "index")}-mean)*scale*{Load(weight, "channel")}+{Load(bias, "channel")}")}}
                    }
                }
            }
            """;
        return new(source,
            LoadHelpers.Append(new("conformer.reduction", "groupshared float conformerReduction[64];")),
            InstructionSynchronization.GroupMemoryBarrier);
    }
}

internal sealed class BiasActivationInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.BiasActivation,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("bias", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["count", "width", "activation"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var count = PositiveInt(attributes, "count");
        var width = PositiveInt(attributes, "width");
        var input = tensors["input"];
        var bias = tensors["bias"];
        var output = tensors["output"];
        var value = $"({Load(input, "i")}+{Load(bias, $"i%{width}u")})";
        var expression = attributes["activation"] switch
        {
            "none" => value,
            "relu" => $"max(0.0f,{value})",
            "swish" => $"({value}/(1.0f+exp(-{value})))",
            "gelu" => $"conformerGelu({value})",
            _ => throw new InstructionAdaptationException(
                CollectionId, Name, "Unsupported activation."),
        };
        var source = $$"""
            {
                if(i<{{count}}u)
                    {{Store(output, "i", expression)}}
            }
            """;
        return new(source, LoadHelpers.Append(new(
            "conformer.gelu",
            "float conformerGelu(float x){float s=x<0.0f?-1.0f:1.0f;float a=abs(x)*0.7071067811865475f;float t=1.0f/(1.0f+0.3275911f*a);float e=1.0f-(((((1.061405429f*t-1.453152027f)*t+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t)*exp(-a*a);return 0.5f*x*(1.0f+s*e);}")));
    }
}

internal sealed class SwiGluInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.SwiGlu,
    [
        new("input", GraphElementType.Float32, GraphResourceAccess.Read),
        new("bias_first", GraphElementType.Float32, GraphResourceAccess.Read),
        new("bias_second", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["rows", "width"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var rows = PositiveInt(attributes, "rows");
        var width = PositiveInt(attributes, "width");
        var input = tensors["input"];
        var first = tensors["bias_first"];
        var second = tensors["bias_second"];
        var output = tensors["output"];
        var source = $$"""
            {
                if(i<{{rows * width}}u) {
                    uint channel=i%{{width}}u;
                    uint baseIndex=(i/{{width}}u)*{{width * 2}}u+channel;
                    float left={{Load(input, "baseIndex")}}+{{Load(first, "channel")}};
                    float right={{Load(input, $"baseIndex+{width}u")}}+{{Load(second, "channel")}};
                    {{Store(output, "i", "left*right/(1.0f+exp(-right))")}}
                }
            }
            """;
        return new(source, LoadHelpers);
    }
}

internal sealed class ResidualInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.Residual,
    [
        new("hidden", GraphElementType.Float32, GraphResourceAccess.Read),
        new("update", GraphElementType.Float32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["count", "scale"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var count = PositiveInt(attributes, "count");
        if (!float.TryParse(attributes["scale"], NumberStyles.Float, CultureInfo.InvariantCulture, out var scale) ||
            !float.IsFinite(scale))
            throw new InstructionAdaptationException(CollectionId, Name, "Scale must be finite.");
        var scaleText = scale == MathF.Truncate(scale)
            ? scale.ToString("0.0", CultureInfo.InvariantCulture)
            : scale.ToString("R", CultureInfo.InvariantCulture);
        var hidden = tensors["hidden"];
        var update = tensors["update"];
        var output = tensors["output"];
        var source = $$"""
            {
                if(i<{{count}}u)
                    {{Store(output, "i", $"{Load(hidden, "i")}+{scaleText}f*{Load(update, "i")}")}}
            }
            """;
        return new(source, LoadHelpers);
    }
}

internal sealed class RelativeAttentionInstruction() : ConformerInstruction(
    Phi4AudioInstructionNames.RelativeAttention,
    [
        new("query", GraphElementType.Float32, GraphResourceAccess.Read),
        new("key", GraphElementType.Float32, GraphResourceAccess.Read),
        new("value", GraphElementType.Float32, GraphResourceAccess.Read),
        new("relative_bias", GraphElementType.Float32, GraphResourceAccess.Read),
        new("frame_count", GraphElementType.Int32, GraphResourceAccess.Read),
        new("output", GraphElementType.Float32, GraphResourceAccess.Write),
    ],
    ["tokens", "width", "heads", "subsampling"])
{
    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = Tensors(parameters);
        var attributes = Attributes(parameters);
        var tokens = PositiveInt(attributes, "tokens");
        var width = PositiveInt(attributes, "width");
        var heads = PositiveInt(attributes, "heads");
        var subsampling = PositiveInt(attributes, "subsampling");
        if (width % heads != 0)
            throw new InstructionAdaptationException(CollectionId, Name, "Width must be divisible by heads.");
        var headWidth = width / heads;
        var scale = (1f / MathF.Sqrt(headWidth)).ToString("R", CultureInfo.InvariantCulture);
        var query = tensors["query"];
        var key = tensors["key"];
        var value = tensors["value"];
        var bias = tensors["relative_bias"];
        var frameCount = tensors["frame_count"];
        var output = tensors["output"];
        var source = $$"""
            {
                uint work=gpu.groupId.x+gpu.groupId.y*gpu.groupCount.x;
                uint lane=gpu.groupIndex;
                if(work<{{tokens * heads}}u) {
                    uint token=work/{{heads}}u;
                    uint head=work%{{heads}}u;
                    uint activeTokens=min(
                        {{tokens}}u,
                        ((uint)max(1,{{Load(frameCount, "0u")}})+
                            {{subsampling - 1}}u)/{{subsampling}}u);
                    if(token>=activeTokens) {
                        for(uint headChannel=lane;headChannel<{{headWidth}}u;headChannel+=64u)
                            {{Store(output,
                                $"token*{width}u+head*{headWidth}u+headChannel",
                                "0.0f")}}
                        return;
                    }
                    precise float localMaximum=-3.402823466e+38f;
                    for(uint source=lane;source<activeTokens;source+=64u) {
                        precise float score=0.0f;
                        for(uint j=0u;j<{{headWidth}}u;j++)
                            score+={{Load(query, $"token*{width}u+head*{headWidth}u+j")}}*
                                {{Load(key, $"source*{width}u+head*{headWidth}u+j")}};
                        int relative=clamp((int)source-(int)token,-500,499)+500;
                        score=score*{{scale}}f+{{Load(bias, $"relative*{heads}u+head")}};
                        conformerAttentionScores[source]=score;
                        localMaximum=max(localMaximum,score);
                    }
                    conformerAttentionReduction[lane]=localMaximum;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            conformerAttentionReduction[lane]=max(
                                conformerAttentionReduction[lane],
                                conformerAttentionReduction[lane+step]);
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float maximum=conformerAttentionReduction[0];
                    GroupMemoryBarrierWithGroupSync();
                    precise float localDenominator=0.0f;
                    for(uint source=lane;source<activeTokens;source+=64u) {
                        precise float probability=exp(
                            conformerAttentionScores[source]-maximum);
                        conformerAttentionScores[source]=probability;
                        localDenominator+=probability;
                    }
                    conformerAttentionReduction[lane]=localDenominator;
                    GroupMemoryBarrierWithGroupSync();
                    for(uint step=32u;step>0u;step>>=1u) {
                        if(lane<step)
                            conformerAttentionReduction[lane]+=
                                conformerAttentionReduction[lane+step];
                        GroupMemoryBarrierWithGroupSync();
                    }
                    precise float denominator=conformerAttentionReduction[0];
                    for(uint headChannel=lane;headChannel<{{headWidth}}u;headChannel+=64u) {
                        precise float result=0.0f;
                        for(uint source=0u;source<activeTokens;source++)
                            result+=conformerAttentionScores[source]*
                                {{Load(value,
                                    $"source*{width}u+head*{headWidth}u+headChannel")}};
                        {{Store(output,
                            $"token*{width}u+head*{headWidth}u+headChannel",
                            "result/denominator")}}
                    }
                }
            }
            """;
        return new(source,
            LoadHelpers.Concat(
            [
                new("conformer.attention.scores",
                    $"groupshared float conformerAttentionScores[{tokens}];"),
                new("conformer.attention.reduction",
                    "groupshared float conformerAttentionReduction[64];"),
            ]),
            InstructionSynchronization.GroupMemoryBarrier);
    }
}
