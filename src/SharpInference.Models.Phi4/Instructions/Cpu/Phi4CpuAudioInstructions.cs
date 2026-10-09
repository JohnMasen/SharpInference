using System.Globalization;
using System.Numerics;
using System.Text;
using SharpInference.Graphs;
using SharpInference.Instructions;

namespace SharpInference.Instructions.Phi4.Cpu;

public sealed class Phi4CpuAudioInstructionCollection : IInstructionCollectionProvider, IGraphInstructionProvider
{
    private static readonly string[] Names =
    [
        Phi4AudioInstructionNames.NormalizeFeatures,
        Phi4AudioInstructionNames.Conv1D,
        Phi4AudioInstructionNames.Conv2D,
        Phi4AudioInstructionNames.FlattenSubsampling,
        Phi4AudioInstructionNames.LayerNorm,
        Phi4AudioInstructionNames.BiasActivation,
        Phi4AudioInstructionNames.SwiGlu,
        Phi4AudioInstructionNames.Residual,
        Phi4AudioInstructionNames.RelativeAttention,
    ];

    private readonly IReadOnlyDictionary<string, Instruction> instructions =
        Names.ToDictionary(name => name, name => (Instruction)new CpuAudioInstruction(name),
            StringComparer.Ordinal);

    public IReadOnlyList<InstructionCollectionDescription> QueryInstructionCollection() =>
        [new(Phi4InstructionCollectionIds.AudioFloat32,
            "Phi-4 FP32 Audio operations", 2, InstructionTarget.Cpu)];

    public IReadOnlyList<Instruction> QueryInstruction(Guid collectionId, string instructionName) =>
        collectionId == Phi4InstructionCollectionIds.AudioFloat32 &&
        instructions.TryGetValue(instructionName, out var instruction)
            ? [instruction]
            : [];

    public IReadOnlyList<GraphInstructionBinding> QueryGraphInstructionBindings() =>
        Phi4AudioGraphOperations.Bindings(InstructionTarget.Cpu);
}

internal sealed class CpuAudioInstruction : Instruction
{
    private static readonly KernelPrecisionProfile Precision =
        new(GraphElementType.Float32, GraphElementType.Float32);
    private readonly string name;

    public CpuAudioInstruction(string name)
    {
        this.name = name;
        Signatures = [CreateSignature(name)];
    }

    public override Guid CollectionId => Phi4InstructionCollectionIds.AudioFloat32;
    public override string Name => name;
    public override InstructionTarget Target => InstructionTarget.Cpu;
    public override IReadOnlyList<InstructionSignature> Signatures { get; }

    protected override InstructionRecording Generate(InstructionParameter[] parameters)
    {
        var tensors = parameters.OfType<InstructionTensorParameter>()
            .ToDictionary(parameter => parameter.Name, StringComparer.Ordinal);
        var attributes = parameters.OfType<InstructionAttributeParameter>()
            .ToDictionary(parameter => parameter.Name, parameter => parameter.Value, StringComparer.Ordinal);
        var code = new StringBuilder("{\n");
        foreach (var tensor in tensors.Values)
        {
            var type = tensor.Tensor.ElementType == GraphElementType.Int32 ? "int" : "float";
            var span = tensor.Access == GraphResourceAccess.Read
                ? "System.ReadOnlySpan"
                : "System.Span";
            code.AppendLine(
                $"{span}<{type}> {tensor.Name} = " +
                $"System.Runtime.InteropServices.MemoryMarshal.Cast<byte,{type}>({tensor.Expression});");
        }
        code.Append("SharpInference.Instructions.Phi4.Cpu.Phi4CpuAudioNumerics.");
        code.Append(Name switch
        {
            Phi4AudioInstructionNames.NormalizeFeatures => "NormalizeFeatures",
            Phi4AudioInstructionNames.Conv1D => "Conv1D",
            Phi4AudioInstructionNames.Conv2D => "Conv2D",
            Phi4AudioInstructionNames.FlattenSubsampling => "FlattenSubsampling",
            Phi4AudioInstructionNames.LayerNorm => "LayerNorm",
            Phi4AudioInstructionNames.BiasActivation => "BiasActivation",
            Phi4AudioInstructionNames.SwiGlu => "SwiGlu",
            Phi4AudioInstructionNames.Residual => "Residual",
            Phi4AudioInstructionNames.RelativeAttention => "RelativeAttention",
            _ => throw new NotSupportedException(Name),
        });
        code.Append('(');
        code.Append(Name switch
        {
            Phi4AudioInstructionNames.NormalizeFeatures =>
                $"input,mean,inverse_standard_deviation,frame_count[0],output,{I("frames")},{I("features")}",
            Phi4AudioInstructionNames.Conv1D =>
                $"input,weight,bias,output,{I("input_length")},{I("input_channels")}," +
                $"{I("output_length")},{I("output_channels")},{I("kernel")}," +
                $"{I("padding")},{I("stride")},{I("groups")},{S("activation")}",
            Phi4AudioInstructionNames.Conv2D =>
                $"input,weight,bias,output,{I("input_height")},{I("input_width")}," +
                $"{I("input_channels")},{I("output_height")},{I("output_width")}," +
                $"{I("output_channels")},{I("kernel_height")},{I("kernel_width")}," +
                $"{I("padding")},{I("stride")},{I("groups")},{S("activation")}",
            Phi4AudioInstructionNames.FlattenSubsampling =>
                $"input,output,{I("tokens")},{I("frequency")},{I("channels")}",
            Phi4AudioInstructionNames.LayerNorm =>
                $"input,weight,bias,output,{I("rows")},{I("width")},{F("epsilon")}",
            Phi4AudioInstructionNames.BiasActivation =>
                $"input,bias,output,{I("count")},{I("width")},{S("activation")}",
            Phi4AudioInstructionNames.SwiGlu =>
                $"input,bias_first,bias_second,output,{I("rows")},{I("width")}",
            Phi4AudioInstructionNames.Residual =>
                $"hidden,update,output,{I("count")},{F("scale")}",
            Phi4AudioInstructionNames.RelativeAttention =>
                $"query,key,value,relative_bias,frame_count[0],output,{I("tokens")}," +
                $"{I("width")},{I("heads")},{I("subsampling")}",
            _ => throw new NotSupportedException(Name),
        });
        code.AppendLine(");");
        code.AppendLine("}");
        return new(code.ToString(), references: [typeof(Phi4CpuAudioNumerics).Assembly]);

        string I(string key) =>
            int.Parse(attributes[key], CultureInfo.InvariantCulture)
                .ToString(CultureInfo.InvariantCulture);
        string F(string key) =>
            float.Parse(attributes[key], CultureInfo.InvariantCulture)
                .ToString("R", CultureInfo.InvariantCulture) + "f";
        string S(string key) => "\"" + attributes[key] + "\"";
    }

    private static InstructionSignature CreateSignature(string name)
    {
        static InstructionPort R(string name, GraphElementType type = GraphElementType.Float32) =>
            new(name, type, GraphResourceAccess.Read);
        static InstructionPort W(string name) =>
            new(name, GraphElementType.Float32, GraphResourceAccess.Write);
        return name switch
        {
            Phi4AudioInstructionNames.NormalizeFeatures => new(
                [R("input"), R("mean"), R("inverse_standard_deviation"),
                    R("frame_count", GraphElementType.Int32), W("output")],
                ["frames", "features"], Precision),
            Phi4AudioInstructionNames.Conv1D => new(
                [R("input"), R("weight"), R("bias"), W("output")],
                ["input_length", "input_channels", "output_length", "output_channels",
                    "kernel", "padding", "stride", "groups", "activation"], Precision),
            Phi4AudioInstructionNames.Conv2D => new(
                [R("input"), R("weight"), R("bias"), W("output")],
                ["input_height", "input_width", "input_channels", "output_height",
                    "output_width", "output_channels", "kernel_height", "kernel_width",
                    "padding", "stride", "groups", "activation"], Precision),
            Phi4AudioInstructionNames.FlattenSubsampling => new(
                [R("input"), W("output")], ["tokens", "frequency", "channels"], Precision),
            Phi4AudioInstructionNames.LayerNorm => new(
                [R("input"), R("weight"), R("bias"), W("output")],
                ["rows", "width", "epsilon"], Precision),
            Phi4AudioInstructionNames.BiasActivation => new(
                [R("input"), R("bias"), W("output")],
                ["count", "width", "activation"], Precision),
            Phi4AudioInstructionNames.SwiGlu => new(
                [R("input"), R("bias_first"), R("bias_second"), W("output")],
                ["rows", "width"], Precision),
            Phi4AudioInstructionNames.Residual => new(
                [R("hidden"), R("update"), W("output")],
                ["count", "scale"], Precision),
            Phi4AudioInstructionNames.RelativeAttention => new(
                [R("query"), R("key"), R("value"), R("relative_bias"),
                    R("frame_count", GraphElementType.Int32), W("output")],
                ["tokens", "width", "heads", "subsampling"], Precision),
            _ => throw new NotSupportedException(name),
        };
    }
}

public static class Phi4CpuAudioNumerics
{
    public static void NormalizeFeatures(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> inverse,
        int frameCount,
        Span<float> output,
        int frames,
        int features)
    {
        for (var frame = 0; frame < Math.Min(frameCount, frames); frame++)
        for (var feature = 0; feature < features; feature++)
        {
            var index = frame * features + feature;
            output[index] = (input[index] - mean[feature]) * inverse[feature];
        }
        output[(Math.Min(frameCount, frames) * features)..].Clear();
    }

    public static void Conv1D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inputLength,
        int inputChannels,
        int outputLength,
        int outputChannels,
        int kernel,
        int padding,
        int stride,
        int groups,
        string activation)
    {
        var inputPerGroup = inputChannels / groups;
        var outputPerGroup = outputChannels / groups;
        for (var position = 0; position < outputLength; position++)
        for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
        {
            var group = outputChannel / outputPerGroup;
            var sum = bias[outputChannel];
            var weightBase = outputChannel * inputPerGroup * kernel;
            for (var inputChannel = 0; inputChannel < inputPerGroup; inputChannel++)
            for (var k = 0; k < kernel; k++)
            {
                var sourcePosition = position * stride + k - padding;
                if ((uint)sourcePosition < (uint)inputLength)
                    sum += input[sourcePosition * inputChannels +
                        group * inputPerGroup + inputChannel] *
                        weight[weightBase + inputChannel * kernel + k];
            }
            output[position * outputChannels + outputChannel] = Activate(sum, activation);
        }
    }

    public static void Conv2D(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int inputHeight,
        int inputWidth,
        int inputChannels,
        int outputHeight,
        int outputWidth,
        int outputChannels,
        int kernelHeight,
        int kernelWidth,
        int padding,
        int stride,
        int groups,
        string activation)
    {
        var inputPerGroup = inputChannels / groups;
        var outputPerGroup = outputChannels / groups;
        for (var outputY = 0; outputY < outputHeight; outputY++)
        for (var outputX = 0; outputX < outputWidth; outputX++)
        for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
        {
            var group = outputChannel / outputPerGroup;
            var sum = bias[outputChannel];
            var weightBase = outputChannel * inputPerGroup * kernelHeight * kernelWidth;
            for (var inputChannel = 0; inputChannel < inputPerGroup; inputChannel++)
            for (var kernelY = 0; kernelY < kernelHeight; kernelY++)
            for (var kernelX = 0; kernelX < kernelWidth; kernelX++)
            {
                var sourceY = outputY * stride + kernelY - padding;
                var sourceX = outputX * stride + kernelX - padding;
                if ((uint)sourceY < (uint)inputHeight && (uint)sourceX < (uint)inputWidth)
                {
                    var source = (sourceY * inputWidth + sourceX) * inputChannels +
                        group * inputPerGroup + inputChannel;
                    var coefficient = weightBase +
                        (inputChannel * kernelHeight + kernelY) * kernelWidth + kernelX;
                    sum += input[source] * weight[coefficient];
                }
            }
            output[(outputY * outputWidth + outputX) * outputChannels + outputChannel] =
                Activate(sum, activation);
        }
    }

    public static void FlattenSubsampling(
        ReadOnlySpan<float> input,
        Span<float> output,
        int tokens,
        int frequency,
        int channels)
    {
        for (var token = 0; token < tokens; token++)
        for (var channel = 0; channel < channels; channel++)
        for (var bin = 0; bin < frequency; bin++)
            output[(token * channels + channel) * frequency + bin] =
                input[(token * frequency + bin) * channels + channel];
    }

    public static void LayerNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int rows,
        int width,
        float epsilon)
    {
        for (var row = 0; row < rows; row++)
        {
            var source = input.Slice(row * width, width);
            var destination = output.Slice(row * width, width);
            var mean = 0f;
            for (var index = 0; index < width; index++) mean += source[index];
            mean /= width;
            var variance = 0f;
            for (var index = 0; index < width; index++)
            {
                var centered = source[index] - mean;
                variance += centered * centered;
            }
            var scale = 1f / MathF.Sqrt(variance / width + epsilon);
            for (var index = 0; index < width; index++)
                destination[index] = (source[index] - mean) * scale * weight[index] + bias[index];
        }
    }

    public static void BiasActivation(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int count,
        int width,
        string activation)
    {
        for (var index = 0; index < count; index++)
            output[index] = Activate(input[index] + bias[index % width], activation);
    }

    public static void SwiGlu(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> biasFirst,
        ReadOnlySpan<float> biasSecond,
        Span<float> output,
        int rows,
        int width)
    {
        for (var row = 0; row < rows; row++)
        for (var channel = 0; channel < width; channel++)
        {
            var inputOffset = row * width * 2 + channel;
            var left = input[inputOffset] + biasFirst[channel];
            var right = input[inputOffset + width] + biasSecond[channel];
            output[row * width + channel] = left * right / (1f + MathF.Exp(-right));
        }
    }

    public static void Residual(
        ReadOnlySpan<float> hidden,
        ReadOnlySpan<float> update,
        Span<float> output,
        int count,
        float scale)
    {
        var width = Vector<float>.Count;
        var scaleVector = new Vector<float>(scale);
        var index = 0;
        for (; index <= count - width; index += width)
            (new Vector<float>(hidden[index..]) +
                new Vector<float>(update[index..]) * scaleVector).CopyTo(output[index..]);
        for (; index < count; index++)
            output[index] = hidden[index] + scale * update[index];
    }

    public static void RelativeAttention(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> key,
        ReadOnlySpan<float> value,
        ReadOnlySpan<float> relativeBias,
        int frameCount,
        Span<float> output,
        int tokens,
        int width,
        int heads,
        int subsampling)
    {
        var headWidth = width / heads;
        var activeTokens = Math.Min(tokens, (Math.Max(1, frameCount) + subsampling - 1) / subsampling);
        var scale = 1f / MathF.Sqrt(headWidth);
        var scores = new float[activeTokens];
        for (var token = 0; token < activeTokens; token++)
        for (var head = 0; head < heads; head++)
        {
            var maximum = float.NegativeInfinity;
            for (var source = 0; source < activeTokens; source++)
            {
                var score = 0f;
                for (var channel = 0; channel < headWidth; channel++)
                    score += query[token * width + head * headWidth + channel] *
                        key[source * width + head * headWidth + channel];
                var relative = Math.Clamp(source - token, -500, 499) + 500;
                scores[source] = score * scale + relativeBias[relative * heads + head];
                maximum = MathF.Max(maximum, scores[source]);
            }
            var denominator = 0f;
            for (var source = 0; source < activeTokens; source++)
            {
                scores[source] = MathF.Exp(scores[source] - maximum);
                denominator += scores[source];
            }
            for (var channel = 0; channel < headWidth; channel++)
            {
                var result = 0f;
                for (var source = 0; source < activeTokens; source++)
                    result += scores[source] *
                        value[source * width + head * headWidth + channel];
                output[token * width + head * headWidth + channel] = result / denominator;
            }
        }
        output[(activeTokens * width)..].Clear();
    }

    private static float Activate(float value, string activation) =>
        activation switch
        {
            "none" => value,
            "relu" => MathF.Max(0, value),
            "swish" => value / (1f + MathF.Exp(-value)),
            "gelu" => Gelu(value),
            _ => throw new ArgumentOutOfRangeException(nameof(activation)),
        };

    private static float Gelu(float value)
    {
        var sign = value < 0 ? -1f : 1f;
        var scaled = MathF.Abs(value) * 0.7071067811865475f;
        var t = 1f / (1f + 0.3275911f * scaled);
        var erf = 1f - (((((1.061405429f * t - 1.453152027f) * t +
            1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t) *
            MathF.Exp(-scaled * scaled);
        return 0.5f * value * (1f + sign * erf);
    }
}
