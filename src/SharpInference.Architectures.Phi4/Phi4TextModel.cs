using System.Buffers;
using System.Numerics.Tensors;
using SharpInference.Gguf;

namespace SharpInference.Architectures.Phi4;

public interface IPhi4MatrixProjector
{
    void Project(GgufModelTensor weight, ReadOnlySpan<float> input, Span<float> output);
}

public sealed class Phi4TextModel
{
    internal const int EmbeddingSize = 3072;
    internal const int HeadCount = 24;
    internal const int KeyValueHeadCount = 8;
    internal const int HeadSize = 128;
    internal const int RotarySize = 96;
    internal const int IntermediateSize = 8192;
    internal const int LayerCount = 32;
    internal const int VocabularySize = 200064;
    private readonly GgufModelFile text;
    private readonly GgufModelFile visionAdapter;
    private readonly GgufModelFile speechAdapter;
    private readonly IPhi4MatrixProjector? matrixProjector;
    private readonly float rmsEpsilon;
    private readonly float ropeBase;
    private readonly float ropeScale;

    internal Phi4TextModel(
        GgufModelFile text,
        GgufModelFile visionAdapter,
        GgufModelFile speechAdapter,
        IPhi4MatrixProjector? matrixProjector)
    {
        this.text = text;
        this.visionAdapter = visionAdapter;
        this.speechAdapter = speechAdapter;
        this.matrixProjector = matrixProjector;
        rmsEpsilon = text.GetMetadata<float>("phi3.attention.layer_norm_rms_epsilon");
        ropeBase = text.GetMetadata<float>("phi3.rope.freq_base");
        ropeScale = text.GetMetadata<float>("phi3.rope.scaling.attn_factor");
    }

    public Phi4TextSession CreateSession(Phi4Adapter adapter = Phi4Adapter.None) =>
        new(this, adapter);

    internal void Embed(int token, Span<float> output)
    {
        if ((uint)token >= VocabularySize)
            throw new ArgumentOutOfRangeException(nameof(token));
        RequireLength(output, EmbeddingSize, nameof(output));
        var values = text.GetTensor("token_embd.weight").HalfValues;
        var offset = checked(token * EmbeddingSize);
        for (var index = 0; index < output.Length; index++)
            output[index] = (float)values[offset + index];
    }

    internal void ForwardLayer(
        int layer,
        int position,
        float[] hidden,
        Phi4Adapter adapter,
        Phi4KeyValueCache cache,
        Phi4TextWorkspace workspace)
    {
        RmsNorm(hidden, Tensor($"blk.{layer}.attn_norm.weight"), workspace.Normalized);
        Linear(workspace.Normalized, 0, EmbeddingSize, Tensor($"blk.{layer}.attn_qkv.weight"),
            AdapterTensor(adapter, $"blk.{layer}.attn_qkv.weight"), workspace.Qkv, workspace, matrixProjector);
        ApplyRotary(workspace.Qkv.AsSpan(0, EmbeddingSize), position, HeadCount);
        ApplyRotary(workspace.Qkv.AsSpan(EmbeddingSize, KeyValueHeadCount * HeadSize), position, KeyValueHeadCount);
        cache.Append(
            workspace.Qkv.AsSpan(EmbeddingSize, KeyValueHeadCount * HeadSize),
            workspace.Qkv.AsSpan(EmbeddingSize + KeyValueHeadCount * HeadSize, KeyValueHeadCount * HeadSize));

        ComputeAttention(workspace.Qkv, cache, workspace.Attention);
        Linear(workspace.Attention, 0, EmbeddingSize, Tensor($"blk.{layer}.attn_output.weight"),
            AdapterTensor(adapter, $"blk.{layer}.attn_output.weight"), workspace.Projected, workspace, matrixProjector);
        Add(hidden, workspace.Projected);

        RmsNorm(hidden, Tensor($"blk.{layer}.ffn_norm.weight"), workspace.Normalized);
        Linear(workspace.Normalized, 0, EmbeddingSize, Tensor($"blk.{layer}.ffn_up.weight"),
            AdapterTensor(adapter, $"blk.{layer}.ffn_up.weight"), workspace.GateUp, workspace, matrixProjector);
        for (var index = 0; index < IntermediateSize; index++)
        {
            var gate = workspace.GateUp[index];
            workspace.GateUp[index] =
                gate / (1f + MathF.Exp(-gate)) * workspace.GateUp[IntermediateSize + index];
        }
        Linear(workspace.GateUp, 0, IntermediateSize, Tensor($"blk.{layer}.ffn_down.weight"),
            AdapterTensor(adapter, $"blk.{layer}.ffn_down.weight"), workspace.Down, workspace, matrixProjector);
        Add(hidden, workspace.Down);
    }

    internal float[] GetLogits(float[] hidden, Phi4TextWorkspace workspace)
    {
        RmsNorm(hidden, Tensor("output_norm.weight"), workspace.Normalized);
        var logits = new float[VocabularySize];
        var weight = Tensor("token_embd.weight");
        if (matrixProjector is null)
            Linear(workspace.Normalized, 0, EmbeddingSize, weight, null, logits, workspace, null);
        else
            matrixProjector.Project(weight, workspace.Normalized, logits);
        return logits;
    }

    private void RmsNorm(
        ReadOnlySpan<float> input,
        GgufModelTensor weight,
        Span<float> output)
    {
        var sum = 0f;
        for (var index = 0; index < input.Length; index++)
            sum += input[index] * input[index];
        var scale = 1f / MathF.Sqrt(sum / input.Length + rmsEpsilon);
        if (weight.Type == GgufModelTensorType.Float32)
        {
            var values = weight.FloatValues;
            for (var index = 0; index < input.Length; index++)
                output[index] = input[index] * scale * values[index];
        }
        else
        {
            var values = weight.HalfValues;
            for (var index = 0; index < input.Length; index++)
                output[index] = input[index] * scale * (float)values[index];
        }
    }

    private static void Linear(
        float[] input,
        int inputOffset,
        int inputLength,
        GgufModelTensor weight,
        (GgufModelTensor A, GgufModelTensor B, float Scale)? adapter,
        float[] output,
        Phi4TextWorkspace workspace,
        IPhi4MatrixProjector? matrixProjector)
    {
        var inputSize = checked((int)weight.Dimensions[0]);
        var outputSize = checked((int)weight.Dimensions[1]);
        if (inputLength != inputSize || inputOffset < 0 || inputOffset > input.Length - inputLength)
            throw new ArgumentException("Linear input does not match the matrix.", nameof(input));
        if (output.Length < outputSize)
            throw new ArgumentException("Linear output does not match the matrix.", nameof(output));
        if (matrixProjector is null)
            Multiply(input, inputOffset, inputLength, weight, output);
        else
            matrixProjector.Project(
                weight,
                input.AsSpan(inputOffset, inputLength),
                output.AsSpan(0, outputSize));
        if (adapter is not { } lora)
            return;
        var rank = checked((int)lora.A.Dimensions[1]);
        Multiply(input, inputOffset, inputLength, lora.A, workspace.LoraTemporary);
        Multiply(workspace.LoraTemporary, 0, rank, lora.B, workspace.LoraUpdate);
        for (var index = 0; index < outputSize; index++)
            output[index] += workspace.LoraUpdate[index] * lora.Scale;
    }

    private static unsafe void Multiply(
        float[] input,
        int inputOffset,
        int inputLength,
        GgufModelTensor matrix,
        float[] output)
    {
        var inputSize = checked((int)matrix.Dimensions[0]);
        var outputSize = checked((int)matrix.Dimensions[1]);
        if (inputLength != inputSize || output.Length < outputSize)
            throw new ArgumentException("Matrix dimensions do not match the supplied buffers.");
        if (matrix.Type == GgufModelTensorType.Float16)
        {
            var address = matrix.DataAddress;
            Parallel.For(
                0,
                outputSize,
                () => ArrayPool<float>.Shared.Rent(inputSize),
                (row, _, scratch) =>
                {
                    TensorPrimitives.ConvertToSingle(
                        new ReadOnlySpan<Half>(
                            (Half*)address + (long)row * inputSize,
                            inputSize),
                        scratch.AsSpan(0, inputSize));
                    output[row] = TensorPrimitives.Dot(
                        scratch.AsSpan(0, inputSize),
                        input.AsSpan(inputOffset, inputSize));
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else if (matrix.Type == GgufModelTensorType.Float32)
        {
            var address = matrix.DataAddress;
            Parallel.For(0, outputSize, row =>
            {
                output[row] = TensorPrimitives.Dot(
                    new ReadOnlySpan<float>(
                        (float*)address + (long)row * inputSize,
                        inputSize),
                    input.AsSpan(inputOffset, inputSize));
            });
        }
        else
        {
            throw new NotSupportedException($"Phi-4 does not support matrix type '{matrix.Type}'.");
        }
    }

    private void ApplyRotary(Span<float> values, int position, int heads)
    {
        for (var head = 0; head < heads; head++)
        {
            var headValues = values.Slice(head * HeadSize, HeadSize);
            for (var index = 0; index < RotarySize / 2; index++)
            {
                var frequency = 1f / MathF.Pow(ropeBase, 2f * index / RotarySize);
                var angle = position * frequency;
                var cosine = MathF.Cos(angle) * ropeScale;
                var sine = MathF.Sin(angle) * ropeScale;
                var left = headValues[index];
                var right = headValues[index + RotarySize / 2];
                headValues[index] = left * cosine - right * sine;
                headValues[index + RotarySize / 2] = right * cosine + left * sine;
            }
        }
    }

    private static void ComputeAttention(
        float[] query,
        Phi4KeyValueCache cache,
        float[] output)
    {
        Array.Clear(output);
        Parallel.For(0, HeadCount, head =>
        {
            var keyValueHead = head / (HeadCount / KeyValueHeadCount);
            var scores = ArrayPool<float>.Shared.Rent(cache.Length);
            try
            {
                var maximum = float.NegativeInfinity;
                for (var token = 0; token < cache.Length; token++)
                {
                    var score = 0f;
                    for (var index = 0; index < HeadSize; index++)
                        score += query[head * HeadSize + index] *
                            cache.GetKey(token, keyValueHead, index);
                    score /= MathF.Sqrt(HeadSize);
                    scores[token] = score;
                    maximum = MathF.Max(maximum, score);
                }
                var denominator = 0f;
                for (var token = 0; token < cache.Length; token++)
                {
                    scores[token] = MathF.Exp(scores[token] - maximum);
                    denominator += scores[token];
                }
                for (var token = 0; token < cache.Length; token++)
                {
                    var probability = scores[token] / denominator;
                    for (var index = 0; index < HeadSize; index++)
                        output[head * HeadSize + index] +=
                            probability * cache.GetValue(token, keyValueHead, index);
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scores);
            }
        });
    }

    private GgufModelTensor Tensor(string name) => text.GetTensor(name);

    private (GgufModelTensor A, GgufModelTensor B, float Scale)? AdapterTensor(
        Phi4Adapter adapter,
        string name)
    {
        var file = adapter switch
        {
            Phi4Adapter.None => null,
            Phi4Adapter.Vision => visionAdapter,
            Phi4Adapter.Speech => speechAdapter,
            _ => throw new ArgumentOutOfRangeException(nameof(adapter)),
        };
        if (file is null)
            return null;
        var a = file.GetTensor(name + ".lora_a");
        var b = file.GetTensor(name + ".lora_b");
        var rank = checked((int)a.Dimensions[1]);
        var alpha = file.GetMetadata<float>("adapter.lora.alpha");
        return (a, b, alpha / rank);
    }

    private static void Add(Span<float> destination, ReadOnlySpan<float> source)
    {
        for (var index = 0; index < destination.Length; index++)
            destination[index] += source[index];
    }

    private static void RequireLength(ReadOnlySpan<float> value, int expected, string parameter)
    {
        if (value.Length != expected)
            throw new ArgumentException($"Expected {expected} values, received {value.Length}.", parameter);
    }
}

public sealed class Phi4TextSession
{
    private readonly Phi4TextModel model;
    private readonly Phi4KeyValueCache[] caches;
    private readonly float[] hidden = new float[Phi4TextModel.EmbeddingSize];
    private readonly Phi4TextWorkspace workspace = new();

    internal Phi4TextSession(Phi4TextModel model, Phi4Adapter adapter)
    {
        this.model = model;
        Adapter = adapter;
        caches = Enumerable.Range(0, Phi4TextModel.LayerCount)
            .Select(_ => new Phi4KeyValueCache())
            .ToArray();
    }

    public Phi4Adapter Adapter { get; }
    public int Position { get; private set; }

    public float[] ForwardToken(int token)
    {
        model.Embed(token, hidden);
        return ForwardEmbedding(hidden);
    }

    public float[] ForwardEmbedding(ReadOnlySpan<float> embedding)
    {
        if (embedding.Length != hidden.Length)
            throw new ArgumentException("Embedding size does not match Phi-4.", nameof(embedding));
        embedding.CopyTo(hidden);
        for (var layer = 0; layer < caches.Length; layer++)
            model.ForwardLayer(layer, Position, hidden, Adapter, caches[layer], workspace);
        Position++;
        return model.GetLogits(hidden, workspace);
    }
}

internal sealed class Phi4TextWorkspace
{
    public float[] Normalized { get; } = new float[Phi4TextModel.EmbeddingSize];
    public float[] Qkv { get; } =
        new float[Phi4TextModel.EmbeddingSize +
            2 * Phi4TextModel.KeyValueHeadCount * Phi4TextModel.HeadSize];
    public float[] Attention { get; } = new float[Phi4TextModel.EmbeddingSize];
    public float[] Projected { get; } = new float[Phi4TextModel.EmbeddingSize];
    public float[] GateUp { get; } = new float[Phi4TextModel.IntermediateSize * 2];
    public float[] Down { get; } = new float[Phi4TextModel.EmbeddingSize];
    public float[] LoraTemporary { get; } = new float[320];
    public float[] LoraUpdate { get; } = new float[Phi4TextModel.IntermediateSize * 2];
}

internal sealed class Phi4KeyValueCache
{
    private const int Width = Phi4TextModel.KeyValueHeadCount * Phi4TextModel.HeadSize;
    private float[] keys = [];
    private float[] values = [];

    public int Length { get; private set; }

    public void Append(ReadOnlySpan<float> key, ReadOnlySpan<float> value)
    {
        if (key.Length != Width || value.Length != Width)
            throw new ArgumentException("KV width does not match Phi-4.");
        var required = checked((Length + 1) * Width);
        if (keys.Length < required)
        {
            var capacity = Math.Max(required, Math.Max(Width * 16, keys.Length * 2));
            Array.Resize(ref keys, capacity);
            Array.Resize(ref values, capacity);
        }
        key.CopyTo(keys.AsSpan(Length * Width, Width));
        value.CopyTo(values.AsSpan(Length * Width, Width));
        Length++;
    }

    public float GetKey(int token, int head, int index) =>
        keys[token * Width + head * Phi4TextModel.HeadSize + index];

    public float GetValue(int token, int head, int index) =>
        values[token * Width + head * Phi4TextModel.HeadSize + index];
}
