using System.Buffers;
using System.Numerics.Tensors;
using SharpInference.Gguf;

namespace SharpInference.Architectures.Phi4;

public sealed class Phi4NativeEmbeddingProvider : IPhi4EmbeddingProvider
{
    private const int TextWidth = 3072;
    private readonly GgufModelFile text;
    private readonly GgufModelFile omni;

    internal Phi4NativeEmbeddingProvider(GgufModelFile text, GgufModelFile omni)
    {
        this.text = text;
        this.omni = omni;
        ValidateModel();
    }

    public async ValueTask<IReadOnlyList<ReadOnlyMemory<float>>> CreateFusedEmbeddingsAsync(
        Phi4PreparedInput input,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.AttentionMask.Length != input.TokenIds.Length)
            throw new ArgumentException("Attention mask length does not match the token sequence.", nameof(input));
        if (input.AudioProjectionMode is not ("speech" or "vision"))
            throw new ArgumentException(
                $"Unknown Phi-4 audio projection mode '{input.AudioProjectionMode}'.", nameof(input));

        var imageTask = input.Images.Count == 0
            ? Task.FromResult(Array.Empty<float[]>())
            : Task.Run(() => EncodeImages(input.Images, cancellationToken), cancellationToken);
        var audioTask = input.Audios.Count == 0
            ? Task.FromResult(Array.Empty<float[]>())
            : Task.Run(
                () => EncodeAudios(input.Audios, input.AudioProjectionMode, cancellationToken),
                cancellationToken);
        await Task.WhenAll(imageTask, audioTask).ConfigureAwait(false);

        var imageEmbeddings = imageTask.Result;
        var audioEmbeddings = audioTask.Result;
        var imageIndex = 0;
        var audioIndex = 0;
        var result = new ReadOnlyMemory<float>[input.TokenIds.Length];
        var tokenEmbedding = text.GetTensor("token_embd.weight");
        for (var index = 0; index < input.TokenIds.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var token = input.TokenIds[index];
            if (token == Phi4Tokenizer.ImageTokenId)
            {
                if (imageIndex == imageEmbeddings.Length)
                    throw new InvalidDataException("Image embeddings are shorter than the image token run.");
                result[index] = imageEmbeddings[imageIndex++];
            }
            else if (token == Phi4Tokenizer.AudioTokenId)
            {
                if (audioIndex == audioEmbeddings.Length)
                    throw new InvalidDataException("Audio embeddings are shorter than the audio token run.");
                result[index] = audioEmbeddings[audioIndex++];
            }
            else
            {
                if ((uint)token >= Phi4TextModel.VocabularySize)
                    throw new InvalidDataException($"Token {token} is outside the Phi-4 vocabulary.");
                var embedding = new float[TextWidth];
                TensorMath.CopyRow(tokenEmbedding, token, embedding);
                result[index] = embedding;
            }
        }
        if (imageIndex != imageEmbeddings.Length || audioIndex != audioEmbeddings.Length)
            throw new InvalidDataException("Projected media embeddings do not match the repeated media tokens.");
        return result;
    }

    private float[][] EncodeImages(
        IReadOnlyList<Phi4ImageFeatures> images,
        CancellationToken cancellationToken)
    {
        var result = new List<float[]>();
        foreach (var image in images)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var projected = EncodeImage(image, cancellationToken);
            if (projected.Length != image.ImageTokenCount)
                throw new InvalidDataException(
                    $"Image encoder produced {projected.Length} embeddings, expected {image.ImageTokenCount}.");
            result.AddRange(projected);
        }
        return result.ToArray();
    }

    private float[][] EncodeImage(Phi4ImageFeatures image, CancellationToken cancellationToken)
    {
        const int cropSize = 448;
        const int patchGrid = 32;
        const int visionWidth = 1152;
        const int compressedGrid = 16;
        var cropValues = checked(3 * cropSize * cropSize);
        if (image.CropCount <= 1 ||
            image.PixelValues.Length != checked(image.CropCount * cropValues) ||
            image.AttentionMask.Length != checked(image.CropCount * patchGrid * patchGrid) ||
            image.TargetHeight % cropSize != 0 ||
            image.TargetWidth % cropSize != 0)
            throw new InvalidDataException("Phi-4 image features have an invalid Golden HD shape.");

        var compressed = new float[image.CropCount][];
        for (var crop = 0; crop < image.CropCount; crop++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var mask = new bool[patchGrid * patchGrid];
            for (var index = 0; index < mask.Length; index++)
                mask[index] = image.AttentionMask[crop * mask.Length + index] != 0;
            var hidden = PatchEmbed(
                image.PixelValues.AsSpan(crop * cropValues, cropValues), mask);
            for (var layer = 0; layer < 26; layer++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                VisionLayer(hidden, mask, layer);
            }
            var pooled = new float[compressedGrid * compressedGrid * visionWidth];
            for (var y = 0; y < compressedGrid; y++)
            for (var x = 0; x < compressedGrid; x++)
            for (var channel = 0; channel < visionWidth; channel++)
            {
                var topLeft = ((y * 2) * patchGrid + x * 2) * visionWidth + channel;
                pooled[(y * compressedGrid + x) * visionWidth + channel] =
                    (hidden[topLeft] +
                     hidden[topLeft + visionWidth] +
                     hidden[topLeft + patchGrid * visionWidth] +
                     hidden[topLeft + (patchGrid + 1) * visionWidth]) * 0.25f;
            }
            compressed[crop] = pooled;
        }

        var horizontalTiles = image.TargetWidth / cropSize;
        var verticalTiles = image.TargetHeight / cropSize;
        var tileCount = checked(horizontalTiles * verticalTiles);
        if (image.CropCount != tileCount + 1)
            throw new InvalidDataException("Image crop count does not match its HD tile grid.");

        var usefulHeight = 0;
        for (var tileY = 0; tileY < verticalTiles; tileY++)
        for (var y = 0; y < compressedGrid; y++)
        {
            var crop = 1 + tileY * horizontalTiles;
            if (image.AttentionMask[crop * patchGrid * patchGrid + (y * 2) * patchGrid] != 0)
                usefulHeight = tileY * compressedGrid + y + 1;
        }
        var usefulWidth = 0;
        for (var tileX = 0; tileX < horizontalTiles; tileX++)
        for (var x = 0; x < compressedGrid; x++)
        {
            var crop = 1 + tileX;
            var offset = crop * patchGrid * patchGrid + x * 2;
            if (image.AttentionMask[offset] != 0)
                usefulWidth = tileX * compressedGrid + x + 1;
        }
        if (usefulHeight == 0 || usefulWidth == 0)
            throw new InvalidDataException("Image attention mask has no useful HD patches.");

        var subSeparator = TensorMath.ToArray(omni.GetTensor("v.sub_GN"));
        var globalSeparator = TensorMath.ToArray(omni.GetTensor("v.glb_GN"));
        var hd = new List<float[]>(image.ImageTokenCount);
        for (var y = 0; y < usefulHeight; y++)
        {
            var tileY = y / compressedGrid;
            var localY = y % compressedGrid;
            for (var x = 0; x < usefulWidth; x++)
            {
                var tileX = x / compressedGrid;
                var localX = x % compressedGrid;
                var crop = 1 + tileY * horizontalTiles + tileX;
                hd.Add(CopyToken(compressed[crop], localY * compressedGrid + localX, visionWidth));
            }
            hd.Add((float[])subSeparator.Clone());
        }
        hd.Add((float[])globalSeparator.Clone());
        for (var y = 0; y < compressedGrid; y++)
        {
            for (var x = 0; x < compressedGrid; x++)
                hd.Add(CopyToken(compressed[0], y * compressedGrid + x, visionWidth));
            hd.Add((float[])subSeparator.Clone());
        }
        if (hd.Count != image.ImageTokenCount)
            throw new InvalidDataException(
                $"HD transform produced {hd.Count} tokens, expected {image.ImageTokenCount}.");
        return Project(hd, "mm.0", "mm.2", cancellationToken);
    }

    private float[] PatchEmbed(ReadOnlySpan<float> pixels, bool[] mask)
    {
        const int cropSize = 448;
        const int patchSize = 14;
        const int grid = 32;
        const int width = 1152;
        var pixelValues = pixels.ToArray();
        var output = new float[grid * grid * width];
        var weight = omni.GetTensor("v.patch_embd.weight");
        var bias = omni.GetTensor("v.patch_embd.bias");
        var position = omni.GetTensor("v.position_embd.weight");
        Parallel.For(0, grid * grid, patch =>
        {
            var patchY = patch / grid;
            var patchX = patch % grid;
            var destination = output.AsSpan(patch * width, width);
            for (var channel = 0; channel < width; channel++)
            {
                var sum = TensorMath.Value(bias, channel);
                var weightBase = channel * 3 * patchSize * patchSize;
                for (var inputChannel = 0; inputChannel < 3; inputChannel++)
                for (var y = 0; y < patchSize; y++)
                for (var x = 0; x < patchSize; x++)
                {
                    var pixel = inputChannel * cropSize * cropSize +
                        (patchY * patchSize + y) * cropSize + patchX * patchSize + x;
                    var coefficient = weightBase + inputChannel * patchSize * patchSize +
                        y * patchSize + x;
                    sum += pixelValues[pixel] * TensorMath.Value(weight, coefficient);
                }
                destination[channel] = sum;
            }
        });

        var validRows = 0;
        while (validRows < grid && mask[validRows * grid])
            validRows++;
        var validColumns = 0;
        while (validColumns < grid && mask[validColumns])
            validColumns++;
        for (var y = 0; y < validRows; y++)
        for (var x = 0; x < validColumns; x++)
        {
            var positionId = (y * grid / validRows) * grid + x * grid / validColumns;
            var token = y * grid + x;
            for (var channel = 0; channel < width; channel++)
                output[token * width + channel] +=
                    TensorMath.Value(position, positionId * width + channel);
        }
        return output;
    }

    private void VisionLayer(float[] hidden, bool[] mask, int layer)
    {
        const int tokens = 1024;
        const int width = 1152;
        const int heads = 16;
        const int headWidth = width / heads;
        var prefix = $"v.blk.{layer}.";
        var normalized = new float[hidden.Length];
        TensorMath.LayerNorm(
            hidden, tokens, width,
            omni.GetTensor(prefix + "ln1.weight"),
            omni.GetTensor(prefix + "ln1.bias"), 1e-6f, normalized);
        var query = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "attn_q.weight"), omni.GetTensor(prefix + "attn_q.bias"));
        var key = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "attn_k.weight"), omni.GetTensor(prefix + "attn_k.bias"));
        var value = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "attn_v.weight"), omni.GetTensor(prefix + "attn_v.bias"));
        var attention = new float[hidden.Length];
        Parallel.For(0, tokens * heads, work =>
        {
            var token = work / heads;
            var head = work % heads;
            var scores = new float[tokens];
            var maximum = float.NegativeInfinity;
            var queryOffset = token * width + head * headWidth;
            for (var source = 0; source < tokens; source++)
            {
                if (!mask[source])
                {
                    scores[source] = float.NegativeInfinity;
                    continue;
                }
                var score = TensorMath.Dot(
                    query, queryOffset, key, source * width + head * headWidth, headWidth) /
                    MathF.Sqrt(headWidth);
                scores[source] = score;
                maximum = MathF.Max(maximum, score);
            }
            var denominator = 0f;
            for (var source = 0; source < tokens; source++)
            {
                if (!mask[source])
                    continue;
                scores[source] = MathF.Exp(scores[source] - maximum);
                denominator += scores[source];
            }
            var destination = token * width + head * headWidth;
            for (var source = 0; source < tokens; source++)
            {
                if (!mask[source])
                    continue;
                var probability = scores[source] / denominator;
                var sourceOffset = source * width + head * headWidth;
                for (var channel = 0; channel < headWidth; channel++)
                    attention[destination + channel] += probability * value[sourceOffset + channel];
            }
        });
        var projected = TensorMath.Linear(attention, tokens,
            omni.GetTensor(prefix + "attn_out.weight"), omni.GetTensor(prefix + "attn_out.bias"));
        TensorMath.Add(hidden, projected, 1f);

        TensorMath.LayerNorm(
            hidden, tokens, width,
            omni.GetTensor(prefix + "ln2.weight"),
            omni.GetTensor(prefix + "ln2.bias"), 1e-6f, normalized);
        var up = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "ffn_up.weight"), omni.GetTensor(prefix + "ffn_up.bias"));
        for (var index = 0; index < up.Length; index++)
            up[index] = TensorMath.GeluTanh(up[index]);
        var down = TensorMath.Linear(up, tokens,
            omni.GetTensor(prefix + "ffn_down.weight"), omni.GetTensor(prefix + "ffn_down.bias"));
        TensorMath.Add(hidden, down, 1f);
    }

    private float[][] EncodeAudios(
        IReadOnlyList<Phi4AudioFeatures> audios,
        string projectionMode,
        CancellationToken cancellationToken)
    {
        var result = new List<float[]>();
        foreach (var audio in audios)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var encoded = EncodeAudio(audio, cancellationToken);
            if (encoded.Length / 1024 != audio.EmbedSize)
                throw new InvalidDataException(
                    $"Audio encoder produced {encoded.Length / 1024} embeddings, expected {audio.EmbedSize}.");
            var tokens = Enumerable.Range(0, audio.EmbedSize)
                .Select(index => CopyToken(encoded, index, 1024))
                .ToList();
            var prefix = projectionMode == "vision" ? "mm.a.vis" : "mm.a.mlp";
            result.AddRange(Project(tokens, prefix + ".0", prefix + ".2", cancellationToken));
        }
        return result.ToArray();
    }

    public float[] EncodeAudioProjected(
        Phi4AudioFeatures audio,
        string projectionMode,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (projectionMode is not ("speech" or "vision"))
            throw new ArgumentException(
                $"Unknown Phi-4 audio projection mode '{projectionMode}'.", nameof(projectionMode));
        var rows = EncodeAudios([audio], projectionMode, cancellationToken);
        var result = new float[checked(rows.Length * TextWidth)];
        for (var index = 0; index < rows.Length; index++)
            rows[index].CopyTo(result, index * TextWidth);
        return result;
    }

    private float[] EncodeAudio(Phi4AudioFeatures audio, CancellationToken cancellationToken)
    {
        const int featureCount = 80;
        const int width = 1024;
        if (audio.FrameCount <= 0 ||
            audio.FeatureCount != featureCount ||
            audio.Values.Length != checked(audio.FrameCount * featureCount))
            throw new InvalidDataException("Phi-4 audio features have an invalid Golden Conformer shape.");

        var normalized = new float[audio.Values.Length];
        var mean = omni.GetTensor("a.global_mean");
        var inverseStandardDeviation = omni.GetTensor("a.global_invstd");
        for (var frame = 0; frame < audio.FrameCount; frame++)
        for (var feature = 0; feature < featureCount; feature++)
        {
            var index = frame * featureCount + feature;
            normalized[index] = (audio.Values[index] - TensorMath.Value(mean, feature)) *
                TensorMath.Value(inverseStandardDeviation, feature);
        }
        var hidden = AudioSubsample(normalized, audio.FrameCount, cancellationToken);
        var tokens = hidden.Length / width;
        if (tokens != audio.EmbedSize)
            throw new InvalidDataException(
                $"Conformer subsampling produced {tokens} frames, expected {audio.EmbedSize}.");

        for (var layer = 0; layer < 24; layer++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            AudioLayer(hidden, tokens, layer);
        }
        return hidden;
    }

    private float[] AudioSubsample(float[] input, int frames, CancellationToken cancellationToken)
    {
        var first = TensorMath.Conv2D(
            input, frames, 80, 1,
            omni.GetTensor("a.conv1d.0.weight"), omni.GetTensor("a.conv1d.0.bias"),
            stride: 2, padding: 1, groups: 1);
        TensorMath.Relu(first);
        cancellationToken.ThrowIfCancellationRequested();
        var secondDepthwise = TensorMath.Conv2D(
            first, DivideRoundUp(frames, 2), 40, 1024,
            omni.GetTensor("a.conv1d.2.weight"), omni.GetTensor("a.conv1d.2.bias"),
            stride: 2, padding: 1, groups: 1024);
        var second = TensorMath.Conv2D(
            secondDepthwise, DivideRoundUp(frames, 4), 20, 1024,
            omni.GetTensor("a.conv1d.3.weight"), omni.GetTensor("a.conv1d.3.bias"),
            stride: 1, padding: 0, groups: 1);
        TensorMath.Relu(second);
        cancellationToken.ThrowIfCancellationRequested();
        var thirdDepthwise = TensorMath.Conv2D(
            second, DivideRoundUp(frames, 4), 20, 1024,
            omni.GetTensor("a.conv1d.5.weight"), omni.GetTensor("a.conv1d.5.bias"),
            stride: 2, padding: 1, groups: 1024);
        var third = TensorMath.Conv2D(
            thirdDepthwise, DivideRoundUp(frames, 8), 10, 1024,
            omni.GetTensor("a.conv1d.6.weight"), omni.GetTensor("a.conv1d.6.bias"),
            stride: 1, padding: 0, groups: 1);
        TensorMath.Relu(third);

        var tokens = DivideRoundUp(frames, 8);
        var flattened = new float[tokens * 10240];
        for (var token = 0; token < tokens; token++)
        for (var channel = 0; channel < 1024; channel++)
        for (var frequency = 0; frequency < 10; frequency++)
            flattened[token * 10240 + channel * 10 + frequency] =
                third[(token * 10 + frequency) * 1024 + channel];
        return TensorMath.Linear(
            flattened, tokens,
            omni.GetTensor("a.conv1d.out.weight"), omni.GetTensor("a.conv1d.out.bias"));
    }

    private void AudioLayer(float[] hidden, int tokens, int layer)
    {
        const int width = 1024;
        const int heads = 16;
        const int headWidth = width / heads;
        var prefix = $"a.blk.{layer}.";
        var feedForward = AudioFeedForward(hidden, tokens, prefix + "ffn_in");
        TensorMath.Add(hidden, feedForward, 0.5f);

        var normalized = new float[hidden.Length];
        TensorMath.LayerNorm(
            hidden, tokens, width,
            omni.GetTensor(prefix + "ln_att.weight"),
            omni.GetTensor(prefix + "ln_att.bias"), 1e-5f, normalized);
        var query = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "attn_q.weight"), omni.GetTensor(prefix + "attn_q.bias"));
        var key = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "attn_k.weight"), omni.GetTensor(prefix + "attn_k.bias"));
        var value = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + "attn_v.weight"), omni.GetTensor(prefix + "attn_v.bias"));
        var bias = omni.GetTensor("a.rel_attn_bias");
        var attention = new float[hidden.Length];
        Parallel.For(0, tokens * heads, work =>
        {
            var token = work / heads;
            var head = work % heads;
            var scores = new float[tokens];
            var maximum = float.NegativeInfinity;
            var queryOffset = token * width + head * headWidth;
            for (var source = 0; source < tokens; source++)
            {
                var relative = Math.Clamp(source - token, -500, 499) + 500;
                var score = TensorMath.Dot(
                    query, queryOffset, key, source * width + head * headWidth, headWidth) /
                    MathF.Sqrt(headWidth);
                score += TensorMath.Value(bias, relative * heads + head);
                scores[source] = score;
                maximum = MathF.Max(maximum, score);
            }
            var denominator = 0f;
            for (var source = 0; source < tokens; source++)
            {
                scores[source] = MathF.Exp(scores[source] - maximum);
                denominator += scores[source];
            }
            var destination = token * width + head * headWidth;
            for (var source = 0; source < tokens; source++)
            {
                var probability = scores[source] / denominator;
                var sourceOffset = source * width + head * headWidth;
                for (var channel = 0; channel < headWidth; channel++)
                    attention[destination + channel] += probability * value[sourceOffset + channel];
            }
        });
        var projected = TensorMath.Linear(attention, tokens,
            omni.GetTensor(prefix + "attn_out.weight"), omni.GetTensor(prefix + "attn_out.bias"));
        TensorMath.Add(hidden, projected, 1f);
        TensorMath.Add(hidden, AudioConvolution(hidden, tokens, prefix), 1f);
        TensorMath.Add(hidden, AudioFeedForward(hidden, tokens, prefix + "ffn_out"), 0.5f);
        TensorMath.LayerNorm(
            hidden, tokens, width,
            omni.GetTensor(prefix + "ln.weight"),
            omni.GetTensor(prefix + "ln.bias"), 1e-5f, normalized);
        normalized.CopyTo(hidden, 0);
    }

    private float[] AudioFeedForward(float[] hidden, int tokens, string prefix)
    {
        const int width = 1024;
        const int intermediate = 1536;
        var normalized = new float[hidden.Length];
        TensorMath.LayerNorm(
            hidden, tokens, width,
            omni.GetTensor(prefix + ".ln.weight"),
            omni.GetTensor(prefix + ".ln.bias"), 1e-5f, normalized);
        var up = TensorMath.Linear(normalized, tokens,
            omni.GetTensor(prefix + ".up.weight"), omni.GetTensor(prefix + ".up.bias"));
        var gated = new float[tokens * intermediate];
        for (var token = 0; token < tokens; token++)
        for (var channel = 0; channel < intermediate; channel++)
        {
            var offset = token * intermediate * 2 + channel;
            gated[token * intermediate + channel] =
                up[offset] * TensorMath.Swish(up[offset + intermediate]);
        }
        return TensorMath.Linear(gated, tokens,
            omni.GetTensor(prefix + ".down.weight"), omni.GetTensor(prefix + ".down.bias"));
    }

    private float[] AudioConvolution(float[] hidden, int tokens, string prefix)
    {
        const int width = 1024;
        var normalized = new float[hidden.Length];
        TensorMath.LayerNorm(
            hidden, tokens, width,
            omni.GetTensor(prefix + "conv.ln.weight"),
            omni.GetTensor(prefix + "conv.ln.bias"), 1e-5f, normalized);
        var pointwise = TensorMath.Conv1D(
            normalized, tokens,
            omni.GetTensor(prefix + "conv.glu.pw.weight"),
            omni.GetTensor(prefix + "conv.glu.pw.bias"), padding: 0, groups: 1);
        var b1 = omni.GetTensor(prefix + "conv.glu.b1");
        var b2 = omni.GetTensor(prefix + "conv.glu.b2");
        var gated = new float[tokens * width];
        for (var token = 0; token < tokens; token++)
        for (var channel = 0; channel < width; channel++)
        {
            var offset = token * width * 2 + channel;
            gated[token * width + channel] =
                (pointwise[offset] + TensorMath.Value(b1, channel)) *
                TensorMath.Swish(pointwise[offset + width] + TensorMath.Value(b2, channel));
        }
        var depthwise = TensorMath.Conv1D(
            gated, tokens,
            omni.GetTensor(prefix + "conv.dw.weight"),
            omni.GetTensor(prefix + "conv.dw.bias"), padding: 2, groups: width);
        Array.Resize(ref depthwise, tokens * width);
        var middle = TensorMath.Conv1D(
            depthwise, tokens,
            omni.GetTensor(prefix + "conv.pw_mid.weight"),
            omni.GetTensor(prefix + "conv.pw_mid.bias"), padding: 0, groups: 1);
        for (var index = 0; index < middle.Length; index++)
            middle[index] = TensorMath.Swish(middle[index]);
        return TensorMath.Conv1D(
            middle, tokens,
            omni.GetTensor(prefix + "conv.pw_ext.weight"),
            omni.GetTensor(prefix + "conv.pw_ext.bias"), padding: 0, groups: 1);
    }

    private float[][] Project(
        IReadOnlyList<float[]> tokens,
        string firstLayer,
        string secondLayer,
        CancellationToken cancellationToken)
    {
        if (tokens.Count == 0)
            return [];
        var inputWidth = tokens[0].Length;
        var flattened = new float[checked(tokens.Count * inputWidth)];
        for (var token = 0; token < tokens.Count; token++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (tokens[token].Length != inputWidth)
                throw new InvalidDataException("Projection input widths are inconsistent.");
            tokens[token].CopyTo(flattened, token * inputWidth);
        }
        var hidden = TensorMath.Linear(
            flattened, tokens.Count,
            omni.GetTensor(firstLayer + ".weight"), omni.GetTensor(firstLayer + ".bias"));
        for (var index = 0; index < hidden.Length; index++)
            hidden[index] = TensorMath.Gelu(hidden[index]);
        var projected = TensorMath.Linear(
            hidden, tokens.Count,
            omni.GetTensor(secondLayer + ".weight"), omni.GetTensor(secondLayer + ".bias"));
        return Enumerable.Range(0, tokens.Count)
            .Select(index => CopyToken(projected, index, TextWidth))
            .ToArray();
    }

    private void ValidateModel()
    {
        if (omni.GetMetadata<string>("general.architecture") != "clip" ||
            !omni.GetMetadata<bool>("clip.has_vision_encoder") ||
            !omni.GetMetadata<bool>("clip.has_audio_encoder") ||
            omni.GetMetadata<uint>("clip.vision.image_size") != 448 ||
            omni.GetMetadata<uint>("clip.vision.patch_size") != 14 ||
            omni.GetMetadata<uint>("clip.vision.embedding_length") != 1152 ||
            omni.GetMetadata<uint>("clip.vision.block_count") != 27 ||
            omni.GetMetadata<uint>("clip.vision.attention.head_count") != 16 ||
            omni.GetMetadata<uint>("clip.vision.projection_dim") != TextWidth)
            throw new InvalidDataException("The Omni GGUF metadata is not the expected Phi-4 vision encoder.");
        foreach (var name in new[]
        {
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "v.blk.25.attn_q.weight",
            "v.sub_GN",
            "v.glb_GN",
            "mm.0.weight",
            "mm.2.weight",
            "a.global_mean",
            "a.conv1d.0.weight",
            "a.conv1d.out.weight",
            "a.blk.23.attn_q.weight",
            "a.rel_attn_bias",
            "mm.a.mlp.0.weight",
            "mm.a.vis.0.weight",
        })
            _ = omni.GetTensor(name);
    }

    private static float[] CopyToken(float[] values, int token, int width)
    {
        var result = new float[width];
        values.AsSpan(token * width, width).CopyTo(result);
        return result;
    }

    private static int DivideRoundUp(int value, int divisor) =>
        checked((value + divisor - 1) / divisor);
}

internal static unsafe class TensorMath
{
    public static float[] ToArray(GgufModelTensor tensor)
    {
        var count = checked((int)(tensor.ByteLength /
            (tensor.Type == GgufModelTensorType.Float32 ? sizeof(float) : sizeof(ushort))));
        var result = new float[count];
        for (var index = 0; index < result.Length; index++)
            result[index] = Value(tensor, index);
        return result;
    }

    public static float Value(GgufModelTensor tensor, int index) =>
        tensor.Type switch
        {
            GgufModelTensorType.Float16 => (float)((Half*)tensor.DataAddress)[index],
            GgufModelTensorType.Float32 => ((float*)tensor.DataAddress)[index],
            _ => throw new InvalidDataException(
                $"Phi-4 Omni tensor '{tensor.Name}' has unsupported type '{tensor.Type}'."),
        };

    public static void CopyRow(GgufModelTensor tensor, int row, Span<float> output)
    {
        var width = checked((int)tensor.Dimensions[0]);
        if (output.Length != width)
            throw new ArgumentException("Tensor row width does not match the output.", nameof(output));
        var offset = checked(row * width);
        for (var index = 0; index < width; index++)
            output[index] = Value(tensor, offset + index);
    }

    public static float[] Linear(
        float[] input,
        int rows,
        GgufModelTensor weight,
        GgufModelTensor bias)
    {
        var inputWidth = checked((int)weight.Dimensions[0]);
        var outputWidth = checked((int)weight.Dimensions[1]);
        if (input.Length != checked(rows * inputWidth) ||
            bias.Dimensions.Count != 1 ||
            bias.Dimensions[0] != (ulong)outputWidth)
            throw new InvalidDataException($"Linear tensor shape is invalid for '{weight.Name}'.");
        var output = new float[checked(rows * outputWidth)];
        if (weight.Type == GgufModelTensorType.Float16 &&
            bias.Type == GgufModelTensorType.Float16)
        {
            var weightAddress = weight.DataAddress;
            var biasAddress = bias.DataAddress;
            Parallel.For(
                0,
                rows,
                () => ArrayPool<float>.Shared.Rent(inputWidth),
                (row, _, scratch) =>
                {
                    var weights = (Half*)weightAddress;
                    var biases = (Half*)biasAddress;
                    var inputOffset = row * inputWidth;
                    var outputOffset = row * outputWidth;
                    var converted = scratch.AsSpan(0, inputWidth);
                    for (var destination = 0; destination < outputWidth; destination++)
                    {
                        TensorPrimitives.ConvertToSingle(
                            new ReadOnlySpan<Half>(
                                weights + (long)destination * inputWidth, inputWidth),
                            converted);
                        output[outputOffset + destination] =
                            (float)biases[destination] +
                            TensorPrimitives.Dot(converted, input.AsSpan(inputOffset, inputWidth));
                    }
                    return scratch;
                },
                scratch => ArrayPool<float>.Shared.Return(scratch));
        }
        else if (weight.Type == GgufModelTensorType.Float32 &&
                 bias.Type == GgufModelTensorType.Float32)
        {
            var weightAddress = weight.DataAddress;
            var biasAddress = bias.DataAddress;
            Parallel.For(0, rows, row =>
            {
                var weights = (float*)weightAddress;
                var biases = (float*)biasAddress;
                var inputOffset = row * inputWidth;
                var outputOffset = row * outputWidth;
                for (var destination = 0; destination < outputWidth; destination++)
                {
                    var sum = biases[destination];
                    var weightOffset = destination * inputWidth;
                    for (var source = 0; source < inputWidth; source++)
                        sum += input[inputOffset + source] * weights[weightOffset + source];
                    output[outputOffset + destination] = sum;
                }
            });
        }
        else
        {
            throw new InvalidDataException(
                $"Linear tensors '{weight.Name}' and '{bias.Name}' must have matching F16 or F32 types.");
        }
        return output;
    }

    public static void LayerNorm(
        float[] input,
        int rows,
        int width,
        GgufModelTensor weight,
        GgufModelTensor bias,
        float epsilon,
        float[] output)
    {
        if (input.Length != checked(rows * width) || output.Length != input.Length)
            throw new ArgumentException("Layer normalization shape is invalid.", nameof(input));
        Parallel.For(0, rows, row =>
        {
            var offset = row * width;
            var mean = 0f;
            for (var channel = 0; channel < width; channel++)
                mean += input[offset + channel];
            mean /= width;
            var variance = 0f;
            for (var channel = 0; channel < width; channel++)
            {
                var centered = input[offset + channel] - mean;
                variance += centered * centered;
            }
            var scale = 1f / MathF.Sqrt(variance / width + epsilon);
            for (var channel = 0; channel < width; channel++)
                output[offset + channel] =
                    (input[offset + channel] - mean) * scale * Value(weight, channel) +
                    Value(bias, channel);
        });
    }

    public static float[] Conv1D(
        float[] input,
        int length,
        GgufModelTensor weight,
        GgufModelTensor bias,
        int padding,
        int groups)
    {
        var kernel = checked((int)weight.Dimensions[0]);
        var inputChannelsPerGroup = checked((int)weight.Dimensions[1]);
        var outputChannels = checked((int)weight.Dimensions[2]);
        var inputChannels = checked(inputChannelsPerGroup * groups);
        var outputLength = checked(length + 2 * padding - kernel + 1);
        if (input.Length != checked(length * inputChannels) || outputChannels % groups != 0)
            throw new InvalidDataException($"Conv1D tensor shape is invalid for '{weight.Name}'.");
        var output = new float[checked(outputLength * outputChannels)];
        var outputsPerGroup = outputChannels / groups;
        Parallel.For(0, outputLength, position =>
        {
            for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
            {
                var group = outputChannel / outputsPerGroup;
                var sum = Value(bias, outputChannel);
                var weightBase = outputChannel * inputChannelsPerGroup * kernel;
                for (var inputChannel = 0; inputChannel < inputChannelsPerGroup; inputChannel++)
                for (var k = 0; k < kernel; k++)
                {
                    var sourcePosition = position + k - padding;
                    if ((uint)sourcePosition >= (uint)length)
                        continue;
                    sum += input[sourcePosition * inputChannels +
                            group * inputChannelsPerGroup + inputChannel] *
                        Value(weight, weightBase + inputChannel * kernel + k);
                }
                output[position * outputChannels + outputChannel] = sum;
            }
        });
        return output;
    }

    public static float[] Conv2D(
        float[] input,
        int inputHeight,
        int inputWidth,
        int inputChannels,
        GgufModelTensor weight,
        GgufModelTensor bias,
        int stride,
        int padding,
        int groups)
    {
        var kernelWidth = checked((int)weight.Dimensions[0]);
        var kernelHeight = checked((int)weight.Dimensions[1]);
        var inputChannelsPerGroup = checked((int)weight.Dimensions[2]);
        var outputChannels = checked((int)weight.Dimensions[3]);
        var outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
        var outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
        if (input.Length != checked(inputHeight * inputWidth * inputChannels) ||
            inputChannelsPerGroup * groups != inputChannels ||
            outputChannels % groups != 0)
            throw new InvalidDataException($"Conv2D tensor shape is invalid for '{weight.Name}'.");
        var output = new float[checked(outputHeight * outputWidth * outputChannels)];
        var outputsPerGroup = outputChannels / groups;
        Parallel.For(0, outputHeight * outputWidth, position =>
        {
            var outputY = position / outputWidth;
            var outputX = position % outputWidth;
            for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
            {
                var group = outputChannel / outputsPerGroup;
                var sum = Value(bias, outputChannel);
                var weightBase = outputChannel *
                    inputChannelsPerGroup * kernelHeight * kernelWidth;
                for (var inputChannel = 0; inputChannel < inputChannelsPerGroup; inputChannel++)
                for (var kernelY = 0; kernelY < kernelHeight; kernelY++)
                for (var kernelX = 0; kernelX < kernelWidth; kernelX++)
                {
                    var sourceY = outputY * stride + kernelY - padding;
                    var sourceX = outputX * stride + kernelX - padding;
                    if ((uint)sourceY >= (uint)inputHeight ||
                        (uint)sourceX >= (uint)inputWidth)
                        continue;
                    var source = (sourceY * inputWidth + sourceX) * inputChannels +
                        group * inputChannelsPerGroup + inputChannel;
                    var coefficient = weightBase +
                        (inputChannel * kernelHeight + kernelY) * kernelWidth + kernelX;
                    sum += input[source] * Value(weight, coefficient);
                }
                output[position * outputChannels + outputChannel] = sum;
            }
        });
        return output;
    }

    public static float Dot(float[] left, int leftOffset, float[] right, int rightOffset, int count)
    {
        var result = 0f;
        for (var index = 0; index < count; index++)
            result += left[leftOffset + index] * right[rightOffset + index];
        return result;
    }

    public static void Add(float[] destination, float[] source, float scale)
    {
        if (destination.Length != source.Length)
            throw new ArgumentException("Tensor lengths do not match.", nameof(source));
        for (var index = 0; index < destination.Length; index++)
            destination[index] += source[index] * scale;
    }

    public static void Relu(float[] values)
    {
        for (var index = 0; index < values.Length; index++)
            values[index] = MathF.Max(0, values[index]);
    }

    public static float Swish(float value) => value / (1f + MathF.Exp(-value));

    public static float Gelu(float value) =>
        0.5f * value * (1f + Erf(value / MathF.Sqrt(2f)));

    public static float GeluTanh(float value)
    {
        const float coefficient = 0.7978845608028654f;
        return 0.5f * value *
            (1f + MathF.Tanh(coefficient * (value + 0.044715f * value * value * value)));
    }

    private static float Erf(float value)
    {
        var sign = MathF.CopySign(1f, value);
        var absolute = MathF.Abs(value);
        var t = 1f / (1f + 0.3275911f * absolute);
        var polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
            1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
        return sign * (1f - polynomial * MathF.Exp(-absolute * absolute));
    }
}
