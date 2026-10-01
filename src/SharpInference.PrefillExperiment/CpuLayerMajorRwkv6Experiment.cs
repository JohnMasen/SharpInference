using System.Numerics;
using System.Numerics.Tensors;
using SharpInference;

namespace SharpInference.PrefillExperiment;

/// <summary>Experimental CPU RWKV-6 prefill: all tokens traverse each layer before the next layer.</summary>
internal static class CpuLayerMajorRwkv6Experiment
{
    public static (float[] Logits, float[] State) Run(
        string modelPath, int[] tokens, float[] initialState, int chunkSize = 64,
        bool chunkedWkv = true, bool batchedProjections = false,
        ReusableGpuProjector? gpuProjector = null, ReusableGpuWkv6Stage? gpuWkv = null)
    {
        using var model = GgmlModelFile.Open(modelPath);
        return Run(model, tokens, initialState, chunkSize, chunkedWkv, batchedProjections,
            gpuProjector, gpuWkv);
    }

    public static (float[] Logits, float[] State) Run(
        GgmlModelFile model, int[] tokens, float[] initialState, int chunkSize = 64,
        bool chunkedWkv = true, bool batchedProjections = false,
        ReusableGpuProjector? gpuProjector = null, ReusableGpuWkv6Stage? gpuWkv = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(initialState);
        if (tokens.Length == 0)
            throw new ArgumentException("At least one token is required.", nameof(tokens));
        if (chunkSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSize));

        var e = model.EmbeddingSize;
        var count = tokens.Length;
        var first = model.GetRequired("blocks.0.att.time_faaaa");
        if (first.Dimensions.Count != 3 || first.Dimensions[0] != 1 ||
            first.Dimensions[1] <= 0 || first.Dimensions[2] <= 0 ||
            checked(first.Dimensions[1] * first.Dimensions[2]) != e)
            throw new InvalidDataException($"Invalid head shape in '{first.Name}'.");
        var heads = first.Dimensions[2];
        var headSize = first.Dimensions[1];
        var wkvSize = checked(e * headSize);
        var stride = checked(2 * e + wkvSize);
        if (initialState.Length != checked(model.LayerCount * stride))
            throw new ArgumentException("Initial state length does not match RWKV-6 [ffn x, att x, WKV] layer layout.", nameof(initialState));
        var state = (float[])initialState.Clone();
        var x = new float[checked(count * e)];
        var embedding = Matrix(model.GetRequired("emb.weight"), e, model.VocabularySize);
        var ln0Weight = Vector(model.GetRequired("blocks.0.ln0.weight"), e);
        var ln0Bias = Vector(model.GetRequired("blocks.0.ln0.bias"), e);
        for (var t = 0; t < count; t++)
        {
            if ((uint)tokens[t] >= (uint)model.VocabularySize)
                throw new ArgumentOutOfRangeException(nameof(tokens), $"Token at index {t} is outside the vocabulary.");
            LayerNorm(embedding.Slice(tokens[t] * e, e), ln0Weight, ln0Bias, x.AsSpan(t * e, e));
        }

        for (var layer = 0; layer < model.LayerCount; layer++)
        {
            string Name(string suffix) => $"blocks.{layer}.{suffix}";
            ReadOnlySpan<float> V(string suffix) => Vector(model.GetRequired(Name(suffix)), e);
            IModelTensor M(string suffix) => model.GetRequired(Name(suffix));
            var offset = layer * stride;
            var attPrevious = state.AsSpan(offset + e, e);
            var ffnPrevious = state.AsSpan(offset, e);
            var normalized = new float[x.Length];
            var delta = new float[x.Length];
            var mixInput = new float[x.Length];
            var maaX = V("att.time_maa_x");
            for (var t = 0; t < count; t++)
            {
                var pos = t * e;
                LayerNorm(x.AsSpan(pos, e), V("ln1.weight"), V("ln1.bias"), normalized.AsSpan(pos, e));
                var previous = t == 0 ? attPrevious : normalized.AsSpan(pos - e, e);
                for (var i = 0; i < e; i++)
                {
                    var d = previous[i] - normalized[pos + i];
                    delta[pos + i] = d;
                    mixInput[pos + i] = normalized[pos + i] + d * maaX[i];
                }
            }
            normalized.AsSpan((count - 1) * e, e).CopyTo(attPrevious);

            var maa = Project(M("att.time_maa_w1"), mixInput, count, e, batchedProjections, gpuProjector);
            if (maa.Length % (count * 5) != 0)
                throw new InvalidDataException($"Invalid RWKV-6 time MAA output in layer {layer}.");
            var maaHidden = maa.Length / (count * 5);
            for (var i = 0; i < maa.Length; i++) maa[i] = MathF.Tanh(maa[i]);
            var w2 = M("att.time_maa_w2");
            if (w2.Dimensions.Count != 3 || w2.Dimensions[0] != maaHidden ||
                w2.Dimensions[1] != e || w2.Dimensions[2] != 5)
                throw new InvalidDataException($"Invalid dimensions for '{w2.Name}'.");
            var mixed = new float[checked(count * 5 * e)];
            var w2Values = w2.FloatValues;
            for (var t = 0; t < count; t++)
            for (var g = 0; g < 5; g++)
            for (var i = 0; i < e; i++)
                mixed[(t * 5 + g) * e + i] = TensorPrimitives.Dot(
                    w2Values.Slice((g * e + i) * maaHidden, maaHidden),
                    maa.AsSpan((t * 5 + g) * maaHidden, maaHidden));

            var mixes = new[]
            {
                V("att.time_maa_w").ToArray(), V("att.time_maa_k").ToArray(),
                V("att.time_maa_v").ToArray(), V("att.time_maa_r").ToArray(),
                V("att.time_maa_g").ToArray()
            };
            var dynamicInputs = new float[5][];
            for (var g = 0; g < 5; g++)
            {
                var input = dynamicInputs[g] = new float[x.Length];
                for (var t = 0; t < count; t++)
                for (var i = 0; i < e; i++)
                {
                    var pos = t * e + i;
                    input[pos] = normalized[pos] +
                        delta[pos] * (mixed[(t * 5 + g) * e + i] + mixes[g][i]);
                }
            }
            var r = Project(M("att.receptance.weight"), dynamicInputs[3], count, e, batchedProjections, gpuProjector);
            var k = Project(M("att.key.weight"), dynamicInputs[1], count, e, batchedProjections, gpuProjector);
            var v = Project(M("att.value.weight"), dynamicInputs[2], count, e, batchedProjections, gpuProjector);
            var gate = Project(M("att.gate.weight"), dynamicInputs[4], count, e, batchedProjections, gpuProjector);
            for (var i = 0; i < gate.Length; i++) gate[i] /= 1f + MathF.Exp(-gate[i]);
            var decayHidden = Project(M("att.time_decay_w1"), dynamicInputs[0], count, e, batchedProjections, gpuProjector);
            for (var i = 0; i < decayHidden.Length; i++) decayHidden[i] = MathF.Tanh(decayHidden[i]);
            var decay = Project(M("att.time_decay_w2"), decayHidden, count, decayHidden.Length / count, batchedProjections, gpuProjector);
            var baseDecay = V("att.time_decay");
            for (var t = 0; t < count; t++)
            for (var i = 0; i < e; i++)
            {
                var pos = t * e + i;
                decay[pos] = MathF.Exp(-MathF.Exp(decay[pos] + baseDecay[i]));
            }

            var wkv = new Wkv6Inputs(heads, headSize, count, initialize: false);
            k.CopyTo(wkv.Keys, 0);
            v.CopyTo(wkv.Values, 0);
            decay.CopyTo(wkv.Decays, 0);
            state.AsSpan(offset + 2 * e, wkvSize).CopyTo(wkv.InitialState);
            var sequence = new Wkv6Sequence(wkv);
            r.CopyTo(sequence.Receptances, 0);
            V("att.time_faaaa").CopyTo(sequence.TimeFirst);
            var attention = gpuWkv is not null
                ? gpuWkv.Run(sequence, chunkSize)
                : chunkedWkv
                    ? CpuWkv6OutputsBenchmark.Chunked(sequence, chunkSize)
                    : CpuWkv6OutputsBenchmark.Sequential(sequence);
            attention.FinalState.CopyTo(state, offset + 2 * e);
            var att = attention.Outputs;
            var lnXWeight = V("att.ln_x.weight");
            var lnXBias = V("att.ln_x.bias");
            for (var t = 0; t < count; t++)
            {
                var pos = t * e;
                GroupNorm(att.AsSpan(pos, e), heads, headSize);
                for (var i = 0; i < e; i++)
                    att[pos + i] = (att[pos + i] * lnXWeight[i] + lnXBias[i]) * gate[pos + i];
            }
            var projected = Project(M("att.output.weight"), att, count, e, batchedProjections, gpuProjector);
            for (var i = 0; i < x.Length; i++) x[i] += projected[i];

            var ffnNormalized = new float[x.Length];
            var ffnKeyInput = new float[x.Length];
            var ffnReceptanceInput = new float[x.Length];
            var ffnMixK = V("ffn.time_maa_k");
            var ffnMixR = V("ffn.time_maa_r");
            for (var t = 0; t < count; t++)
            {
                var pos = t * e;
                LayerNorm(x.AsSpan(pos, e), V("ln2.weight"), V("ln2.bias"), ffnNormalized.AsSpan(pos, e));
                var previous = t == 0 ? ffnPrevious : ffnNormalized.AsSpan(pos - e, e);
                for (var i = 0; i < e; i++)
                {
                    var d = previous[i] - ffnNormalized[pos + i];
                    ffnKeyInput[pos + i] = ffnNormalized[pos + i] + d * ffnMixK[i];
                    ffnReceptanceInput[pos + i] = ffnNormalized[pos + i] + d * ffnMixR[i];
                }
            }
            ffnNormalized.AsSpan((count - 1) * e, e).CopyTo(ffnPrevious);
            var ffnR = Project(M("ffn.receptance.weight"), ffnReceptanceInput, count, e, batchedProjections, gpuProjector);
            var ffnHidden = Project(M("ffn.key.weight"), ffnKeyInput, count, e, batchedProjections, gpuProjector);
            for (var i = 0; i < ffnHidden.Length; i++)
            {
                var value = MathF.Max(0, ffnHidden[i]);
                ffnHidden[i] = value * value;
            }
            var ffnOutput = Project(M("ffn.value.weight"), ffnHidden, count, ffnHidden.Length / count, batchedProjections, gpuProjector);
            for (var i = 0; i < x.Length; i++)
                x[i] += ffnOutput[i] * (1f / (1f + MathF.Exp(-ffnR[i])));
        }

        var last = new float[e];
        LayerNorm(x.AsSpan((count - 1) * e, e),
            Vector(model.GetRequired("ln_out.weight"), e),
            Vector(model.GetRequired("ln_out.bias"), e), last);
        return (Project(model.GetRequired("head.weight"), last, 1, e, batchedProjections, gpuProjector), state);
    }

    private static ReadOnlySpan<float> Vector(IModelTensor tensor, int length)
    {
        if (tensor.FloatValues.Length != length)
            throw new InvalidDataException($"Tensor '{tensor.Name}' is not a vector of length {length}.");
        return tensor.FloatValues;
    }

    private static ReadOnlySpan<float> Matrix(IModelTensor tensor, int input, int output)
    {
        if (tensor.Dimensions.Count != 2 || tensor.Dimensions[0] != input ||
            tensor.Dimensions[1] != output)
            throw new InvalidDataException($"Tensor '{tensor.Name}' must have dimensions [{input}, {output}].");
        return tensor.FloatValues;
    }

    private static float[] Project(IModelTensor tensor, float[] input, int count, int inputWidth,
        bool batchedProjections = false, ReusableGpuProjector? gpuProjector = null)
    {
        if (input.Length != checked(count * inputWidth) || tensor.Dimensions.Count != 2 ||
            tensor.Dimensions[0] != inputWidth)
            throw new InvalidDataException($"Tensor '{tensor.Name}' cannot project width {inputWidth}.");
        if (gpuProjector is not null)
            return gpuProjector.Project(tensor, input, count, inputWidth);
        var outputWidth = tensor.Dimensions[1];
        var output = new float[checked(count * outputWidth)];
        var weights = tensor.FloatValues;
        if (batchedProjections && count > 1)
        {
            const int batchSize = 16;
            var vectorWidth = Vector<float>.Count;
            var stride = checked((batchSize + vectorWidth - 1) / vectorWidth * vectorWidth);
            var transposed = new float[checked(inputWidth * stride)];
            var accumulators = new Vector<float>[stride / vectorWidth];
            var partial = new float[stride];
            for (var start = 0; start < count; start += batchSize)
            {
                var batchCount = Math.Min(batchSize, count - start);
                for (var col = 0; col < inputWidth; col++)
                for (var token = 0; token < batchCount; token++)
                    transposed[col * stride + token] = input[(start + token) * inputWidth + col];

                for (var row = 0; row < outputWidth; row++)
                {
                    Array.Clear(accumulators);
                    for (var col = 0; col < inputWidth; col++)
                    {
                        var weight = new Vector<float>(weights[row * inputWidth + col]);
                        for (var lane = 0; lane < accumulators.Length; lane++)
                            accumulators[lane] += weight *
                                new Vector<float>(transposed, col * stride + lane * vectorWidth);
                    }
                    for (var lane = 0; lane < accumulators.Length; lane++)
                        accumulators[lane].CopyTo(partial, lane * vectorWidth);
                    for (var token = 0; token < batchCount; token++)
                        output[(start + token) * outputWidth + row] = partial[token];
                }
            }
            return output;
        }

        for (var t = 0; t < count; t++)
        for (var row = 0; row < outputWidth; row++)
            output[t * outputWidth + row] = TensorPrimitives.Dot(
                weights.Slice(row * inputWidth, inputWidth),
                input.AsSpan(t * inputWidth, inputWidth));
        return output;
    }

    private static void LayerNorm(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> output)
    {
        float mean = 0;
        for (var i = 0; i < input.Length; i++) mean += input[i];
        mean /= input.Length;
        float variance = 0;
        for (var i = 0; i < input.Length; i++)
        {
            var difference = input[i] - mean;
            variance += difference * difference;
        }
        var inverse = 1f / MathF.Sqrt(variance / input.Length + 1e-5f);
        for (var i = 0; i < input.Length; i++)
            output[i] = (input[i] - mean) * inverse * weight[i] + bias[i];
    }

    private static void GroupNorm(Span<float> values, int heads, int size)
    {
        for (var head = 0; head < heads; head++)
        {
            var offset = head * size;
            float mean = 0;
            for (var i = 0; i < size; i++) mean += values[offset + i];
            mean /= size;
            float variance = 0;
            for (var i = 0; i < size; i++)
            {
                var difference = values[offset + i] - mean;
                variance += difference * difference;
            }
            var inverse = 1f / MathF.Sqrt(variance / size + 64e-5f);
            for (var i = 0; i < size; i++)
                values[offset + i] = (values[offset + i] - mean) * inverse;
        }
    }
}
