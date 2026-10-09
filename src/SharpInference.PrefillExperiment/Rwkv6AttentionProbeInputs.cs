using SharpInference;
using System.Numerics.Tensors;

namespace SharpInference.PrefillExperiment;

internal sealed record Rwkv6AttentionProbeInputs(
    int HeadCount,
    int HeadSize,
    Half[] KeyWeights,
    Half[] ValueWeights,
    Half[] ReceptanceWeights,
    float[] KeyActivations,
    float[] ValueActivations,
    float[] ReceptanceActivations,
    float[] Decays,
    float[] TimeFirst,
    float[] InitialState)
{
    public float[] Activations => KeyActivations;

    public static Rwkv6AttentionProbeInputs FromFirstLayer(GgmlModelFile model, int tokens)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (tokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(tokens));
        var width = model.EmbeddingSize;
        var first = model.GetRequired("blocks.0.att.time_faaaa");
        if (first.Dimensions.Count != 3 || first.Dimensions[0] != 1 ||
            first.Dimensions[1] <= 0 || first.Dimensions[2] <= 0 ||
            checked(first.Dimensions[1] * first.Dimensions[2]) != width)
            throw new InvalidDataException("The first RWKV-6 layer has invalid head dimensions.");

        Half[] Weights(string name)
        {
            var tensor = model.GetRequired($"blocks.0.att.{name}.weight");
            if (tensor.Dimensions.Count != 2 ||
                tensor.Dimensions[0] != width || tensor.Dimensions[1] != width)
                throw new InvalidDataException($"The attention projection '{tensor.Name}' must be {width}x{width}.");
            return tensor.DataType == TensorDataType.Float16
                ? tensor.HalfValues.ToArray()
                : tensor.FloatValues.ToArray().Select(value => (Half)value).ToArray();
        }

        var embedding = model.GetRequired("emb.weight");
        if (embedding.Dimensions.Count != 2 ||
            embedding.Dimensions[0] != width ||
            embedding.Dimensions[1] != model.VocabularySize)
            throw new InvalidDataException("The RWKV-6 embedding tensor has invalid dimensions.");
        var ln0Weight = model.GetRequired("blocks.0.ln0.weight").FloatValues;
        var ln0Bias = model.GetRequired("blocks.0.ln0.bias").FloatValues;
        var ln1Weight = model.GetRequired("blocks.0.ln1.weight").FloatValues;
        var ln1Bias = model.GetRequired("blocks.0.ln1.bias").FloatValues;
        var maaX = model.GetRequired("blocks.0.att.time_maa_x").FloatValues;
        var maaW1 = model.GetRequired("blocks.0.att.time_maa_w1");
        var maaW2 = model.GetRequired("blocks.0.att.time_maa_w2");
        var mixK = model.GetRequired("blocks.0.att.time_maa_k").FloatValues;
        var mixV = model.GetRequired("blocks.0.att.time_maa_v").FloatValues;
        var mixR = model.GetRequired("blocks.0.att.time_maa_r").FloatValues;
        var mixW = model.GetRequired("blocks.0.att.time_maa_w").FloatValues;
        var decayW1 = model.GetRequired("blocks.0.att.time_decay_w1");
        var decayW2 = model.GetRequired("blocks.0.att.time_decay_w2");
        var baseDecay = model.GetRequired("blocks.0.att.time_decay").FloatValues;
        var timeFirst = first.FloatValues.ToArray();
        if (ln0Weight.Length != width || ln0Bias.Length != width ||
            ln1Weight.Length != width || ln1Bias.Length != width ||
            baseDecay.Length != width || timeFirst.Length != width ||
            maaX.Length != width || mixK.Length != width || mixV.Length != width ||
            mixR.Length != width || mixW.Length != width)
            throw new InvalidDataException("First-layer normalization or mixing vectors have invalid dimensions.");
        if (maaW1.Dimensions.Count != 2 || maaW1.Dimensions[0] != width ||
            maaW1.Dimensions[1] % 5 != 0 || maaW2.Dimensions.Count != 3 ||
            maaW2.Dimensions[0] * 5 != maaW1.Dimensions[1] ||
            maaW2.Dimensions[1] != width || maaW2.Dimensions[2] != 5)
            throw new InvalidDataException("First-layer RWKV-6 dynamic time-mix tensors have invalid dimensions.");
        if (decayW1.Dimensions.Count != 2 || decayW1.Dimensions[0] != width ||
            decayW2.Dimensions.Count != 2 ||
            decayW2.Dimensions[0] != decayW1.Dimensions[1] ||
            decayW2.Dimensions[1] != width)
            throw new InvalidDataException("First-layer RWKV-6 decay tensors have invalid dimensions.");

        var keyInputs = new float[checked(tokens * width)];
        var valueInputs = new float[keyInputs.Length];
        var receptanceInputs = new float[keyInputs.Length];
        var decays = new float[keyInputs.Length];
        var raw = new float[width];
        var x = new float[width];
        var norm = new float[width];
        var previous = new float[width];
        var mixedInput = new float[width];
        var decayInput = new float[width];
        var maa = new float[maaW1.Dimensions[1]];
        var mixed = new float[checked(5 * width)];
        var hidden = new float[decayW1.Dimensions[1]];
        var decayOutput = new float[width];
        for (var token = 0; token < tokens; token++)
        {
            var id = token % model.VocabularySize;
            var offset = checked(id * width);
            if (embedding.DataType == TensorDataType.Float16)
            {
                var values = embedding.HalfValues.Slice(offset, width);
                for (var i = 0; i < width; i++) raw[i] = (float)values[i];
            }
            else
                embedding.FloatValues.Slice(offset, width).CopyTo(raw);
            Normalize(raw, ln0Weight, ln0Bias, x);
            Normalize(x, ln1Weight, ln1Bias, norm);
            for (var i = 0; i < width; i++)
                mixedInput[i] = norm[i] + (previous[i] - norm[i]) * maaX[i];
            MatVec(maaW1, mixedInput, maa);
            for (var i = 0; i < maa.Length; i++) maa[i] = MathF.Tanh(maa[i]);
            var maaHidden = maa.Length / 5;
            var w2 = maaW2.FloatValues;
            for (var group = 0; group < 5; group++)
            for (var i = 0; i < width; i++)
                mixed[group * width + i] = TensorPrimitives.Dot(
                    w2.Slice((group * width + i) * maaHidden, maaHidden),
                    maa.AsSpan(group * maaHidden, maaHidden));
            for (var i = 0; i < width; i++)
            {
                var delta = previous[i] - norm[i];
                keyInputs[token * width + i] = norm[i] + delta * (mixed[width + i] + mixK[i]);
                valueInputs[token * width + i] = norm[i] + delta * (mixed[2 * width + i] + mixV[i]);
                receptanceInputs[token * width + i] = norm[i] + delta * (mixed[3 * width + i] + mixR[i]);
                decayInput[i] = norm[i] + delta * (mixed[i] + mixW[i]);
            }
            MatVec(decayW1, decayInput, hidden);
            for (var i = 0; i < hidden.Length; i++) hidden[i] = MathF.Tanh(hidden[i]);
            MatVec(decayW2, hidden, decayOutput);
            for (var i = 0; i < width; i++)
                decays[token * width + i] = MathF.Exp(-MathF.Exp(decayOutput[i] + baseDecay[i]));
            norm.CopyTo(previous, 0);
        }

        var state = new float[checked(width * first.Dimensions[1])];
        var random = new Random(1701);
        for (var i = 0; i < state.Length; i++)
            state[i] = (float)(random.NextDouble() * 0.002 - 0.001);
        return new Rwkv6AttentionProbeInputs(
            first.Dimensions[2], first.Dimensions[1],
            Weights("key"), Weights("value"), Weights("receptance"),
            keyInputs, valueInputs, receptanceInputs, decays, timeFirst, state);
    }

    private static void MatVec(IModelTensor tensor, ReadOnlySpan<float> input, Span<float> output)
    {
        if (tensor.Dimensions.Count != 2 ||
            tensor.Dimensions[0] != input.Length || tensor.Dimensions[1] != output.Length)
            throw new InvalidDataException($"'{tensor.Name}' cannot project [{input.Length}] to [{output.Length}].");
        var weights = tensor.FloatValues;
        for (var row = 0; row < output.Length; row++)
            output[row] = TensorPrimitives.Dot(weights.Slice(row * input.Length, input.Length), input);
    }

    private static void Normalize(ReadOnlySpan<float> input, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> output)
    {
        var mean = 0f;
        for (var i = 0; i < input.Length; i++) mean += input[i];
        mean /= input.Length;
        var variance = 0f;
        for (var i = 0; i < input.Length; i++)
        {
            var delta = input[i] - mean;
            variance += delta * delta;
        }
        var inverse = 1f / MathF.Sqrt(variance / input.Length + 1e-5f);
        for (var i = 0; i < input.Length; i++)
            output[i] = (input[i] - mean) * inverse * weight[i] + bias[i];
    }
}
