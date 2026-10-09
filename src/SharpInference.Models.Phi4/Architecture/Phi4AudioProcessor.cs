namespace SharpInference.Architectures.Phi4;

public sealed record Phi4AudioFeatures(float[] Values, int FrameCount, int FeatureCount, int EmbedSize)
{
    public ReadOnlySpan<float> GetFrame(int index)
    {
        if ((uint)index >= (uint)FrameCount)
            throw new ArgumentOutOfRangeException(nameof(index));
        return Values.AsSpan(index * FeatureCount, FeatureCount);
    }
}

public sealed class Phi4AudioProcessor
{
    private const int TargetSampleRate = 16000;
    private const int FftSize = 512;
    private const int WindowLength = 400;
    private const int HopLength = 160;
    private const int MelCount = 80;
    private const float Preemphasis = 0.97f;
    private static readonly (double[] Real, double[] Imaginary) Dft = CreateDft();
    private readonly double[] hamming = CreateHamming(WindowLength);
    private readonly float[] mel = CreateMelBank();

    public Phi4AudioFeatures Process(
        ReadOnlySpan<float> interleavedSamples,
        int sampleRate,
        int channels = 1)
    {
        if (sampleRate < TargetSampleRate || sampleRate >= TargetSampleRate * 2)
            throw new NotSupportedException(
                $"Phi-4 currently accepts PCM rates from {TargetSampleRate} through {TargetSampleRate * 2 - 1} Hz; received {sampleRate} Hz.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels));
        if (interleavedSamples.Length % channels != 0)
            throw new ArgumentException("Interleaved PCM length is not divisible by channel count.", nameof(interleavedSamples));

        var monoLength = interleavedSamples.Length / channels;
        if (monoLength < WindowLength)
            throw new ArgumentException($"Audio requires at least {WindowLength} samples.", nameof(interleavedSamples));
        var mono = new float[monoLength];
        if (channels == 1)
        {
            interleavedSamples.CopyTo(mono);
        }
        else
        {
            for (var frame = 0; frame < mono.Length; frame++)
            {
                var sum = 0f;
                for (var channel = 0; channel < channels; channel++)
                    sum += interleavedSamples[frame * channels + channel];
                mono[frame] = sum / channels;
            }
        }

        var frameCount = (mono.Length - WindowLength) / HopLength + 1;
        var features = new float[checked(frameCount * MelCount)];
        var windowed = new double[WindowLength];
        var power = new float[FftSize / 2 + 1];
        for (var frame = 0; frame < frameCount; frame++)
        {
            var offset = frame * HopLength;
            for (var index = 0; index < WindowLength; index++)
            {
                var previous = index == 0 ? mono[offset] : mono[offset + index - 1];
                var emphasized = (mono[offset + index] - Preemphasis * previous) * 32768f;
                windowed[index] = emphasized * hamming[index];
            }
            for (var bin = 0; bin < power.Length; bin++)
            {
                var real = 0.0;
                var imaginary = 0.0;
                var dftOffset = bin * WindowLength;
                for (var index = 0; index < WindowLength; index++)
                {
                    real += windowed[index] * Dft.Real[dftOffset + index];
                    imaginary += windowed[index] * Dft.Imaginary[dftOffset + index];
                }
                var magnitude = (float)Math.Sqrt(real * real + imaginary * imaginary);
                power[bin] = magnitude * magnitude;
            }
            for (var filter = 0; filter < MelCount; filter++)
            {
                var sum = 0f;
                for (var bin = 0; bin < power.Length; bin++)
                    sum += power[bin] * mel[bin * MelCount + filter];
                features[frame * MelCount + filter] = MathF.Log(MathF.Max(1f, sum));
            }
        }

        return new Phi4AudioFeatures(
            features,
            frameCount,
            MelCount,
            DivideRoundUp(frameCount, 8));
    }

    private static int DivideRoundUp(int value, int divisor) =>
        checked((value + divisor - 1) / divisor);

    private static double[] CreateHamming(int length)
    {
        var result = new double[length];
        for (var index = 0; index < result.Length; index++)
            result[index] = 0.54 - 0.46 * Math.Cos(2 * Math.PI * index / (length - 1));
        return result;
    }

    private static float[] CreateMelBank()
    {
        const int sampleRate = TargetSampleRate;
        const double maximumFrequency = 7690;
        var result = new float[(FftSize / 2 + 1) * MelCount];
        var lowMel = Mel(0);
        var highMel = Mel(maximumFrequency);
        var centers = new double[MelCount + 2];
        for (var index = 0; index < centers.Length; index++)
            centers[index] = lowMel + (highMel - lowMel) * index / (MelCount + 1);
        var width = (highMel - lowMel) / (MelCount + 1);
        var lowBin = FrequencyToBin(0) + 1;
        var highBin = Math.Max(FrequencyToBin(maximumFrequency), lowBin);
        for (var filter = 0; filter < MelCount; filter++)
        {
            var left = centers[filter];
            var center = centers[filter + 1];
            var right = centers[filter + 2];
            for (var bin = lowBin; bin < highBin; bin++)
            {
                var mel = BinToMel(bin);
                if (left < mel && mel < right)
                    result[bin * MelCount + filter] = (float)(1 - Math.Abs(center - mel) / width);
            }
        }
        return result;

        static double Mel(double frequency) => 1127.0 * Math.Log(1.0 + frequency / 700.0);
        static double BinToMel(int bin) =>
            1127.0 * Math.Log(1.0 + bin * sampleRate / (FftSize * 700.0));
        static int FrequencyToBin(double frequency) =>
            (int)(frequency * FftSize / sampleRate + 0.5);
    }

    private static (double[] Real, double[] Imaginary) CreateDft()
    {
        var real = new double[(FftSize / 2 + 1) * WindowLength];
        var imaginary = new double[real.Length];
        for (var bin = 0; bin <= FftSize / 2; bin++)
        for (var sample = 0; sample < WindowLength; sample++)
        {
            var angle = -2 * Math.PI * bin * sample / FftSize;
            real[bin * WindowLength + sample] = Math.Cos(angle);
            imaginary[bin * WindowLength + sample] = Math.Sin(angle);
        }
        return (real, imaginary);
    }
}
