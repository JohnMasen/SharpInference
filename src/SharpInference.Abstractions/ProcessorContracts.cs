namespace SharpInference;

[Flags]
public enum ProcessorInputModality
{
    None = 0,
    Text = 1,
    Image = 2,
    Audio = 4,
}

[Flags]
public enum ProcessorOutputModality
{
    None = 0,
    Text = 1,
    Tensor = 2,
}

public enum ProcessorPixelFormat
{
    Gray8,
    Rgb24,
    Bgr24,
    Rgba32,
    Bgra32,
}

public enum ProcessorAudioSampleFormat
{
    Signed16,
    Float32,
}

public sealed record ProcessorTextCapabilities
{
    public ProcessorTextCapabilities(int maximumContextLength, int maximumBatchSize = 1)
    {
        if (maximumContextLength <= 0) throw new ArgumentOutOfRangeException(nameof(maximumContextLength));
        if (maximumBatchSize <= 0) throw new ArgumentOutOfRangeException(nameof(maximumBatchSize));
        MaximumContextLength = maximumContextLength;
        MaximumBatchSize = maximumBatchSize;
    }

    public int MaximumContextLength { get; }
    public int MaximumBatchSize { get; }
}

public sealed record ProcessorImageCapabilities
{
    public ProcessorImageCapabilities(
        IReadOnlySet<ProcessorPixelFormat> pixelFormats,
        int maximumWidth,
        int maximumHeight,
        int maximumImageCount,
        int maximumTileCount = 1)
    {
        ArgumentNullException.ThrowIfNull(pixelFormats);
        if (pixelFormats.Count == 0 || pixelFormats.Any(format => !Enum.IsDefined(format)))
            throw new ArgumentException("At least one valid pixel format is required.", nameof(pixelFormats));
        if (maximumWidth <= 0) throw new ArgumentOutOfRangeException(nameof(maximumWidth));
        if (maximumHeight <= 0) throw new ArgumentOutOfRangeException(nameof(maximumHeight));
        if (maximumImageCount <= 0) throw new ArgumentOutOfRangeException(nameof(maximumImageCount));
        if (maximumTileCount <= 0) throw new ArgumentOutOfRangeException(nameof(maximumTileCount));
        PixelFormats = new HashSet<ProcessorPixelFormat>(pixelFormats);
        MaximumWidth = maximumWidth;
        MaximumHeight = maximumHeight;
        MaximumImageCount = maximumImageCount;
        MaximumTileCount = maximumTileCount;
    }

    public IReadOnlySet<ProcessorPixelFormat> PixelFormats { get; }
    public int MaximumWidth { get; }
    public int MaximumHeight { get; }
    public int MaximumImageCount { get; }
    public int MaximumTileCount { get; }
}

public sealed record ProcessorAudioCapabilities
{
    public ProcessorAudioCapabilities(
        IReadOnlySet<ProcessorAudioSampleFormat> sampleFormats,
        IReadOnlySet<int> sampleRates,
        int maximumChannels,
        TimeSpan maximumDuration)
    {
        ArgumentNullException.ThrowIfNull(sampleFormats);
        ArgumentNullException.ThrowIfNull(sampleRates);
        if (sampleFormats.Count == 0 || sampleFormats.Any(format => !Enum.IsDefined(format)))
            throw new ArgumentException("At least one valid sample format is required.", nameof(sampleFormats));
        if (sampleRates.Count == 0 || sampleRates.Any(rate => rate <= 0))
            throw new ArgumentException("At least one positive sample rate is required.", nameof(sampleRates));
        if (maximumChannels <= 0) throw new ArgumentOutOfRangeException(nameof(maximumChannels));
        if (maximumDuration <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(maximumDuration));
        SampleFormats = new HashSet<ProcessorAudioSampleFormat>(sampleFormats);
        SampleRates = new HashSet<int>(sampleRates);
        MaximumChannels = maximumChannels;
        MaximumDuration = maximumDuration;
    }

    public IReadOnlySet<ProcessorAudioSampleFormat> SampleFormats { get; }
    public IReadOnlySet<int> SampleRates { get; }
    public int MaximumChannels { get; }
    public TimeSpan MaximumDuration { get; }
}

public sealed record ProcessorExecutionCapabilities
{
    public ProcessorExecutionCapabilities(
        int maximumConcurrentSessions,
        int maximumParallelComponents,
        IReadOnlySet<string> storageDomains)
    {
        if (maximumConcurrentSessions <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumConcurrentSessions));
        if (maximumParallelComponents <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumParallelComponents));
        ArgumentNullException.ThrowIfNull(storageDomains);
        if (storageDomains.Any(string.IsNullOrWhiteSpace))
            throw new ArgumentException("Storage domain names must be non-empty.", nameof(storageDomains));
        MaximumConcurrentSessions = maximumConcurrentSessions;
        MaximumParallelComponents = maximumParallelComponents;
        StorageDomains = new HashSet<string>(storageDomains, StringComparer.Ordinal);
    }

    public int MaximumConcurrentSessions { get; }
    public int MaximumParallelComponents { get; }
    public IReadOnlySet<string> StorageDomains { get; }
}

public sealed class ProcessorCapabilities
{
    public ProcessorCapabilities(
        IEnumerable<ProcessorInputModality> inputCombinations,
        ProcessorOutputModality outputModalities,
        ProcessorTextCapabilities? text = null,
        ProcessorImageCapabilities? image = null,
        ProcessorAudioCapabilities? audio = null,
        ProcessorExecutionCapabilities? execution = null)
    {
        ArgumentNullException.ThrowIfNull(inputCombinations);
        var combinations = inputCombinations.ToArray();
        if (combinations.Length == 0 ||
            combinations.Any(value => value == ProcessorInputModality.None || !Enum.IsDefined(value & ~AllInputs)) ||
            combinations.Distinct().Count() != combinations.Length)
        {
            throw new ArgumentException("Input combinations must be unique, non-empty modality sets.",
                nameof(inputCombinations));
        }
        if (outputModalities == ProcessorOutputModality.None ||
            !Enum.IsDefined(outputModalities & ~AllOutputs))
        {
            throw new ArgumentOutOfRangeException(nameof(outputModalities));
        }

        var supported = combinations.Aggregate(ProcessorInputModality.None, static (value, next) => value | next);
        if (supported.HasFlag(ProcessorInputModality.Text) != (text is not null) ||
            supported.HasFlag(ProcessorInputModality.Image) != (image is not null) ||
            supported.HasFlag(ProcessorInputModality.Audio) != (audio is not null))
        {
            throw new ArgumentException("Every supported input modality must have matching capabilities.");
        }

        InputCombinations = Array.AsReadOnly(combinations);
        OutputModalities = outputModalities;
        Text = text;
        Image = image;
        Audio = audio;
        Execution = execution ?? new ProcessorExecutionCapabilities(1, 1, new HashSet<string>());
    }

    private const ProcessorInputModality AllInputs =
        ProcessorInputModality.Text | ProcessorInputModality.Image | ProcessorInputModality.Audio;
    private const ProcessorOutputModality AllOutputs =
        ProcessorOutputModality.Text | ProcessorOutputModality.Tensor;

    public IReadOnlyList<ProcessorInputModality> InputCombinations { get; }
    public ProcessorOutputModality OutputModalities { get; }
    public ProcessorTextCapabilities? Text { get; }
    public ProcessorImageCapabilities? Image { get; }
    public ProcessorAudioCapabilities? Audio { get; }
    public ProcessorExecutionCapabilities Execution { get; }
    public bool Supports(ProcessorInputModality combination) => InputCombinations.Contains(combination);
}

public abstract record ProcessorInputPart;

public sealed record TextInputPart : ProcessorInputPart
{
    public TextInputPart(string text)
    {
        Text = text ?? throw new ArgumentNullException(nameof(text));
    }

    public string Text { get; }
}

public sealed record ImageInputPart : ProcessorInputPart
{
    public ImageInputPart(
        ReadOnlyMemory<byte> pixels,
        int width,
        int height,
        int stride,
        ProcessorPixelFormat pixelFormat)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (!Enum.IsDefined(pixelFormat)) throw new ArgumentOutOfRangeException(nameof(pixelFormat));
        var minimumStride = checked(width * BytesPerPixel(pixelFormat));
        if (stride < minimumStride)
            throw new ArgumentException("Image stride is smaller than the packed row size.", nameof(stride));
        if (pixels.Length < checked(stride * height))
            throw new ArgumentException("The pixel buffer does not contain every image row.", nameof(pixels));

        Pixels = pixels;
        Width = width;
        Height = height;
        Stride = stride;
        PixelFormat = pixelFormat;
    }

    public ReadOnlyMemory<byte> Pixels { get; }
    public int Width { get; }
    public int Height { get; }
    public int Stride { get; }
    public ProcessorPixelFormat PixelFormat { get; }

    private static int BytesPerPixel(ProcessorPixelFormat format) => format switch
    {
        ProcessorPixelFormat.Gray8 => 1,
        ProcessorPixelFormat.Rgb24 or ProcessorPixelFormat.Bgr24 => 3,
        ProcessorPixelFormat.Rgba32 or ProcessorPixelFormat.Bgra32 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };
}

public sealed record AudioInputPart : ProcessorInputPart
{
    public AudioInputPart(
        ReadOnlyMemory<byte> samples,
        ProcessorAudioSampleFormat sampleFormat,
        int sampleRate,
        int channels)
    {
        if (!Enum.IsDefined(sampleFormat)) throw new ArgumentOutOfRangeException(nameof(sampleFormat));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        var frameSize = checked(channels * (sampleFormat == ProcessorAudioSampleFormat.Signed16 ? 2 : 4));
        if (samples.Length == 0 || samples.Length % frameSize != 0)
            throw new ArgumentException("PCM data must contain a whole number of non-empty sample frames.",
                nameof(samples));

        Samples = samples;
        SampleFormat = sampleFormat;
        SampleRate = sampleRate;
        Channels = channels;
    }

    public ReadOnlyMemory<byte> Samples { get; }
    public ProcessorAudioSampleFormat SampleFormat { get; }
    public int SampleRate { get; }
    public int Channels { get; }
    public int FrameCount => Samples.Length /
        (Channels * (SampleFormat == ProcessorAudioSampleFormat.Signed16 ? 2 : 4));
}

public sealed class ProcessorInput
{
    public ProcessorInput(IEnumerable<ProcessorInputPart> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        var values = parts.ToArray();
        if (values.Length == 0 || values.Any(static part => part is null))
            throw new ArgumentException("Processor input requires at least one ordered part.", nameof(parts));
        Parts = Array.AsReadOnly(values);
        Modalities = values.Aggregate(ProcessorInputModality.None, static (result, part) => result | (part switch
        {
            TextInputPart => ProcessorInputModality.Text,
            ImageInputPart => ProcessorInputModality.Image,
            AudioInputPart => ProcessorInputModality.Audio,
            _ => throw new ArgumentException($"Unknown processor input part '{part.GetType().Name}'."),
        }));
    }

    public IReadOnlyList<ProcessorInputPart> Parts { get; }
    public ProcessorInputModality Modalities { get; }
}

public interface IProcessor : IDisposable
{
    ProcessorCapabilities Capabilities { get; }
    IProcessorSession CreateSession();
}

public interface IProcessorSession : IDisposable
{
    IProcessor Processor { get; }
    void Reset();
}
