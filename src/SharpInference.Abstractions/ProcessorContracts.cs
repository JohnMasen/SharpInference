namespace SharpInference;

/// <summary>Identifies the input modalities present in a processor request.</summary>
[Flags]
public enum ProcessorInputModality
{
    /// <summary>No input modality.</summary>
    None = 0,
    /// <summary>Text input.</summary>
    Text = 1,
    /// <summary>Pixel image input.</summary>
    Image = 2,
    /// <summary>PCM audio input.</summary>
    Audio = 4,
    /// <summary>Named raw tensor input.</summary>
    Tensor = 8,
}

/// <summary>Identifies the output modalities supported by a processor.</summary>
[Flags]
public enum ProcessorOutputModality
{
    /// <summary>No output modality.</summary>
    None = 0,
    /// <summary>Generated text output.</summary>
    Text = 1,
    /// <summary>Named tensor output.</summary>
    Tensor = 2,
}

/// <summary>Specifies the byte channel order of an image pixel.</summary>
public enum ProcessorPixelFormat
{
    /// <summary>One 8-bit grayscale channel.</summary>
    Gray8,
    /// <summary>8-bit red, green and blue channels.</summary>
    Rgb24,
    /// <summary>8-bit blue, green and red channels.</summary>
    Bgr24,
    /// <summary>8-bit red, green, blue and alpha channels.</summary>
    Rgba32,
    /// <summary>8-bit blue, green, red and alpha channels.</summary>
    Bgra32,
}

/// <summary>Specifies the storage format of an interleaved PCM sample.</summary>
public enum ProcessorAudioSampleFormat
{
    /// <summary>Signed 16-bit integer samples.</summary>
    Signed16,
    /// <summary>Single-precision floating-point samples.</summary>
    Float32,
}

/// <summary>Declares the processor's text context and batch limits.</summary>
public sealed record ProcessorTextCapabilities
{
    /// <summary>Creates positive text context and batch limits.</summary>
    /// <param name="maximumContextLength">The maximum number of context tokens.</param>
    /// <param name="maximumBatchSize">The maximum number of items in a text batch.</param>
    public ProcessorTextCapabilities(int maximumContextLength, int maximumBatchSize = 1)
    {
        if (maximumContextLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumContextLength));
        }

        if (maximumBatchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumBatchSize));
        }

        MaximumContextLength = maximumContextLength;
        MaximumBatchSize = maximumBatchSize;
    }

    /// <summary>Gets the maximum number of context tokens.</summary>
    public int MaximumContextLength { get; }
    /// <summary>Gets the maximum text batch size.</summary>
    public int MaximumBatchSize { get; }
}

/// <summary>Declares supported image formats and dimensional limits.</summary>
public sealed record ProcessorImageCapabilities
{
    /// <summary>Creates image limits and validates the supported formats.</summary>
    /// <param name="pixelFormats">The accepted pixel channel formats.</param>
    /// <param name="maximumWidth">The maximum image width in pixels.</param>
    /// <param name="maximumHeight">The maximum image height in pixels.</param>
    /// <param name="maximumImageCount">The maximum images in one request.</param>
    /// <param name="maximumTileCount">The maximum tiles used to represent an image.</param>
    public ProcessorImageCapabilities(
        IReadOnlySet<ProcessorPixelFormat> pixelFormats,
        int maximumWidth,
        int maximumHeight,
        int maximumImageCount,
        int maximumTileCount = 1)
    {
        ArgumentNullException.ThrowIfNull(pixelFormats);
        if (pixelFormats.Count == 0 || pixelFormats.Any(format => !Enum.IsDefined(format)))
        {
            throw new ArgumentException("At least one valid pixel format is required.", nameof(pixelFormats));
        }

        if (maximumWidth <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumWidth));
        }

        if (maximumHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumHeight));
        }

        if (maximumImageCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumImageCount));
        }

        if (maximumTileCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumTileCount));
        }

        PixelFormats = new HashSet<ProcessorPixelFormat>(pixelFormats);
        MaximumWidth = maximumWidth;
        MaximumHeight = maximumHeight;
        MaximumImageCount = maximumImageCount;
        MaximumTileCount = maximumTileCount;
    }

    /// <summary>Gets the accepted pixel formats.</summary>
    public IReadOnlySet<ProcessorPixelFormat> PixelFormats { get; }
    /// <summary>Gets the maximum image width in pixels.</summary>
    public int MaximumWidth { get; }
    /// <summary>Gets the maximum image height in pixels.</summary>
    public int MaximumHeight { get; }
    /// <summary>Gets the maximum images in a request.</summary>
    public int MaximumImageCount { get; }
    /// <summary>Gets the maximum tile count.</summary>
    public int MaximumTileCount { get; }
}

/// <summary>Declares accepted PCM formats, rates and audio limits.</summary>
public sealed record ProcessorAudioCapabilities
{
    /// <summary>Creates and validates the supported audio formats and limits.</summary>
    /// <param name="sampleFormats">The accepted PCM sample formats.</param>
    /// <param name="sampleRates">The accepted sample rates in hertz.</param>
    /// <param name="maximumChannels">The maximum interleaved channel count.</param>
    /// <param name="maximumDuration">The maximum duration of an audio input.</param>
    public ProcessorAudioCapabilities(
        IReadOnlySet<ProcessorAudioSampleFormat> sampleFormats,
        IReadOnlySet<int> sampleRates,
        int maximumChannels,
        TimeSpan maximumDuration)
    {
        ArgumentNullException.ThrowIfNull(sampleFormats);
        ArgumentNullException.ThrowIfNull(sampleRates);
        if (sampleFormats.Count == 0 || sampleFormats.Any(format => !Enum.IsDefined(format)))
        {
            throw new ArgumentException("At least one valid sample format is required.", nameof(sampleFormats));
        }

        if (sampleRates.Count == 0 || sampleRates.Any(rate => rate <= 0))
        {
            throw new ArgumentException("At least one positive sample rate is required.", nameof(sampleRates));
        }

        if (maximumChannels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumChannels));
        }

        if (maximumDuration <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumDuration));
        }

        SampleFormats = new HashSet<ProcessorAudioSampleFormat>(sampleFormats);
        SampleRates = new HashSet<int>(sampleRates);
        MaximumChannels = maximumChannels;
        MaximumDuration = maximumDuration;
    }

    /// <summary>Gets the accepted PCM sample formats.</summary>
    public IReadOnlySet<ProcessorAudioSampleFormat> SampleFormats { get; }
    /// <summary>Gets accepted sample rates in hertz.</summary>
    public IReadOnlySet<int> SampleRates { get; }
    /// <summary>Gets the maximum channel count.</summary>
    public int MaximumChannels { get; }
    /// <summary>Gets the maximum input duration.</summary>
    public TimeSpan MaximumDuration { get; }
}

/// <summary>Declares concurrency limits and supported execution storage domains.</summary>
public sealed record ProcessorExecutionCapabilities
{
    /// <summary>Creates execution limits and validates storage domain names.</summary>
    /// <param name="maximumConcurrentSessions">The maximum concurrently active sessions.</param>
    /// <param name="maximumParallelComponents">The maximum components that can execute in parallel.</param>
    /// <param name="storageDomains">The supported host or device storage domains.</param>
    /// <param name="requiresResidentSessionAdmission">Whether resource-retaining sessions require bounded admission.</param>
    public ProcessorExecutionCapabilities(
        int maximumConcurrentSessions,
        int maximumParallelComponents,
        IReadOnlySet<string> storageDomains,
        bool requiresResidentSessionAdmission = false)
    {
        if (maximumConcurrentSessions <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumConcurrentSessions));
        }

        if (maximumParallelComponents <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumParallelComponents));
        }

        ArgumentNullException.ThrowIfNull(storageDomains);
        if (storageDomains.Any(string.IsNullOrWhiteSpace))
        {
            throw new ArgumentException("Storage domain names must be non-empty.", nameof(storageDomains));
        }

        MaximumConcurrentSessions = maximumConcurrentSessions;
        MaximumParallelComponents = maximumParallelComponents;
        StorageDomains = new HashSet<string>(storageDomains, StringComparer.Ordinal);
        RequiresResidentSessionAdmission = requiresResidentSessionAdmission;
    }

    /// <summary>Gets the maximum concurrently active sessions.</summary>
    public int MaximumConcurrentSessions { get; }
    /// <summary>Gets the maximum components that can execute in parallel.</summary>
    public int MaximumParallelComponents { get; }
    /// <summary>Gets the supported execution storage domains.</summary>
    public IReadOnlySet<string> StorageDomains { get; }
    /// <summary>Requires bounded admission before constructing sessions that retain execution resources.</summary>
    public bool RequiresResidentSessionAdmission { get; }
}

/// <summary>Describes valid input combinations, outputs and execution limits.</summary>
public sealed class ProcessorCapabilities
{
    /// <summary>Creates a capability contract with matching modality-specific limits.</summary>
    /// <param name="inputCombinations">The distinct, non-empty accepted modality combinations.</param>
    /// <param name="outputModalities">The supported output modalities.</param>
    /// <param name="text">Limits required when text inputs are accepted.</param>
    /// <param name="image">Limits required when image inputs are accepted.</param>
    /// <param name="audio">Limits required when audio inputs are accepted.</param>
    /// <param name="execution">Execution limits, or conservative single-session defaults.</param>
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
        ProcessorInputModality.Text | ProcessorInputModality.Image | ProcessorInputModality.Audio | ProcessorInputModality.Tensor;
    private const ProcessorOutputModality AllOutputs =
        ProcessorOutputModality.Text | ProcessorOutputModality.Tensor;

    /// <summary>Gets the accepted complete modality combinations.</summary>
    public IReadOnlyList<ProcessorInputModality> InputCombinations { get; }
    /// <summary>Gets the supported output modalities.</summary>
    public ProcessorOutputModality OutputModalities { get; }
    /// <summary>Gets the text limits, or null when text input is unsupported.</summary>
    public ProcessorTextCapabilities? Text { get; }
    /// <summary>Gets the image limits, or null when image input is unsupported.</summary>
    public ProcessorImageCapabilities? Image { get; }
    /// <summary>Gets the audio limits, or null when audio input is unsupported.</summary>
    public ProcessorAudioCapabilities? Audio { get; }
    /// <summary>Gets the concurrency and storage limits.</summary>
    public ProcessorExecutionCapabilities Execution { get; }
    /// <summary>Checks whether a complete input modality combination is accepted.</summary>
    /// <param name="combination">The modalities present in a request.</param>
    /// <returns>Whether the combination is explicitly supported.</returns>
    public bool Supports(ProcessorInputModality combination) => InputCombinations.Contains(combination);
}

/// <summary>Represents one ordered part of a multimodal processor input.</summary>
public abstract record ProcessorInputPart;

/// <summary>Supplies raw bytes for a named tensor input resource.</summary>
public sealed record TensorInputPart : ProcessorInputPart
{
    /// <summary>Creates a named tensor input without copying its bytes.</summary>
    /// <param name="resource">The non-empty graph input resource name.</param>
    /// <param name="data">The tensor bytes to retain for the request.</param>
    public TensorInputPart(string resource, ReadOnlyMemory<byte> data)
    {
        Resource = string.IsNullOrWhiteSpace(resource)
            ? throw new ArgumentException("A tensor resource name is required.", nameof(resource))
            : resource;
        Data = data;
    }

    /// <summary>Gets the graph input resource name.</summary>
    public string Resource { get; }
    /// <summary>Gets the supplied tensor bytes.</summary>
    public ReadOnlyMemory<byte> Data { get; }
}

/// <summary>Supplies one text segment in an ordered request.</summary>
public sealed record TextInputPart : ProcessorInputPart
{
    /// <summary>Creates a text segment.</summary>
    /// <param name="text">The non-null input text.</param>
    public TextInputPart(string text)
    {
        Text = text ?? throw new ArgumentNullException(nameof(text));
    }

    /// <summary>Gets the input text.</summary>
    public string Text { get; }
}

/// <summary>Supplies a validated image buffer and its row layout.</summary>
public sealed record ImageInputPart : ProcessorInputPart
{
    /// <summary>Creates an image input and validates row stride and buffer capacity.</summary>
    /// <param name="pixels">The image pixel bytes, retained without copying.</param>
    /// <param name="width">The width in pixels.</param>
    /// <param name="height">The height in pixels.</param>
    /// <param name="stride">The number of bytes between row starts.</param>
    /// <param name="pixelFormat">The channel order of each pixel.</param>
    public ImageInputPart(
        ReadOnlyMemory<byte> pixels,
        int width,
        int height,
        int stride,
        ProcessorPixelFormat pixelFormat)
    {
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }

        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }

        if (stride <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stride));
        }

        if (!Enum.IsDefined(pixelFormat))
        {
            throw new ArgumentOutOfRangeException(nameof(pixelFormat));
        }

        var minimumStride = checked(width * BytesPerPixel(pixelFormat));
        if (stride < minimumStride)
        {
            throw new ArgumentException("Image stride is smaller than the packed row size.", nameof(stride));
        }

        if (pixels.Length < checked(stride * height))
        {
            throw new ArgumentException("The pixel buffer does not contain every image row.", nameof(pixels));
        }

        Pixels = pixels;
        Width = width;
        Height = height;
        Stride = stride;
        PixelFormat = pixelFormat;
    }

    /// <summary>Gets the supplied pixel bytes.</summary>
    public ReadOnlyMemory<byte> Pixels { get; }
    /// <summary>Gets the width in pixels.</summary>
    public int Width { get; }
    /// <summary>Gets the height in pixels.</summary>
    public int Height { get; }
    /// <summary>Gets the row stride in bytes.</summary>
    public int Stride { get; }
    /// <summary>Gets the pixel channel format.</summary>
    public ProcessorPixelFormat PixelFormat { get; }

    private static int BytesPerPixel(ProcessorPixelFormat format) => format switch
    {
        ProcessorPixelFormat.Gray8 => 1,
        ProcessorPixelFormat.Rgb24 or ProcessorPixelFormat.Bgr24 => 3,
        ProcessorPixelFormat.Rgba32 or ProcessorPixelFormat.Bgra32 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };
}

/// <summary>Supplies non-empty interleaved PCM audio frames.</summary>
public sealed record AudioInputPart : ProcessorInputPart
{
    /// <summary>Creates audio input and validates complete PCM frames.</summary>
    /// <param name="samples">The interleaved PCM sample bytes.</param>
    /// <param name="sampleFormat">The sample storage format.</param>
    /// <param name="sampleRate">The sample rate in hertz.</param>
    /// <param name="channels">The number of interleaved channels.</param>
    public AudioInputPart(
        ReadOnlyMemory<byte> samples,
        ProcessorAudioSampleFormat sampleFormat,
        int sampleRate,
        int channels)
    {
        if (!Enum.IsDefined(sampleFormat))
        {
            throw new ArgumentOutOfRangeException(nameof(sampleFormat));
        }

        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        }

        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels));
        }

        var frameSize = checked(channels * (sampleFormat == ProcessorAudioSampleFormat.Signed16 ? 2 : 4));
        if (samples.Length == 0 || samples.Length % frameSize != 0)
        {
            throw new ArgumentException("PCM data must contain a whole number of non-empty sample frames.",
                nameof(samples));
        }

        Samples = samples;
        SampleFormat = sampleFormat;
        SampleRate = sampleRate;
        Channels = channels;
    }

    /// <summary>Gets the interleaved PCM bytes.</summary>
    public ReadOnlyMemory<byte> Samples { get; }
    /// <summary>Gets the PCM sample format.</summary>
    public ProcessorAudioSampleFormat SampleFormat { get; }
    /// <summary>Gets the sample rate in hertz.</summary>
    public int SampleRate { get; }
    /// <summary>Gets the interleaved channel count.</summary>
    public int Channels { get; }
    /// <summary>Gets the number of complete PCM frames.</summary>
    public int FrameCount => Samples.Length /
        (Channels * (SampleFormat == ProcessorAudioSampleFormat.Signed16 ? 2 : 4));
}

/// <summary>Stores an ordered, non-empty multimodal input request.</summary>
public sealed class ProcessorInput
{
    /// <summary>Creates a request and computes its combined input modalities.</summary>
    /// <param name="parts">The non-null input parts in processing order.</param>
    public ProcessorInput(IEnumerable<ProcessorInputPart> parts)
    {
        ArgumentNullException.ThrowIfNull(parts);
        var values = parts.ToArray();
        if (values.Length == 0 || values.Any(static part => part is null))
        {
            throw new ArgumentException("Processor input requires at least one ordered part.", nameof(parts));
        }

        Parts = Array.AsReadOnly(values);
        Modalities = values.Aggregate(ProcessorInputModality.None, static (result, part) => result | (part switch
        {
            TextInputPart => ProcessorInputModality.Text,
            ImageInputPart => ProcessorInputModality.Image,
            AudioInputPart => ProcessorInputModality.Audio,
            TensorInputPart => ProcessorInputModality.Tensor,
            _ => throw new ArgumentException($"Unknown processor input part '{part.GetType().Name}'."),
        }));
    }

    /// <summary>Gets the input parts in request order.</summary>
    public IReadOnlyList<ProcessorInputPart> Parts { get; }
    /// <summary>Gets the union of all input part modalities.</summary>
    public ProcessorInputModality Modalities { get; }
}

/// <summary>Exposes processor capabilities and creates independent execution sessions.</summary>
public interface IProcessor : IDisposable
{
    /// <summary>Gets the accepted modalities and execution limits.</summary>
    ProcessorCapabilities Capabilities { get; }
    /// <summary>Creates an independent session owned by the caller.</summary>
    /// <returns>The new processor session.</returns>
    IProcessorSession CreateSession();
}

/// <summary>Represents mutable execution state belonging to a processor.</summary>
public interface IProcessorSession : IDisposable
{
    /// <summary>Gets the processor that owns this session.</summary>
    IProcessor Processor { get; }
    /// <summary>Restores the session's initial state.</summary>
    void Reset();
}

/// <summary>Executes named tensor inputs and returns named tensor outputs.</summary>
public interface ITensorProcessorSession : IProcessorSession
{
    /// <summary>Executes a complete tensor request against this session's state.</summary>
    /// <param name="inputs">Input bytes keyed by graph resource name.</param>
    /// <param name="cancellationToken">A token used to cancel execution.</param>
    /// <returns>Owned output buffers keyed by graph resource name.</returns>
    IReadOnlyDictionary<string, byte[]> Execute(
        IReadOnlyDictionary<string, ReadOnlyMemory<byte>> inputs, CancellationToken cancellationToken = default);
}
