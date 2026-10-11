using System.Runtime.InteropServices;

namespace SharpInference.Architectures.Phi4;

/// <summary>Contains a tokenized prompt with expanded media placeholders, processed media features, and adapter selection.</summary>
/// <param name="TokenIds">The prompt token IDs with media placeholders expanded to their embedding lengths.</param>
/// <param name="AttentionMask">The validity mask corresponding to the prompt token IDs.</param>
/// <param name="Images">The processed images in prompt order.</param>
/// <param name="Audios">The processed audio clips in prompt order.</param>
/// <param name="Adapter">The LoRA adapter selected for decoding.</param>
/// <param name="AudioProjectionMode">The audio projection mode, either speech or vision.</param>
public sealed record Phi4PreparedInput(
    int[] TokenIds,
    bool[] AttentionMask,
    IReadOnlyList<Phi4ImageFeatures> Images,
    IReadOnlyList<Phi4AudioFeatures> Audios,
    Phi4Adapter Adapter,
    string AudioProjectionMode);

/// <summary>Contains the prompt tokens, newly generated tokens, and decoded generated text.</summary>
/// <param name="InputTokenIds">The prepared prompt token IDs.</param>
/// <param name="GeneratedTokenIds">The newly generated token IDs.</param>
/// <param name="Text">The decoded generated text.</param>
public sealed record Phi4GenerationResult(
    IReadOnlyList<int> InputTokenIds,
    IReadOnlyList<int> GeneratedTokenIds,
    string Text);

/// <summary>Defines creation of a prompt embedding sequence with media embeddings replacing expanded placeholder tokens.</summary>
public interface IPhi4EmbeddingProvider
{
    /// <summary>Creates one text-width embedding per prepared prompt token, replacing media placeholders with projected media embeddings.</summary>
    ValueTask<IReadOnlyList<ReadOnlyMemory<float>>> CreateFusedEmbeddingsAsync(
        Phi4PreparedInput input,
        CancellationToken cancellationToken);
}

/// <summary>Provides Phi-4 multimodal input capabilities and creates sessions for preprocessing and text generation.</summary>
public sealed class Phi4Processor : IProcessor
{
    private static readonly ProcessorCapabilities Phi4Capabilities = new(
        [
            ProcessorInputModality.Text,
            ProcessorInputModality.Text | ProcessorInputModality.Image,
            ProcessorInputModality.Text | ProcessorInputModality.Audio,
            ProcessorInputModality.Text | ProcessorInputModality.Image | ProcessorInputModality.Audio,
        ],
        ProcessorOutputModality.Text,
        new ProcessorTextCapabilities(131072),
        new ProcessorImageCapabilities(
            new HashSet<ProcessorPixelFormat>
            {
                ProcessorPixelFormat.Rgb24,
                ProcessorPixelFormat.Bgr24,
                ProcessorPixelFormat.Rgba32,
                ProcessorPixelFormat.Bgra32,
            },
            int.MaxValue, int.MaxValue, 64, 36),
        new ProcessorAudioCapabilities(
            new HashSet<ProcessorAudioSampleFormat>
            {
                ProcessorAudioSampleFormat.Signed16,
                ProcessorAudioSampleFormat.Float32,
            },
            new HashSet<int> { 16000, 22050, 24000 },
            8,
            TimeSpan.FromHours(1)),
        new ProcessorExecutionCapabilities(
            maximumConcurrentSessions: 1,
            maximumParallelComponents: 3,
            new HashSet<string>(StringComparer.Ordinal) { "cpu/default/host" }));

    private readonly Phi4ModelPackage package;
    private readonly IPhi4EmbeddingProvider? embeddingProvider;
    private bool disposed;

    /// <summary>Initializes a processor for the supplied package, using its native embedding provider when none is supplied.</summary>
    public Phi4Processor(Phi4ModelPackage package, IPhi4EmbeddingProvider? embeddingProvider = null)
    {
        this.package = package ?? throw new ArgumentNullException(nameof(package));
        this.embeddingProvider = embeddingProvider ?? package.CreateEmbeddingProvider();
    }

    /// <inheritdoc/>
    public ProcessorCapabilities Capabilities => Phi4Capabilities;
    /// <summary>Gets the tokenizer associated with the model package.</summary>
    public Phi4Tokenizer Tokenizer => package.Tokenizer;

    /// <summary>Creates a preprocessing and generation session over this processor's package.</summary>
    public Phi4ProcessorSession CreateSession()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new Phi4ProcessorSession(this, package.CreateTextModel(), embeddingProvider);
    }

    IProcessorSession IProcessor.CreateSession() => CreateSession();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (disposed)
            return;
        package.Dispose();
        disposed = true;
    }
}

/// <summary>Prepares multimodal prompts and generates text using the associated Phi-4 processor.</summary>
public sealed class Phi4ProcessorSession : IProcessorSession
{
    private readonly Phi4Processor processor;
    private readonly Phi4TextModel textModel;
    private readonly IPhi4EmbeddingProvider? embeddingProvider;
    private bool disposed;

    internal Phi4ProcessorSession(
        Phi4Processor processor,
        Phi4TextModel textModel,
        IPhi4EmbeddingProvider? embeddingProvider)
    {
        this.processor = processor;
        this.textModel = textModel;
        this.embeddingProvider = embeddingProvider;
    }

    /// <summary>Gets the Phi-4 processor that created this session.</summary>
    public Phi4Processor Processor => processor;
    IProcessor IProcessorSession.Processor => processor;

    /// <summary>Checks that the session is active; generation uses fresh decoder state for each request.</summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }

    /// <summary>Tokenizes the input prompt, preprocesses media, expands media placeholders, and selects the decoding adapter.</summary>
    public async ValueTask<Phi4PreparedInput> PrepareAsync(
        ProcessorInput input,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);
        if (!processor.Capabilities.Supports(input.Modalities))
            throw new NotSupportedException($"Phi-4 does not support input combination '{input.Modalities}'.");

        var imageParts = input.Parts.OfType<ImageInputPart>().ToArray();
        var audioParts = input.Parts.OfType<AudioInputPart>().ToArray();
        var imageTasks = imageParts.Select(part => Task.Run(
            () => ProcessImage(part), cancellationToken)).ToArray();
        var audioTasks = audioParts.Select(part => Task.Run(
            () => ProcessAudio(part), cancellationToken)).ToArray();
        await Task.WhenAll(imageTasks.Cast<Task>().Concat(audioTasks)).ConfigureAwait(false);
        var images = imageTasks.Select(static task => task.Result).ToArray();
        var audios = audioTasks.Select(static task => task.Result).ToArray();

        var prompt = BuildPrompt(input.Parts);
        var tokens = processor.Tokenizer.Encode(prompt);
        var expanded = ExpandMediaTokens(tokens, images, audios);
        if (expanded.Length > processor.Capabilities.Text!.MaximumContextLength)
            throw new ArgumentException("Prepared input exceeds the Phi-4 context length.", nameof(input));

        var adapter = images.Length != 0
            ? Phi4Adapter.Vision
            : audios.Length != 0 ? Phi4Adapter.Speech : Phi4Adapter.None;
        return new Phi4PreparedInput(
            expanded,
            Enumerable.Repeat(true, expanded.Length).ToArray(),
            images,
            audios,
            adapter,
            images.Length != 0 ? "vision" : "speech");
    }

    /// <summary>Prepares the input and greedily generates up to the requested number of new tokens.</summary>
    public async ValueTask<Phi4GenerationResult> GenerateAsync(
        ProcessorInput input,
        int maximumNewTokens,
        CancellationToken cancellationToken = default)
    {
        if (maximumNewTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumNewTokens));
        var prepared = await PrepareAsync(input, cancellationToken).ConfigureAwait(false);
        var session = textModel.CreateSession(prepared.Adapter);
        float[] logits;
        if (prepared.Images.Count == 0 && prepared.Audios.Count == 0)
        {
            logits = [];
            foreach (var token in prepared.TokenIds)
            {
                cancellationToken.ThrowIfCancellationRequested();
                logits = session.ForwardToken(token);
            }
        }
        else
        {
            var embeddings = await embeddingProvider!.CreateFusedEmbeddingsAsync(prepared, cancellationToken)
                .ConfigureAwait(false);
            if (embeddings.Count != prepared.TokenIds.Length)
                throw new InvalidDataException("The embedding provider returned a different sequence length.");
            logits = [];
            foreach (var embedding in embeddings)
            {
                cancellationToken.ThrowIfCancellationRequested();
                logits = session.ForwardEmbedding(embedding.Span);
            }
        }

        var generated = new List<int>(maximumNewTokens);
        for (var index = 0; index < maximumNewTokens; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var token = ArgMax(logits);
            if (token == Phi4Tokenizer.EndOfTextTokenId)
                break;
            generated.Add(token);
            logits = session.ForwardToken(token);
        }
        return new Phi4GenerationResult(
            prepared.TokenIds,
            generated,
            processor.Tokenizer.Decode(generated, skipSpecialTokens: true));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        disposed = true;
    }

    private static string BuildPrompt(IReadOnlyList<ProcessorInputPart> parts)
    {
        var builder = new System.Text.StringBuilder();
        var image = 0;
        var audio = 0;
        foreach (var part in parts)
        {
            switch (part)
            {
                case TextInputPart text:
                    builder.Append(text.Text);
                    break;
                case ImageInputPart:
                    image++;
                    builder.Append("<|endoftext10|>");
                    break;
                case AudioInputPart:
                    audio++;
                    builder.Append("<|endoftext11|>");
                    break;
                default:
                    throw new ArgumentException($"Unsupported input part '{part.GetType().Name}'.", nameof(parts));
            }
        }
        return builder.ToString();
    }

    private static int[] ExpandMediaTokens(
        IReadOnlyList<int> tokens,
        IReadOnlyList<Phi4ImageFeatures> images,
        IReadOnlyList<Phi4AudioFeatures> audios)
    {
        var result = new List<int>(tokens.Count);
        var image = 0;
        var audio = 0;
        foreach (var token in tokens)
        {
            if (token == Phi4Tokenizer.ImageTokenId)
            {
                if (image == images.Count)
                    throw new InvalidDataException("The prompt has more image placeholders than image inputs.");
                result.AddRange(Enumerable.Repeat(token, images[image++].ImageTokenCount));
            }
            else if (token == Phi4Tokenizer.AudioTokenId)
            {
                if (audio == audios.Count)
                    throw new InvalidDataException("The prompt has more audio placeholders than audio inputs.");
                result.AddRange(Enumerable.Repeat(token, audios[audio++].EmbedSize));
            }
            else
            {
                result.Add(token);
            }
        }
        if (image != images.Count || audio != audios.Count)
            throw new InvalidDataException("Media input count does not match generated placeholders.");
        return result.ToArray();
    }

    private static Phi4ImageFeatures ProcessImage(ImageInputPart input) =>
        new Phi4ImageProcessor().Process(
            input.Pixels.Span,
            input.Width,
            input.Height,
            input.Stride,
            input.PixelFormat switch
            {
                ProcessorPixelFormat.Rgb24 => Phi4PixelFormat.Rgb24,
                ProcessorPixelFormat.Bgr24 => Phi4PixelFormat.Bgr24,
                ProcessorPixelFormat.Rgba32 => Phi4PixelFormat.Rgba32,
                ProcessorPixelFormat.Bgra32 => Phi4PixelFormat.Bgra32,
                _ => throw new NotSupportedException($"Pixel format '{input.PixelFormat}' is unsupported."),
            });

    private static Phi4AudioFeatures ProcessAudio(AudioInputPart input)
    {
        float[] samples;
        if (input.SampleFormat == ProcessorAudioSampleFormat.Float32)
        {
            samples = MemoryMarshal.Cast<byte, float>(input.Samples.Span).ToArray();
        }
        else
        {
            var source = MemoryMarshal.Cast<byte, short>(input.Samples.Span);
            samples = new float[source.Length];
            for (var index = 0; index < source.Length; index++)
                samples[index] = source[index] / 32768f;
        }
        return new Phi4AudioProcessor().Process(samples, input.SampleRate, input.Channels);
    }

    private static int ArgMax(IReadOnlyList<float> values)
    {
        if (values.Count == 0)
            throw new InvalidOperationException("Cannot select a token before prefill.");
        var result = 0;
        for (var index = 1; index < values.Count; index++)
            if (values[index] > values[result])
                result = index;
        return result;
    }
}
