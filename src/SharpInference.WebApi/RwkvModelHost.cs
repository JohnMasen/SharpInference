using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Options;
using SharpInference;
using SharpInference.Runtime;
using SharpInference.Applications;

namespace SharpInference.WebApi;

/// <summary>Configures HTTP server listen addresses.</summary>
public sealed class ServerOptions
{
    /// <summary>Gets the server listen URLs, defaulting to all network interfaces on port 9841.</summary>
    public string Urls { get; init; } = "http://0.0.0.0:9841";
}

/// <summary>Configures the served RWKV model, token limits, API authentication, admission, and state caching.</summary>
public sealed class RwkvWebOptions
{
    /// <summary>Gets the path of the RWKV GGML model file to serve.</summary>
    public string ModelPath { get; init; } = string.Empty;
    /// <summary>Gets the default completion token limit when a request omits max_tokens.</summary>
    public int DefaultMaxTokens { get; init; } = 4096;
    /// <summary>Gets the logical token budget shared by prompt and completion.</summary>
    public int ContextWindowTokens { get; init; } = 12800;
    /// <summary>Gets the optional API key required for authenticated requests.</summary>
    public string? ApiKey { get; init; }
    /// <summary>Gets resident GPU session and generation admission settings.</summary>
    public GpuBatchServiceOptions GpuBatchService { get; init; } = new();
    /// <summary>Gets prompt-state cache enablement and capacity settings.</summary>
    public PromptStateManagerOptions StateManager { get; init; } = new();
}

/// <summary>Validates model-serving options and derives the public model identifier from the model file name.</summary>
public sealed class RwkvModelDescriptor
{
    /// <summary>Creates a model descriptor after validating the path, token limits, admission settings, and cache settings.</summary>
    /// <param name="configuredOptions">The configured model-serving options to validate.</param>
    public RwkvModelDescriptor(IOptions<RwkvWebOptions> configuredOptions)
    {
        var options = configuredOptions.Value;
        if (string.IsNullOrWhiteSpace(options.ModelPath))
        {
            throw new InvalidOperationException("Configure Rwkv:ModelPath before starting the server.");
        }

        ModelPath = options.ModelPath;
        ModelId = CreateModelId(ModelPath);
        if (string.IsNullOrEmpty(ModelId))
        {
            throw new InvalidOperationException(
                "Rwkv:ModelPath file name must contain at least one ASCII letter, digit, '.', '-', or '_' for the model ID.");
        }

        if (options.DefaultMaxTokens <= 0)
        {
            throw new InvalidOperationException("Rwkv:DefaultMaxTokens must be positive.");
        }
        if (options.ContextWindowTokens <= 0)
        {
            throw new InvalidOperationException("Rwkv:ContextWindowTokens must be positive.");
        }
        options.GpuBatchService.Validate();
        options.StateManager.Validate();
        DefaultMaxTokens = options.DefaultMaxTokens;
        ContextWindowTokens = options.ContextWindowTokens;
    }

    /// <summary>Gets the configured model file path.</summary>
    public string ModelPath { get; }
    /// <summary>Gets the sanitized public model identifier derived from the file name.</summary>
    public string ModelId { get; }
    /// <summary>Gets the validated default completion token limit.</summary>
    public int DefaultMaxTokens { get; }
    /// <summary>Gets the validated logical context-window size.</summary>
    public int ContextWindowTokens { get; }

    /// <summary>Checks whether the requested model identifier exactly matches the served identifier.</summary>
    /// <param name="requestedModel">The model identifier requested by the client.</param>
    public bool MatchesModel(string? requestedModel) =>
        string.Equals(requestedModel, ModelId, StringComparison.Ordinal);

    private static string CreateModelId(string modelPath)
    {
        var fileName = Path.GetFileName(modelPath);
        var modelId = new StringBuilder(fileName.Length);
        var needsSeparator = false;
        foreach (var character in fileName)
        {
            if (character is >= 'A' and <= 'Z' or >= 'a' and <= 'z' or >= '0' and <= '9' or '.' or '-' or '_')
            {
                if (needsSeparator && modelId.Length > 0)
                {
                    modelId.Append('-');
                }

                modelId.Append(character);
                needsSeparator = false;
            }
            else
            {
                needsSeparator = true;
            }
        }

        return modelId.ToString().Trim('-');
    }
}

/// <summary>Validates that the configured RWKV model file is accessible and has a supported GGML header.</summary>
public static class RwkvModelStartupValidator
{
    /// <summary>Checks the configured model header and reports inaccessible or invalid files as startup errors.</summary>
    /// <param name="descriptor">The descriptor containing the model file path to validate.</param>
    public static void Validate(RwkvModelDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        try
        {
            GgmlModelFile.ValidateHeader(descriptor.ModelPath);
        }
        catch (Exception exception) when (exception is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            throw new InvalidOperationException(
                $"The configured RWKV model file '{descriptor.ModelPath}' is unavailable or invalid: {exception.Message}",
                exception);
        }
    }
}

/// <summary>Hosts a compiled RWKV processor, tokenizer, prompt transfer, generation admission, and reusable state cache.</summary>
public sealed class RwkvModelHost : IDisposable
{
    private readonly Processor model;
    private readonly RwkvWorldTokenizer tokenizer;
    private readonly IModelTextTransfer textTransfer;
    private readonly GpuBatchScheduler generationScheduler;
    private readonly PromptStateManager stateManager;
    private readonly ILogger<RwkvModelHost> logger;

    /// <summary>Loads and binds the configured RWKV model, validates its tokenizer, and initializes generation services.</summary>
    /// <param name="descriptor">The validated descriptor of the model to serve.</param>
    /// <param name="configuration">The application configuration containing the RWKV runtime section.</param>
    /// <param name="runtimeFactory">The application composition resolving the model module, tokenizer, and backend.</param>
    /// <param name="gpuBatchOptions">The resident-session and generation admission limits.</param>
    /// <param name="textTransferResolver">The resolver selecting the sole prompt transfer compatible with the model.</param>
    /// <param name="stateManager">The shared prompt-state snapshot cache.</param>
    /// <param name="logger">The logger receiving generation and state-cache timing information.</param>
    public RwkvModelHost(
        RwkvModelDescriptor descriptor,
        IConfiguration configuration,
        RwkvApplicationComposition runtimeFactory,
        GpuBatchServiceOptions gpuBatchOptions,
        ModelTextTransferResolver textTransferResolver,
        PromptStateManager stateManager,
        ILogger<RwkvModelHost> logger)
    {
        using var catalog = GgmlModelFile.Open(descriptor.ModelPath);
        var runtimeSection = configuration.GetSection("Rwkv:Runtime");
        var runtime = runtimeFactory.CreateRuntime(runtimeSection, catalog);
        model = new ProcessorPipelineBuilder(descriptor.ModelPath)
            .UseReader(new GgmlModelReader())
            .UseProvider(runtime.Provider)
            .UseBackend(_ => runtime.CreateBackend())
            .UsePortableGraphArchitecture()
            .Build();
        try
        {
            tokenizer = runtime.Tokenizer;
            var metadata = RwkvModelMetadata.FromModelMetadata(model.Metadata);
            if (tokenizer.TokenIds[^1] >= metadata.VocabularySize)
                throw new InvalidOperationException("The configured tokenizer contains token IDs outside the configured model vocabulary.");

            ModelId = descriptor.ModelId;
            DefaultMaxTokens = descriptor.DefaultMaxTokens;
            ContextWindowTokens = descriptor.ContextWindowTokens;
            textTransfer = textTransferResolver.Resolve(metadata);
            generationScheduler = new GpuBatchScheduler(gpuBatchOptions, model.Capabilities.Execution, useVmQueues: true);
            this.stateManager = stateManager;
            this.logger = logger;
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>Gets the public identifier of the served model.</summary>
    public string ModelId { get; }
    /// <summary>Gets the architecture identifier reported by the bound model.</summary>
    public string ArchitectureId => model.Metadata.ArchitectureId;
    /// <summary>Gets the default completion token limit for this host.</summary>
    public int DefaultMaxTokens { get; }
    /// <summary>Gets the logical context-window size shared by prompt and completion.</summary>
    public int ContextWindowTokens { get; }

    /// <summary>Checks whether a requested model identifier exactly matches this host&apos;s model.</summary>
    /// <param name="requestedModel">The model identifier requested by the client.</param>
    public bool MatchesModel(string? requestedModel) => string.Equals(requestedModel, ModelId, StringComparison.Ordinal);

    /// <summary>Applies the model&apos;s chat template and tokenizes the resulting prompt text.</summary>
    /// <param name="messages">The ordered text conversation messages to format.</param>
    public EncodedChatPrompt EncodePrompt(IReadOnlyList<ChatMessage> messages)
    {
        var prompt = textTransfer.Transfer(messages);
        return new EncodedChatPrompt(prompt, tokenizer.Encode(prompt));
    }

    /// <summary>Streams generated text using admitted sessions, reusable prompt state, sampling options, and stop sequences.</summary>
    /// <param name="prompt">The canonical prompt text and its encoded token identifiers.</param>
    /// <param name="options">The sampling and completion token limits used for generation.</param>
    /// <param name="stops">The text sequences that terminate assistant generation.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    public async IAsyncEnumerable<string> GenerateAsync(
        EncodedChatPrompt prompt,
        RwkvGenerationOptions options,
        IReadOnlyList<string> stops,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var waitStarted = Stopwatch.GetTimestamp();
        var generationLease = await generationScheduler.AcquireAsync(cancellationToken);
        logger.LogInformation(
            "Generation admission: wait {WaitMilliseconds:F1} ms.",
            Stopwatch.GetElapsedTime(waitStarted).TotalMilliseconds);
        await using (generationLease)
        {
            var sessionStarted = Stopwatch.GetTimestamp();
            using var session = model.CreateSession();
            logger.LogInformation(
                "Generation session created in {SessionMilliseconds:F1} ms.",
                Stopwatch.GetElapsedTime(sessionStarted).TotalMilliseconds);
            var cacheStarted = Stopwatch.GetTimestamp();
            logger.LogInformation(
                "Prompt state cache lookup: prompt characters {PromptCharacterCount}, entries {EntryCount}/{Capacity}.",
                prompt.Text.Length,
                stateManager.Count,
                stateManager.Capacity);
            var match = stateManager.FindLongestPrefix(prompt.Text);
            IReadOnlyList<int> suffixTokens;
            if (match is null)
            {
                suffixTokens = prompt.TokenIds;
                logger.LogInformation(
                    "Prompt state cache selected None: full prefill of {PromptTokenCount} tokens for {PromptCharacterCount} characters.",
                    suffixTokens.Count,
                    prompt.Text.Length);
            }
            else
            {
                using var cachedState = new MemoryStream(match.State.ToArray(), writable: false);
                session.LoadState(cachedState);
                suffixTokens = tokenizer.Encode(prompt.Text[match.StartCharacterIndex..]);
                logger.LogInformation(
                    "Prompt state cache selected {SelectedState}: reused {CachedCharacterCount} characters, prefill starts at character {StartCharacterIndex}, suffix tokens {SuffixTokenCount}, score {Score}.",
                    match.Kind,
                    match.StartCharacterIndex,
                    match.StartCharacterIndex,
                    suffixTokens.Count,
                    match.Score);
            }

            logger.LogInformation(
                "Prompt state cache lookup and restore completed in {CacheMilliseconds:F1} ms.",
                Stopwatch.GetElapsedTime(cacheStarted).TotalMilliseconds);
            var prefillStarted = Stopwatch.GetTimestamp();
            var logits = await session.PrefillAsync(suffixTokens.ToArray(), cancellationToken).ConfigureAwait(false);
            logger.LogInformation(
                "Prompt suffix prefill: {SuffixTokenCount} tokens in {PrefillMilliseconds:F1} ms.",
                suffixTokens.Count,
                Stopwatch.GetElapsedTime(prefillStarted).TotalMilliseconds);
            byte[]? prefillState = null;
            if (stateManager.Capacity > 0)
            {
                var prefillSnapshotStarted = Stopwatch.GetTimestamp();
                using var snapshot = new MemoryStream();
                session.SaveState(snapshot);
                prefillState = snapshot.ToArray();
                logger.LogInformation(
                    "Prefill state snapshot exported in {SnapshotMilliseconds:F1} ms.",
                    Stopwatch.GetElapsedTime(prefillSnapshotStarted).TotalMilliseconds);
                stateManager.Store(prompt.Text, prefillState, PromptStateKind.Prefill);
                logger.LogInformation(
                    "Prefill state cache stored: {PromptCharacterCount} characters, entries {EntryCount}/{Capacity}.",
                    prompt.Text.Length,
                    stateManager.Count,
                    stateManager.Capacity);
            }
            var effectiveOptions = options with { StopStrings = stops.ToArray() };
            var generationStarted = Stopwatch.GetTimestamp();
            var visibleAnswer = new StringBuilder();
            await foreach (var text in RwkvTextGenerator.GenerateFromPrefilledAsync(
                session,
                tokenizer,
                logits,
                effectiveOptions,
                cancellationToken))
            {
                visibleAnswer.Append(text);
                yield return text;
            }

            logger.LogInformation(
                "Generation stream completed in {GenerationMilliseconds:F1} ms.",
                Stopwatch.GetElapsedTime(generationStarted).TotalMilliseconds);
            if (stateManager.Capacity == 0)
            {
                logger.LogInformation("Answer state cache disabled: no state stored for {PromptCharacterCount} prompt characters.", prompt.Text.Length);
            }
            else if (AnswerStateCacheKey.TryCreate(prompt.Text, visibleAnswer.ToString(), out var answeredPrefix, out var requiresReplay))
            {
                if (requiresReplay)
                {
                    var replayStarted = Stopwatch.GetTimestamp();
                    using var replayState = new MemoryStream(
                        prefillState ?? throw new InvalidOperationException("The prefill state was not captured for answer replay."),
                        writable: false);
                    session.LoadState(replayState);
                    var canonicalSuffix = answeredPrefix[prompt.Text.Length..];
                    var replayTokens = tokenizer.Encode(canonicalSuffix);
                    await session.PrefillAsync(replayTokens.ToArray(), cancellationToken).ConfigureAwait(false);
                    logger.LogInformation(
                        "Answer state cache replayed {AnswerTokenCount} normalized answer tokens in {ReplayMilliseconds:F1} ms.",
                        replayTokens.Count,
                        Stopwatch.GetElapsedTime(replayStarted).TotalMilliseconds);
                }

                var answerSnapshotStarted = Stopwatch.GetTimestamp();
                using var answerSnapshot = new MemoryStream();
                session.SaveState(answerSnapshot);
                var answeredState = answerSnapshot.ToArray();
                logger.LogInformation(
                    "Answer state snapshot exported in {SnapshotMilliseconds:F1} ms.",
                    Stopwatch.GetElapsedTime(answerSnapshotStarted).TotalMilliseconds);
                stateManager.Store(answeredPrefix, answeredState, PromptStateKind.Complete);
                logger.LogInformation(
                    "Answer state cache stored: answer characters {AnswerCharacterCount}, prefix characters {PrefixCharacterCount}, entries {EntryCount}/{Capacity}; hidden stop tokens are not checked when the answer needs no replay.",
                    visibleAnswer.Length,
                    answeredPrefix.Length,
                    stateManager.Count,
                    stateManager.Capacity);
            }
            else
            {
                logger.LogInformation("Answer state cache skipped: visible answer cannot form a message prefix.");
            }
        }
    }

    /// <summary>Releases the hosted processor and its backend resources.</summary>
    public void Dispose() => model.Dispose();

}

/// <summary>Parses OpenAI-compatible prompt controls from JSON request values.</summary>
public static class OpenAiPrompt
{
    /// <summary>Parses an absent stop value, a non-empty string, or an array of non-empty stop strings.</summary>
    /// <param name="stop">The JSON stop value to parse.</param>
    public static IReadOnlyList<string> ParseStops(JsonElement stop)
    {
        if (stop.ValueKind is JsonValueKind.Undefined or JsonValueKind.Null) return [];
        if (stop.ValueKind == JsonValueKind.String) return [ValidateStop(stop.GetString())];
        if (stop.ValueKind != JsonValueKind.Array) throw new ArgumentException("The stop value must be a string or an array of strings.");

        var result = new List<string>();
        foreach (var item in stop.EnumerateArray())
        {
            if (item.ValueKind != JsonValueKind.String) throw new ArgumentException("Every stop value must be a string.");
            result.Add(ValidateStop(item.GetString()));
        }

        return result;
    }

    private static string ValidateStop(string? stop) =>
        !string.IsNullOrEmpty(stop) ? stop : throw new ArgumentException("A stop value cannot be empty.");
}

/// <summary>Writes OpenAI-compatible JSON errors and server-sent event payloads.</summary>
public static class OpenAiResponses
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    /// <summary>Creates a JSON invalid-request error result with the supplied HTTP status and error code.</summary>
    /// <param name="statusCode">The HTTP status code to return.</param>
    /// <param name="message">The human-readable invalid-request error message.</param>
    /// <param name="code">The machine-readable error code returned to the client.</param>
    public static IResult Error(int statusCode, string message, string code) =>
        Results.Json(new { error = new { message, type = "invalid_request_error", code } }, statusCode: statusCode, options: JsonOptions);

    /// <summary>Writes an invalid-request JSON error directly to the HTTP response.</summary>
    /// <param name="response">The HTTP response receiving the serialized payload.</param>
    /// <param name="statusCode">The HTTP status code to return.</param>
    /// <param name="message">The human-readable invalid-request error message.</param>
    /// <param name="code">The machine-readable error code returned to the client.</param>
    public static Task WriteErrorAsync(HttpResponse response, int statusCode, string message, string code)
    {
        response.StatusCode = statusCode;
        return response.WriteAsJsonAsync(new { error = new { message, type = "invalid_request_error", code } }, JsonOptions);
    }

    /// <summary>Writes and flushes a JSON payload as a server-sent data event.</summary>
    /// <param name="response">The HTTP response receiving the serialized payload.</param>
    /// <param name="value">The object serialized as an SSE JSON data payload.</param>
    /// <param name="cancellationToken">The token used to cancel the operation or admission wait.</param>
    public static async Task WriteSseAsync(HttpResponse response, object value, CancellationToken cancellationToken)
    {
        await response.WriteAsync("data: ", cancellationToken);
        await JsonSerializer.SerializeAsync(response.Body, value, JsonOptions, cancellationToken);
        await response.WriteAsync("\n\n", cancellationToken);
        await response.Body.FlushAsync(cancellationToken);
    }
}
