using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Options;
using SharpInference;
using SharpInference.Runtime;

namespace SharpInference.WebApi;

public sealed class ServerOptions
{
    public string Urls { get; init; } = "http://0.0.0.0:9841";
}

public sealed class RwkvWebOptions
{
    public string ModelPath { get; init; } = string.Empty;
    public int DefaultMaxTokens { get; init; } = 4096;
    public int ContextWindowTokens { get; init; } = 12800;
    public string? ApiKey { get; init; }
    public GpuBatchServiceOptions GpuBatchService { get; init; } = new();
    public PromptStateManagerOptions StateManager { get; init; } = new();
}

public sealed class RwkvModelDescriptor
{
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

    public string ModelPath { get; }
    public string ModelId { get; }
    public int DefaultMaxTokens { get; }
    public int ContextWindowTokens { get; }

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

public static class RwkvModelStartupValidator
{
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

public sealed class RwkvModelHost : IDisposable
{
    private readonly Processor model;
    private readonly RwkvWorldTokenizer tokenizer;
    private readonly IModelTextTransfer textTransfer;
    private readonly GpuBatchScheduler generationScheduler;
    private readonly PromptStateManager stateManager;
    private readonly ILogger<RwkvModelHost> logger;

    public RwkvModelHost(
        RwkvModelDescriptor descriptor,
        IConfiguration configuration,
        RwkvRuntimeFactory runtimeFactory,
        GpuBatchScheduler gpuBatchScheduler,
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
        tokenizer = runtime.Tokenizer;
        if (tokenizer.TokenIds[^1] >= model.Metadata.VocabularySize)
        {
            model.Dispose();
            throw new InvalidOperationException("The configured tokenizer contains token IDs outside the configured model vocabulary.");
        }

        ModelId = descriptor.ModelId;
        DefaultMaxTokens = descriptor.DefaultMaxTokens;
        ContextWindowTokens = descriptor.ContextWindowTokens;
        textTransfer = textTransferResolver.Resolve(model.Metadata);
        generationScheduler = gpuBatchScheduler;
        this.stateManager = stateManager;
        this.logger = logger;
    }

    public string ModelId { get; }
    public string ArchitectureId => model.Metadata.ArchitectureId;
    public int DefaultMaxTokens { get; }
    public int ContextWindowTokens { get; }

    public bool MatchesModel(string? requestedModel) => string.Equals(requestedModel, ModelId, StringComparison.Ordinal);

    public EncodedChatPrompt EncodePrompt(IReadOnlyList<ChatMessage> messages)
    {
        var prompt = textTransfer.Transfer(messages);
        return new EncodedChatPrompt(prompt, tokenizer.Encode(prompt));
    }

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
            var logits = session.Prefill(suffixTokens.ToArray());
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
                    session.Prefill(replayTokens.ToArray());
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

    public void Dispose() => model.Dispose();

}

public static class OpenAiPrompt
{
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

public static class OpenAiResponses
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public static IResult Error(int statusCode, string message, string code) =>
        Results.Json(new { error = new { message, type = "invalid_request_error", code } }, statusCode: statusCode, options: JsonOptions);

    public static Task WriteErrorAsync(HttpResponse response, int statusCode, string message, string code)
    {
        response.StatusCode = statusCode;
        return response.WriteAsJsonAsync(new { error = new { message, type = "invalid_request_error", code } }, JsonOptions);
    }

    public static async Task WriteSseAsync(HttpResponse response, object value, CancellationToken cancellationToken)
    {
        await response.WriteAsync("data: ", cancellationToken);
        await JsonSerializer.SerializeAsync(response.Body, value, JsonOptions, cancellationToken);
        await response.WriteAsync("\n\n", cancellationToken);
        await response.Body.FlushAsync(cancellationToken);
    }
}
