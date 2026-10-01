using System.Text.Json;
using System.Text.Json.Serialization;
using System.Diagnostics;
using System.Text.Encodings.Web;
using Microsoft.Extensions.Configuration.CommandLine;
using Microsoft.Extensions.Configuration.EnvironmentVariables;
using SharpInference;
using SharpInference.Architectures.Rwkv6;
using SharpInference.Runtime;
using SharpInference.WebApi;

if (args.Length == 0 || args.Any(static argument => argument is "--help" or "-h"))
{
    Console.WriteLine(WebApiCommandLine.Usage);
    return;
}

WebApiCommandLine commandLine;
try
{
    commandLine = WebApiCommandLine.Parse(args);
}
catch (Exception exception) when (exception is ArgumentException or FileNotFoundException)
{
    Console.Error.WriteLine($"Error: {exception.Message}");
    Environment.ExitCode = 2;
    return;
}

var builder = WebApplication.CreateBuilder();
builder.Logging.AddSimpleConsole(options => options.TimestampFormat = "yyyy-MM-dd HH:mm:ss.fff zzz ");
for (var index = builder.Configuration.Sources.Count - 1; index >= 0; index--)
{
    if (builder.Configuration.Sources[index] is EnvironmentVariablesConfigurationSource or CommandLineConfigurationSource)
    {
        builder.Configuration.Sources.RemoveAt(index);
    }
}

if (commandLine.ConfigurationPath is not null)
{
    builder.Configuration.AddJsonFile(commandLine.ConfigurationPath, optional: false, reloadOnChange: false);
}

builder.Configuration.AddInMemoryCollection(commandLine.Overrides);
var serverOptions = builder.Configuration.GetSection("Server").Get<ServerOptions>() ?? new ServerOptions();
builder.WebHost.UseUrls(serverOptions.Urls);
builder.Services.Configure<RwkvWebOptions>(builder.Configuration.GetSection("Rwkv"));
builder.Services.AddSingleton(serviceProvider =>
    new GpuBatchScheduler(
        serviceProvider.GetRequiredService<Microsoft.Extensions.Options.IOptions<RwkvWebOptions>>().Value.GpuBatchService,
        builder.Configuration["Rwkv:Runtime:Kind"]));
builder.Services.AddSingleton<RwkvRuntimeFactory>();
builder.Services.AddSingleton<IModelTextTransfer, Rwkv6WorldTextTransfer>();
builder.Services.AddSingleton<IModelTextTransfer, Rwkv7G1TextTransfer>();
builder.Services.AddSingleton<ModelTextTransferResolver>();
builder.Services.AddSingleton(serviceProvider =>
    new PromptStateManager(
        serviceProvider.GetRequiredService<Microsoft.Extensions.Options.IOptions<RwkvWebOptions>>().Value.StateManager));
builder.Services.AddSingleton<RwkvModelDescriptor>();
builder.Services.AddSingleton<RwkvModelHost>();

var app = builder.Build();
var modelDescriptor = app.Services.GetRequiredService<RwkvModelDescriptor>();
RwkvModelStartupValidator.Validate(modelDescriptor);
var webOptions = app.Services.GetRequiredService<Microsoft.Extensions.Options.IOptions<RwkvWebOptions>>().Value;
var diagnosticJsonOptions = new JsonSerializerOptions
{
    Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
};

app.Use(async (context, next) =>
{
    var stopwatch = Stopwatch.StartNew();
    app.Logger.LogInformation(
        "HTTP request started: {Method} {Path} from {RemoteIp}",
        context.Request.Method,
        context.Request.Path,
        context.Connection.RemoteIpAddress?.ToString() ?? "unknown");
    try
    {
        await next();
    }
    finally
    {
        stopwatch.Stop();
        app.Logger.LogInformation(
            "HTTP request completed: {Method} {Path} status {StatusCode} in {ElapsedMilliseconds} ms ({ContentType})",
            context.Request.Method,
            context.Request.Path,
            context.Response.StatusCode,
            stopwatch.ElapsedMilliseconds,
            context.Response.ContentType ?? "no-content-type");
    }
});

app.Use(async (context, next) =>
{
    if (context.Request.Path == "/health" || string.IsNullOrEmpty(webOptions.ApiKey))
    {
        await next();
        return;
    }

    var bearer = context.Request.Headers.Authorization.ToString();
    var apiKey = context.Request.Headers["X-Api-Key"].ToString();
    if (bearer == $"Bearer {webOptions.ApiKey}" || apiKey == webOptions.ApiKey)
    {
        await next();
        return;
    }

    await OpenAiResponses.WriteErrorAsync(context.Response, StatusCodes.Status401Unauthorized, "Invalid or missing API key.", "invalid_api_key");
});

app.MapGet("/health", () => Results.Ok(new { status = "ok" }));

app.MapGet("/v1/models", (RwkvModelDescriptor model) => Results.Ok(new
{
    @object = "list",
    data = new[]
    {
        new
        {
            id = model.ModelId,
            @object = "model",
            created = 0,
            owned_by = "sharpinference",
        },
    },
}));

app.MapGet("/v1/models/{modelId}", (string modelId, RwkvModelDescriptor model) =>
    model.MatchesModel(modelId)
        ? Results.Ok(new { id = model.ModelId, @object = "model", created = 0, owned_by = "sharpinference" })
        : OpenAiResponses.Error(StatusCodes.Status404NotFound, $"The model '{modelId}' does not exist.", "model_not_found"));

app.MapPost("/v1/chat/completions", async (
    HttpContext context,
    ChatCompletionRequest request,
    RwkvModelDescriptor model,
    IServiceProvider serviceProvider,
    CancellationToken cancellationToken) =>
{
    var handlerStarted = Stopwatch.GetTimestamp();
    if (!model.MatchesModel(request.Model))
    {
        app.Logger.LogWarning("Rejected chat completion for unconfigured model {RequestedModel}", request.Model ?? "null");
        await OpenAiResponses.WriteErrorAsync(context.Response, StatusCodes.Status404NotFound, $"The configured model is '{model.ModelId}'.", "model_not_found");
        return;
    }

    try
    {
        var host = serviceProvider.GetRequiredService<RwkvModelHost>();
        app.Logger.LogInformation(
            "Chat completion model host ready in {HostMilliseconds:F1} ms.",
            Stopwatch.GetElapsedTime(handlerStarted).TotalMilliseconds);
        var requestPayload = JsonSerializer.Serialize(new
        {
            request.Model,
            request.Messages,
            request.Stream,
            request.MaxTokens,
            request.Temperature,
            request.TopP,
            request.TopK,
            request.Seed,
            Stop = request.Stop.ValueKind == JsonValueKind.Undefined
                ? (JsonElement?)null
                : request.Stop,
        }, diagnosticJsonOptions);
        app.Logger.LogInformation("Chat completion request payload: {RequestPayload}", requestPayload);
        var prompt = host.EncodePrompt(request.Messages);
        var stops = OpenAiPrompt.ParseStops(request.Stop)
            .Concat(host.ArchitectureId == "rwkv-7"
                ? Rwkv7G1TextTransfer.StopStrings
                : Rwkv6WorldChatTemplateTransfer.StopStrings)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        var requestedMaxTokens = request.MaxTokens ?? host.DefaultMaxTokens;
        var effectiveMaxTokens = CompletionTokenBudget.Calculate(
            host.ContextWindowTokens,
            prompt.TokenIds.Count,
            requestedMaxTokens);
        if (effectiveMaxTokens != requestedMaxTokens)
        {
            app.Logger.LogWarning(
                "Requested max_tokens {RequestedMaxTokens} was limited to {EffectiveMaxTokens} by the {ContextWindowTokens}-token context window for a {PromptTokenCount}-token prompt.",
                requestedMaxTokens,
                effectiveMaxTokens,
                host.ContextWindowTokens,
                prompt.TokenIds.Count);
        }

        var options = new RwkvGenerationOptions
        {
            MaxTokens = effectiveMaxTokens,
            Temperature = request.Temperature ?? 1.0f,
            TopP = request.TopP ?? 0.8f,
            TopK = request.TopK ?? 0,
            Seed = request.Seed,
        };
        options.Validate();
        app.Logger.LogInformation(
            "Chat completion started: model {ModelId}, stream {Stream}, messages {MessageCount}, prompt characters {PromptLength}, prompt tokens {PromptTokenCount}, max tokens {MaxTokens}, temperature {Temperature}, top-p {TopP}, top-k {TopK}",
            host.ModelId,
            request.Stream,
            request.Messages.Count,
            prompt.Text.Length,
            prompt.TokenIds.Count,
            options.MaxTokens,
            options.Temperature,
            options.TopP,
            options.TopK);

        var completionId = $"chatcmpl-{Guid.NewGuid():N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var firstContentLogged = false;
        void LogFirstContent(string text)
        {
            if (firstContentLogged || text.Length == 0) return;
            firstContentLogged = true;
            app.Logger.LogInformation(
                "Chat completion first content: id {CompletionId}, handler-to-first-content {FirstContentMilliseconds:F1} ms.",
                completionId,
                Stopwatch.GetElapsedTime(handlerStarted).TotalMilliseconds);
        }

        if (request.Stream)
        {
            var streamingCompletion = new System.Text.StringBuilder();
            context.Response.StatusCode = StatusCodes.Status200OK;
            context.Response.ContentType = "text/event-stream";
            context.Response.Headers.CacheControl = "no-cache";
            context.Response.Headers.Append("X-Accel-Buffering", "no");

            await OpenAiResponses.WriteSseAsync(context.Response, new ChatCompletionChunk(
                completionId, created, host.ModelId,
                [new ChatCompletionChunkChoice(0, new ChatDelta("assistant", null), null)]), cancellationToken);

            await foreach (var text in host.GenerateAsync(prompt, options, stops, cancellationToken))
            {
                LogFirstContent(text);
                streamingCompletion.Append(text);
                await OpenAiResponses.WriteSseAsync(context.Response, new ChatCompletionChunk(
                    completionId, created, host.ModelId,
                    [new ChatCompletionChunkChoice(0, new ChatDelta(null, text), null)]), cancellationToken);
            }

            await OpenAiResponses.WriteSseAsync(context.Response, new ChatCompletionChunk(
                completionId, created, host.ModelId,
                [new ChatCompletionChunkChoice(0, new ChatDelta(null, null), "stop")]), cancellationToken);
            await context.Response.WriteAsync("data: [DONE]\n\n", cancellationToken);
            await context.Response.Body.FlushAsync(cancellationToken);
            app.Logger.LogInformation(
                "Chat completion finished: id {CompletionId}, model {ModelId}, stream true, generated characters {CharacterCount}, output: {Output}",
                completionId,
                host.ModelId,
                streamingCompletion.Length,
                streamingCompletion.ToString());
            return;
        }

        var completion = new System.Text.StringBuilder();
        await foreach (var text in host.GenerateAsync(prompt, options, stops, cancellationToken))
        {
            LogFirstContent(text);
            completion.Append(text);
        }

        await context.Response.WriteAsJsonAsync(new ChatCompletionResponse(
            completionId,
            created,
            host.ModelId,
            [new ChatCompletionChoice(0, new ChatMessageText("assistant", completion.ToString()), "stop")],
            new Usage(prompt.TokenIds.Count, 0, prompt.TokenIds.Count)),
            cancellationToken);
        app.Logger.LogInformation(
            "Chat completion finished: id {CompletionId}, model {ModelId}, stream false, generated characters {CharacterCount}, output: {Output}",
            completionId,
            host.ModelId,
            completion.Length,
            completion.ToString());
    }
    catch (ContextWindowExceededException exception)
    {
        app.Logger.LogWarning(exception, "Rejected chat completion because the prompt exceeds the context window");
        await OpenAiResponses.WriteErrorAsync(
            context.Response,
            StatusCodes.Status400BadRequest,
            exception.Message,
            "context_length_exceeded");
    }
    catch (ArgumentException exception)
    {
        app.Logger.LogWarning(exception, "Rejected invalid chat completion request");
        await OpenAiResponses.WriteErrorAsync(context.Response, StatusCodes.Status400BadRequest, exception.Message, "invalid_request_error");
    }
});

app.Run();
