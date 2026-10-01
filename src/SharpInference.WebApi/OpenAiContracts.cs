using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInference.WebApi;

public sealed class ChatCompletionRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }
    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; init; } = [];
    [JsonPropertyName("stream")]
    public bool Stream { get; init; }
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }
    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }
    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }
    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }
    [JsonPropertyName("seed")]
    public int? Seed { get; init; }
    [JsonPropertyName("stop")]
    public JsonElement Stop { get; init; }
}

public sealed record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] JsonElement Content);

public sealed record ChatCompletionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<ChatCompletionChoice> Choices,
    [property: JsonPropertyName("usage")] Usage Usage)
{
    [JsonPropertyName("object")]
    public string Object { get; } = "chat.completion";
}

public sealed record ChatCompletionChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("message")] ChatMessageText Message,
    [property: JsonPropertyName("finish_reason")] string FinishReason);

public sealed record ChatMessageText(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

public sealed record ChatCompletionChunk(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<ChatCompletionChunkChoice> Choices)
{
    [JsonPropertyName("object")]
    public string Object { get; } = "chat.completion.chunk";
}

public sealed record ChatCompletionChunkChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("delta")] ChatDelta Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

public sealed record ChatDelta(
    [property: JsonPropertyName("role")] string? Role,
    [property: JsonPropertyName("content")] string? Content);

public sealed record Usage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);
