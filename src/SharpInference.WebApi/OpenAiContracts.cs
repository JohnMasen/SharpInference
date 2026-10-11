using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpInference.WebApi;

/// <summary>Represents an OpenAI-compatible chat completion request and its sampling controls.</summary>
public sealed class ChatCompletionRequest
{
    /// <summary>Gets the requested served model identifier.</summary>
    [JsonPropertyName("model")]
    public string? Model { get; init; }
    /// <summary>Gets the ordered conversation messages to encode as the prompt.</summary>
    [JsonPropertyName("messages")]
    public List<ChatMessage> Messages { get; init; } = [];
    /// <summary>Gets whether completion text is returned as server-sent events.</summary>
    [JsonPropertyName("stream")]
    public bool Stream { get; init; }
    /// <summary>Gets the optional completion token limit, otherwise the server default is used.</summary>
    [JsonPropertyName("max_tokens")]
    public int? MaxTokens { get; init; }
    /// <summary>Gets the optional temperature controlling sampling randomness.</summary>
    [JsonPropertyName("temperature")]
    public float? Temperature { get; init; }
    /// <summary>Gets the optional cumulative probability threshold for nucleus sampling.</summary>
    [JsonPropertyName("top_p")]
    public float? TopP { get; init; }
    /// <summary>Gets the optional maximum number of highest-probability sampling candidates.</summary>
    [JsonPropertyName("top_k")]
    public int? TopK { get; init; }
    /// <summary>Gets the optional random seed for token sampling.</summary>
    [JsonPropertyName("seed")]
    public int? Seed { get; init; }
    /// <summary>Gets stop sequences encoded as a string, array of strings, or absent JSON value.</summary>
    [JsonPropertyName("stop")]
    public JsonElement Stop { get; init; }
}

/// <summary>Represents a conversation message with a role and JSON text content or content parts.</summary>
/// <param name="Role">The conversation role associated with the message or delta.</param>
/// <param name="Content">The message text or JSON content parts.</param>
public sealed record ChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] JsonElement Content);

/// <summary>Represents a complete non-streaming OpenAI-compatible chat response.</summary>
/// <param name="Id">The unique identifier shared by the completion response and its streamed chunks.</param>
/// <param name="Created">The response creation time as Unix seconds.</param>
/// <param name="Model">The served model identifier included in the response.</param>
/// <param name="Choices">The ordered completion choices included in this response.</param>
/// <param name="Usage">The prompt and completion token usage for this response.</param>
public sealed record ChatCompletionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<ChatCompletionChoice> Choices,
    [property: JsonPropertyName("usage")] Usage Usage)
{
    /// <summary>Gets the fixed object discriminator for a complete chat completion.</summary>
    [JsonPropertyName("object")]
    public string Object { get; } = "chat.completion";
}

/// <summary>Represents one completed assistant message and its termination reason.</summary>
/// <param name="Index">The zero-based choice index.</param>
/// <param name="Message">The completed assistant message.</param>
/// <param name="FinishReason">The reason generation finished, or null while a streamed choice is still active.</param>
public sealed record ChatCompletionChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("message")] ChatMessageText Message,
    [property: JsonPropertyName("finish_reason")] string FinishReason);

/// <summary>Represents a text-only response message with a conversation role.</summary>
/// <param name="Role">The conversation role associated with the message or delta.</param>
/// <param name="Content">The text content of the response message.</param>
public sealed record ChatMessageText(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

/// <summary>Represents one OpenAI-compatible streamed chat completion event.</summary>
/// <param name="Id">The unique identifier shared by the completion response and its streamed chunks.</param>
/// <param name="Created">The response creation time as Unix seconds.</param>
/// <param name="Model">The served model identifier included in the response.</param>
/// <param name="Choices">The ordered completion choices included in this response.</param>
public sealed record ChatCompletionChunk(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] IReadOnlyList<ChatCompletionChunkChoice> Choices)
{
    /// <summary>Gets the fixed object discriminator for a streamed chat completion chunk.</summary>
    [JsonPropertyName("object")]
    public string Object { get; } = "chat.completion.chunk";
}

/// <summary>Represents one streamed choice delta and its optional termination reason.</summary>
/// <param name="Index">The zero-based choice index.</param>
/// <param name="Delta">The incremental role or text update for this streamed choice.</param>
/// <param name="FinishReason">The reason generation finished, or null while a streamed choice is still active.</param>
public sealed record ChatCompletionChunkChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("delta")] ChatDelta Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

/// <summary>Represents optional role and text updates within a streamed chat choice.</summary>
/// <param name="Role">The conversation role associated with the message or delta.</param>
/// <param name="Content">The incremental text, or null when the delta contains no text.</param>
public sealed record ChatDelta(
    [property: JsonPropertyName("role")] string? Role,
    [property: JsonPropertyName("content")] string? Content);

/// <summary>Reports prompt, completion, and combined token usage for a chat completion.</summary>
/// <param name="PromptTokens">The number of tokens used to encode the prompt.</param>
/// <param name="CompletionTokens">The number of tokens generated for the completion.</param>
/// <param name="TotalTokens">The sum of prompt and completion token counts.</param>
public sealed record Usage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);
