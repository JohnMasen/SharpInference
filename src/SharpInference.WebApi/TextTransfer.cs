using System.Text;
using System.Text.Json;
using SharpInference;

namespace SharpInference.WebApi;

/// <summary>Transforms input into text suitable for subsequent prompt-processing stages.</summary>
/// <typeparam name="TInput">The type of input transformed into prompt text.</typeparam>
public interface ITextTransfer<in TInput>
{
    /// <summary>Converts the supplied input into prompt text.</summary>
    /// <param name="input">The input value to convert into prompt text.</param>
    string Transfer(TInput input);
}

/// <summary>Transforms input into text suitable for subsequent prompt-processing stages.</summary>
public interface ITextTransfer : ITextTransfer<string>
{
}

/// <summary>Converts input to text and applies an ordered sequence of additional text transformations.</summary>
/// <typeparam name="TInput">The type of input transformed into prompt text.</typeparam>
public sealed class TextTransferChain<TInput> : ITextTransfer<TInput>
{
    private readonly ITextTransfer<TInput> source;
    private readonly IReadOnlyList<ITextTransfer> transfers;

    /// <summary>Creates a transfer chain with an initial conversion and optional subsequent text transformations.</summary>
    /// <param name="source">The initial transformation from input values to text.</param>
    /// <param name="transfers">The optional ordered text transformations applied after the initial conversion.</param>
    public TextTransferChain(ITextTransfer<TInput> source, IEnumerable<ITextTransfer>? transfers = null)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));
        this.transfers = transfers?.ToArray() ?? [];
    }

    /// <inheritdoc/>
    public string Transfer(TInput input)
    {
        var text = source.Transfer(input);
        foreach (var transfer in transfers)
        {
            text = transfer.Transfer(text)
                ?? throw new InvalidOperationException(
                    $"Text transfer '{transfer.GetType().Name}' returned null.");
        }

        return text;
    }
}

/// <summary>Converts chat messages into a prompt for a compatible RWKV model architecture.</summary>
public interface IModelTextTransfer : ITextTransfer<IReadOnlyList<ChatMessage>>
{
    /// <summary>Checks whether this prompt transfer supports the supplied model metadata.</summary>
    /// <param name="metadata">The model metadata used to select or validate prompt formatting.</param>
    bool Supports(RwkvModelMetadata metadata);
}

/// <summary>Selects exactly one registered prompt transfer matching a model architecture.</summary>
public sealed class ModelTextTransferResolver
{
    private readonly IReadOnlyList<IModelTextTransfer> transfers;

    /// <summary>Registers a non-empty set of model prompt transfers.</summary>
    /// <param name="transfers">The registered model-specific prompt transfers.</param>
    public ModelTextTransferResolver(IEnumerable<IModelTextTransfer> transfers)
    {
        ArgumentNullException.ThrowIfNull(transfers);
        this.transfers = transfers.ToArray();
        if (this.transfers.Count == 0)
        {
            throw new InvalidOperationException("At least one model text transfer must be registered.");
        }
    }

    /// <summary>Returns the sole compatible transfer, rejecting unsupported or ambiguous architecture matches.</summary>
    /// <param name="metadata">The model metadata used to select or validate prompt formatting.</param>
    public IModelTextTransfer Resolve(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        var matches = transfers.Where(transfer => transfer.Supports(metadata)).ToArray();
        return matches.Length switch
        {
            1 => matches[0],
            0 => throw new NotSupportedException(
                $"No text transfer chain is registered for model architecture '{metadata.ArchitectureId}'."),
            _ => throw new InvalidOperationException(
                $"More than one text transfer chain matches model architecture '{metadata.ArchitectureId}'."),
        };
    }
}

/// <summary>Formats RWKV-6 World chat prompts through the registered text transfer chain.</summary>
public sealed class Rwkv6WorldTextTransfer : IModelTextTransfer
{
    private readonly TextTransferChain<IReadOnlyList<ChatMessage>> chain;

    /// <summary>Creates a text transfer using the RWKV-6 World chat template.</summary>
    public Rwkv6WorldTextTransfer()
    {
        chain = new TextTransferChain<IReadOnlyList<ChatMessage>>(
            new Rwkv6WorldChatTemplateTransfer());
    }

    /// <inheritdoc/>
    public bool Supports(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return string.Equals(metadata.ArchitectureId, "rwkv-6", StringComparison.Ordinal);
    }

    /// <inheritdoc/>
    public string Transfer(IReadOnlyList<ChatMessage> input) => chain.Transfer(input);
}

/// <summary>Formats RWKV-7 G1 prompts using the World-style chat template.</summary>
public sealed class Rwkv7G1TextTransfer : IModelTextTransfer
{
    private readonly Rwkv6WorldChatTemplateTransfer template = new("RWKV-7 G1");

    /// <summary>Gets the role-boundary strings that stop assistant generation.</summary>
    public static IReadOnlyList<string> StopStrings => Rwkv6WorldChatTemplateTransfer.StopStrings;

    /// <inheritdoc/>
    public bool Supports(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return string.Equals(metadata.ArchitectureId, "rwkv-7", StringComparison.Ordinal);
    }

    /// <inheritdoc/>
    public string Transfer(IReadOnlyList<ChatMessage> input) => template.Transfer(input);
}

/// <summary>Formats supported text chat messages as World-style role blocks ending with an assistant prompt.</summary>
public sealed class Rwkv6WorldChatTemplateTransfer : ITextTransfer<IReadOnlyList<ChatMessage>>
{
    private readonly string modelName;

    /// <summary>Creates a chat template transfer using the RWKV-6 World model name in validation errors.</summary>
    public Rwkv6WorldChatTemplateTransfer() : this("RWKV-6 World") { }

    internal Rwkv6WorldChatTemplateTransfer(string modelName) => this.modelName = modelName;

    /// <summary>Gets the user, system, and assistant role-boundary stop sequences.</summary>
    public static IReadOnlyList<string> StopStrings { get; } =
        ["\n\nUser:", "\n\nSystem:", "\n\nAssistant:"];

    /// <summary>Normalizes text messages, maps supported roles, and appends the assistant generation marker.</summary>
    /// <param name="messages">The ordered text conversation messages to format.</param>
    public string Transfer(IReadOnlyList<ChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("At least one chat message is required.", nameof(messages));
        }

        var prompt = new StringBuilder();
        foreach (var message in messages)
        {
            var role = message.Role switch
            {
                "system" or "developer" => "System",
                "user" => "User",
                "assistant" => "Assistant",
                _ => throw new ArgumentException(
                    $"The {modelName} chat template does not support the '{message.Role}' message role."),
            };

            prompt
                .Append(role)
                .Append(": ")
                .Append(NormalizeMessageContent(GetTextContent(message.Content)))
                .Append("\n\n");
        }

        prompt.Append("Assistant:");
        return prompt.ToString();
    }

    private string GetTextContent(JsonElement content)
    {
        if (content.ValueKind == JsonValueKind.String)
        {
            return content.GetString()!;
        }

        if (content.ValueKind != JsonValueKind.Array)
        {
            throw new ArgumentException(
                $"The {modelName} chat template supports string content or an array of text content parts.");
        }

        var text = new StringBuilder();
        foreach (var part in content.EnumerateArray())
        {
            if (part.ValueKind != JsonValueKind.Object ||
                !part.TryGetProperty("type", out var type) ||
                type.ValueKind != JsonValueKind.String ||
                type.GetString() is not ("text" or "input_text" or "output_text") ||
                !part.TryGetProperty("text", out var value) ||
                value.ValueKind != JsonValueKind.String)
            {
                throw new ArgumentException(
                    $"The {modelName} chat template supports only text content parts; image, audio, file, and tool content are not supported.");
            }

            text.Append(value.GetString());
        }

        return text.ToString();
    }

    internal static string NormalizeMessageContent(string content)
    {
        var normalized = content.ReplaceLineEndings("\n").Trim();
        var result = new StringBuilder(normalized.Length);
        var previousWasNewLine = false;
        foreach (var character in normalized)
        {
            if (character == '\n')
            {
                if (previousWasNewLine)
                {
                    continue;
                }

                previousWasNewLine = true;
            }
            else
            {
                previousWasNewLine = false;
            }

            result.Append(character);
        }

        return result.ToString();
    }
}

/// <summary>Carries canonical prompt text together with its tokenizer-produced token identifiers.</summary>
/// <param name="Text">The canonical prompt text produced by the chat template.</param>
/// <param name="TokenIds">The token identifiers obtained by encoding the prompt text.</param>
public sealed record EncodedChatPrompt(string Text, IReadOnlyList<int> TokenIds);
