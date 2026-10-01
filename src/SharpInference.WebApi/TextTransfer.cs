using System.Text;
using System.Text.Json;
using SharpInference;

namespace SharpInference.WebApi;

public interface ITextTransfer<in TInput>
{
    string Transfer(TInput input);
}

public interface ITextTransfer : ITextTransfer<string>
{
}

public sealed class TextTransferChain<TInput> : ITextTransfer<TInput>
{
    private readonly ITextTransfer<TInput> source;
    private readonly IReadOnlyList<ITextTransfer> transfers;

    public TextTransferChain(ITextTransfer<TInput> source, IEnumerable<ITextTransfer>? transfers = null)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));
        this.transfers = transfers?.ToArray() ?? [];
    }

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

public interface IModelTextTransfer : ITextTransfer<IReadOnlyList<ChatMessage>>
{
    bool Supports(RwkvModelMetadata metadata);
}

public sealed class ModelTextTransferResolver
{
    private readonly IReadOnlyList<IModelTextTransfer> transfers;

    public ModelTextTransferResolver(IEnumerable<IModelTextTransfer> transfers)
    {
        ArgumentNullException.ThrowIfNull(transfers);
        this.transfers = transfers.ToArray();
        if (this.transfers.Count == 0)
        {
            throw new InvalidOperationException("At least one model text transfer must be registered.");
        }
    }

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

public sealed class Rwkv6WorldTextTransfer : IModelTextTransfer
{
    private readonly TextTransferChain<IReadOnlyList<ChatMessage>> chain;

    public Rwkv6WorldTextTransfer()
    {
        chain = new TextTransferChain<IReadOnlyList<ChatMessage>>(
            new Rwkv6WorldChatTemplateTransfer());
    }

    public bool Supports(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return string.Equals(metadata.ArchitectureId, "rwkv-6", StringComparison.Ordinal);
    }

    public string Transfer(IReadOnlyList<ChatMessage> input) => chain.Transfer(input);
}

public sealed class Rwkv7G1TextTransfer : IModelTextTransfer
{
    private readonly Rwkv6WorldChatTemplateTransfer template = new("RWKV-7 G1");

    public static IReadOnlyList<string> StopStrings => Rwkv6WorldChatTemplateTransfer.StopStrings;

    public bool Supports(RwkvModelMetadata metadata)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        return string.Equals(metadata.ArchitectureId, "rwkv-7", StringComparison.Ordinal);
    }

    public string Transfer(IReadOnlyList<ChatMessage> input) => template.Transfer(input);
}

public sealed class Rwkv6WorldChatTemplateTransfer : ITextTransfer<IReadOnlyList<ChatMessage>>
{
    private readonly string modelName;

    public Rwkv6WorldChatTemplateTransfer() : this("RWKV-6 World") { }

    internal Rwkv6WorldChatTemplateTransfer(string modelName) => this.modelName = modelName;

    public static IReadOnlyList<string> StopStrings { get; } =
        ["\n\nUser:", "\n\nSystem:", "\n\nAssistant:"];

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

public sealed record EncodedChatPrompt(string Text, IReadOnlyList<int> TokenIds);
