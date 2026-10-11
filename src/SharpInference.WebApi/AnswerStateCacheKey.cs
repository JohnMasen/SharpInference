namespace SharpInference.WebApi;

/// <summary>Builds canonical answered-prompt prefixes suitable for reusing cached RWKV session state.</summary>
public static class AnswerStateCacheKey
{
    /// <summary>Attempts to append a normalized non-empty answer to an assistant prompt and identifies whether canonical-token replay is needed.</summary>
    /// <param name="prompt">The canonical prompt ending with the assistant marker.</param>
    /// <param name="answer">The visible assistant answer to normalize for caching.</param>
    /// <param name="answeredPrefix">Receives the canonical prompt and answer prefix, or an empty string when creation fails.</param>
    public static bool TryCreate(string prompt, string answer, out string answeredPrefix) =>
        TryCreate(prompt, answer, out answeredPrefix, out _);

    /// <summary>Attempts to append a normalized non-empty answer to an assistant prompt and identifies whether canonical-token replay is needed.</summary>
    /// <param name="prompt">The canonical prompt ending with the assistant marker.</param>
    /// <param name="answer">The visible assistant answer to normalize for caching.</param>
    /// <param name="answeredPrefix">Receives the canonical prompt and answer prefix, or an empty string when creation fails.</param>
    /// <param name="requiresReplay">Receives whether the visible answer differs from the canonical cached suffix and needs token replay.</param>
    public static bool TryCreate(string prompt, string answer, out string answeredPrefix, out bool requiresReplay)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(answer);
        answeredPrefix = string.Empty;
        requiresReplay = false;
        if (!prompt.EndsWith("Assistant:", StringComparison.Ordinal))
        {
            return false;
        }

        var normalizedAnswer = Rwkv6WorldChatTemplateTransfer.NormalizeMessageContent(answer);
        if (normalizedAnswer.Length == 0)
        {
            return false;
        }

        var canonicalSuffix = " " + normalizedAnswer;
        answeredPrefix = prompt + canonicalSuffix;
        requiresReplay = !string.Equals(answer, canonicalSuffix, StringComparison.Ordinal);
        return true;
    }
}
