namespace SharpInference.WebApi;

public static class AnswerStateCacheKey
{
    public static bool TryCreate(string prompt, string answer, out string answeredPrefix) =>
        TryCreate(prompt, answer, out answeredPrefix, out _);

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
